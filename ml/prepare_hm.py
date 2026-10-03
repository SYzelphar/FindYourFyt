"""Build the FindYourFit catalog and interaction data from the Kaggle H&M dataset.

Catalog ("current assortment"):
  1. product group is a garment (upper body / lower body / full body)
  2. not Baby/Children
  3. product image exists and decodes
  4. sold at least MIN_SALES times in the last ASSORTMENT_WEEKS weeks
  5. capped at MAX_ITEMS, most-sold first

Interactions: purchases of catalog items in the last HISTORY_WEEKS weeks, one row per
(customer, day, item), with a time-based split: train < VAL_START <= val < TEST_START <= test.

Outputs: data/processed/{catalog,transactions}.parquet, prepare_report.json and
web-sized images in data/images/<article_id>.jpg. Raw files are never modified.

Usage:
    python ml/prepare_hm.py [--workers 12]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import polars as pl
from PIL import Image

import config as C

ARTICLE_COLUMNS = [
    "article_id", "product_code", "prod_name", "product_type_name", "product_group_name",
    "graphical_appearance_name", "colour_group_name", "perceived_colour_value_name",
    "perceived_colour_master_name", "department_name", "index_name", "index_group_name",
    "section_name", "garment_group_name", "detail_desc",
]


def resize_image(article_id):
    """Write a web-sized copy. Returns (article_id, error or None)."""
    out = C.image_path(article_id)
    if out.exists():
        return article_id, None
    try:
        with Image.open(C.raw_image_path(article_id)) as im:
            im.draft("RGB", (C.WEB_IMAGE_MAX_SIDE, C.WEB_IMAGE_MAX_SIDE))  # fast JPEG downscale on decode
            im = im.convert("RGB")
            im.thumbnail((C.WEB_IMAGE_MAX_SIDE, C.WEB_IMAGE_MAX_SIDE), Image.LANCZOS)
            im.save(out, "JPEG", quality=88, optimize=True)
        return article_id, None
    except Exception as e:  # missing or corrupt file
        return article_id, type(e).__name__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}

    # ------------------------------------------------------------ candidate articles
    articles = pl.read_csv(C.ARTICLES_CSV, schema_overrides={"article_id": pl.Utf8, "product_code": pl.Utf8})
    counts["raw_articles"] = articles.height
    articles = articles.select(ARTICLE_COLUMNS).with_columns(
        pl.col("article_id").cast(pl.Int64), pl.col("product_code").cast(pl.Int64),
        pl.col("detail_desc").fill_null(""),
    )
    articles = articles.filter(pl.col("product_group_name").is_in(C.GARMENT_GROUPS))
    counts["garments"] = articles.height
    articles = articles.filter(~pl.col("index_group_name").is_in(C.EXCLUDED_INDEX_GROUPS))
    counts["adult_garments"] = articles.height
    articles = articles.filter(
        pl.col("article_id").map_elements(lambda a: C.raw_image_path(a).exists(), return_dtype=pl.Boolean))
    counts["with_image"] = articles.height

    # ------------------------------------------------------------ transactions window
    history_start = C.LAST_DAY - timedelta(weeks=C.HISTORY_WEEKS) + timedelta(days=1)
    assortment_start = C.LAST_DAY - timedelta(weeks=C.ASSORTMENT_WEEKS) + timedelta(days=1)
    tx = (
        pl.scan_csv(C.TRANSACTIONS_CSV, schema_overrides={"article_id": pl.Utf8, "customer_id": pl.Utf8},
                    try_parse_dates=True)
        .filter(pl.col("t_dat") >= history_start)
        .with_columns(pl.col("article_id").cast(pl.Int64))
        .collect()
    )
    counts["transactions_in_window_all_products"] = tx.height

    sales = (tx.filter(pl.col("t_dat") >= assortment_start)
             .group_by("article_id").agg(pl.len().alias("recent_sales")))
    catalog = (articles.join(sales, on="article_id", how="inner")
               .filter(pl.col("recent_sales") >= C.MIN_SALES)
               .sort(["recent_sales", "article_id"], descending=[True, False]))
    counts["in_current_assortment"] = catalog.height
    catalog = catalog.head(C.MAX_ITEMS)

    # ------------------------------------------------------------ images
    print(f"Resizing {catalog.height} images with {args.workers} workers ...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(resize_image, catalog["article_id"].to_list(), chunksize=64))
    failed = {aid: err for aid, err in results if err}
    catalog = catalog.filter(~pl.col("article_id").is_in(list(failed)))
    counts["image_failures"] = len(failed)
    counts["final_catalog"] = catalog.height

    catalog = catalog.sort("article_id").with_row_index("item_idx").with_columns(pl.col("item_idx").cast(pl.Int32))
    catalog.write_parquet(C.CATALOG)

    # ------------------------------------------------------------ interactions on catalog items
    tx = (tx.join(catalog.select("article_id", "item_idx"), on="article_id", how="inner")
            .unique(subset=["customer_id", "t_dat", "article_id"], keep="first")   # quantity duplicates
            .sort(["customer_id", "t_dat"]))
    cust = tx.select("customer_id").unique(maintain_order=True).with_row_index("customer_idx")
    tx = (tx.join(cust, on="customer_id")
            .with_columns(
                pl.col("customer_idx").cast(pl.Int32),
                pl.when(pl.col("t_dat") >= C.TEST_START).then(pl.lit("test"))
                  .when(pl.col("t_dat") >= C.VAL_START).then(pl.lit("val"))
                  .otherwise(pl.lit("train")).alias("split"))
            .select("t_dat", "customer_idx", "item_idx", "article_id", "price", "sales_channel_id", "split"))
    tx.write_parquet(C.TRANSACTIONS)

    split_counts = tx.group_by("split").agg(pl.len().alias("rows"),
                                            pl.col("customer_idx").n_unique().alias("customers"))
    report = {
        "counts": counts,
        "history_window": [str(history_start), str(C.LAST_DAY)],
        "assortment_window": [str(assortment_start), str(C.LAST_DAY)],
        "interactions": tx.height,
        "customers": cust.height,
        "splits": {r["split"]: {"rows": r["rows"], "customers": r["customers"]} for r in split_counts.to_dicts()},
        "index_groups": dict(catalog.group_by("index_group_name").len().iter_rows()),
        "product_types": dict(catalog.group_by("product_type_name").len().sort("len", descending=True).iter_rows()),
        "image_failures": failed,
    }
    C.PREPARE_REPORT.write_text(json.dumps(report, indent=2, default=str))

    print("\nCatalog")
    for k, v in counts.items():
        print(f"  {k:<38}{v:,}")
    print(f"\nInteractions: {tx.height:,} rows, {cust.height:,} customers")
    for s, d in sorted(report["splits"].items()):
        print(f"  {s:<6}{d['rows']:>12,} rows {d['customers']:>10,} customers")
    print(f"\nDone in {time.time() - t0:.0f}s -> {C.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
