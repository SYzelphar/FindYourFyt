"""Assemble the serving bundle (backend/models/) from the offline outputs.

    catalog.parquet       item metadata, row order = every vector file below
    taste_vectors.npy     two-tower item vectors (128-d)       + faiss_taste.index  (inner product)
    visual_vectors.npy    FashionCLIP image vectors (512-d)    + faiss_visual.index (inner product)
    popularity.npy        z-scored log(recent sales)
    prototypes.npy        onboarding "style prototypes": spherical k-means (k=64) on visual vectors,
                          the 3 most-sold items closest to each centroid
    user_tower.pt         two-tower user tower (runs on liked items at serving time)
    manifest.json         dims, model names, feature calibration statistics

Usage:
    python ml/build_artifacts.py
"""
import json
import shutil

import faiss
import numpy as np
import polars as pl

import config as C

K_PROTOTYPES = 64
PER_PROTOTYPE = 3


def main():
    rng = np.random.default_rng(C.SEED)
    out = C.MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)

    catalog = pl.read_parquet(C.CATALOG).sort("item_idx")
    n = catalog.height
    taste = np.load(C.TWO_TOWER_DIR / "item_vectors.npy").astype(np.float32)
    visual = np.load(C.EMBEDDINGS_DIR / f"{C.PRIMARY_ENCODER}_image.npy").astype(np.float32)
    assert taste.shape[0] == visual.shape[0] == n, (taste.shape, visual.shape, n)

    for name, vecs in [("taste", taste), ("visual", visual)]:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, str(out / f"faiss_{name}.index"))
        np.save(out / f"{name}_vectors.npy", vecs)

    sales = np.log1p(catalog["recent_sales"].to_numpy().astype(np.float64))
    np.save(out / "popularity.npy", ((sales - sales.mean()) / sales.std()).astype(np.float32))

    km = faiss.Kmeans(visual.shape[1], K_PROTOTYPES, niter=30, spherical=True, seed=C.SEED)
    km.train(visual)
    index = faiss.IndexFlatIP(visual.shape[1])
    index.add(visual)
    _, near = index.search(km.centroids.astype(np.float32), 40)
    sold = catalog["recent_sales"].to_numpy().astype(np.int64)
    prototypes = []
    for rows in near:
        best = sorted(rows.tolist(), key=lambda r: -sold[r])[:PER_PROTOTYPE]
        prototypes.extend(best)
    prototypes = np.array(sorted(set(prototypes)), dtype=np.int64)
    np.save(out / "prototypes.npy", prototypes)

    # Calibration for the session model's scalar features (z-scores).
    a, b = rng.integers(0, n, 20_000), rng.integers(0, n, 20_000)
    taste_cos = (taste[a] * taste[b]).sum(1)
    q = rng.integers(0, n, 4_000)
    sets = rng.integers(0, n, (4_000, 5))
    visual_max = np.einsum("id,ijd->ij", visual[q], visual[sets]).max(1)
    centroids = visual[sets].mean(1)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_cos = (visual[q] * centroids).sum(1)
    stats = {
        "taste_cos": {"mean": float(taste_cos.mean()), "std": float(taste_cos.std())},
        "visual_max_cos": {"mean": float(visual_max.mean()), "std": float(visual_max.std())},
        "visual_centroid_cos": {"mean": float(centroid_cos.mean()), "std": float(centroid_cos.std())},
    }

    tt = json.loads((C.TWO_TOWER_DIR / "vocab.json").read_text())
    shutil.copy(C.TWO_TOWER_DIR / "user_tower.pt", out / "user_tower.pt")
    catalog.drop("item_idx").to_pandas().to_parquet(out / "catalog.parquet", index=False)
    emb_manifest = json.loads((C.EMBEDDINGS_DIR / "manifest.json").read_text())
    manifest = {
        "dataset": "H&M Personalized Fashion Recommendations (Kaggle)",
        "items": n,
        "visual_encoder": {"name": C.PRIMARY_ENCODER, **emb_manifest[C.PRIMARY_ENCODER]},
        "two_tower": {"dim": tt["dim"], "max_len": tt["max_len"]},
        "prototypes": int(len(prototypes)),
        "feature_stats": stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
