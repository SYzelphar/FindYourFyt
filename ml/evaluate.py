"""Offline evaluation of FindYourFit v2.

1. Embedding probe (catalog metadata, no users)
   For each encoder: do an item's nearest visual neighbours (other products, not colour variants)
   share its product type / colour / pattern? Plus zero-shot text->image retrieval for the
   CLIP models, e.g. "black trousers" -> are the top 20 items black trousers? (measures steering).

2. Real held-out purchases (H&M customers, test week 2020-09-16..22)
   History = each customer's purchases before the test week (last 20 items). Predict the 12 new
   items they will buy next week. Recall@12, MAP@12 (the Kaggle competition metric), NDCG@12,
   HitRate@12, catalog coverage. Baselines: popularity, visual kNN (VGG16, CLIP, FashionCLIP).

3. Simulated swipe sessions (real customers as simulated users)
   A simulated user swipes right on a card if its (product type, colour family) pair occurs in
   that customer's real purchases, with 10% label noise. The rule uses metadata, which none of
   the image models see, so it isn't circular. Every policy plays 30 swipes per user through the
   same interface the app uses (the first card of each response is the one swiped).

Purchases are a proxy for preference and simulated users are not people: these are offline
estimates, not online A/B results.

Usage:
    python ml/evaluate.py [--customers 20000] [--sim-users 300] [--swipes 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import faiss
import numpy as np
import polars as pl
import torch

import config as C

sys.path.insert(0, str(C.ROOT / "backend"))
from recsys import Bundle, RecConfig, Recommender  # noqa: E402
from recsys.session_store import SessionState  # noqa: E402

K = 12


# ===================================================================== helpers
def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def ap_at_k(recs, targets, k=K):
    hits, score = 0, 0.0
    for i, r in enumerate(recs[:k]):
        if r in targets:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(targets), k)


def ndcg_at_k(recs, targets, k=K):
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(recs[:k]) if r in targets)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(targets), k)))
    return dcg / idcg


def diversity(V, rows):
    if len(rows) < 2:
        return None
    S = V[rows] @ V[rows].T
    n = len(rows)
    return float(1 - (S.sum() - n) / (n * (n - 1)))


# ===================================================================== 1. embedding probe
def embedding_probe(catalog: pl.DataFrame, rng, n_queries=2000):
    out = {}
    codes = catalog["product_code"].to_numpy()
    attrs = {a: catalog[a].to_numpy() for a in ("product_type_name", "colour_group_name", "graphical_appearance_name")}
    queries = rng.choice(catalog.height, size=n_queries, replace=False)
    combos = (catalog.group_by(["colour_group_name", "product_type_name"]).len()
              .filter(pl.col("len") >= 40)
              .sort(["len", "colour_group_name", "product_type_name"], descending=[True, False, False]).head(40))
    for name in C.ENCODERS:
        path = C.EMBEDDINGS_DIR / f"{name}_image.npy"
        if not path.exists():
            continue
        V = np.load(path)
        index = faiss.IndexFlatIP(V.shape[1])
        index.add(V)
        _, nn = index.search(V[queries], 60)
        res = {}
        for a, lab in attrs.items():
            p = []
            for q, row in zip(queries, nn):
                top = [r for r in row if codes[r] != codes[q]][:10]   # skip colour variants of itself
                p.append(np.mean(lab[top] == lab[q]))
            res[f"knn_precision@10_{a}"] = float(np.mean(p))
        text_path = C.EMBEDDINGS_DIR / f"{name}_text.npy"
        if text_path.exists() and name != "vgg16":
            from encoders import load_encoder
            enc = load_encoder(name)
            prompts = [f"a photo of {c.lower()} {t.lower()} clothing" for c, t in
                       zip(combos["colour_group_name"], combos["product_type_name"])]
            T = enc.encode_text(prompts)
            del enc
            torch.cuda.empty_cache()
            precs = []
            for (c, t), tv in zip(zip(combos["colour_group_name"], combos["product_type_name"]), T):
                top = np.argsort(-(V @ tv))[:20]
                precs.append(np.mean((attrs["colour_group_name"][top] == c) & (attrs["product_type_name"][top] == t)))
            res["text_to_image_precision@20"] = float(np.mean(precs))
            res["text_queries"] = len(prompts)
        out[name] = res
        print(f"  probe {name}: {json.dumps({k: round(v, 3) for k, v in res.items()})}")
    return out


# ===================================================================== 2. real purchases
def purchase_eval(catalog, tx, bundle: Bundle, n_customers, seed):
    hist_tx = tx.filter(pl.col("split") != "test").sort("t_dat")
    hist = (hist_tx.group_by("customer_idx", maintain_order=True)
            .agg(pl.col("item_idx").reverse().unique(maintain_order=True).head(20).alias("hist")))
    tgt = tx.filter(pl.col("split") == "test").group_by("customer_idx").agg(pl.col("item_idx").unique().alias("t"))
    df = hist.join(tgt, on="customer_idx").sort("customer_idx")   # deterministic order before sampling
    df = df.sample(min(n_customers, df.height), seed=seed)
    H = [list(h) for h in df["hist"].to_list()]
    T = [set(t) - set(h) for t, h in zip(df["t"].to_list(), H)]
    keep = [i for i, t in enumerate(T) if t]
    H, T = [H[i] for i in keep], [T[i] for i in keep]
    print(f"  {len(H):,} test customers with new purchases")

    last_week_pop = (tx.filter(pl.col("split") == "val").group_by("item_idx").len().sort("len", descending=True))
    pop_order = last_week_pop["item_idx"].to_list()
    n_items = catalog.height

    def rank_by_vectors(V, user_vecs):
        out = []
        for i in range(0, len(user_vecs), 2048):
            S = torch.as_tensor(user_vecs[i:i + 2048], device="cuda") @ torch.as_tensor(V, device="cuda").T
            for j, h in enumerate(H[i:i + 2048]):
                S[j, h] = float("-inf")
            out.extend(S.topk(K, dim=1).indices.cpu().numpy().tolist())
        return out

    methods = {}
    def top_unseen(h):
        hs, out = set(h), []
        for r in pop_order:
            if r not in hs:
                out.append(r)
                if len(out) == K:
                    break
        return out

    methods["popularity (last week)"] = [top_unseen(h) for h in H]
    for name in C.ENCODERS:
        p = C.EMBEDDINGS_DIR / f"{name}_image.npy"
        if p.exists():
            V = np.load(p)
            w = lambda n: 0.85 ** np.arange(n)   # noqa: E731 - recency weights, most recent first
            U = l2n(np.stack([(V[h] * w(len(h))[:, None]).sum(0) for h in H])).astype(np.float32)
            methods[f"visual kNN ({name})"] = rank_by_vectors(V, U)
    from recsys.two_tower import user_vector
    U = np.stack([user_vector(bundle.user_tower, bundle.taste, list(reversed(h))) for h in H]).astype(np.float32)
    methods["two-tower (FashionCLIP + purchases)"] = rank_by_vectors(bundle.taste, U)

    visual = bundle.visual
    results = {}
    for name, recs in methods.items():
        r = {
            "recall@12": float(np.mean([len(set(x) & t) / len(t) for x, t in zip(recs, T)])),
            "map@12": float(np.mean([ap_at_k(x, t) for x, t in zip(recs, T)])),
            "ndcg@12": float(np.mean([ndcg_at_k(x, t) for x, t in zip(recs, T)])),
            "hit_rate@12": float(np.mean([bool(set(x) & t) for x, t in zip(recs, T)])),
            "coverage": len({r for x in recs for r in x}) / n_items,
            "diversity": float(np.mean([diversity(visual, x[:K]) for x in recs[:3000]])),
        }
        results[name] = r
        print(f"  {name:<38} " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()))
    return {"customers": len(H), "methods": results}


# ===================================================================== 3. simulated sessions
class V1MeanVector:
    """The v1 algorithm: query = norm(mean(liked) - 0.5 mean(disliked)); random cards until a like."""

    def __init__(self, V, rng):
        self.V, self.rng = V, rng
        self.index = faiss.IndexFlatIP(V.shape[1])
        self.index.add(V)

    def next(self, liked, disliked, seen):
        if not liked:
            while True:
                r = int(self.rng.integers(len(self.V)))
                if r not in seen:
                    return r
        q = self.V[liked].mean(0) - (0.5 * self.V[disliked].mean(0) if disliked else 0)
        _, rows = self.index.search(l2n(q)[None].astype(np.float32), len(seen) + 10)
        return next(int(r) for r in rows[0] if r not in seen)


def default_policies(bundle):
    return {
        "random": None,
        "popularity": None,
        "v1: VGG16 mean vector": ("v1", np.load(C.EMBEDDINGS_DIR / "vgg16_image.npy")),
        "v1 algorithm on FashionCLIP": ("v1", bundle.visual),
        "v2 full": RecConfig(),
        "v2 greedy (no Thompson)": RecConfig(use_thompson=False),
        "v2 standard Thompson (temperature 1)": RecConfig(thompson_temperature=1.0),
        "v2 without two-tower": RecConfig(use_two_tower=False),
        "v2 without visual kNN features": RecConfig(use_visual_knn=False),
        "v2 single interest": RecConfig(use_interests=False),
        "v2 without UCB onboarding": RecConfig(explore_bonus=0.0),
    }


def simulate(bundle: Bundle, catalog, tx, n_users, swipes, seed, policies=None):
    ptype = catalog["product_type_name"].to_numpy()
    colour = catalog["perceived_colour_master_name"].to_numpy()
    pairs = np.array([f"{a}|{b}" for a, b in zip(ptype, colour)])
    per_cust = (tx.group_by("customer_idx").agg(pl.col("item_idx").unique().alias("items"))
                .filter(pl.col("items").list.len().is_between(8, 60)))
    test_custs = tx.filter(pl.col("split") == "test")["customer_idx"].unique()
    per_cust = per_cust.filter(pl.col("customer_idx").is_in(test_custs.implode())).sort("customer_idx")
    users = per_cust.sample(n_users, seed=seed)["items"].to_list()   # sorted first: group_by order is random
    users = [sorted(items) for items in users]
    tastes = [set(pairs[items]) for items in users]
    prevalence = float(np.mean([np.isin(pairs, list(t)).mean() for t in tastes]))
    print(f"  {len(users)} simulated users; mean share of catalog they'd like: {prevalence:.3f}")

    def oracle(taste, row, r):
        like = pairs[row] in taste
        return (not like) if r.random() < 0.10 else like

    policies = policies or default_policies(bundle)
    trending = np.argsort(-bundle.popularity_z)
    results, curves, per_user_rates = {}, {}, {}
    for name, spec in policies.items():
        t0 = time.time()
        like_matrix, first_likes, shown_all, divs = [], [], set(), []
        for u, taste in enumerate(tastes):
            r = np.random.default_rng([seed, u])
            liked, disliked, seen, likes = [], [], set(), []
            if isinstance(spec, RecConfig):
                rec = Recommender(bundle, text_encoder=None, config=spec)
                state = SessionState(session_id=f"sim{u}", seed=seed + u)
            elif isinstance(spec, tuple):
                v1 = V1MeanVector(spec[1], r)
            for _ in range(swipes):
                if spec is None and name == "random":
                    row = int(r.integers(len(pairs)))
                    while row in seen:
                        row = int(r.integers(len(pairs)))
                elif spec is None:
                    row = int(next(x for x in trending if x not in seen))
                elif isinstance(spec, tuple):
                    row = v1.next(liked, disliked, seen)
                else:
                    items, _ = rec.recommend(state, k=5)
                    row = bundle.row_of[items[0]["id"]]
                y = oracle(taste, row, r)
                seen.add(row)
                (liked if y else disliked).append(row)
                likes.append(y)
                if isinstance(spec, RecConfig):
                    rec.record_swipe(state, row, y)
            like_matrix.append(likes)
            first_likes.append(next((i for i, x in enumerate(likes) if x), None))
            shown_all |= seen
            divs.append(diversity(bundle.visual, list(seen)))
        L = np.array(like_matrix, dtype=float)
        fl = [x for x in first_likes if x is not None]
        per_user = L.mean(1)
        results[name] = {
            "like_rate": float(L.mean()),
            "like_rate_se": float(per_user.std(ddof=1) / np.sqrt(len(per_user))),
            "like_rate_swipes_11_30": float(L[:, 10:].mean()),
            "median_swipes_to_first_like": float(np.median(fl)) + 1 if fl else None,
            "sessions_without_like": int(sum(x is None for x in first_likes)),
            "session_diversity": float(np.mean([d for d in divs if d is not None])),
            "catalog_coverage": len(shown_all) / len(pairs),
            "seconds": round(time.time() - t0, 1),
        }
        curves[name] = L.mean(0).tolist()
        per_user_rates[name] = per_user
        print(f"  {name:<34} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in results[name].items()))
    # Paired comparison: every policy sees the same simulated users, so per-user differences
    # cancel most of the between-user variance.
    paired = {}
    base = "v1 algorithm on FashionCLIP"
    if base in per_user_rates:
        for name, rates in per_user_rates.items():
            if name.startswith("v2"):
                d = rates - per_user_rates[base]
                se = d.std(ddof=1) / np.sqrt(len(d))
                paired[name] = {"vs": base, "diff": float(d.mean()),
                                "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}
    return {"users": len(users), "swipes": swipes, "label_noise": 0.10, "mean_like_prevalence": prevalence,
            "policies": results, "paired_vs_v1": paired, "curves": curves}


# ===================================================================== plots
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"v2 full": "#2a78d6", "v1 algorithm on FashionCLIP": "#eb6834", "v1: VGG16 mean vector": "#1baf7a",
          "random": MUTED}


def _style(ax, fig):
    import matplotlib.pyplot as plt  # noqa: F401
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(sim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    _style(ax, fig)
    window = 5
    for name, color in SERIES.items():
        y = np.array(sim["curves"][name])
        smooth = np.convolve(y, np.ones(window) / window, mode="valid")
        x = np.arange(window, len(y) + 1)
        ax.plot(x, smooth, color=color, linewidth=2, solid_capstyle="round")
        ax.annotate(name, (x[-1], smooth[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel("Swipe number", color=INK2, fontsize=9)
    ax.set_ylabel("Share of cards liked (5-swipe moving average)", color=INK2, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=sim["swipes"] + 9)
    ax.set_title("Simulated swipe sessions: how fast each policy learns", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_retrieval(purchase, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = sorted(purchase["methods"].items(), key=lambda kv: kv[1]["recall@12"])
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=150)
    _style(ax, fig)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    names = [n for n, _ in items]
    vals = [m["recall@12"] for _, m in items]
    ax.barh(names, vals, color="#2a78d6", height=0.6)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Recall@12 on real next-week purchases (higher is better)", color=INK2, fontsize=9)
    ax.set_title(f"Predicting H&M customers' purchases ({purchase['customers']:,} customers)",
                 loc="left", color=INK, fontsize=11)
    ax.set_xlim(right=max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def markdown(report) -> str:
    lines = ["# FindYourFit evaluation", "", f"_Generated by `ml/evaluate.py` (seed {report['seed']})._", ""]
    lines += ["## Embedding probe", "",
              "| Encoder | kNN P@10 type | kNN P@10 colour | kNN P@10 pattern | text→image P@20 |",
              "|---|---|---|---|---|"]
    for n, r in report["embedding_probe"].items():
        t = r.get("text_to_image_precision@20")
        lines.append(f"| {n} | {r['knn_precision@10_product_type_name']:.3f} "
                     f"| {r['knn_precision@10_colour_group_name']:.3f} "
                     f"| {r['knn_precision@10_graphical_appearance_name']:.3f} | {'–' if t is None else f'{t:.3f}'} |")
    p = report["purchases"]
    lines += ["", f"## Real next-week purchases ({p['customers']:,} customers)", "",
              "| Method | Recall@12 | MAP@12 | NDCG@12 | HitRate@12 | Coverage |", "|---|---|---|---|---|---|"]
    for n, r in sorted(p["methods"].items(), key=lambda kv: kv[1]["recall@12"]):
        lines.append(f"| {n} | {r['recall@12']:.4f} | {r['map@12']:.4f} | {r['ndcg@12']:.4f} | "
                     f"{r['hit_rate@12']:.4f} | {r['coverage']:.1%} |")
    s = report["simulation"]
    lines += ["", f"## Simulated swipe sessions ({s['users']} users × {s['swipes']} swipes, "
              f"{s['label_noise']:.0%} label noise)", "",
              "| Policy | Like rate ± s.e. | Like rate, swipes 11–30 | Swipes to first like (median) "
              "| No-like sessions | Diversity | Coverage |",
              "|---|---|---|---|---|---|---|"]
    for n, r in s["policies"].items():
        lines.append(f"| {n} | {r['like_rate']:.3f} ± {r['like_rate_se']:.3f} | {r['like_rate_swipes_11_30']:.3f} | "
                     f"{r['median_swipes_to_first_like']} | {r['sessions_without_like']} | "
                     f"{r['session_diversity']:.3f} | {r['catalog_coverage']:.1%} |")
    if s.get("paired_vs_v1"):
        lines += ["", "Paired difference in like rate vs the v1 algorithm on FashionCLIP (same users, 95% CI):", ""]
        for n, r in s["paired_vs_v1"].items():
            lines.append(f"- {n}: {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=20_000)
    ap.add_argument("--sim-users", type=int, default=300)
    ap.add_argument("--swipes", type=int, default=30)
    ap.add_argument("--skip", nargs="*", default=[], choices=["probe", "purchases", "simulation"])
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    C.REPORTS_DIR.mkdir(exist_ok=True)
    catalog = pl.read_parquet(C.CATALOG).sort("item_idx")
    tx = pl.read_parquet(C.TRANSACTIONS)
    bundle = Bundle(C.MODELS_DIR)
    out_json = C.REPORTS_DIR / "eval.json"
    report = json.loads(out_json.read_text()) if out_json.exists() else {}
    report["seed"] = C.SEED
    report["disclaimer"] = ("Offline estimates: purchases are a proxy for preference and simulated users "
                            "are not people. No online A/B test has been run.")
    if "probe" not in args.skip:
        print("Embedding probe")
        report["embedding_probe"] = embedding_probe(catalog, rng)
    if "purchases" not in args.skip:
        print("Real purchases")
        report["purchases"] = purchase_eval(catalog, tx, bundle, args.customers, C.SEED)
    if "simulation" not in args.skip:
        print("Simulated sessions")
        report["simulation"] = simulate(bundle, catalog, tx, args.sim_users, args.swipes, C.SEED)
    out_json.write_text(json.dumps(report, indent=2))
    (C.REPORTS_DIR / "eval.md").write_text(markdown(report), encoding="utf-8")
    plot_curves(report["simulation"], C.REPORTS_DIR / "swipe_learning_curve.png")
    plot_retrieval(report["purchases"], C.REPORTS_DIR / "purchase_recall.png")
    print(f"\nWrote reports/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
