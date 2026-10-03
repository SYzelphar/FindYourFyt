"""Tune the session model's hyperparameters with simulated swipe sessions.

Uses a different set of simulated users (seed 7) from the final evaluation (seed 42), so the
numbers reported in reports/eval.md are not tuned on their own test users.

Stages (each writes reports/tuning_<stage>.json):
  exploration  prior variance of the taste weights x UCB bonus before the first like
  priors       prior means of the two-tower and looks-like-liked weights x taste variance
  thompson     temperature of the Thompson posterior (1 = standard, 0 = greedy). Selection rule,
               fixed before running: highest like rate whose catalog coverage is >= 1.5x greedy's
  centroid     prior variance of the scalar weights (Thompson noise on z-scored features) x prior
               means of the fits-a-style and looks-like-liked weights, with the v1 algorithm on
               the same users as a reference

Usage:
    python ml/tune_session.py --stage centroid [--users 80] [--swipes 30]
"""
import argparse
import json
import sys

import polars as pl

import config as C
from evaluate import simulate

sys.path.insert(0, str(C.ROOT / "backend"))
from recsys import Bundle, RecConfig  # noqa: E402
from recsys.session_model import Prior  # noqa: E402

TUNING_SEED = 7


def grid(stage):
    if stage == "exploration":
        return {f"taste_var={tv} ucb={k}": RecConfig(prior=Prior(taste_var=tv), explore_bonus=k)
                for tv in (1.0, 0.3, 0.1) for k in (0.0, 1.0, 2.0)}
    if stage == "thompson":
        return {f"temperature={t}": RecConfig(thompson_temperature=t) for t in (1.0, 0.5, 0.25, 0.1, 0.0)}
    if stage == "centroid":
        out = {"v1 algorithm on FashionCLIP (reference)": ("v1", None)}
        for sv in (0.5, 0.1, 0.03):
            for mi in (0.0, 1.0):
                for ll in (0.4, 0.8):
                    prior = Prior(taste_var=0.1, scalar_var=sv, scalar_means=(0.3, ll, -0.8, 0.2, mi))
                    out[f"scalar_var={sv} fits_style={mi} looks_like_liked={ll}"] = RecConfig(prior=prior)
        return out
    out = {}
    for tt in (0.0, 0.3, 0.6, 1.0):
        for ll in (0.8, 1.5):
            for tv in (0.1, 0.3):
                prior = Prior(taste_var=tv, scalar_means=(tt, ll, -0.8, 0.2, 0.0))
                out[f"two_tower={tt} looks_like_liked={ll} taste_var={tv}"] = RecConfig(prior=prior)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["exploration", "priors", "centroid", "thompson"], default="thompson")
    ap.add_argument("--users", type=int, default=80)
    ap.add_argument("--swipes", type=int, default=30)
    args = ap.parse_args()
    bundle = Bundle(C.MODELS_DIR)
    catalog = pl.read_parquet(C.CATALOG).sort("item_idx")
    tx = pl.read_parquet(C.TRANSACTIONS)
    policies = grid(args.stage)
    for name, spec in policies.items():
        if isinstance(spec, tuple):
            policies[name] = ("v1", bundle.visual)
    res = simulate(bundle, catalog, tx, args.users, args.swipes, TUNING_SEED, policies=policies)
    if args.stage == "thompson":
        greedy_cov = res["policies"]["temperature=0.0"]["catalog_coverage"]
        ok = {n: r for n, r in res["policies"].items() if r["catalog_coverage"] >= 1.5 * greedy_cov}
        best = max(ok, key=lambda n: ok[n]["like_rate"]) if ok else None
        print(f"\nSelected by rule (coverage >= 1.5x greedy = {1.5 * greedy_cov:.3f}): {best}")
        res["selected"] = best
    ranked = sorted(res["policies"].items(), key=lambda kv: -kv[1]["like_rate"])
    print("\nBest by like rate:")
    for name, r in ranked[:5]:
        print(f"  {name}: like_rate={r['like_rate']:.3f} diversity={r['session_diversity']:.3f} "
              f"coverage={r['catalog_coverage']:.3f}")
    C.REPORTS_DIR.mkdir(exist_ok=True)
    (C.REPORTS_DIR / f"tuning_{args.stage}.json").write_text(json.dumps({"seed": TUNING_SEED, **res}, indent=2))


if __name__ == "__main__":
    main()
