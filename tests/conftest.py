"""Synthetic serving bundle for fast tests (no model downloads, no H&M data).

Four visual "styles" (clusters) x 3 colours; items in the same style have close visual and
taste vectors, so personalisation behaviour can be asserted deterministically.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from recsys import Bundle, RecConfig, Recommender  # noqa: E402
from recsys.events import EventLog  # noqa: E402

STYLES = ["Trousers", "Dress", "T-shirt", "Jacket"]
COLOURS = ["Black", "White", "Red"]
N_PER_STYLE = 60


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("models")
    rows, taste, visual = [], [], []
    taste_centres = _unit(rng.standard_normal((len(STYLES), 16)))
    visual_centres = _unit(rng.standard_normal((len(STYLES), 32)))
    colour_dirs = _unit(rng.standard_normal((len(COLOURS), 32)))
    for s, style in enumerate(STYLES):
        for i in range(N_PER_STYLE):
            c = i % len(COLOURS)
            aid = 100_000_000 + s * 1000 + i
            rows.append({
                "article_id": aid, "product_code": aid // 3,   # every 3 items share a product code
                "prod_name": f"{COLOURS[c]} {style} {i}", "product_type_name": style,
                "product_group_name": "Garment", "graphical_appearance_name": "Solid",
                "colour_group_name": COLOURS[c], "perceived_colour_master_name": COLOURS[c],
                "index_group_name": "Menswear" if s % 2 else "Ladieswear", "section_name": "S",
                "garment_group_name": "G", "detail_desc": f"A {COLOURS[c].lower()} {style.lower()}",
                "recent_sales": int(rng.integers(3, 100)),
            })
            taste.append(taste_centres[s] + 0.25 * rng.standard_normal(16))
            visual.append(visual_centres[s] + 0.6 * colour_dirs[c] + 0.15 * rng.standard_normal(32))
    taste, visual = _unit(np.array(taste)), _unit(np.array(visual))
    cat = pd.DataFrame(rows)
    cat.to_parquet(d / "catalog.parquet", index=False)
    np.save(d / "taste_vectors.npy", taste)
    np.save(d / "visual_vectors.npy", visual)
    sales = np.log1p(cat["recent_sales"].to_numpy())
    np.save(d / "popularity.npy", ((sales - sales.mean()) / sales.std()).astype(np.float32))
    np.save(d / "prototypes.npy", np.arange(0, len(cat), 7))
    a, b = rng.integers(0, len(cat), 5000), rng.integers(0, len(cat), 5000)
    tc = (taste[a] * taste[b]).sum(1)
    sets = visual[rng.integers(0, len(cat), (1000, 5))]
    vm = np.einsum("id,ijd->ij", visual[a[:1000]], sets).max(1)
    cen = sets.mean(1)
    cc = (visual[a[:1000]] * (cen / np.linalg.norm(cen, axis=1, keepdims=True))).sum(1)
    (d / "manifest.json").write_text(json.dumps({
        "items": len(cat), "visual_encoder": {"model": "synthetic"}, "two_tower": {"dim": 16, "max_len": 20},
        "feature_stats": {"taste_cos": {"mean": float(tc.mean()), "std": float(tc.std())},
                          "visual_max_cos": {"mean": float(vm.mean()), "std": float(vm.std())},
                          "visual_centroid_cos": {"mean": float(cc.mean()), "std": float(cc.std())}},
    }))
    np.save(d / "_colour_dirs.npy", colour_dirs)   # used by FakeText
    return d


class FakeText:
    """'red' -> the red colour direction in visual space; anything else -> random."""

    def __init__(self, colour_dirs):
        self.dirs = dict(zip([c.lower() for c in COLOURS], colour_dirs))

    def encode(self, text):
        for name, v in self.dirs.items():
            if name in text.lower():
                return v
        v = np.random.default_rng(len(text)).standard_normal(len(next(iter(self.dirs.values()))))
        return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture(scope="session")
def bundle(bundle_dir):
    return Bundle(bundle_dir, load_user_tower=False)


@pytest.fixture()
def recommender(bundle, bundle_dir):
    return Recommender(bundle, text_encoder=FakeText(np.load(bundle_dir / "_colour_dirs.npy")),
                       config=RecConfig(taste_pool=60, interest_pool=40, thompson_pool=40, steer_pool=60,
                                        explore_pool=20, trending_pool=20, rerank_pool=40))


@pytest.fixture()
def client(recommender, tmp_path):
    from app import create_app
    app = create_app(recommender, events=EventLog(tmp_path / "events"))
    app.config["TESTING"] = True
    return app.test_client()


def style_of(bundle, item):
    return bundle.record(bundle.row_of[item["id"]])["productType"]
