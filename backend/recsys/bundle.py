"""Serving artifacts produced by ml/build_artifacts.py (backend/models/)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
META_COLUMNS = ["article_id", "product_code", "prod_name", "product_type_name", "product_group_name",
                "graphical_appearance_name", "colour_group_name", "perceived_colour_master_name",
                "index_group_name", "section_name", "garment_group_name", "detail_desc"]


class BundleUnavailable(RuntimeError):
    pass


class Bundle:
    """Catalog metadata + item vectors + FAISS indices + user tower, all aligned by row."""

    def __init__(self, models_dir: Path | str = DEFAULT_MODELS_DIR, load_user_tower=True):
        d = Path(models_dir)
        required = ["manifest.json", "catalog.parquet", "taste_vectors.npy", "visual_vectors.npy",
                    "popularity.npy", "prototypes.npy"]
        missing = [f for f in required if not (d / f).exists()]
        if missing:
            raise BundleUnavailable(f"missing {missing} in {d}; run the ml/ pipeline (see README)")
        self.dir = d
        self.manifest = json.loads((d / "manifest.json").read_text())
        self.catalog = pd.read_parquet(d / "catalog.parquet").reset_index(drop=True)
        self.taste = np.load(d / "taste_vectors.npy").astype(np.float32)
        self.visual = np.load(d / "visual_vectors.npy").astype(np.float32)
        self.popularity_z = np.load(d / "popularity.npy").astype(np.float32)
        self.prototypes = np.load(d / "prototypes.npy").astype(np.int64)
        n = len(self.catalog)
        for name, arr in [("taste", self.taste), ("visual", self.visual), ("popularity", self.popularity_z)]:
            if len(arr) != n:
                raise BundleUnavailable(f"{name} has {len(arr)} rows but catalog has {n}")

        self.taste_index = self._index(d / "faiss_taste.index", self.taste)
        self.visual_index = self._index(d / "faiss_visual.index", self.visual)
        self.article_ids = self.catalog["article_id"].to_numpy(dtype=np.int64)
        self.row_of = {int(a): i for i, a in enumerate(self.article_ids)}
        self.product_codes = self.catalog["product_code"].to_numpy(dtype=np.int64)
        self.index_groups = self.catalog["index_group_name"].to_numpy()
        self.stats = self.manifest["feature_stats"]
        self._meta = self.catalog[META_COLUMNS].to_dict("records")
        self._records = [self._record(r) for r in self._meta]

        self.user_tower = None
        if load_user_tower and (d / "user_tower.pt").exists():
            import torch

            from .two_tower import UserTower
            cfg = self.manifest["two_tower"]
            self.user_tower = UserTower(dim=cfg["dim"], max_len=cfg["max_len"])
            self.user_tower.load_state_dict(torch.load(d / "user_tower.pt", map_location="cpu"))
            self.user_tower.eval()
        log.info("Bundle loaded: %d items, taste dim %d, visual dim %d, user tower %s",
                 n, self.taste.shape[1], self.visual.shape[1], "on" if self.user_tower else "off")

    @staticmethod
    def _index(path: Path, vectors: np.ndarray):
        if path.exists():
            index = faiss.read_index(str(path))
        else:
            log.warning("%s missing; building in memory", path.name)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        if index.ntotal != len(vectors) or index.d != vectors.shape[1]:
            raise BundleUnavailable(f"{path.name} does not match its vectors; rerun ml/build_artifacts.py")
        return index

    @staticmethod
    def _record(r: dict) -> dict:
        aid = int(r["article_id"])
        return {
            "id": aid,
            "productCode": int(r["product_code"]),
            "name": r["prod_name"],
            "productType": r["product_type_name"],
            "productGroup": r["product_group_name"],
            "colour": r["colour_group_name"],
            "colourFamily": r["perceived_colour_master_name"],
            "pattern": r["graphical_appearance_name"],
            "department": r["index_group_name"],
            "section": r["section_name"],
            "garmentGroup": r["garment_group_name"],
            "description": r["detail_desc"] or "",
            "image": f"/images/{aid:010d}.jpg",
        }

    def __len__(self):
        return len(self.article_ids)

    def record(self, row: int) -> dict:
        return dict(self._records[row])

    def meta(self, row: int) -> dict:
        return self._meta[row]

    def search(self, index, query: np.ndarray, k: int):
        k = min(k, index.ntotal)
        scores, rows = index.search(np.ascontiguousarray(query, dtype=np.float32)[None, :], k)
        return [(int(r), float(s)) for r, s in zip(rows[0], scores[0]) if r >= 0]
