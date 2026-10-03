"""Multi-interest user modelling (PinnerSage-style).

A single average of everything you liked blurs distinct tastes together (black trousers +
floral dresses -> a muddy midpoint). Instead, liked items are clustered in FashionCLIP visual
space with average-linkage agglomerative clustering; every cluster is an "interest" with
  - a medoid (the most central liked item, used as a retrieval query),
  - a centroid (recency-weighted mean of its members, used for the "fits your style" feature),
  - a weight (recency-weighted number of likes),
  - a human-readable label from the members' metadata (e.g. "Black trousers").
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering


@dataclass
class Interest:
    rows: list
    medoid: int
    centroid: np.ndarray
    weight: float
    label: str


def label_for(meta_rows: list[dict], colour_share=0.6) -> str:
    def dominant(key, share):
        values = Counter(m[key] for m in meta_rows)
        value, n = values.most_common(1)[0]
        return value if n / len(meta_rows) >= share else None

    ptype = dominant("product_type_name", 0.0)
    colour = dominant("colour_group_name", colour_share)
    pattern = dominant("graphical_appearance_name", 0.6)
    words = []
    if colour:
        words.append(colour)
    if pattern and pattern not in ("Solid", "Unknown"):
        words.append(pattern.lower())
    words.append(ptype.lower())
    label = " ".join(words)
    return label[0].upper() + label[1:]


def find_interests(liked_rows: list[int], visual: np.ndarray, meta, max_interests=4,
                   distance_threshold=0.22, recency_decay=0.92) -> list[Interest]:
    """liked_rows oldest -> newest. meta(row) -> dict of attributes."""
    if not liked_rows:
        return []
    rows = list(dict.fromkeys(liked_rows))
    n = len(rows)
    recency = {r: recency_decay ** (n - 1 - i) for i, r in enumerate(rows)}
    V = visual[rows]
    if n == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
                                         metric="cosine", linkage="average").fit_predict(V)
    interests = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        members = [rows[i] for i in idx]
        sims = V[idx] @ V[idx].T
        medoid = members[int(sims.mean(axis=1).argmax())]
        w = np.array([recency[r] for r in members], dtype=np.float32)
        centroid = (V[idx] * w[:, None]).sum(axis=0)
        centroid /= np.linalg.norm(centroid)
        interests.append(Interest(rows=members, medoid=medoid, centroid=centroid,
                                  weight=float(sum(recency[r] for r in members)),
                                  label=label_for([meta(r) for r in members])))
    interests.sort(key=lambda it: -it.weight)
    interests = interests[:max_interests]
    # Two visually different clusters can share a coarse label; name them by their main colour.
    seen = set()
    for it in interests:
        if it.label in seen:
            it.label = label_for([meta(r) for r in it.rows], colour_share=0.0)
        seen.add(it.label)
    return interests
