"""FindYourFit v2 recommender: candidate generation -> Bayesian ranking -> diversity re-ranking.

Per request (one per swipe):

  1. Candidate generation (FAISS, ~1k items from several sources)
       taste_profile   two-tower user vector of your likes      -> taste index
       interest:i      medoid of each of your interests          -> visual index
       learned_style   Thompson sample of the session weights    -> taste index (max inner product)
       steer           "like these but <text>" composed query    -> visual index
                       + the text alone (FashionCLIP text -> image search)
       explore         unseen style prototypes (k-means medoids)
       trending        most-bought items right now
  2. Steering filter: with an active steer, keep only items that match the words (z-scored
     text->image similarity >= steer_min_z, roughly the top 10% of the catalog)
  3. Ranking: one Thompson draw  w ~ posterior,  score = w . f(item) + steering bonus
  3. Re-ranking: MMR in visual space; at most one colour variant per product; nothing already swiped
  4. Explanation for every card (sources, the liked items it resembles, score breakdown, p(like))
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .bundle import Bundle
from .interests import Interest, find_interests
from .session_model import SCALAR_FEATURES, BayesianLogistic, Prior
from .session_store import MAX_STEERS, SessionState, Steer, Swipe


@dataclass
class RecConfig:
    batch_size: int = 5
    taste_pool: int = 300
    interest_pool: int = 150
    thompson_pool: int = 200
    steer_pool: int = 300
    explore_pool: int = 80
    trending_pool: int = 100
    rerank_pool: int = 80
    mmr_lambda: float = 0.75
    steer_weight: float = 1.5
    steer_gamma: float = 1.0
    steer_min_z: float = 1.3        # a steer is an instruction: only items this close to the words
    explore_bonus: float = 2.0      # UCB bonus on posterior std before the first like (tuned)
    near_duplicate: float = 0.97    # visual cosine above which a candidate counts as already seen
    max_interests: int = 4
    use_two_tower: bool = True
    use_visual_knn: bool = True
    use_interests: bool = True
    use_thompson: bool = True       # False: rank by the posterior mean (greedy)
    thompson_temperature: float = 0.25  # tempered Thompson: sample from N(mu, 0.25 Sigma) (tuned)
    prior: Prior = field(default_factory=Prior)


HEADLINES = {
    "looks_like_liked": "Looks like the {liked} you liked",
    "matches_interest": "Fits your “{interest}” style",
    "two_tower_match": "Shoppers with your taste often buy this",
    "style_learned_from_swipes": "Fits the style you keep swiping right on",
    "popularity": "Trending at H&M right now",
}


class Recommender:
    def __init__(self, bundle: Bundle, text_encoder=None, config: RecConfig | None = None):
        self.b = bundle
        self.text = text_encoder
        self.cfg = config or RecConfig()
        self.taste_dim = bundle.taste.shape[1]
        self.trending_order = np.argsort(-bundle.popularity_z, kind="stable")

    # ================================================================ features
    def user_vector(self, liked_rows):
        if not liked_rows or not self.cfg.use_two_tower:
            return None
        if self.b.user_tower is not None:
            from .two_tower import user_vector
            return user_vector(self.b.user_tower, self.b.taste, liked_rows)
        v = self.b.taste[liked_rows].mean(axis=0)
        return v / np.linalg.norm(v)

    def scalar_features(self, rows, liked_rows, disliked_rows, u, interests=()) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        st = self.b.stats
        out = np.zeros((len(rows), len(SCALAR_FEATURES)), dtype=np.float32)
        if u is not None:
            out[:, 0] = (self.b.taste[rows] @ u - st["taste_cos"]["mean"]) / st["taste_cos"]["std"]
        if self.cfg.use_visual_knn:
            V = self.b.visual[rows]
            if liked_rows:
                m = (V @ self.b.visual[liked_rows].T).max(axis=1)
                out[:, 1] = (m - st["visual_max_cos"]["mean"]) / st["visual_max_cos"]["std"]
            if disliked_rows:
                m = (V @ self.b.visual[disliked_rows].T).max(axis=1)
                out[:, 2] = (m - st["visual_max_cos"]["mean"]) / st["visual_max_cos"]["std"]
        out[:, 3] = self.b.popularity_z[rows]
        if interests and self.cfg.use_visual_knn:
            cents = np.stack([it.centroid for it in interests])
            m = (self.b.visual[rows] @ cents.T).max(axis=1)
            out[:, 4] = (m - st["visual_centroid_cos"]["mean"]) / st["visual_centroid_cos"]["std"]
        return np.clip(out, -4, 6)

    def interests(self, liked_rows):
        if not liked_rows:
            return []
        if not self.cfg.use_interests:     # ablation: one interest = plain mean of all likes
            return find_interests(liked_rows, self.b.visual, self.b.meta, 1, distance_threshold=2.0)
        return find_interests(liked_rows, self.b.visual, self.b.meta, self.cfg.max_interests)

    def fit_model(self, state: SessionState) -> BayesianLogistic:
        model = BayesianLogistic(self.taste_dim, self.cfg.prior)
        if state.swipes:
            rows = [s.row for s in state.swipes]
            F = BayesianLogistic.features(self.b.taste[rows], np.stack([s.scalars for s in state.swipes]))
            model.fit(F, np.array([1.0 if s.liked else 0.0 for s in state.swipes]))
        return model

    # ================================================================ state changes
    def record_swipe(self, state: SessionState, row: int, liked: bool):
        """Scalar features are frozen as they were when the card was shown (no label leakage)."""
        liked_rows, disliked_rows = state.liked_rows, state.disliked_rows
        scalars = self.scalar_features([row], liked_rows, disliked_rows, self.user_vector(liked_rows),
                                       self.interests(liked_rows))[0]
        state.swipes.append(Swipe(row=row, liked=liked, scalars=scalars))
        state.seen_codes.add(int(self.b.product_codes[row]))

    def add_steer(self, state: SessionState, text: str):
        if self.text is None:
            raise RuntimeError("text steering unavailable (no text encoder loaded)")
        vec = self.text.encode(text)
        cos = self.b.visual @ vec
        z = ((cos - cos.mean()) / (cos.std() + 1e-6)).astype(np.float32)
        state.steers = [s for s in state.steers if s.text.lower() != text.lower()][-(MAX_STEERS - 1):]
        state.steers.append(Steer(text=text, vector=vec, z=z))

    def remove_steer(self, state: SessionState, text: str | None = None):
        state.steers = [] if text is None else [s for s in state.steers if s.text.lower() != text.lower()]

    # ================================================================ recommendation
    def _allowed(self, state: SessionState, rows):
        seen = state.seen_rows
        groups = set(state.index_groups) if state.index_groups else None
        for r in rows:
            if r in seen or int(self.b.product_codes[r]) in state.seen_codes:
                continue
            if groups and self.b.index_groups[r] not in groups:
                continue
            yield r

    def _candidates(self, state, model, w, interests: list[Interest], u, rng) -> dict:
        cand: dict[int, set] = {}
        b, cfg = self.b, self.cfg
        scale = 3 if state.index_groups else 1    # filters discard results; search wider

        def add(pairs, source):
            for r in self._allowed(state, (p[0] for p in pairs)):
                cand.setdefault(r, set()).add(source)

        if u is not None:
            add(b.search(b.taste_index, u, cfg.taste_pool * scale), "taste_profile")
        if interests:
            for i, it in enumerate(interests):
                add(b.search(b.visual_index, b.visual[it.medoid], cfg.interest_pool * scale), f"interest:{i}")
        if model.n and cfg.use_thompson:
            q = w[: self.taste_dim]
            if np.linalg.norm(q) > 1e-6:
                add(b.search(b.taste_index, q, cfg.thompson_pool * scale), "learned_style")
        liked = state.liked_rows
        for steer in state.steers:
            q = steer.vector.copy()
            if liked:
                m = b.visual[liked].mean(axis=0)
                q = m / np.linalg.norm(m) + cfg.steer_gamma * q
            add(b.search(b.visual_index, q / np.linalg.norm(q), cfg.steer_pool * scale), "steer")
            add(b.search(b.visual_index, steer.vector, cfg.steer_pool * scale), "steer")
        protos = list(self._allowed(state, b.prototypes.tolist()))
        if protos:
            n = len(protos) if not liked else min(cfg.explore_pool, len(protos))   # onboarding: all of them
            pick = rng.choice(len(protos), size=n, replace=False)
            add([(protos[i], 0.0) for i in pick], "explore")
        trending = []
        for r in self._allowed(state, self.trending_order[: cfg.trending_pool * 4 * scale].tolist()):
            trending.append((r, 0.0))
            if len(trending) >= cfg.trending_pool:
                break
        add(trending, "trending")
        return cand

    def _mmr(self, rows, scores, k):
        order = np.argsort(-scores)[: self.cfg.rerank_pool]
        rows, scores = rows[order], scores[order]
        rel = (scores - scores.min()) / (np.ptp(scores) + 1e-9)
        V = self.b.visual[rows]
        chosen, codes = [], set()
        max_sim = np.zeros(len(rows), dtype=np.float32)
        available = np.ones(len(rows), dtype=bool)
        lam = self.cfg.mmr_lambda
        while len(chosen) < k and available.any():
            mmr = lam * rel - (1 - lam) * max_sim
            mmr[~available] = -np.inf
            i = int(np.argmax(mmr))
            available[i] = False
            code = int(self.b.product_codes[rows[i]])
            if code in codes:            # one colour variant per product per batch
                continue
            codes.add(code)
            chosen.append((int(rows[i]), float(scores[i])))
            max_sim = np.maximum(max_sim, V @ V[i])
        return chosen

    def recommend(self, state: SessionState, k: int | None = None):
        """Next cards for this session -> (items, info)."""
        k = k or self.cfg.batch_size
        state.requests += 1
        rng = np.random.default_rng([state.seed, len(state.swipes), state.requests])
        liked, disliked = state.liked_rows, state.disliked_rows
        model = self.fit_model(state)
        w = model.sample(rng, self.cfg.thompson_temperature) if self.cfg.use_thompson else model.mu
        u = self.user_vector(liked)
        interests = self.interests(liked)
        cand = self._candidates(state, model, w, interests, u, rng)
        rows = np.fromiter(cand.keys(), dtype=np.int64)
        if state.swipes and len(rows):
            # Re-listed products often reuse the same photo under a new product code: drop
            # candidates that are visually near-identical to anything already swiped.
            seen = [s.row for s in state.swipes]
            dup = (self.b.visual[rows] @ self.b.visual[seen].T).max(axis=1) >= self.cfg.near_duplicate
            rows = rows[~dup]
        if state.steers and len(rows):
            # A steer filters ("jackets" -> only jacket-like items); taste then ranks within it.
            # If too few candidates match, fall back to the soft bonus alone.
            match = np.mean([s.z[rows] for s in state.steers], axis=0) >= self.cfg.steer_min_z
            if match.sum() >= k:
                rows = rows[match]
        if not len(rows):
            return [], {"mode": "exhausted", "interests": []}

        F = BayesianLogistic.features(self.b.taste[rows],
                                      self.scalar_features(rows, liked, disliked, u, interests))
        scores = F @ w
        if not liked and self.cfg.explore_bonus:
            # Before the first like: optimism under uncertainty (UCB). Passes shrink the posterior
            # variance around what was rejected, so the deck moves on to unexplored styles.
            scores = scores + self.cfg.explore_bonus * np.sqrt(np.einsum("ij,jk,ik->i", F, model.cov, F))
        steer_z = None
        if state.steers:
            steer_z = np.mean([s.z[rows] for s in state.steers], axis=0)
            scores = scores + self.cfg.steer_weight * steer_z
        picks = self._mmr(rows, scores, k)

        pos = {r: i for i, r in enumerate(rows)}
        p_like, logit_std = model.predictive(F[[pos[r] for r, _ in picks]])
        items, served = [], {}
        for j, (r, score) in enumerate(picks):
            i = pos[r]
            expl = self._explain(r, cand[r], F[i], model, interests, liked, disliked, state,
                                 None if steer_z is None else float(steer_z[i]))
            item = self.b.record(r)
            item.update(score=round(score, 4), pLike=round(float(p_like[j]), 4),
                        uncertainty=round(float(logit_std[j]), 4), sources=sorted(cand[r]), explanation=expl)
            items.append(item)
            served[r] = {"rank": j, "sources": sorted(cand[r]), "p_like": float(p_like[j])}
        state.served = served
        mode = "personalized" if liked else ("exploring" if disliked else "onboarding")
        return items, {"mode": mode, "steers": [s.text for s in state.steers],
                       "interests": [it.label for it in interests]}

    # ================================================================ explanations
    def _explain(self, row, sources, f_row, model, interests, liked, disliked, state, steer_z):
        parts = model.contributions(f_row, self.taste_dim)
        if not liked:
            parts["two_tower_match"] = parts["looks_like_liked"] = parts["matches_interest"] = 0.0
        if steer_z is not None:
            parts["steer"] = self.cfg.steer_weight * steer_z
        expl = {"contributions": {k: round(v, 3) for k, v in parts.items()}}

        similar = []
        if liked:
            sims = self.b.visual[liked] @ self.b.visual[row]
            for j in np.argsort(-sims)[:2]:
                rec = self.b.record(liked[j])
                similar.append({"id": rec["id"], "name": rec["name"], "image": rec["image"],
                                "similarity": round(float(sims[j]), 3)})
            mine = self.b.meta(row)
            nearest = self.b.meta(liked[int(np.argmax(sims))])
            expl["sharedAttributes"] = [mine[a] for a in ("colour_group_name", "graphical_appearance_name",
                                                          "product_type_name") if mine[a] == nearest[a]]
        expl["becauseYouLiked"] = similar

        interest = next((interests[int(s.split(":")[1])].label for s in sources if s.startswith("interest:")), None)
        if interest is None and interests:
            interest = max(interests, key=lambda it: float(self.b.visual[row] @ it.centroid)).label
        expl["interest"] = interest

        steer_text = state.steers[-1].text if state.steers else None
        if steer_z is not None:
            expl["steer"] = {"text": steer_text, "z": round(steer_z, 2)}

        # Headline: the biggest positive reason, in plain words.
        candidates = {k: v for k, v in parts.items() if k in HEADLINES or k == "steer"}
        best = max(candidates, key=candidates.get) if candidates else None
        if not liked:
            headline = ("Something different from what you passed on" if disliked
                        else "Exploring styles to learn your taste")
            if "trending" in sources and best == "popularity":
                headline = HEADLINES["popularity"]
        elif best == "steer" and candidates["steer"] > 0.5:
            headline = f"Matches “{steer_text}”"
        elif best is None or candidates[best] <= 0.1 or sources <= {"explore", "trending"}:
            headline = "Something new: testing a different style"
        else:
            headline = HEADLINES[best].format(liked=similar[0]["name"] if similar else "item",
                                              interest=(interest or "favourite").lower())
        expl["headline"] = headline
        return expl

    # ================================================================ profile & item APIs
    def profile(self, state: SessionState) -> dict:
        liked, disliked = state.liked_rows, state.disliked_rows
        model = self.fit_model(state)
        shown = None
        if state.swipes:
            rows = [s.row for s in state.swipes[-30:]]
            shown = BayesianLogistic.features(self.b.taste[rows], np.stack([s.scalars for s in state.swipes[-30:]]))
        interests = self.interests(liked)
        total_w = sum(it.weight for it in interests) or 1.0
        affinities = {}
        for key, attr in [("productType", "product_type_name"), ("colour", "colour_group_name"),
                          ("pattern", "graphical_appearance_name"), ("department", "index_group_name")]:
            likes = Counter(self.b.meta(r)[attr] for r in liked)
            passes = Counter(self.b.meta(r)[attr] for r in disliked)
            scored = [(v, float(np.log((likes[v] + 0.5) / (passes[v] + 0.5))), likes[v], passes[v])
                      for v in set(likes) | set(passes)]
            scored.sort(key=lambda t: -t[1])
            affinities[key] = {
                "loves": [{"value": v, "score": round(s, 2), "likes": lk, "passes": ps}
                          for v, s, lk, ps in scored if s > 0][:3],
                "avoids": [{"value": v, "score": round(s, 2), "likes": lk, "passes": ps}
                           for v, s, lk, ps in reversed(scored) if s < 0][:3],
            }
        weights = model.weights()
        return {
            "sessionId": state.session_id,
            "swipes": len(state.swipes), "likes": len(liked), "passes": len(disliked),
            "confidence": round(1 - model.uncertainty(shown), 3),
            "interests": [{"label": it.label, "share": round(it.weight / total_w, 3), "size": len(it.rows),
                           "items": [self.b.record(r) for r in it.rows[-4:]]} for it in interests],
            "affinities": affinities,
            "learnedWeights": {k: round(v, 3) for k, v in weights.items()},
            "steers": [s.text for s in state.steers],
            "departments": state.index_groups,
        }

    def similar(self, row: int, k=10):
        code = int(self.b.product_codes[row])
        out = []
        for r, s in self.b.search(self.b.visual_index, self.b.visual[row], k * 4 + 10):
            if r == row or int(self.b.product_codes[r]) == code:
                continue
            item = self.b.record(r)
            item["score"] = round(s, 4)
            out.append(item)
            if len(out) == k:
                break
        return out

    def search(self, text: str, k=20):
        if self.text is None:
            raise RuntimeError("text search unavailable (no text encoder loaded)")
        cos = self.b.visual @ self.text.encode(text)
        out = []
        for r in np.argsort(-cos)[:k]:
            item = self.b.record(int(r))
            item["score"] = round(float(cos[r]), 4)
            out.append(item)
        return out
