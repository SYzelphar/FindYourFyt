"""Per-session Bayesian logistic regression with Thompson sampling.

    p(right swipe | item) = sigmoid(w . f(item))

f(item) = [ taste vector (128-d, two-tower item tower)          learned weights, prior mean 0
            two-tower match   cos(user_tower(likes), item)      prior mean > 0 (trust the population model)
            looks-like-liked  max visual cos to liked items     prior mean > 0
            looks-like-passed max visual cos to disliked items  prior mean < 0
            popularity        log recent sales                  prior mean > 0 (small)
            fits-a-style      max visual cos to an interest centroid   prior mean > 0
            bias ]

The scalar features are z-scored with catalog statistics (manifest.json), so prior weights
are on a common scale. The posterior is a Laplace approximation (Newton's method to the MAP,
covariance = inverse Hessian) refit from scratch after every swipe; with d = 133 and at most a
few hundred swipes this takes milliseconds. Thompson sampling draws w ~ N(mu, Sigma), which
explores where the model is uncertain and exploits where it is confident.

Feature values for past swipes are the ones the item had *when it was shown*, so the
similarity-to-likes features never leak the swipe's own label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCALAR_FEATURES = ["two_tower_match", "looks_like_liked", "looks_like_passed", "popularity", "matches_interest"]


@dataclass
class Prior:
    # Defaults tuned with ml/tune_session.py on held-out simulated users (reports/tuning_*.json).
    taste_var: float = 0.1
    scalar_means: tuple = (0.3, 0.8, -0.8, 0.2, 1.0)
    scalar_var: float = 0.03     # small: scalar features are z-scores up to ~5, so Thompson noise scales fast
    bias_mean: float = -0.5      # most cards are passed on
    bias_var: float = 1.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class BayesianLogistic:
    def __init__(self, taste_dim: int, prior: Prior | None = None):
        self.prior = prior or Prior()
        p = self.prior
        self.dim = taste_dim + len(SCALAR_FEATURES) + 1
        self.m0 = np.concatenate([np.zeros(taste_dim), np.array(p.scalar_means), [p.bias_mean]])
        self.v0 = np.concatenate([np.full(taste_dim, p.taste_var), np.full(len(SCALAR_FEATURES), p.scalar_var),
                                  [p.bias_var]])
        self.mu = self.m0.copy()
        self.cov = np.diag(self.v0)
        self.n = 0

    @staticmethod
    def features(taste: np.ndarray, scalars: np.ndarray) -> np.ndarray:
        """taste [n, d], scalars [n, k] -> [n, d + k + 1] (with bias column)."""
        return np.hstack([taste, scalars, np.ones((len(taste), 1))])

    def fit(self, F: np.ndarray, y: np.ndarray, iters: int = 25, tol: float = 1e-6):
        """MAP + Laplace covariance from all swipes so far (F [n, dim], y in {0, 1})."""
        self.n = len(y)
        prec0 = 1.0 / self.v0
        if self.n == 0:
            self.mu, self.cov = self.m0.copy(), np.diag(self.v0)
            return self
        w = self.m0.copy()
        for _ in range(iters):
            p = _sigmoid(F @ w)
            grad = prec0 * (w - self.m0) - F.T @ (y - p)
            H = (F.T * (p * (1 - p))) @ F + np.diag(prec0)
            step = np.linalg.solve(H, grad)
            w -= step
            if np.abs(step).max() < tol:
                break
        p = _sigmoid(F @ w)
        H = (F.T * (p * (1 - p))) @ F + np.diag(prec0)
        self.mu = w
        self.cov = np.linalg.inv(H)
        self.cov = (self.cov + self.cov.T) / 2
        return self

    def sample(self, rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
        """Thompson sample of the weight vector from a tempered posterior N(mu, temperature * Sigma).

        temperature = 1 is standard Thompson sampling, 0 is greedy (posterior mean).
        """
        if temperature <= 0:
            return self.mu.copy()
        L = np.linalg.cholesky(self.cov + 1e-9 * np.eye(self.dim))
        return self.mu + np.sqrt(temperature) * (L @ rng.standard_normal(self.dim))

    def predictive(self, F: np.ndarray):
        """Posterior predictive p(like) (probit approximation) and logit std per row."""
        mean = F @ self.mu
        var = np.einsum("ij,jk,ik->i", F, self.cov, F)
        p = _sigmoid(mean / np.sqrt(1 + np.pi * var / 8))
        return p, np.sqrt(var)

    def contributions(self, F_row: np.ndarray, taste_dim: int) -> dict:
        """Split the posterior-mean logit into named parts (used for explanations)."""
        parts = {"style_learned_from_swipes": float(F_row[:taste_dim] @ self.mu[:taste_dim])}
        for j, name in enumerate(SCALAR_FEATURES):
            parts[name] = float(F_row[taste_dim + j] * self.mu[taste_dim + j])
        parts["baseline"] = float(self.mu[-1])
        return parts

    def uncertainty(self, F: np.ndarray | None = None) -> float:
        """Fraction of prior predictive variance left (1 = knows nothing, -> 0 as it learns).

        Measured on the given items (e.g. the ones the user has been shown): with 133 weights,
        the total variance barely moves after a few swipes even though the model is already
        confident about the kinds of items the user is actually seeing.
        """
        if F is None or not len(F):
            return float(np.trace(self.cov) / self.v0.sum())
        post = np.einsum("ij,jk,ik->i", F, self.cov, F)
        prior = (F * F) @ self.v0
        return float(np.mean(post / prior))

    def weights(self) -> dict:
        d = self.dim - len(SCALAR_FEATURES) - 1
        return {name: float(self.mu[d + j]) for j, name in enumerate(SCALAR_FEATURES)}
