import numpy as np

from recsys.interests import find_interests
from recsys.session_model import BayesianLogistic, Prior


def _features(taste):
    return BayesianLogistic.features(taste, np.zeros((len(taste), 5)))


def test_posterior_learns_direction_from_likes_and_dislikes():
    rng = np.random.default_rng(0)
    d = 8
    likes = np.eye(d)[0] + 0.1 * rng.standard_normal((6, d))
    dislikes = -np.eye(d)[0] + 0.1 * rng.standard_normal((6, d))
    model = BayesianLogistic(d, Prior(taste_var=1.0))   # mechanics test, not the tuned prior
    model.fit(_features(np.vstack([likes, dislikes])), np.r_[np.ones(6), np.zeros(6)])
    assert model.mu[0] > 0.5
    p, _ = model.predictive(_features(np.stack([np.eye(d)[0], -np.eye(d)[0]])))
    assert p[0] > 0.6 > 0.4 > p[1]


def test_uncertainty_shrinks_with_data():
    rng = np.random.default_rng(1)
    d = 8
    model = BayesianLogistic(d)
    before = model.uncertainty()
    X = rng.standard_normal((30, d))
    model.fit(_features(X), (X[:, 0] > 0).astype(float))
    assert model.uncertainty() < before


def test_thompson_samples_vary_but_centre_on_mean():
    model = BayesianLogistic(4)
    rng = np.random.default_rng(2)
    samples = np.stack([model.sample(rng) for _ in range(2000)])
    assert samples.std(axis=0).min() > 0.1
    assert np.allclose(samples.mean(axis=0), model.mu, atol=0.1)


def test_contributions_sum_to_logit():
    model = BayesianLogistic(4)
    f = _features(np.array([[0.3, -0.2, 0.1, 0.5]]))[0]
    parts = model.contributions(f, 4)
    assert np.isclose(sum(parts.values()), f @ model.mu)


def test_interests_split_distinct_styles(bundle):
    trousers = [r for r in range(len(bundle)) if bundle.meta(r)["product_type_name"] == "Trousers"][:4]
    dresses = [r for r in range(len(bundle)) if bundle.meta(r)["product_type_name"] == "Dress"][:4]
    interests = find_interests(trousers + dresses, bundle.visual, bundle.meta, distance_threshold=0.5)
    labels = " ".join(i.label.lower() for i in interests)
    assert len(interests) >= 2
    assert "trousers" in labels and "dress" in labels
