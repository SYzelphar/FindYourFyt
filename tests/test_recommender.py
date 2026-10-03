import copy
from collections import Counter

from conftest import style_of
from recsys.session_store import SessionState


def new_state(seed=0):
    return SessionState(session_id="t", seed=seed)


def swipe_loop(rec, state, likes_style, n=20, dislike_style=None):
    """Simulated user: likes one style, optionally dislikes another explicitly."""
    for _ in range(n):
        items, _ = rec.recommend(state)
        top = items[0]
        style = style_of(rec.b, top)
        rec.record_swipe(state, rec.b.row_of[top["id"]], style == likes_style)
    return state


def test_onboarding_batch_is_diverse_and_explained(recommender):
    items, info = recommender.recommend(new_state())
    assert info["mode"] == "onboarding"
    assert len(items) == 5
    assert len({i["productCode"] for i in items}) == 5
    assert all(i["explanation"]["headline"] for i in items)


def test_never_repeats_items_or_colour_variants(recommender):
    state = new_state()
    shown = []
    for _ in range(40):
        items, _ = recommender.recommend(state)
        top = items[0]
        assert top["id"] not in {s["id"] for s in shown}
        assert top["productCode"] not in {s["productCode"] for s in shown}
        shown.append(top)
        recommender.record_swipe(state, recommender.b.row_of[top["id"]], len(shown) % 3 == 0)


def test_learns_liked_style(recommender):
    # The fixture has 20 distinct dress products; stop well before they run out.
    state = swipe_loop(recommender, new_state(1), "Dress", n=12)
    items, info = recommender.recommend(state)
    assert info["mode"] == "personalized"
    styles = Counter(style_of(recommender.b, i) for i in items)
    assert styles["Dress"] >= 3
    later = [recommender.b.meta(s.row)["product_type_name"] == "Dress" for s in state.swipes[4:]]
    assert sum(later) / len(later) > 0.6


def test_explanations_reference_liked_items(recommender):
    state = swipe_loop(recommender, new_state(2), "Jacket", n=15)
    items, _ = recommender.recommend(state)
    expl = items[0]["explanation"]
    liked_ids = {int(recommender.b.article_ids[r]) for r in state.liked_rows}
    assert expl["becauseYouLiked"]
    assert {x["id"] for x in expl["becauseYouLiked"]} <= liked_ids
    assert set(expl["contributions"]) >= {"style_learned_from_swipes", "looks_like_liked", "baseline"}


def test_steering_shifts_the_deck(recommender):
    b = recommender.b

    def red_share(state):
        items, _ = recommender.recommend(state)
        flags = [b.record(b.row_of[i["id"]])["colour"] == "Red" for i in items]
        return sum(flags) / len(flags), flags[0]

    plain, steered, top_red = [], [], 0
    for seed in range(3, 8):
        base = swipe_loop(recommender, new_state(seed), "T-shirt", n=10)
        plain.append(red_share(copy.deepcopy(base))[0])
        recommender.add_steer(base, "in red")
        share, first = red_share(base)
        steered.append(share)
        top_red += first
    assert sum(steered) / 5 > sum(plain) / 5 + 0.3
    assert min(steered) >= 0.8          # a steer filters the deck, it doesn't just nudge it
    assert top_red == 5
    recommender.remove_steer(base)
    assert base.steers == []


def test_department_filter(recommender):
    state = new_state(4)
    state.index_groups = ["Menswear"]
    for _ in range(5):
        items, _ = recommender.recommend(state)
        assert all(i["department"] == "Menswear" for i in items)
        recommender.record_swipe(state, recommender.b.row_of[items[0]["id"]], True)


def test_profile_summarises_session(recommender):
    state = swipe_loop(recommender, new_state(5), "Trousers", n=15)
    profile = recommender.profile(state)
    assert profile["swipes"] == 15
    assert profile["interests"] and "trousers" in profile["interests"][0]["label"].lower()
    assert 0 <= profile["confidence"] <= 1
    assert profile["affinities"]["productType"]["loves"][0]["value"] == "Trousers"


def test_no_near_duplicates_of_swiped_items(recommender):
    state = new_state(6)
    for _ in range(15):
        items, _ = recommender.recommend(state)
        recommender.record_swipe(state, recommender.b.row_of[items[0]["id"]], True)
    seen = [s.row for s in state.swipes]
    items, _ = recommender.recommend(state)
    rows = [recommender.b.row_of[i["id"]] for i in items]
    sims = recommender.b.visual[rows] @ recommender.b.visual[seen].T
    assert sims.max() < recommender.cfg.near_duplicate
