"""FindYourFit API (Flask).

    GET    /api/health                          service + model status
    POST   /api/session                         {departments?: [...]} -> session + onboarding cards
    GET    /api/session/<id>                    swipe counts, mode
    GET    /api/session/<id>/profile            learned interests, attribute affinities, model confidence
    GET    /api/recommendations?session_id=..   next cards for the current profile
    POST   /api/swipe                           {session_id, product_id, direction} -> next cards
    POST   /api/steer                           {session_id, text} -> steer the deck with words
    DELETE /api/steer                           {session_id, text?} -> remove one / all steers
    GET    /api/search?q=..&k=20                text -> products (FashionCLIP)
    GET    /api/products/<id>                   product metadata
    GET    /api/products/<id>/similar?k=10      visually similar products
    GET    /images/<id>.jpg                     product images

Run:  python app.py   (http://127.0.0.1:5000)
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from recsys import VALID_DIRECTIONS, Bundle, BundleUnavailable, InMemorySessionStore, Recommender
from recsys.bundle import DEFAULT_MODELS_DIR
from recsys.events import EventLog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("findyourfit")

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = Path(os.environ.get("FYF_IMAGES_DIR", ROOT / "data" / "images"))
DEPARTMENTS = ["Ladieswear", "Menswear", "Divided", "Sport"]
MAX_K = 50
MAX_STEER_CHARS = 80


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def load_recommender():
    bundle = Bundle(os.environ.get("FYF_MODELS_DIR") or DEFAULT_MODELS_DIR)
    text = None
    if os.environ.get("FYF_TEXT_ENCODER", "fashion_clip") == "fashion_clip":
        try:
            from recsys.steering import FashionClipText
            text = FashionClipText()
        except Exception as e:  # steering/search degrade gracefully
            log.warning("Text encoder unavailable (%s); steering and search disabled", e)
    return Recommender(bundle, text_encoder=text)


def create_app(recommender: Recommender | None = None, sessions=None, events=None, load_error=None):
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": r"http://(localhost|127\.0\.0\.1):\d+"}})
    sessions = sessions or InMemorySessionStore()
    events = events or EventLog()
    started = time.time()

    # ------------------------------------------------------------ errors
    @app.errorhandler(ApiError)
    def _api_error(e):
        return jsonify({"error": {"code": e.code, "message": e.message}}), e.status

    @app.errorhandler(HTTPException)
    def _http_error(e):
        return jsonify({"error": {"code": (e.name or "error").lower().replace(" ", "_"),
                                  "message": e.description}}), e.code

    @app.errorhandler(Exception)
    def _unexpected(e):
        log.exception("Unhandled error")
        return jsonify({"error": {"code": "internal_error", "message": "internal server error"}}), 500

    # ------------------------------------------------------------ helpers
    def rec() -> Recommender:
        if recommender is None:
            raise ApiError(503, "model_unavailable", load_error or "recommender not loaded")
        return recommender

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError(400, "malformed_request", "expected a JSON object body")
        return data

    def session(session_id):
        if not session_id:
            raise ApiError(400, "missing_session_id", "session_id is required")
        state = sessions.get(session_id)
        if state is None:
            raise ApiError(404, "session_not_found", "unknown or expired session_id; create a new session")
        return state

    def product_row(value) -> int:
        try:
            f = float(value)
        except (TypeError, ValueError):
            f = float("nan")
        if isinstance(value, bool) or not f.is_integer():
            raise ApiError(400, "invalid_product_id", "product_id must be an integer")
        row = rec().b.row_of.get(int(f))
        if row is None:
            raise ApiError(404, "product_not_found", f"product {int(f)} is not in the catalog")
        return row

    def k_param(default):
        try:
            return max(1, min(int(request.args.get("k", default)), MAX_K))
        except ValueError:
            raise ApiError(400, "invalid_k", "k must be an integer") from None

    def steer_text(data) -> str:
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, "invalid_text", "text must be a non-empty string")
        if len(text) > MAX_STEER_CHARS:
            raise ApiError(400, "invalid_text", f"text must be at most {MAX_STEER_CHARS} characters")
        return text.strip()

    def deck(state, k=None):
        items, info = rec().recommend(state, k)
        return {"session_id": state.session_id, "recommendations": items, **info,
                "stats": {"swipes": len(state.swipes), "likes": len(state.liked_rows)}}

    # ------------------------------------------------------------ routes
    @app.get("/api/health")
    def health():
        ok = recommender is not None
        payload = {"status": "ok" if ok else "degraded", "model_loaded": ok, "active_sessions": len(sessions),
                   "uptime_s": round(time.time() - started)}
        if ok:
            b = recommender.b
            payload.update(items=len(b), taste_dim=int(b.taste.shape[1]), visual_dim=int(b.visual.shape[1]),
                           visual_encoder=b.manifest["visual_encoder"]["model"],
                           user_tower=b.user_tower is not None, text_encoder=recommender.text is not None)
        else:
            payload["error"] = load_error
        return jsonify(payload), (200 if ok else 503)

    @app.post("/api/session")
    def create_session():
        rec()
        data = request.get_json(silent=True) or {}
        groups = data.get("departments") if isinstance(data, dict) else None
        if groups is not None:
            if not isinstance(groups, list) or not set(groups) <= set(DEPARTMENTS):
                raise ApiError(400, "invalid_departments", f"departments must be a subset of {DEPARTMENTS}")
            groups = groups or None
        state = sessions.create(index_groups=groups)
        events.write("session_start", session=state.session_id, departments=groups)
        return jsonify(deck(state)), 201

    @app.get("/api/session/<session_id>")
    def get_session(session_id):
        state = session(session_id)
        return jsonify({"session_id": state.session_id, "swipes": len(state.swipes),
                        "likes": len(state.liked_rows), "passes": len(state.disliked_rows),
                        "steers": [s.text for s in state.steers], "departments": state.index_groups})

    @app.get("/api/session/<session_id>/profile")
    def get_profile(session_id):
        return jsonify(rec().profile(session(session_id)))

    @app.get("/api/recommendations")
    def recommendations():
        state = session(request.args.get("session_id"))
        with sessions.lock():
            payload = deck(state, k_param(None) if "k" in request.args else None)
            sessions.save(state)
        return jsonify(payload)

    @app.post("/api/swipe")
    def swipe():
        data = body()
        direction = data.get("direction")
        if direction not in VALID_DIRECTIONS:
            raise ApiError(400, "invalid_direction", f"direction must be one of {list(VALID_DIRECTIONS)}")
        row = product_row(data.get("product_id"))
        with sessions.lock():
            state = session(data.get("session_id"))
            served = state.served.get(row, {})
            rec().record_swipe(state, row, direction == "right")
            payload = deck(state)
            sessions.save(state)
        events.write("swipe", session=state.session_id, product_id=int(rec().b.article_ids[row]),
                     direction=direction, n_swipe=len(state.swipes), shown_rank=served.get("rank"),
                     sources=served.get("sources"), predicted_p_like=served.get("p_like"),
                     steers=[s.text for s in state.steers])
        return jsonify(payload)

    @app.post("/api/steer")
    def add_steer():
        data = body()
        text = steer_text(data)
        if rec().text is None:
            raise ApiError(503, "text_unavailable", "text steering is not available on this server")
        with sessions.lock():
            state = session(data.get("session_id"))
            rec().add_steer(state, text)
            payload = deck(state)
            sessions.save(state)
        events.write("steer", session=state.session_id, text=text)
        return jsonify(payload)

    @app.delete("/api/steer")
    def remove_steer():
        data = body()
        with sessions.lock():
            state = session(data.get("session_id"))
            rec().remove_steer(state, data.get("text"))
            payload = deck(state)
            sessions.save(state)
        return jsonify(payload)

    @app.get("/api/search")
    def search():
        q = request.args.get("q", "")
        if not q.strip() or len(q) > MAX_STEER_CHARS:
            raise ApiError(400, "invalid_query", f"q must be 1-{MAX_STEER_CHARS} characters")
        if rec().text is None:
            raise ApiError(503, "text_unavailable", "text search is not available on this server")
        return jsonify({"query": q, "results": rec().search(q, k_param(20))})

    @app.get("/api/products/<product_id>")
    def product(product_id):
        return jsonify(rec().b.record(product_row(product_id)))

    @app.get("/api/products/<product_id>/similar")
    def similar(product_id):
        row = product_row(product_id)
        return jsonify({"product_id": int(rec().b.article_ids[row]),
                        "recommendations": rec().similar(row, k_param(10))})

    @app.get("/images/<path:filename>")
    def images(filename):
        return send_from_directory(IMAGES_DIR, filename, max_age=7 * 24 * 3600)

    return app


def build_app():
    """Production entry point (gunicorn "app:build_app()")."""
    try:
        recommender, error = load_recommender(), None
    except BundleUnavailable as e:
        recommender, error = None, str(e)
        log.error("Recommender unavailable: %s", e)
    return create_app(recommender, load_error=error)


def main():
    build_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)


if __name__ == "__main__":
    main()
