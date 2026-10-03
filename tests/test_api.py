import json


def start(client, **body):
    r = client.post("/api/session", json=body or None)
    assert r.status_code == 201, r.json
    return r.json


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json["model_loaded"] and r.json["items"] == 240


def test_swipe_flow_returns_new_cards(client):
    s = start(client)
    first = s["recommendations"][0]
    r = client.post("/api/swipe", json={"session_id": s["session_id"], "product_id": first["id"], "direction": "right"})
    assert r.status_code == 200
    body = r.json
    assert body["stats"] == {"swipes": 1, "likes": 1}
    assert first["id"] not in {i["id"] for i in body["recommendations"]}
    assert all({"id", "name", "image", "explanation", "pLike"} <= set(i) for i in body["recommendations"])


def test_swipe_validation_errors(client):
    sid = start(client)["session_id"]
    pid = start(client)["recommendations"][0]["id"]
    cases = [
        ({"session_id": sid, "product_id": pid, "direction": "up"}, 400, "invalid_direction"),
        ({"session_id": sid, "product_id": "abc", "direction": "left"}, 400, "invalid_product_id"),
        ({"session_id": sid, "product_id": 1, "direction": "left"}, 404, "product_not_found"),
        ({"session_id": "nope", "product_id": pid, "direction": "left"}, 404, "session_not_found"),
        ({"product_id": pid, "direction": "left"}, 400, "missing_session_id"),
    ]
    for body, status, code in cases:
        r = client.post("/api/swipe", json=body)
        assert (r.status_code, r.json["error"]["code"]) == (status, code), body
    r = client.post("/api/swipe", data="not json", content_type="application/json")
    assert r.json["error"]["code"] == "malformed_request"


def test_departments(client):
    s = start(client, departments=["Menswear"])
    assert all(i["department"] == "Menswear" for i in s["recommendations"])
    r = client.post("/api/session", json={"departments": ["Kids"]})
    assert r.status_code == 400


def test_steer_and_profile(client):
    sid = start(client)["session_id"]
    r = client.post("/api/steer", json={"session_id": sid, "text": "in red"})
    assert r.status_code == 200 and r.json["steers"] == ["in red"]
    assert client.post("/api/steer", json={"session_id": sid, "text": ""}).status_code == 400
    r = client.delete("/api/steer", json={"session_id": sid})
    assert r.json["steers"] == []
    p = client.get(f"/api/session/{sid}/profile")
    assert p.status_code == 200 and p.json["swipes"] == 0


def test_search_similar_and_products(client):
    r = client.get("/api/search?q=red&k=5")
    assert r.status_code == 200 and len(r.json["results"]) == 5
    pid = r.json["results"][0]["id"]
    assert client.get(f"/api/products/{pid}").json["id"] == pid
    sim = client.get(f"/api/products/{pid}/similar?k=4").json["recommendations"]
    assert len(sim) == 4 and pid not in {x["id"] for x in sim}


def test_swipes_are_logged(client, tmp_path):
    s = start(client)
    pid = s["recommendations"][0]["id"]
    client.post("/api/swipe", json={"session_id": s["session_id"], "product_id": pid, "direction": "left"})
    lines = [json.loads(line) for f in (tmp_path / "events").glob("*.jsonl") for line in f.read_text().splitlines()]
    swipe = next(e for e in lines if e["event"] == "swipe")
    assert swipe["product_id"] == pid and swipe["shown_rank"] == 0 and swipe["sources"]


def test_model_unavailable():
    from app import create_app
    client = create_app(None, load_error="no bundle").test_client()
    assert client.get("/api/health").status_code == 503
    assert client.post("/api/session").json["error"]["code"] == "model_unavailable"
