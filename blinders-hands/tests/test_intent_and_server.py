import time

from blinders_hands.config import Settings
from blinders_hands.intent import rule_based
from blinders_hands.pipeline import build_assistant, direction_phrase
from blinders_hands.server import create_app


def test_rule_based_intent():
    i = rule_based("帮我看看布洛芬还能不能吃")
    assert i.task == "check_expiry" and "布洛芬" in i.target
    assert rule_based("哪瓶是洗发水").task == "identify"
    assert rule_based("降压药一次吃几片").task == "read_usage"
    assert rule_based("我的眼药水掉了在哪").target == "眼药水"
    assert rule_based("布洛芬一次吃几片").target == "布洛芬"
    i = rule_based("孕妇能不能吃布洛芬")
    assert i.task == "ask" and i.target == "布洛芬" and i.question
    i = rule_based("把降压药递给我")
    assert i.target == "降压药" and i.handover
    assert rule_based("帮我整理一下药箱").task == "scan_box"
    assert rule_based("有哪些药快过期了").task == "expiring"


def test_direction():
    assert direction_phrase((0.0, 0.7, 0.2, 0.9)) == "在您左边、靠近您"
    assert direction_phrase((0.45, 0.45, 0.55, 0.55)) == "在您正前方"


def test_end_to_end_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("BH_MOCK", "1")
    monkeypatch.setenv("INVENTORY_FILE", str(tmp_path / "inv.json"))
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("ARM_MOVE_WAIT", "0")
    s = Settings()
    app = create_app(build_assistant(s), settings=s)
    c = app.test_client()

    u = c.post("/api/understand", json={"text": "布洛芬过期了吗"}).get_json()
    assert u["understood"] and u["task"] == "check_expiry"

    job = c.post("/api/tasks", json={"target": u["target"], "task": u["task"]}).get_json()
    for _ in range(100):
        st = c.get(f"/api/tasks/{job['id']}").get_json()
        if st["status"] != "running":
            break
        time.sleep(0.05)
    assert st["status"] == "done", st
    assert "有效期" in st["message"] and "2027年2月28日" in st["message"]
    assert st["location"].startswith("在您")
    assert any("转到第" in p for p in st["progress"])


def test_rejects_bad_task(monkeypatch):
    monkeypatch.setenv("BH_MOCK", "1")
    s = Settings()
    c = create_app(build_assistant(s), settings=s).test_client()
    assert c.post("/api/tasks", json={"target": "", "task": "check_expiry"}).status_code == 400


def _wait(c, job_id, until=("done", "error")):
    for _ in range(200):
        st = c.get(f"/api/tasks/{job_id}").get_json()
        if st["status"] in until:
            return st
        time.sleep(0.02)
    raise AssertionError(st)


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("BH_MOCK", "1")
    monkeypatch.setenv("ARM_MOVE_WAIT", "0")
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "caps"))
    monkeypatch.setenv("INVENTORY_FILE", str(tmp_path / "inv.json"))
    s = Settings()
    a = build_assistant(s)
    return create_app(a, settings=s).test_client(), a


def test_handover_waits_for_release(monkeypatch, tmp_path):
    c, a = _client(monkeypatch, tmp_path)
    job = c.post("/api/tasks", json={"target": "降压药", "task": "identify", "handover": True}).get_json()
    st = _wait(c, job["id"], until=("waiting_release",))
    assert "递到您手边" in st["progress"][-1]
    c.post("/api/release")
    st = _wait(c, job["id"])
    assert st["status"] == "done"
    assert a.arm.history[-1][0] == "gripper" and a.arm.history[-1][1] > 0   # 松开夹爪


def test_scan_then_query_and_reminder(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path)
    job = c.post("/api/tasks", json={"task": "scan_box"}).get_json()
    st = _wait(c, job["id"])
    assert "2样记下了有效期" in st["message"]
    inv = c.get("/api/inventory").get_json()
    assert {it["name"] for it in inv["items"]} == {"布洛芬缓释胶囊", "阿莫西林胶囊"}


def test_ask_and_usage(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path)
    job = c.post("/api/tasks", json={"target": "布洛芬", "task": "ask", "question": "孕妇能不能吃"}).get_json()
    assert "孕妇" in _wait(c, job["id"])["message"]
    job = c.post("/api/tasks", json={"target": "布洛芬", "task": "read_usage"}).get_json()
    assert "一次1粒" in _wait(c, job["id"])["message"]


def test_stop_blocks_until_resume(monkeypatch, tmp_path):
    c, a = _client(monkeypatch, tmp_path)
    c.post("/api/stop")
    assert ("stop", None) in a.arm.history
    assert c.post("/api/tasks", json={"target": "布洛芬", "task": "check_expiry"}).status_code == 409
    c.post("/api/resume")
    job = c.post("/api/tasks", json={"target": "布洛芬", "task": "check_expiry"}).get_json()
    assert _wait(c, job["id"])["status"] == "done"


def test_guide_until_reached(monkeypatch, tmp_path):
    c, _ = _client(monkeypatch, tmp_path)
    for _ in range(10):
        r = c.post("/api/guide", json={"target": "布洛芬"}).get_json()
        if r["done"]:
            break
    assert r["done"] and "到了" in r["say"]
