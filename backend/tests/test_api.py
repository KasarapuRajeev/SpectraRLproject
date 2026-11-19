import time
import numpy as np
import pytest
from fastapi.testclient import TestClient
from spectrarl import api as api_mod, db
from spectrarl.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait_run(c, rid, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        r = c.get(f"/api/runs/{rid}").json()
        if r["status"] not in ("queued", "running"):
            return r
        time.sleep(0.2)
    raise TimeoutError


@pytest.fixture(scope="module")
def run_id(client):
    r = client.post("/api/runs", json={"name": "api-test", "env": {"N": 6}, "ql": {"episodes": 80, "slots": 200}})
    assert r.status_code == 201
    done = wait_run(client, r.json()["id"])
    assert done["status"] == "done", done
    return done["id"]


def test_health_and_meta(client):
    assert client.get("/api/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert m["defaults"]["env"]["K"] == 16 and m["defaults"]["ql"]["alpha"] == 0.1
    assert m["defaults"]["ql"]["gamma"] == 0.95 and m["defaults"]["ql"]["eps_decay"] == 0.995
    assert m["paper"]["throughput"]["ql"][-1] == 88.6 and len(m["assumptions"]) >= 5


def test_run_lifecycle_persists_metrics_and_policy(client, run_id):
    r = client.get(f"/api/runs/{run_id}").json()
    assert r["episodes_done"] == 80 and r["policy_path"] and r["summary_json"]
    m = client.get(f"/api/runs/{run_id}/metrics").json()
    assert len(m) == 80 * 3 and {x["policy"] for x in m} == {"ql", "greedy", "random"}
    assert len(client.get(f"/api/runs/{run_id}/metrics", params={"since": 70}).json()) == 10 * 3
    assert any(x["id"] == run_id for x in client.get("/api/runs").json())
    p = client.get(f"/api/runs/{run_id}/policy")
    assert p.status_code == 200 and len(p.content) > 1000


def test_simulate_and_evaluate(client, run_id):
    for pol in ("ql", "greedy", "random"):
        s = client.post(f"/api/runs/{run_id}/simulate", json={"policy": pol, "slots": 60}).json()
        occ, act, code = np.array(s["occ"]), np.array(s["act"]), np.array(s["code"])
        assert occ.shape == (60, 16) and act.shape == (60, 6) and set(np.unique(occ)) <= {0, 1}
        assert act.max() <= 16 and set(np.unique(code)) <= {0, 1, 2, 3}
        # outcome codes must be consistent with the world: code 2 (PU collision) only on occupied channels
        for t, i in zip(*np.where(code == 2)):
            assert occ[t, act[t, i] - 1] == 1
        for t, i in zip(*np.where(code == 1)):
            assert occ[t, act[t, i] - 1] == 0
    e = client.post(f"/api/runs/{run_id}/evaluate", json={"episodes": 5}).json()
    assert set(e) == {"ql", "greedy", "random"} and 0 <= e["ql"]["utilisation"] <= 1
    e2 = client.post(f"/api/runs/{run_id}/evaluate", json={"episodes": 5, "eta_pu": 0.8}).json()
    assert e2["random"]["pu_collision"] > e["random"]["pu_collision"]  # more PU activity -> more random collisions


def test_qvalues_policymap_bellman(client, run_id):
    q = client.get(f"/api/runs/{run_id}/qvalues", params={"su": 0, "c_prev": 3, "occupied": 0, "best": 7}).json()
    assert len(q["q_values"]) == 17 and q["action"] == int(np.argmax(q["q_values"]))
    assert client.get(f"/api/runs/{run_id}/qvalues", params={"su": 99}).status_code == 422
    assert client.get(f"/api/runs/{run_id}/qvalues", params={"c_prev": 40}).status_code == 422
    pm = client.get(f"/api/runs/{run_id}/policy-map").json()
    assert np.array(pm["free"]["matrix"]).shape == (17, 17) and 0 < pm["visited_fraction"] <= 1 and "behaviour" in pm
    b = client.get(f"/api/runs/{run_id}/bellman", params={"steps": 8}).json()
    assert len(b["rows"]) == 8 and b["alpha"] == 0.1


def test_decide_edge_inference(client, run_id):
    body = {"su": 1, "prev_channel": 2, "sensed_busy": [i % 3 == 0 for i in range(16)], "snr": [float(i + 1) for i in range(16)], "queue": 4}
    d = client.post(f"/api/runs/{run_id}/decide", json=body).json()
    assert 0 <= d["action"] <= 16 and d["latency_ms"] < 50 and len(d["q_values"]) == 17
    bad = client.post(f"/api/runs/{run_id}/decide", json={**body, "snr": [1.0]})
    assert bad.status_code == 422


def test_reward_endpoint_matches_equation3(client):
    r = client.post("/api/reward", json={"action": "transmit", "capacity_ratio": 0.8, "switched": True}).json()
    assert r["R"] == pytest.approx(0.8 - 0.3)
    r = client.post("/api/reward", json={"action": "transmit", "pu_active": True, "switched": True}).json()
    assert r["R"] == pytest.approx(-1.5 - 0.3) and r["throughput"] == 0
    r = client.post("/api/reward", json={"action": "backoff", "all_busy": True}).json()
    assert r["R"] == pytest.approx(0.1)
    r = client.post("/api/reward", json={"action": "backoff", "all_busy": False}).json()
    assert r["R"] == pytest.approx(-0.2)
    r = client.post("/api/reward", json={"action": "transmit", "su_collision": True, "ql": {"rho_su_collision": 0}}).json()
    assert r["R"] == 0  # strict paper reward


def test_channel_endpoint_rayleigh_physics(client):
    r = client.post("/api/channel", json={"distance_m": 50}).json()
    assert r["mean_snr_db"] == pytest.approx(22.4, abs=0.5) and r["noise_dbm"] == pytest.approx(-104, abs=0.01)
    assert r["mean_capacity_mbps"] < r["jensen_bound_mbps"]              # Jensen: E[log(1+x)] <= log(1+E[x])
    assert r["outage_prob_3db"] == pytest.approx(1 - np.exp(-2 / (10 ** (r["mean_snr_db"] / 10))), abs=0.01)  # Rayleigh outage law
    far = client.post("/api/channel", json={"distance_m": 200}).json()
    assert far["mean_capacity_mbps"] < r["mean_capacity_mbps"]
    pu = client.post("/api/channel", json={"distance_m": 50, "pu_active_interference_dbm": -90}).json()
    assert pu["mean_capacity_mbps"] < r["mean_capacity_mbps"]


def test_validation_errors(client):
    assert client.post("/api/runs", json={"ql": {"episodes": 999999}}).status_code == 422
    assert client.post("/api/runs", json={"ql": {"eps_min": 0.9, "eps0": 0.5}}).status_code == 422
    assert client.post("/api/runs", json={"env": {"K": 1}}).status_code == 422
    big = client.post("/api/runs", json={"env": {"N": 64, "K": 32}, "ql": {"episodes": 20000, "slots": 5000}})
    assert big.status_code == 422 and "too large" in big.json()["detail"]
    assert client.get("/api/runs/99999").status_code == 404
    assert client.get("/api/runs/99999/metrics").status_code == 404
    assert client.post("/api/experiments", json={"suite": "nope"}).status_code == 422


def test_stop_and_delete_run(client):
    r = client.post("/api/runs", json={"ql": {"episodes": 5000, "slots": 500}, "env": {"N": 30}}).json()
    time.sleep(0.5)
    assert client.delete(f"/api/runs/{r['id']}").status_code == 409  # cannot delete a running job
    assert client.post(f"/api/runs/{r['id']}/stop").json()["stopping"] is True
    done = wait_run(client, r["id"])
    assert done["status"] == "stopped" and 0 < done["episodes_done"] < 5000
    assert client.get(f"/api/runs/{r['id']}/qvalues").status_code == 200   # partial policy is usable
    assert client.delete(f"/api/runs/{r['id']}").status_code == 204
    assert client.get(f"/api/runs/{r['id']}").status_code == 404


def test_policy_unavailable_while_untrained(client):
    rid = db.create_run("never ran", api_mod.EnvConfig(), api_mod.QLConfig(), True)
    assert client.post(f"/api/runs/{rid}/simulate", json={}).status_code == 409


def test_experiment_workflow_and_latest_results(client):
    e = client.post("/api/experiments", json={"suite": "convergence", "quick": True}).json()
    t = time.time()
    while time.time() - t < 120:
        s = client.get(f"/api/experiments/{e['id']}").json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert s["status"] == "done", s
    r = s["result"]
    assert len(r["ql"]["mean"]) == len(r["x"]) and r["_meta"]["quick"] is True
    latest = client.get("/api/experiments/results/latest").json()
    assert "convergence" in latest["results"] and latest["sources"]["convergence"]["experiment_id"] == e["id"]
    assert any(x["id"] == e["id"] for x in client.get("/api/experiments").json())


def test_experiment_all_suites_quick(client):
    e = client.post("/api/experiments", json={"suite": "all", "quick": True}).json()
    t = time.time()
    while time.time() - t < 300:
        s = client.get(f"/api/experiments/{e['id']}").json()
        if s["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert s["status"] == "done", s
    assert set(s["result"]) == set(client.get("/api/meta").json()["suites"])
    r = s["result"]
    assert r["throughput_vs_n"]["N"] == [5, 10, 20, 40] and len(r["alpha"]["table"]) == 4
    assert r["runtime"]["python_ms"]["ql"] < 5 and r["ablation"]["variants"][0]["name"].startswith("MO-QL")
    latest = client.get("/api/experiments/results/latest").json()["results"]
    assert "ablation" in latest and "runtime" in latest


def test_api_key_protection(client, monkeypatch):
    monkeypatch.setattr(api_mod, "API_KEY", "s3cret")
    assert client.post("/api/runs", json={}).status_code == 401
    assert client.post("/api/runs", json={"ql": {"episodes": 2, "slots": 10}}, headers={"X-API-Key": "s3cret"}).status_code == 201
    assert client.get("/api/runs").status_code == 200  # reads stay open


def test_interrupted_jobs_are_flagged_on_restart():
    rid = db.create_run("zombie", api_mod.EnvConfig(), api_mod.QLConfig(), True)
    db.update_run(rid, status="running")
    db.mark_interrupted()
    assert db.get_run(rid)["status"] == "interrupted"


def test_database_copied_from_another_machine():
    """A run whose stored policy path is from another machine must not break the API (regression: Internal Server Error)."""
    from spectrarl import db
    from spectrarl.config import EnvConfig, QLConfig
    rid = db.create_run("foreign", EnvConfig(), QLConfig(episodes=10, slots=10), False)
    db.update_run(rid, status="done", policy_path="/nonexistent/other-machine/run_x.npz")
    db.drop_missing_policies()
    r = db.get_run(rid)
    assert r["policy_path"] is None and r["status"] == "failed"
    db.delete_run(rid)
