"""FastAPI application: REST API + static front end."""
from __future__ import annotations

import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, core as C, db, jobs
from .config import ASSUMPTIONS, EnvConfig, QLConfig, check_budget
from .experiments import SUITES
from .paper_reference import PAPER
from .runner import bellman_trace, behaviour_stats, decide, evaluate, observation_to_state, run_one, simulate
from .world import build_world, capacity_bps, path_gain_const, sinr

API_KEY = os.environ.get("SPECTRARL_API_KEY")  # optional: when set, mutating endpoints require X-API-Key


def require_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    db.mark_interrupted()
    db.drop_missing_policies()
    yield


app = FastAPI(title="SpectraRL API", version=__version__, lifespan=lifespan,
              description="RL-based dynamic spectrum allocation (MO-QL-DSA) for cognitive radio networks")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("SPECTRARL_CORS", "*").split(","), allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ValueError)
async def _value_error(_, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _run_or_404(rid: int) -> dict:
    r = db.get_run(rid)
    if not r:
        raise HTTPException(404, f"run {rid} not found")
    return r


def _policy_or_404(rid: int):
    _run_or_404(rid)
    try:
        return jobs.get_policy(rid)
    except KeyError:
        raise HTTPException(409, "this run has no trained policy yet (still training or failed)")


# ---------------------------------------------------------------- meta
@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "time": time.time()}


@app.get("/api/meta")
def meta():
    return {"version": __version__, "defaults": {"env": EnvConfig().model_dump(), "ql": QLConfig().model_dump()},
            "assumptions": ASSUMPTIONS, "suites": SUITES, "paper": PAPER,
            "state_size": C.n_states(EnvConfig().K), "auth_required": bool(API_KEY)}


# ---------------------------------------------------------------- training runs
class TrainRequest(BaseModel):
    name: str = Field("Training run", max_length=80)
    env: EnvConfig = EnvConfig()
    ql: QLConfig = QLConfig()
    baselines: bool = True


@app.post("/api/runs", status_code=201, dependencies=[Depends(require_key)])
def create_run(req: TrainRequest):
    rid = jobs.submit_run(req.name, req.env, req.ql, req.baselines)
    return db.get_run(rid)


@app.get("/api/runs")
def list_runs():
    return db.list_runs()


@app.get("/api/runs/{rid}")
def get_run(rid: int):
    return _run_or_404(rid)


@app.get("/api/runs/{rid}/metrics")
def run_metrics(rid: int, since: int = Query(0, ge=0)):
    _run_or_404(rid)
    return db.get_metrics(rid, since)


@app.post("/api/runs/{rid}/stop", dependencies=[Depends(require_key)])
def stop(rid: int):
    _run_or_404(rid)
    return {"stopping": jobs.stop_run(rid)}


@app.delete("/api/runs/{rid}", status_code=204, dependencies=[Depends(require_key)])
def delete_run(rid: int):
    r = _run_or_404(rid)
    if r["status"] in ("queued", "running"):
        raise HTTPException(409, "stop the run before deleting it")
    db.delete_run(rid)
    return Response(status_code=204)


@app.get("/api/runs/{rid}/policy")
def download_policy(rid: int):
    r = _run_or_404(rid)
    if not r.get("policy_path"):
        raise HTTPException(409, "no policy yet")
    return FileResponse(db.policy_file(rid), media_type="application/octet-stream", filename=f"spectrarl_policy_run{rid}.npz")


class SimRequest(BaseModel):
    policy: Literal["ql", "greedy", "random"] = "ql"
    slots: int = Field(120, ge=10, le=600)
    seed: int = Field(1234, ge=0)
    eta_pu: Optional[float] = Field(None, ge=0.02, le=0.95)


@app.post("/api/runs/{rid}/simulate")
def simulate_run(rid: int, req: SimRequest):
    pol = _policy_or_404(rid)
    env = pol.env if req.eta_pu is None else pol.env.model_copy(update={"eta_pu": req.eta_pu})
    res = simulate(req.policy, env, pol.ql, pol, req.slots, req.seed)
    res["K"], res["N"] = env.K, env.N
    return res


class EvalRequest(BaseModel):
    episodes: int = Field(20, ge=1, le=200)
    seed: int = 90000
    eta_pu: Optional[float] = Field(None, ge=0.02, le=0.95)


@app.post("/api/runs/{rid}/evaluate")
def evaluate_run(rid: int, req: EvalRequest):
    pol = _policy_or_404(rid)
    env = pol.env if req.eta_pu is None else pol.env.model_copy(update={"eta_pu": req.eta_pu})
    w = build_world(env)
    return {k: evaluate(k, env, pol.ql, pol if k == "ql" else None, req.episodes, req.seed, w) for k in ("ql", "greedy", "random")}


@app.get("/api/runs/{rid}/qvalues")
def qvalues(rid: int, su: int = Query(0, ge=0), c_prev: int = Query(0, ge=0), occupied: int = Query(0, ge=0, le=1),
            best: int = Query(0, ge=0), queue_bucket: int = Query(2, ge=0, le=2), free_bucket: int = Query(2, ge=0, le=2)):
    pol = _policy_or_404(rid)
    K = pol.env.K
    if su >= pol.env.N or c_prev > K or best > K:
        raise HTTPException(422, "su / channel index out of range")
    s = int(C.encode_state(c_prev, occupied, best, queue_bucket, free_bucket, K))
    d = decide(pol, su, s)
    return {**d, "K": K, "decoded": {"c_prev": c_prev, "occupied": occupied, "best": best, "queue_bucket": queue_bucket, "free_bucket": free_bucket}}


def run_one_trace(pol, slots: int = 500, seed: int = 4242):
    return run_one(C.POLICY_QL, build_world(pol.env), pol.ql, pol.q, 0.0, False, seed, seed + 1, T=slots, trace=True)["trace"]


@app.get("/api/runs/{rid}/policy-map")
def policy_map(rid: int, su: int = Query(0, ge=0)):
    """Mean Q(c_prev, a) over visited states (rows with any non-zero Q), split by whether C_{t-1} is sensed occupied."""
    pol = _policy_or_404(rid)
    K = pol.env.K
    if su >= pol.env.N:
        raise HTTPException(422, "su out of range")
    q = pol.q[su]
    S = q.shape[0]
    visited = np.abs(q).sum(1) > 0
    cp = np.zeros(S, int); oc = np.zeros(S, int)
    for s in range(S):
        a, b, *_ = C.decode_state(s, K)
        cp[s], oc[s] = a, b
    out = {}
    for o in (0, 1):
        m = np.zeros((K + 1, K + 1))
        cnt = np.zeros(K + 1)
        for c in range(K + 1):
            sel = visited & (cp == c) & (oc == o)
            cnt[c] = sel.sum()
            if sel.any():
                m[c] = q[sel].mean(0)
        out["free" if o == 0 else "occupied"] = {"matrix": m.tolist(), "visited_states": cnt.tolist()}
    out["K"] = K
    out["visited_fraction"] = float(visited.mean())
    tr = run_one_trace(pol)
    out["behaviour"] = behaviour_stats(tr["act"], tr["code"])
    return out


@app.get("/api/runs/{rid}/bellman")
def bellman(rid: int, steps: int = Query(12, ge=1, le=60), seed: int = 3):
    pol = _policy_or_404(rid)
    return {"rows": bellman_trace(pol, pol.env, pol.ql, steps, seed), "alpha": pol.ql.alpha, "gamma": pol.ql.gamma}


class DecideRequest(BaseModel):
    su: int = Field(0, ge=0)
    prev_channel: int = Field(0, ge=0)
    sensed_busy: list[bool]
    snr: list[float]
    queue: int = Field(1, ge=0)


@app.post("/api/runs/{rid}/decide")
def decide_run(rid: int, req: DecideRequest):
    """Real single-decision inference, as an edge radio would call it. Latency is measured server-side."""
    pol = _policy_or_404(rid)
    K = pol.env.K
    if len(req.sensed_busy) != K or len(req.snr) != K or req.prev_channel > K or req.su >= pol.env.N:
        raise HTTPException(422, f"need exactly K={K} sensing/SNR entries and valid indices")
    t = time.perf_counter()
    s = observation_to_state(K, req.prev_channel, req.sensed_busy, req.snr, req.queue)
    d = decide(pol, req.su, s)
    d["latency_ms"] = (time.perf_counter() - t) * 1e3
    return d


# ---------------------------------------------------------------- analytic model endpoints
class RewardRequest(BaseModel):
    action: Literal["transmit", "backoff"] = "transmit"
    pu_active: bool = False
    su_collision: bool = False
    switched: bool = False
    all_busy: bool = False
    capacity_ratio: float = Field(0.7, ge=0, le=1)
    ql: QLConfig = QLConfig()


@app.post("/api/reward")
def reward(req: RewardRequest):
    """Eq. (3) evaluated by the *same* compiled function the simulator uses."""
    P = np.zeros(C.P_COUNT)
    q = req.ql
    P[C.P_W1], P[C.P_RCOLL], P[C.P_RSW], P[C.P_RI], P[C.P_RP], P[C.P_RSU] = q.w_throughput, q.rho_collision, q.rho_switch, q.idle_reward, q.idle_penalty, q.rho_su_collision
    if req.action == "backoff":
        R, T, I, H, Pi = C.reward_terms(False, False, False, False, True, req.all_busy, 0.0, P)
    else:
        success = not req.pu_active and not req.su_collision
        R, T, I, H, Pi = C.reward_terms(success, req.pu_active, req.su_collision, req.switched, False, False, req.capacity_ratio, P)
    return {"R": R, "throughput": q.w_throughput * T, "collision": I, "switch": H, "idle": Pi}


class ChannelRequest(BaseModel):
    env: EnvConfig = EnvConfig()
    distance_m: float = Field(50.0, ge=1, le=1000)
    pu_active_interference_dbm: Optional[float] = Field(None, description="active-PU interference power at the SU receiver")
    samples: int = Field(20000, ge=100, le=200000)
    seed: int = 1


@app.post("/api/channel")
def channel(req: ChannelRequest):
    """Eq. (1) + (2) with Rayleigh fading (|h|^2 ~ Exp(1)), Monte-Carlo."""
    e = req.env
    pl0 = path_gain_const(e)
    ipu = 0.0 if req.pu_active_interference_dbm is None else 10 ** (req.pu_active_interference_dbm / 10) / 1000
    h2 = np.random.default_rng(req.seed).exponential(1.0, req.samples)
    g = e.ptx_w * pl0 * h2 * req.distance_m ** (-e.path_loss_exp) / (e.noise_w + ipu)
    cap = e.bandwidth_hz * np.log2(1 + g) / 1e6
    mean_snr = e.ptx_w * pl0 * req.distance_m ** (-e.path_loss_exp) / (e.noise_w + ipu)
    hist, edges = np.histogram(cap, bins=40)
    return {"mean_snr_db": 10 * math.log10(mean_snr), "mean_capacity_mbps": float(cap.mean()),
            "jensen_bound_mbps": capacity_bps(e.bandwidth_hz, mean_snr) / 1e6, "p5_mbps": float(np.percentile(cap, 5)),
            "p95_mbps": float(np.percentile(cap, 95)), "outage_prob_3db": float((g < 2.0).mean()),
            "hist": hist.tolist(), "edges": edges.tolist(), "noise_dbm": 10 * math.log10(e.noise_w * 1000)}


# ---------------------------------------------------------------- experiments
class ExpRequest(BaseModel):
    suite: str = "all"
    quick: bool = False
    env: EnvConfig = EnvConfig()
    ql: QLConfig = QLConfig()


@app.post("/api/experiments", status_code=201, dependencies=[Depends(require_key)])
def create_experiment(req: ExpRequest):
    eid = jobs.submit_experiment(req.suite, req.env, req.ql, req.quick)
    return db.get_experiment(eid, with_result=False)


@app.get("/api/experiments")
def list_experiments():
    return db.list_experiments()


@app.get("/api/experiments/results/latest")
def latest_results(quick: Optional[bool] = None):
    """Newest finished result for every suite (whether it ran alone or as part of 'all')."""
    out, src = {}, {}
    for e in db.list_experiments():
        if e["status"] != "done" or (quick is not None and bool(e["quick"]) != quick):
            continue
        full = db.get_experiment(e["id"])
        res = full["result"]
        parts = res if full["suite"] == "all" else {full["suite"]: res}
        for k, v in parts.items():
            if k not in out:
                out[k], src[k] = v, {"experiment_id": e["id"], "quick": bool(e["quick"]), "finished_at": e["finished_at"]}
    return {"results": out, "sources": src}


@app.get("/api/experiments/{eid}")
def get_experiment(eid: int):
    e = db.get_experiment(eid)
    if not e:
        raise HTTPException(404, "experiment not found")
    return e


@app.delete("/api/experiments/{eid}", status_code=204, dependencies=[Depends(require_key)])
def delete_experiment(eid: int):
    db.delete_experiment(eid)
    return Response(status_code=204)


# ---------------------------------------------------------------- static front end
DIST = Path(os.environ.get("SPECTRARL_DIST", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
