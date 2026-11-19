"""Background execution of training runs and experiment suites."""
from __future__ import annotations
import json, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import db
from .config import EnvConfig, QLConfig, check_budget
from .experiments import run_suite, SUITES
from .runner import Policy, train

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="spectrarl-job")
_stops: dict[int, threading.Event] = {}
_exp_stops: dict[int, threading.Event] = {}


def submit_run(name: str, env: EnvConfig, ql: QLConfig, baselines: bool = True) -> int:
    check_budget(env, ql, 3 if baselines else 1)
    rid = db.create_run(name, env, ql, baselines)
    _stops[rid] = threading.Event()
    _pool.submit(_run_train, rid, env, ql, baselines)
    return rid


def stop_run(rid: int) -> bool:
    ev = _stops.get(rid)
    if ev:
        ev.set()
        return True
    return False


def _summ(hist_tail: list[dict]) -> dict:
    out = {}
    for pol in ("ql", "greedy", "random"):
        rows = [h[pol] for h in hist_tail if pol in h]
        if rows:
            out[pol] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0] if m != "channel_util"}
    return out


def _run_train(rid: int, env: EnvConfig, ql: QLConfig, baselines: bool):
    stop = _stops[rid]
    buf: list[dict] = []
    tail: list[dict] = []
    try:
        db.update_run(rid, status="running")

        def cb(rec):
            buf.append(rec)
            tail.append(rec)
            del tail[:-20]
            if len(buf) >= 5 or rec["episode"] == ql.episodes:
                db.add_metrics(rid, buf)
                db.update_run(rid, episodes_done=rec["episode"])
                buf.clear()

        pol = train(env, ql, baselines, cb, stop.is_set)
        if buf:
            db.add_metrics(rid, buf)
        done = pol.meta["episodes_trained"]
        path = db.policies_dir() / f"run_{rid}.npz"
        path.write_bytes(pol.to_bytes())
        db.update_run(rid, status="stopped" if stop.is_set() and done < ql.episodes else "done", episodes_done=done,
                      finished_at=time.time(), policy_path=str(path), summary_json=json.dumps(_summ(tail)))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        db.update_run(rid, status="failed", error=str(e), finished_at=time.time())
    finally:
        _stops.pop(rid, None)


@lru_cache(maxsize=8)
def load_policy(rid: int, mtime_ns: int = 0) -> Policy:
    r = db.get_run(rid)
    if not r or not r.get("policy_path"):
        raise KeyError("no trained policy for this run")
    return Policy.from_bytes(db.policy_file(rid).read_bytes())


def get_policy(rid: int) -> Policy:
    r = db.get_run(rid)
    if not r or not r.get("policy_path"):
        raise KeyError("no trained policy for this run")
    return load_policy(rid, db.policy_file(rid).stat().st_mtime_ns)


def submit_experiment(suite: str, env: EnvConfig, ql: QLConfig, quick: bool) -> int:
    if suite != "all" and suite not in SUITES:
        raise ValueError(f"unknown suite {suite}")
    eid = db.create_experiment(suite, quick, env, ql)
    _pool.submit(_run_experiment, eid, suite, env, ql, quick)
    return eid


def _run_experiment(eid: int, suite: str, env: EnvConfig, ql: QLConfig, quick: bool):
    try:
        db.update_experiment(eid, status="running")
        names = SUITES if suite == "all" else [suite]
        res = {}
        for i, n in enumerate(names):
            db.update_experiment(eid, progress=i / len(names), message=f"running {n} ({i + 1}/{len(names)})")
            res[n] = run_suite(n, env, ql, quick)
        db.update_experiment(eid, status="done", progress=1.0, message="complete", finished_at=time.time(),
                             result_json=res if suite == "all" else res[suite])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        db.update_experiment(eid, status="failed", error=str(e), finished_at=time.time())
