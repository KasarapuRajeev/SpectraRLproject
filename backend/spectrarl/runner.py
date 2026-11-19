"""Training / evaluation / tracing drivers around the numba kernel."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import core as C
from .config import EnvConfig, QLConfig, check_budget
from .world import World, build_world

POLICY_NAMES = {C.POLICY_QL: "ql", C.POLICY_GREEDY: "greedy", C.POLICY_RANDOM: "random"}
METRICS = ["reward", "throughput_mbps", "pu_collision", "su_collision", "utilisation", "handovers_per_min",
           "tx_rate", "pu_flips_per_1000"]


def _state(seed: int, *stream: int) -> np.ndarray:
    ss = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, *[int(s) & 0xFFFFFFFF for s in stream]])
    return ss.generate_state(1, dtype=np.uint64)


def new_qtable(cfg: EnvConfig) -> np.ndarray:
    return np.zeros((cfg.N, C.n_states(cfg.K), cfg.K + 1), dtype=np.float32)


@dataclass
class Policy:
    q: np.ndarray
    env: EnvConfig
    ql: QLConfig
    meta: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, q=self.q, env=self.env.model_dump_json(), ql=self.ql.model_dump_json(), meta=str(self.meta))
        return buf.getvalue()

    @staticmethod
    def from_bytes(b: bytes) -> "Policy":
        z = np.load(io.BytesIO(b), allow_pickle=False)
        return Policy(z["q"], EnvConfig.model_validate_json(str(z["env"])), QLConfig.model_validate_json(str(z["ql"])))


def _dummy_trace():
    return (np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.int64), np.zeros((1, 1), np.int8), np.zeros((1, C.TRACE_COLS)))


def run_one(policy: int, world: World, ql: QLConfig, Q: np.ndarray, eps: float, learn: bool, env_seed: int,
            agent_seed: int, T: Optional[int] = None, trace: bool = False) -> dict:
    cfg = world.cfg
    T = T or ql.slots
    P = world.params(ql)
    out = np.zeros(C.O_COUNT)
    used = np.zeros(cfg.K)
    opp = np.zeros(cfg.K)
    if trace:
        tr = (np.zeros((T, cfg.K), np.uint8), np.zeros((T, cfg.N), np.int64), np.zeros((T, cfg.N), np.int8),
              np.zeros((T, C.TRACE_COLS)))
    else:
        tr = _dummy_trace()
    C.run_episode(policy, cfg.N, cfg.K, cfg.M, T, world.pu_a, world.pu_b, world.owner, world.rx_mean, world.intf_mean,
                  P, Q, ql.alpha, ql.gamma, eps, learn, _state(env_seed, 1), _state(agent_seed, 2), out, used, opp,
                  trace, tr[0], tr[1], tr[2], tr[3])
    res = derive(out, used, opp, cfg, T)
    if trace:
        res["trace"] = {"occ": tr[0], "act": tr[1], "code": tr[2], "su0": tr[3]}
    return res


def derive(out, used, opp, cfg: EnvConfig, T: int) -> dict:
    minutes = T * cfg.slot_ms / 60000.0
    tx = max(out[C.O_TX], 1.0)
    return {
        "reward": float(out[C.O_REWARD]),
        "throughput_mbps": float(out[C.O_THR_BITS] / T / 1e6),
        "pu_collision": float(out[C.O_PU_COLL] / tx),
        "su_collision": float(out[C.O_SU_COLL] / tx),
        "utilisation": float(used.sum() / max(opp.sum(), 1.0)),
        "handovers_per_min": float(out[C.O_SWITCH] / cfg.N / minutes),
        "tx_rate": float(out[C.O_TX] / (cfg.N * T)),
        "pu_flips_per_1000": float(out[C.O_FLIPS] / cfg.M / T * 1000.0),
        "channel_util": (used / np.maximum(opp, 1.0)).tolist(),
    }


def train(env_cfg: EnvConfig, ql: QLConfig, baselines: bool = True, callback: Optional[Callable[[dict], None]] = None,
          should_stop: Optional[Callable[[], bool]] = None, world: Optional[World] = None) -> Policy:
    """Algorithm 1 of the presentation. Greedy/Random run on the *same* episode seeds for a paired comparison."""
    check_budget(env_cfg, ql, 3 if baselines else 1)
    world = world or build_world(env_cfg)
    Q = new_qtable(env_cfg)
    eps = ql.eps0
    t0 = time.time()
    for ep in range(1, ql.episodes + 1):
        if should_stop and should_stop():
            break
        env_seed = ql.seed * 1_000_003 + ep
        rec = {"episode": ep, "eps": eps}
        rec["ql"] = run_one(C.POLICY_QL, world, ql, Q, eps, True, env_seed, env_seed + 17)
        if baselines:
            rec["greedy"] = run_one(C.POLICY_GREEDY, world, ql, Q, 0.0, False, env_seed, env_seed + 17)
            rec["random"] = run_one(C.POLICY_RANDOM, world, ql, Q, 0.0, False, env_seed, env_seed + 17)
        eps = max(ql.eps_min, eps * ql.eps_decay)  # eps <- max(eps_min, eps * delta)
        rec["elapsed_s"] = time.time() - t0
        if callback:
            callback(rec)
    return Policy(Q, env_cfg, ql, {"episodes_trained": ep if not (should_stop and should_stop()) else ep - 1})


def evaluate(kind: str, env_cfg: EnvConfig, ql: QLConfig, policy: Optional[Policy] = None, n_eps: int = 30,
             seed_base: int = 90_000, world: Optional[World] = None) -> dict:
    """Frozen-policy evaluation (eps = 0, no learning) on fresh seeds. kind in {ql, greedy, random}."""
    world = world or build_world(env_cfg)
    pid = {"ql": C.POLICY_QL, "greedy": C.POLICY_GREEDY, "random": C.POLICY_RANDOM}[kind]
    if kind == "ql" and policy is None:
        raise ValueError("ql evaluation needs a trained policy")
    Q = policy.q if policy is not None else new_qtable(env_cfg)
    rows = []
    for i in range(n_eps):
        rows.append(run_one(pid, world, ql, Q, 0.0, False, seed_base + i, seed_base + i + 5))
    agg = {}
    for m in METRICS:
        v = np.array([r[m] for r in rows])
        agg[m] = float(v.mean())
        agg[m + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    agg["channel_util"] = np.mean([r["channel_util"] for r in rows], axis=0).tolist()
    return agg


def simulate(kind: str, env_cfg: EnvConfig, ql: QLConfig, policy: Optional[Policy], slots: int, seed: int) -> dict:
    world = build_world(env_cfg)
    pid = {"ql": C.POLICY_QL, "greedy": C.POLICY_GREEDY, "random": C.POLICY_RANDOM}[kind]
    Q = policy.q if (kind == "ql" and policy is not None) else new_qtable(env_cfg)
    res = run_one(pid, world, ql, Q, 0.0, False, seed, seed + 5, T=slots, trace=True)
    tr = res.pop("trace")
    res["occ"] = tr["occ"].tolist()
    res["act"] = tr["act"].tolist()
    res["code"] = tr["code"].tolist()
    res["K"], res["N"], res["slots"] = env_cfg.K, env_cfg.N, slots
    return res


def bellman_trace(policy: Policy, env_cfg: EnvConfig, ql: QLConfig, steps: int = 12, seed: int = 3) -> list:
    """Run a *learning* episode on a copy of the table and return SU0's real (s, a, R, Q_old, Q_new) tuples."""
    world = build_world(env_cfg)
    Q = policy.q.copy()
    res = run_one(C.POLICY_QL, world, ql, Q, ql.eps_min, True, seed, seed + 1, T=steps + 2, trace=True)
    su0 = res["trace"]["su0"]
    rows = []
    for t in range(steps):
        s, a, r, qo, qn, ex = su0[t]
        cp, oc, best, qb, nf = C.decode_state(int(s), env_cfg.K)
        rows.append({"t": t, "state": int(s), "c_prev": int(cp), "occupied": int(oc), "best_free": int(best),
                     "queue_bucket": int(qb), "free_bucket": int(nf), "action": int(a), "reward": float(r),
                     "q_old": float(qo), "q_new": float(qn), "explored": bool(ex)})
    return rows


# ---- single-decision inference (what an edge radio would run) ------------------
def observation_to_state(K: int, prev_channel: int, sensed_busy: list, snr: list, queue: int) -> int:
    free = [k for k in range(K) if not sensed_busy[k]]
    best = 0
    if free:
        best = 1 + max(free, key=lambda k: snr[k])
    occ = 1 if (prev_channel >= 1 and sensed_busy[prev_channel - 1]) else 0
    return int(C.encode_state(prev_channel, occ, best, int(C.queue_bucket(queue)), int(C.free_bucket(len(free))), K))


def decide(policy: Policy, su: int, state: int) -> dict:
    row = policy.q[su, state]
    m = float(row.max())
    a = int(np.flatnonzero(row == m)[0])
    return {"action": a, "q_values": row.astype(float).tolist(), "state": state}


def behaviour_stats(act: np.ndarray, code: np.ndarray) -> dict:
    """Interpretable behaviour measured on a rollout: how often SUs keep their channel, and whether they leave after a PU collision.
    act/code are [T, N] arrays from a trace (code 1=success, 2=PU collision, 3=SU collision)."""
    T = act.shape[0]
    prev, cur = act[:-1], act[1:]
    tx = (prev > 0) & (cur > 0)
    stay_all = float((cur[tx] == prev[tx]).mean()) if tx.any() else None
    ok = (code[:-1] == 1) & (prev > 0)
    stay_after_success = float((cur[ok] == prev[ok]).mean()) if ok.any() else None
    pu = (code[:-1] == 2) & (prev > 0)
    leave_after_pu = float((cur[pu] != prev[pu]).mean()) if pu.any() else None
    su = (code[:-1] == 3) & (prev > 0)
    leave_after_su = float((cur[su] != prev[su]).mean()) if su.any() else None
    return {"stay_rate": stay_all, "stay_after_success": stay_after_success, "leave_after_pu_collision": leave_after_pu,
            "leave_after_su_collision": leave_after_su, "idle_rate": float((act == 0).mean())}
