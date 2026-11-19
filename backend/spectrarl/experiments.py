"""Experiment suites that regenerate every results slide with *measured* numbers.

convergence | throughput_vs_n | collision_vs_eta | utilisation | handovers | alpha | runtime | ablation
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

from . import core as C
from .config import EnvConfig, QLConfig
from .runner import Policy, decide, evaluate, new_qtable, observation_to_state, train, METRICS
from .world import build_world

SUITES = ["convergence", "throughput_vs_n", "collision_vs_eta", "utilisation", "handovers", "alpha", "ablation", "pareto", "runtime"]
KINDS = ("ql", "greedy", "random")


def episodes_to_fraction(curve, frac=0.9, win=10):
    """First episode at which the smoothed reward has closed `frac` of the gap between its start and plateau."""
    c = np.asarray(curve, float)
    if len(c) < 3 * win:
        return len(c)
    sm = np.convolve(c, np.ones(win) / win, mode="valid")
    start, final = sm[:5].mean(), c[-max(win, len(c) // 10):].mean()
    if abs(final - start) < 1e-9:
        return 1
    prog = (sm - start) / (final - start)
    idx = np.flatnonzero(prog >= frac)
    return int(idx[0] + win) if len(idx) else len(c)


def _scale(episodes: int, quick: bool) -> tuple[int, int]:
    return (150, 200) if quick else (episodes, 500)


def _pmap(fn, items, workers=2):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def _agg(rows: list[dict]) -> dict:
    return {m: float(np.mean([r[m] for r in rows])) for m in rows[0] if m != "channel_util"}


def convergence(env: EnvConfig, ql: QLConfig, quick: bool, seeds: int = 3, progress=None) -> dict:
    E, T = _scale(ql.episodes, quick)
    curves = {k: [] for k in KINDS}
    for sd in range(seeds):
        q = ql.model_copy(update={"episodes": E, "slots": T, "seed": ql.seed + sd})
        hist = []
        train(env, q, True, hist.append)
        for k in KINDS:
            curves[k].append([h[k]["reward"] for h in hist])
        if progress:
            progress((sd + 1) / seeds, f"convergence seed {sd + 1}/{seeds}")
    out = {"episodes": E, "slots": T, "seeds": seeds}
    marks = [m for m in [100, 200, 350, 500, 750, 1000] if m <= E] if not quick else [25, 50, 100, 150]
    for k in KINDS:
        arr = np.array(curves[k])
        out[k] = {"mean": arr.mean(0)[::2].tolist(), "std": arr.std(0)[::2].tolist(),
                  "table": [{"episode": m, "mean": float(arr[:, max(0, m - 10):m].mean()), "std": float(arr[:, max(0, m - 10):m].mean(1).std())} for m in marks]}
    out["x"] = list(range(1, E + 1))[::2]
    out["episodes_to_90"] = [episodes_to_fraction(c) for c in curves["ql"]]
    out["final_reward"] = {k: float(np.array(curves[k])[:, -50:].mean()) for k in KINDS}
    return out


def _train_eval(env: EnvConfig, ql: QLConfig, n_eval: int):
    pol = train(env, ql, baselines=False)
    w = build_world(env)
    res = {"ql": evaluate("ql", env, ql, pol, n_eval, world=w)}
    for k in ("greedy", "random"):
        res[k] = evaluate(k, env, ql, None, n_eval, world=w)
    return res


def throughput_vs_n(env, ql, quick, progress=None):
    E, T = _scale(ql.episodes, quick)
    Ns = [5, 10, 20, 40] if quick else [5, 10, 15, 20, 25, 30, 35, 40]
    q = ql.model_copy(update={"episodes": E, "slots": T})
    rows = _pmap(lambda n: _train_eval(env.model_copy(update={"N": n}), q, 15 if quick else 30), Ns)
    if progress:
        progress(1.0, "throughput sweep done")
    return {"N": Ns, "episodes": E, **{k: {m: [r[k][m] for r in rows] for m in METRICS + [m + "_std" for m in METRICS]} for k in KINDS}}


def collision_vs_eta(env, ql, quick, progress=None):
    E, T = _scale(ql.episodes, quick)
    etas = [0.1, 0.4, 0.7, 0.9] if quick else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q = ql.model_copy(update={"episodes": E, "slots": T})
    rows = _pmap(lambda e: _train_eval(env.model_copy(update={"eta_pu": e}), q, 15 if quick else 30), etas)
    return {"eta": etas, "episodes": E, **{k: {m: [r[k][m] for r in rows] for m in METRICS + [m + "_std" for m in METRICS]} for k in KINDS}}


def utilisation(env, ql, quick, progress=None):
    E, T = _scale(ql.episodes, quick)
    q = ql.model_copy(update={"episodes": E, "slots": T})
    res = _train_eval(env, q, 20 if quick else 60)
    return {"K": env.K, "episodes": E, **{k: {"channel_util": res[k]["channel_util"], "avg": res[k]["utilisation"], "std": res[k]["utilisation_std"]} for k in KINDS}}


def handovers(env, ql, quick, progress=None):
    E, T = _scale(ql.episodes, quick)
    dyns = [0.15, 0.6, 1.5, 3.0] if quick else [0.15, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0]
    q = ql.model_copy(update={"episodes": E, "slots": T})
    rows = _pmap(lambda d: _train_eval(env.model_copy(update={"pu_dynamics": d}), q, 15 if quick else 30), dyns)
    return {"pu_dynamics": dyns, "pu_flips_per_1000": [r["ql"]["pu_flips_per_1000"] for r in rows], "episodes": E,
            **{k: {"handovers_per_min": [r[k]["handovers_per_min"] for r in rows], "throughput_mbps": [r[k]["throughput_mbps"] for r in rows]} for k in KINDS}}


def alpha(env, ql, quick, seeds: int = 3, progress=None):
    E, T = _scale(ql.episodes, quick)
    alphas = [0.01, 0.1, 0.3, 0.5]

    def one(args):
        a, sd = args
        q = ql.model_copy(update={"episodes": E, "slots": T, "alpha": a, "seed": ql.seed + sd})
        h = []
        train(env, q, False, h.append)
        return [x["ql"]["reward"] for x in h]

    jobs = [(a, s) for a in alphas for s in range(seeds)]
    res = _pmap(one, jobs)
    out = {"alpha": alphas, "episodes": E, "curves": [], "table": []}
    for i, a in enumerate(alphas):
        arr = np.array(res[i * seeds:(i + 1) * seeds])
        late = arr[:, int(E * 0.7):]
        out["curves"].append({"alpha": a, "mean": arr.mean(0)[::2].tolist(), "std": arr.std(0)[::2].tolist()})
        out["table"].append({"alpha": a, "episodes_to_90": float(np.mean([episodes_to_fraction(c) for c in arr])),
                             "variance": float(late.var(axis=1).mean()), "final_reward": float(arr[:, -50:].mean())})
    out["x"] = list(range(1, E + 1))[::2]
    return out


ABLATIONS = {
    "MO-QL (full, extended reward)": {},
    "MO-QL (strict paper reward, no SU-collision term)": {"rho_su_collision": 0.0},
    "No switching penalty": {"rho_switch": 0.0},
    "No PU-collision penalty": {"rho_collision": 0.0},
    "Throughput-only Q-learning (simple QL baseline)": {"rho_collision": 0.0, "rho_switch": 0.0, "rho_su_collision": 0.0, "idle_reward": 0.0, "idle_penalty": 0.0},
}


def ablation(env, ql, quick, progress=None):
    E, T = _scale(ql.episodes, quick)

    def one(item):
        name, upd = item
        q = ql.model_copy(update={"episodes": E, "slots": T, **upd})
        pol = train(env, q, baselines=False)
        # evaluate with the *full* reward weights so rewards are comparable across variants
        r = evaluate("ql", env, ql, pol, 15 if quick else 40)
        return name, r

    rows = _pmap(one, list(ABLATIONS.items()))
    w = build_world(env)
    base = {k: evaluate(k, env, ql, None, 15 if quick else 40, world=w) for k in ("greedy", "random")}
    out = {"variants": [{"name": n, **{m: r[m] for m in METRICS}} for n, r in rows], "episodes": E}
    out["variants"] += [{"name": "Greedy", **{m: base["greedy"][m] for m in METRICS}}, {"name": "Random", **{m: base["random"][m] for m in METRICS}}]
    return out


def pareto(env, ql, quick, progress=None):
    """Multi-objective trade-off: sweep the PU-collision weight rho_coll and record the resulting operating points."""
    E, T = _scale(ql.episodes, quick)
    ws = [0.5, 1.5, 5.0] if quick else [0.25, 0.75, 1.5, 3.0, 5.0, 8.0]

    def one(w):
        q = ql.model_copy(update={"episodes": E, "slots": T, "rho_collision": w})
        pol = train(env, q, baselines=False)
        return evaluate("ql", env, ql, pol, 15 if quick else 40)  # judged with the common reference weights

    rows = _pmap(one, ws)
    g = evaluate("greedy", env, ql, None, 15 if quick else 40)
    keep = ["throughput_mbps", "pu_collision", "su_collision", "utilisation", "handovers_per_min", "tx_rate", "reward"]
    return {"rho_collision": ws, "episodes": E, "ql": {m: [r[m] for r in rows] for m in keep}, "greedy": {m: g[m] for m in keep}}


def runtime(env, ql, quick, progress=None):
    """Per-decision cost. (a) compiled kernel decision rule, (b) the Python inference path an edge process would call."""
    K = env.K
    Q = np.random.default_rng(0).normal(size=(1, C.n_states(K), K + 1)).astype(np.float32)
    n = 2_000_000
    C.bench_decisions(0, Q, 10, K, 1)
    res = {"kernel_us": {}, "python_ms": {}}
    for pid, name in enumerate(KINDS):
        t = time.perf_counter()
        C.bench_decisions(pid, Q, n, K, 1)
        res["kernel_us"][name] = (time.perf_counter() - t) / n * 1e6
    pol = Policy(np.repeat(Q, env.N, 0), env, ql)
    rng = np.random.default_rng(1)
    reps = 3000 if not quick else 500
    obs = [(int(rng.integers(0, K + 1)), (rng.random(K) < 0.4).tolist(), rng.exponential(100, K).tolist(), int(rng.integers(1, 10))) for _ in range(64)]
    t = time.perf_counter()
    for i in range(reps):
        c, sb, snr, qq = obs[i % 64]
        decide(pol, 0, observation_to_state(K, c, sb, snr, qq))
    res["python_ms"]["ql"] = (time.perf_counter() - t) / reps * 1e3
    t = time.perf_counter()
    for i in range(reps):
        c, sb, snr, qq = obs[i % 64]
        free = [k for k in range(K) if not sb[k]]
        _ = 1 + max(free, key=lambda k: snr[k]) if free else 0
    res["python_ms"]["greedy"] = (time.perf_counter() - t) / reps * 1e3
    t = time.perf_counter()
    for i in range(reps):
        _ = 1 + int(rng.integers(0, K))
    res["python_ms"]["random"] = (time.perf_counter() - t) / reps * 1e3
    res["table_bytes_per_su"] = int(Q[0].nbytes)
    res["table_bytes_per_su_int8"] = int(Q[0].size)
    return res


FUNCS: dict[str, Callable] = {"convergence": convergence, "throughput_vs_n": throughput_vs_n, "collision_vs_eta": collision_vs_eta,
                              "utilisation": utilisation, "handovers": handovers, "alpha": alpha, "ablation": ablation, "pareto": pareto, "runtime": runtime}


def run_suite(name: str, env: EnvConfig, ql: QLConfig, quick: bool = False, progress: Optional[Callable] = None) -> dict:
    if name not in FUNCS:
        raise ValueError(f"unknown suite {name}; choose from {SUITES}")
    t0 = time.time()
    res = FUNCS[name](env, ql, quick, progress=progress)
    res["_meta"] = {"suite": name, "quick": quick, "seconds": time.time() - t0, "env": env.model_dump(), "ql": ql.model_dump()}
    return res
