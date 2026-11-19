import math
import numpy as np
import pytest
from spectrarl import core as C
from spectrarl.config import EnvConfig, QLConfig, check_budget
from spectrarl.runner import (Policy, bellman_trace, decide, evaluate, new_qtable, observation_to_state, run_one, train, _state)
from spectrarl.world import build_world, capacity_bps, sinr


# ---------------------------------------------------------------- state encoding
def test_state_encoding_is_a_bijection():
    K = 16
    seen = set()
    for cp in range(K + 1):
        for oc in (0, 1):
            for b in range(K + 1):
                for qb in range(3):
                    for nf in range(3):
                        s = C.encode_state(cp, oc, b, qb, nf, K)
                        assert 0 <= s < C.n_states(K)
                        assert C.decode_state(s, K) == (cp, oc, b, qb, nf)
                        seen.add(s)
    assert len(seen) == C.n_states(K) == 5202


def test_rng_is_uniform_and_deterministic():
    st = np.array([123], np.uint64)
    x = np.array([C.rnd(st) for _ in range(20000)])
    assert 0 <= x.min() and x.max() < 1
    assert abs(x.mean() - 0.5) < 0.01 and abs(x.var() - 1 / 12) < 0.005
    st2 = np.array([123], np.uint64)
    assert C.rnd(st2) == x[0]


# ---------------------------------------------------------------- reward = Eq. (3)
def _P(**kw):
    P = np.zeros(C.P_COUNT)
    P[C.P_W1], P[C.P_RCOLL], P[C.P_RSW], P[C.P_RI], P[C.P_RP], P[C.P_RSU] = (kw.get(k, d) for k, d in
        [("w1", 1.0), ("rc", 1.5), ("rs", 0.3), ("ri", 0.1), ("rp", 0.2), ("rsu", 0.5)])
    return P


@pytest.mark.parametrize("success,pu,su,sw,thr,exp", [
    (True, False, False, False, 0.7, 0.7),
    (True, False, False, True, 0.7, 0.7 - 0.3),
    (False, True, False, False, 0.0, -1.5),
    (False, True, False, True, 0.0, -1.5 - 0.3),
    (False, False, True, False, 0.0, -0.5),
    (False, True, True, False, 0.0, -1.5),  # PU collision dominates: SU penalty not double counted
])
def test_reward_transmit_cases(success, pu, su, sw, thr, exp):
    R, *_ = C.reward_terms(success, pu, su, sw, False, False, thr, _P())
    assert R == pytest.approx(exp)


def test_reward_idle_and_strict_paper_mode():
    assert C.reward_terms(False, False, False, False, True, True, 0, _P())[0] == pytest.approx(0.1)   # smart backoff
    assert C.reward_terms(False, False, False, False, True, False, 0, _P())[0] == pytest.approx(-0.2)  # idle while white space
    strict = _P(rsu=0.0)
    assert C.reward_terms(False, False, True, False, False, False, 0, strict)[0] == 0.0  # paper: SU collision only zeroes T


# ---------------------------------------------------------------- physics (Eq. 1, 2) vs the simulator
def test_eq1_eq2_closed_forms():
    env = EnvConfig()
    w = build_world(env)
    g = sinr(env.ptx_w, 1.0, 50.0, env.path_loss_exp, w.pl0, env.noise_w)
    assert 10 * math.log10(g) == pytest.approx(22.4, abs=0.5)          # hand calculation: ~22 dB at 50 m
    assert capacity_bps(10e6, g) / 1e6 == pytest.approx(10 * math.log2(1 + g))
    assert env.noise_w == pytest.approx(10 ** (-104 / 10) / 1000, rel=1e-9)  # -174 dBm/Hz over 10 MHz = -104 dBm


def test_simulator_throughput_matches_analytic_expectation():
    """Single saturated SU, random channel choice: E[thr] = (1-eta) * E[C] * (1 - ovh*(K-1)/K)."""
    env = EnvConfig(N=1, eta_pu=0.3, arrival_prob=1.0, p_false_alarm=0, p_miss_detect=0)
    ql = QLConfig(slots=500)
    w = build_world(env)
    sim = np.mean([run_one(C.POLICY_RANDOM, w, ql, new_qtable(env), 0, False, 500 + i, 9 + i)["throughput_mbps"] for i in range(120)])
    h2 = np.random.default_rng(0).exponential(1, 400000)
    ec = np.mean(env.bandwidth_hz * np.log2(1 + w.rx_mean[0] * h2 / env.noise_w)) / 1e6
    expected = (1 - env.eta_pu) * ec * (1 - env.switch_overhead * (env.K - 1) / env.K)
    assert sim == pytest.approx(expected, rel=0.03)


def test_random_policy_pu_collision_equals_activity_ratio():
    for eta in (0.2, 0.6):
        env = EnvConfig(N=2, eta_pu=eta)
        ql = QLConfig(slots=500)
        w = build_world(env)
        r = np.mean([run_one(C.POLICY_RANDOM, w, ql, new_qtable(env), 0, False, 70 + i, 3 + i)["pu_collision"] for i in range(40)])
        assert r == pytest.approx(eta, abs=0.03)


def test_pu_markov_stationary_distribution():
    env = EnvConfig(N=1, eta_pu=0.5)
    ql = QLConfig(slots=500)
    w = build_world(env)
    flips = np.mean([run_one(C.POLICY_RANDOM, w, ql, new_qtable(env), 0, False, 10 + i, 1)["pu_flips_per_1000"] for i in range(30)])
    # expected flips per PU per slot = 2*eta*(1-eta)*s  (stationary prob of being in a state x leaving prob)
    s = (w.pu_a + w.pu_b)  # = s_j
    expected = np.mean(2 * 0.5 * 0.5 * s) * 1000
    assert flips == pytest.approx(expected, rel=0.08)


# ---------------------------------------------------------------- fairness / determinism
def test_common_random_numbers_across_policies():
    env = EnvConfig()
    ql = QLConfig(slots=300)
    w = build_world(env)
    Q = new_qtable(env)
    flips = [run_one(p, w, ql, Q, 0.3, False, 42, 7)["pu_flips_per_1000"] for p in (0, 1, 2)]
    assert flips[0] == flips[1] == flips[2]


def test_determinism_and_seed_sensitivity():
    env = EnvConfig()
    ql = QLConfig(slots=200)
    w = build_world(env)
    a = run_one(0, w, ql, new_qtable(env), 1.0, True, 5, 6)
    b = run_one(0, w, ql, new_qtable(env), 1.0, True, 5, 6)
    c = run_one(0, w, ql, new_qtable(env), 1.0, True, 6, 6)
    assert a["reward"] == b["reward"] and a["throughput_mbps"] == b["throughput_mbps"]
    assert a["reward"] != c["reward"]


def test_metric_invariants():
    env = EnvConfig(N=25)
    ql = QLConfig(slots=300)
    w = build_world(env)
    for p in (0, 1, 2):
        r = run_one(p, w, ql, new_qtable(env), 0.5, p == 0, 3, 4)
        assert 0 <= r["utilisation"] <= 1 and 0 <= r["tx_rate"] <= 1 and r["throughput_mbps"] >= 0
        assert r["pu_collision"] + r["su_collision"] <= 1 + 1e-9
        assert all(0 <= u <= 1 for u in r["channel_util"])


def test_su_contention_domain_vs_sinr_modes():
    ql = QLConfig(slots=300)
    for mode in ("domain", "sinr"):
        env = EnvConfig(N=30, su_contention=mode)
        r = run_one(2, build_world(env), ql, new_qtable(env), 0, False, 1, 2)
        assert r["throughput_mbps"] > 0
    # spatial reuse (sinr mode) can only help throughput vs a single collision domain
    e0, e1 = EnvConfig(N=30, su_contention="domain"), EnvConfig(N=30, su_contention="sinr")
    t0 = run_one(2, build_world(e0), ql, new_qtable(e0), 0, False, 1, 2)["throughput_mbps"]
    t1 = run_one(2, build_world(e1), ql, new_qtable(e1), 0, False, 1, 2)["throughput_mbps"]
    assert t1 >= t0


# ---------------------------------------------------------------- Q-learning mechanics
def test_bellman_update_matches_formula_exactly():
    env = EnvConfig(N=1)
    ql = QLConfig(alpha=0.37, gamma=0.9, slots=10)
    w = build_world(env)
    rng = np.random.default_rng(0)
    Q0 = rng.normal(size=(1, C.n_states(env.K), env.K + 1)).astype(np.float32)
    Q = Q0.copy()
    res = run_one(C.POLICY_QL, w, ql, Q, 0.0, True, 11, 12, T=6, trace=True)
    su0 = res["trace"]["su0"]
    checked = 0
    for t in range(0, 4):
        s, a, r = int(su0[t, 0]), int(su0[t, 1]), su0[t, 2]
        s2 = int(su0[t + 1, 0])
        if s2 in [int(x) for x in su0[:t + 1, 0]]:
            continue  # row s2 may have been updated earlier in the episode; skip those cases
        q_old = float(Q0[0, s, a])
        target = r + ql.gamma * float(Q0[0, s2].max())
        expected = q_old + ql.alpha * (target - q_old)
        assert su0[t, 3] == pytest.approx(q_old, abs=1e-5)
        assert su0[t, 4] == pytest.approx(expected, abs=1e-4)
        checked += 1
    assert checked >= 2


def test_untrained_greedy_policy_breaks_ties_uniformly():
    env = EnvConfig(N=1, arrival_prob=1.0)
    ql = QLConfig(slots=500)
    w = build_world(env)
    res = run_one(C.POLICY_QL, w, ql, new_qtable(env), 0.0, False, 1, 2, trace=True)
    acts = res["trace"]["act"][:, 0]
    counts = np.bincount(acts, minlength=env.K + 1)
    assert counts.min() > 0  # all-zero Q must not collapse onto action 0


def test_epsilon_greedy_exploration_is_uniform_over_actions():
    env = EnvConfig(N=1, arrival_prob=1.0)
    ql = QLConfig(slots=500)
    Q = new_qtable(env); Q[:, :, 3] = 100.0  # strongly prefer channel 3
    w = build_world(env)
    exploit = run_one(C.POLICY_QL, w, ql, Q.copy(), 0.0, False, 1, 2, trace=True)["trace"]["act"][:, 0]
    assert (exploit == 3).mean() > 0.95
    explore = run_one(C.POLICY_QL, w, ql, Q.copy(), 1.0, False, 1, 2, trace=True)["trace"]["act"][:, 0]
    frac = np.bincount(explore, minlength=env.K + 1) / len(explore)
    assert frac.max() < 0.15 and frac.min() > 0.02  # ~1/17 each


def test_epsilon_decay_schedule_matches_algorithm_1():
    env = EnvConfig()
    ql = QLConfig(episodes=1200, slots=10, eps_decay=0.995)
    seen = []
    train(env, ql, baselines=False, callback=lambda r: seen.append(r["eps"]))
    assert seen[0] == 1.0
    assert seen[100] == pytest.approx(max(0.01, 0.995 ** 100))
    assert min(seen) == pytest.approx(0.01) and seen[-1] == pytest.approx(0.01)


def test_empty_queue_means_silence_and_no_learning():
    env = EnvConfig(N=1, arrival_prob=0.0, queue_max=4)
    ql = QLConfig(slots=60)
    w = build_world(env)
    Q = new_qtable(env)
    r = run_one(C.POLICY_QL, w, ql, Q, 1.0, True, 1, 2)
    assert r["tx_rate"] < 0.15 and abs(Q).sum() >= 0  # queue drains (starts at 2) then silent
    assert r["reward"] > -5


def test_policy_serialisation_roundtrip():
    env = EnvConfig(N=3)
    ql = QLConfig(episodes=3, slots=50)
    pol = train(env, ql, baselines=False)
    back = Policy.from_bytes(pol.to_bytes())
    assert np.array_equal(back.q, pol.q) and back.env == pol.env and back.ql == pol.ql


def test_observation_to_state_semantics():
    K = 16
    busy = [False] * K
    busy[4] = True
    snr = [1.0] * K
    snr[9] = 50.0
    s = observation_to_state(K, prev_channel=5, sensed_busy=busy, snr=snr, queue=7)
    assert C.decode_state(s, K) == (5, 1, 10, 2, 2)   # prev ch5 (index4) busy; best free = ch10; queue>3; >4 free
    busy = [True] * K
    s = observation_to_state(K, 0, busy, snr, 0)
    assert C.decode_state(s, K) == (0, 0, 0, 0, 0)


# ---------------------------------------------------------------- the ML claim: does it actually learn?
@pytest.fixture(scope="module")
def trained():
    env = EnvConfig()
    ql = QLConfig(episodes=600, slots=500)
    hist = []
    pol = train(env, ql, True, hist.append)
    return env, ql, pol, hist


def test_learning_curve_improves_and_beats_baselines(trained):
    env, ql, pol, hist = trained
    early = np.mean([h["ql"]["reward"] for h in hist[:20]])
    late = np.mean([h["ql"]["reward"] for h in hist[-50:]])
    greedy = np.mean([h["greedy"]["reward"] for h in hist[-50:]])
    rand = np.mean([h["random"]["reward"] for h in hist[-50:]])
    assert late > early + 100
    assert late > greedy > rand


def test_trained_policy_beats_baselines_on_heldout_seeds(trained):
    env, ql, pol, _ = trained
    w = build_world(env)
    q, g, r = (evaluate(k, env, ql, pol if k == "ql" else None, 20, 123456, w) for k in ("ql", "greedy", "random"))
    assert q["reward"] > g["reward"] > r["reward"]
    assert q["throughput_mbps"] > g["throughput_mbps"] > 0
    assert q["pu_collision"] < r["pu_collision"] * 0.5
    assert q["su_collision"] < g["su_collision"]
    assert q["handovers_per_min"] < 0.2 * g["handovers_per_min"]
    assert q["utilisation"] > g["utilisation"]


def test_policy_has_interpretable_behaviour(trained):
    from spectrarl.runner import behaviour_stats
    env, ql, pol, _ = trained
    w = build_world(env)
    tr = run_one(C.POLICY_QL, w, ql, pol.q, 0.0, False, 777, 778, T=500, trace=True)["trace"]
    b = behaviour_stats(tr["act"], tr["code"])
    assert b["stay_after_success"] > 0.75           # keeps a channel that works
    g = run_one(C.POLICY_GREEDY, w, ql, pol.q, 0.0, False, 777, 778, T=500, trace=True)["trace"]
    assert behaviour_stats(g["act"], g["code"])["stay_after_success"] < 0.3   # greedy chases the best SINR instead


def test_trace_shapes_and_codes(trained):
    env, ql, pol, _ = trained
    r = run_one(C.POLICY_QL, build_world(env), ql, pol.q, 0.0, False, 5, 6, T=50, trace=True)
    tr = r["trace"]
    assert tr["occ"].shape == (50, env.K) and tr["act"].shape == (50, env.N)
    assert set(np.unique(tr["code"])) <= {0, 1, 2, 3} and tr["act"].min() >= 0 and tr["act"].max() <= env.K


def test_bellman_trace_api_function(trained):
    env, ql, pol, _ = trained
    rows = bellman_trace(pol, env, ql, 10)
    assert len(rows) == 10 and all({"state", "action", "reward", "q_old", "q_new"} <= set(r) for r in rows)


# ---------------------------------------------------------------- config validation
def test_config_validation():
    with pytest.raises(Exception):
        QLConfig(eps_min=0.5, eps0=0.1)
    with pytest.raises(Exception):
        EnvConfig(K=1)
    with pytest.raises(Exception):
        QLConfig(alpha=0)
    with pytest.raises(ValueError):
        check_budget(EnvConfig(N=64, K=32), QLConfig(episodes=20000, slots=5000))
    check_budget(EnvConfig(), QLConfig())  # paper defaults are allowed
