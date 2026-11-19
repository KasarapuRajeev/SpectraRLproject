"""Numba-compiled cognitive-radio environment + agents.

One kernel (`run_episode`) simulates a full episode for one policy:
  0 = MO-QL-DSA (tabular Q-learning, one Q-table per SU, decentralised)
  1 = Greedy    (sense, then take the idle channel with the highest SINR)
  2 = Random    (uniformly random channel, no sensing)

Environment randomness (PU Markov chains, Rayleigh fading, sensing errors, arrivals) uses its own RNG stream
whose consumption does not depend on the policy, so all policies see *identical* radio conditions for the
same episode seed (common random numbers -> fair comparison). Agent randomness uses a separate stream.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

# ---- parameter vector layout -------------------------------------------------
(P_B, P_NOISE, P_CMAX, P_LAM, P_PFA, P_PMD, P_W1, P_RCOLL, P_RSW, P_RI, P_RP, P_OVH, P_QMAX,
 P_SUMODE, P_THR, P_RSU, P_COUNT) = range(17)

# ---- per-episode output layout ----------------------------------------------
(O_REWARD, O_THR_BITS, O_TX, O_PU_COLL, O_SU_COLL, O_SUCCESS, O_SWITCH, O_FLIPS, O_ACTIVE, O_IDLE, O_TSUM,
 O_COUNT) = range(12)

POLICY_QL, POLICY_GREEDY, POLICY_RANDOM = 0, 1, 2
TRACE_COLS = 6  # SU0 step trace: state, action, reward, q_old, q_new, explored

U64 = np.uint64


@njit(cache=True, inline="always")
def rnd(st):
    """splitmix64 -> float64 in [0,1). `st` is a uint64[1] state array."""
    st[0] = st[0] + U64(0x9E3779B97F4A7C15)
    z = st[0]
    z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
    z = z ^ (z >> U64(31))
    return (z >> U64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def n_states(K):
    return (K + 1) * 2 * (K + 1) * 3 * 3


@njit(cache=True)
def encode_state(cprev, occ, best, qb, nf, K):
    """S_t = (C_{t-1}, O[C_{t-1}], best free channel, queue bucket, free-count bucket) -> table row."""
    return ((((cprev * 2 + occ) * (K + 1) + best) * 3 + qb) * 3 + nf)


@njit(cache=True)
def decode_state(s, K):
    nf = s % 3
    s //= 3
    qb = s % 3
    s //= 3
    best = s % (K + 1)
    s //= K + 1
    occ = s % 2
    cprev = s // 2
    return cprev, occ, best, qb, nf


@njit(cache=True, inline="always")
def queue_bucket(q):
    return 0 if q <= 0 else (1 if q <= 3 else 2)


@njit(cache=True, inline="always")
def free_bucket(n):
    return 0 if n <= 0 else (1 if n <= 4 else 2)


@njit(cache=True, inline="always")
def argmax_tiebreak(row, agent):
    """argmax with uniformly random tie-breaking (an untrained all-zero row must not always pick action 0)."""
    best = row[0]
    idx = 0
    ties = 1
    for a in range(1, row.shape[0]):
        v = row[a]
        if v > best:
            best = v
            idx = a
            ties = 1
        elif v == best:
            ties += 1
            if rnd(agent) * ties < 1.0:
                idx = a
    return idx


@njit(cache=True)
def reward_terms(success, pu_hit, su_hit, switched, active_idle, all_busy, thr_norm, P):
    """Eq. (3): R = w1*T - rho_coll*I_coll - rho_sw*H_switch + P_idle (+ optional -rho_su*[SU-SU collision]).
    Returns (R, T, I, H, Pidle). With P[P_RSU] = 0 this is exactly the presentation's reward."""
    T = thr_norm if success else 0.0
    I = P[P_RCOLL] if pu_hit else 0.0
    if su_hit and not pu_hit:
        I += P[P_RSU]
    H = P[P_RSW] if switched else 0.0
    Pi = 0.0
    if active_idle:
        Pi = P[P_RI] if all_busy else -P[P_RP]
    return P[P_W1] * T - I - H + Pi, T, I, H, Pi


@njit(cache=True, nogil=True)
def run_episode(policy, N, K, M, T, pu_a, pu_b, owner, rx_mean, intf_mean, P, Q, alpha, gamma, eps, learn,
                env, agent, out, used, opp, trace_on, tr_occ, tr_act, tr_code, tr_su0):
    A = K + 1
    noise = P[P_NOISE]
    B = P[P_B]
    cmax = P[P_CMAX]
    occ = np.zeros(K, np.uint8)
    occ_prev = np.zeros(K, np.uint8)
    pu = np.zeros(M, np.uint8)
    for j in range(M):
        pu[j] = 1 if rnd(env) < pu_a[j] / (pu_a[j] + pu_b[j]) else 0
    for k in range(K):
        occ_prev[k] = pu[owner[k]]
    q = np.full(N, int(P[P_QMAX]) // 2, np.int64)
    cprev = np.zeros(N, np.int64)
    s_prev = np.full(N, -1, np.int64)
    a_prev = np.zeros(N, np.int64)
    r_prev = np.zeros(N, np.float64)
    valid = np.zeros(N, np.bool_)
    sensed = np.zeros((N, K), np.uint8)
    h2 = np.zeros((N, K))
    g2 = np.zeros((N, N))
    arr = np.zeros(N, np.bool_)
    act = np.zeros(N, np.int64)
    nk = np.zeros(K, np.int64)
    explored = np.zeros(N, np.bool_)
    got = np.zeros(K, np.bool_)
    sumo = int(P[P_SUMODE])
    qold0 = 0.0
    for i in range(O_COUNT):
        out[i] = 0.0
    for k in range(K):
        used[k] = 0.0
        opp[k] = 0.0

    for t in range(T):
        # ---- PU Markov ON/OFF transitions ----------------------------------
        for j in range(M):
            u = rnd(env)
            if pu[j] == 1:
                if u < pu_b[j]:
                    pu[j] = 0
                    out[O_FLIPS] += 1
            else:
                if u < pu_a[j]:
                    pu[j] = 1
                    out[O_FLIPS] += 1
        n_busy = 0
        for k in range(K):
            occ[k] = pu[owner[k]]
            n_busy += occ[k]
        all_busy = n_busy == K
        # ---- fading, sensing, arrivals (fixed number of draws per slot) -----
        for i in range(N):
            for k in range(K):
                h2[i, k] = -math.log(1.0 - rnd(env))
                u = rnd(env)
                if occ_prev[k] == 1:
                    sensed[i, k] = 0 if u < P[P_PMD] else 1
                else:
                    sensed[i, k] = 1 if u < P[P_PFA] else 0
            arr[i] = rnd(env) < P[P_LAM]
        if sumo == 1:
            for i in range(N):
                for j in range(N):
                    g2[i, j] = -math.log(1.0 - rnd(env))
        # ---- observe, learn from previous transition, decide ---------------
        for i in range(N):
            best = 0
            bsnr = -1.0
            nfree = 0
            for k in range(K):
                if sensed[i, k] == 0:
                    nfree += 1
                    snr = rx_mean[i] * h2[i, k] / noise
                    if snr > bsnr:
                        bsnr = snr
                        best = k + 1
            cp = cprev[i]
            oc = 1 if (cp >= 1 and sensed[i, cp - 1] == 1) else 0
            qb = queue_bucket(q[i])
            s = encode_state(cp, oc, best, qb, free_bucket(nfree), K)
            if policy == 0 and learn and valid[i]:
                m = Q[i, s, 0]
                for a in range(1, A):
                    if Q[i, s, a] > m:
                        m = Q[i, s, a]
                old = Q[i, s_prev[i], a_prev[i]]
                new = old + alpha * (r_prev[i] + gamma * m - old)
                Q[i, s_prev[i], a_prev[i]] = new
                if i == 0 and trace_on:
                    tr_su0[t - 1, 3] = old
                    tr_su0[t - 1, 4] = new
            a = 0
            explored[i] = False
            if q[i] > 0:
                if policy == 0:
                    if rnd(agent) < eps:
                        a = int(rnd(agent) * A)
                        if a >= A:
                            a = A - 1
                        explored[i] = True
                    else:
                        a = argmax_tiebreak(Q[i, s], agent)
                elif policy == 1:
                    a = best
                else:
                    a = 1 + int(rnd(agent) * K)
                    if a > K:
                        a = K
            act[i] = a
            s_prev[i] = s
            a_prev[i] = a
            valid[i] = q[i] > 0
            if i == 0 and trace_on:
                tr_su0[t, 0] = s
                tr_su0[t, 1] = a
                tr_su0[t, 5] = 1.0 if explored[i] else 0.0
        # ---- resolve the slot ------------------------------------------------
        for k in range(K):
            nk[k] = 0
        for i in range(N):
            if act[i] > 0:
                nk[act[i] - 1] += 1
        for k in range(K):
            got[k] = False
        for i in range(N):
            a = act[i]
            R = 0.0
            code = 0
            if a == 0:
                if q[i] > 0:
                    out[O_IDLE] += 1
                    R, _, _, _, _ = reward_terms(False, False, False, False, True, all_busy, 0.0, P)
            else:
                k = a - 1
                out[O_TX] += 1
                pu_hit = occ[k] == 1
                interf = 0.0
                su_fail = False
                if nk[k] >= 2:
                    if sumo == 0:
                        su_fail = True
                    else:
                        for j in range(N):
                            if j != i and act[j] == a:
                                interf += intf_mean[j, i] * g2[j, i]
                        sinr = rx_mean[i] * h2[i, k] / (noise + interf)
                        su_fail = sinr < P[P_THR]
                switched = cprev[i] >= 1 and a != cprev[i]
                success = (not pu_hit) and (not su_fail)
                thr_norm = 0.0
                if success:
                    sinr = rx_mean[i] * h2[i, k] / (noise + interf)
                    c = B * math.log2(1.0 + sinr)
                    if switched:
                        c *= 1.0 - P[P_OVH]
                    out[O_THR_BITS] += c
                    thr_norm = c / cmax
                    if thr_norm > 1.0:
                        thr_norm = 1.0
                    out[O_SUCCESS] += 1
                    got[k] = True
                    code = 1
                    out[O_TSUM] += thr_norm
                elif pu_hit:
                    out[O_PU_COLL] += 1
                    code = 2
                else:
                    out[O_SU_COLL] += 1
                    code = 3
                if switched:
                    out[O_SWITCH] += 1
                R, _, _, _, _ = reward_terms(success, pu_hit, su_fail, switched, False, all_busy, thr_norm, P)
            if q[i] > 0:
                out[O_ACTIVE] += 1
                out[O_REWARD] += R / N
            r_prev[i] = R
            if i == 0 and trace_on:
                tr_su0[t, 2] = R
            if trace_on:
                tr_act[t, i] = a
                tr_code[t, i] = code
            # queue dynamics
            if code == 1 and q[i] > 0:
                q[i] -= 1
            if arr[i] and q[i] < int(P[P_QMAX]):
                q[i] += 1
            cprev[i] = a
        for k in range(K):
            if occ[k] == 0:
                opp[k] += 1.0
                if got[k]:
                    used[k] += 1.0
            occ_prev[k] = occ[k]
            if trace_on:
                tr_occ[t, k] = occ[k]


@njit(cache=True, nogil=True)
def bench_decisions(policy, Q, n, K, seed):
    """Micro-benchmark of the decision rule alone (state lookup + selection); returns a checksum."""
    agent = np.zeros(1, np.uint64)
    agent[0] = U64(seed)
    A = K + 1
    S = Q.shape[1]
    chk = 0
    for it in range(n):
        if policy == 0:
            s = int(rnd(agent) * S)
            chk += argmax_tiebreak(Q[0, s], agent)
        elif policy == 1:
            best = 0
            bs = -1.0
            for k in range(K):
                v = rnd(agent)
                if v > bs:
                    bs = v
                    best = k + 1
            chk += best
        else:
            chk += 1 + int(rnd(agent) * K)
    return chk + A - A
