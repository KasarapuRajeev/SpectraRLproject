"""World generation (geometry, PU dynamics) and the analytical channel model of Eqs. (1)-(2)."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .config import EnvConfig, QLConfig
from . import core as C


@dataclass
class World:
    cfg: EnvConfig
    pu_a: np.ndarray       # P(OFF->ON) per PU
    pu_b: np.ndarray       # P(ON->OFF) per PU
    owner: np.ndarray      # channel -> PU index
    rx_mean: np.ndarray    # mean received SU power on own link [N] (W)
    intf_mean: np.ndarray  # mean interference power tx j -> rx i [N,N] (W)
    link_d: np.ndarray     # own link distance [N] (m)
    su_pos: np.ndarray
    pu_pos: np.ndarray
    pl0: float
    cmax: float

    def params(self, ql: QLConfig) -> np.ndarray:
        c = self.cfg
        P = np.zeros(C.P_COUNT)
        P[C.P_B] = c.bandwidth_hz
        P[C.P_NOISE] = c.noise_w
        P[C.P_CMAX] = self.cmax
        P[C.P_LAM] = c.arrival_prob
        P[C.P_PFA] = c.p_false_alarm
        P[C.P_PMD] = c.p_miss_detect
        P[C.P_W1] = ql.w_throughput
        P[C.P_RCOLL] = ql.rho_collision
        P[C.P_RSW] = ql.rho_switch
        P[C.P_RI] = ql.idle_reward
        P[C.P_RP] = ql.idle_penalty
        P[C.P_OVH] = c.switch_overhead
        P[C.P_QMAX] = c.queue_max
        P[C.P_SUMODE] = 1 if c.su_contention == "sinr" else 0
        P[C.P_THR] = 10 ** (c.sinr_threshold_db / 10)
        P[C.P_RSU] = ql.rho_su_collision
        return P


def path_gain_const(cfg: EnvConfig) -> float:
    lam = 3e8 / cfg.fc_hz
    return (lam / (4 * math.pi)) ** 2  # free-space gain at 1 m reference distance


def build_world(cfg: EnvConfig) -> World:
    rng = np.random.default_rng(cfg.world_seed)
    N, M, K = cfg.N, cfg.M, cfg.K
    s = np.clip(rng.uniform(0.06, 0.5, size=M) * cfg.pu_dynamics, 1e-3, 1.0)
    pu_a = s * cfg.eta_pu
    pu_b = s * (1.0 - cfg.eta_pu)
    owner = np.arange(K) % M
    tx = rng.uniform(0, cfg.area_m, size=(N, 2))
    d = rng.uniform(cfg.link_dist_min_m, cfg.link_dist_max_m, size=N)
    ang = rng.uniform(0, 2 * math.pi, size=N)
    rx = tx + np.stack([d * np.cos(ang), d * np.sin(ang)], axis=1)
    pl0 = path_gain_const(cfg)
    rx_mean = cfg.ptx_w * pl0 * d ** (-cfg.path_loss_exp)
    D = np.maximum(np.linalg.norm(tx[:, None, :] - rx[None, :, :], axis=2), 1.0)  # tx j -> rx i : D[j, i]
    intf = cfg.ptx_w * pl0 * D ** (-cfg.path_loss_exp)
    pu_pos = rng.uniform(0, cfg.area_m, size=(M, 2))
    cmax = cfg.bandwidth_hz * math.log2(1 + cfg.ptx_w * pl0 * cfg.link_dist_min_m ** (-cfg.path_loss_exp) / cfg.noise_w)
    return World(cfg, pu_a, pu_b, owner.astype(np.int64), rx_mean, intf, d, tx, pu_pos, pl0, cmax)


# ---- analytical channel model: Eq. (1) and (2) -------------------------------
def sinr(ptx: float, h2: float, d: float, alpha: float, pl0: float, noise: float,
         su_interf: float = 0.0, pu_interf: float = 0.0) -> float:
    """Eq. (1): gamma = Ptx |h|^2 d^-alpha / (sigma^2 + sum SU interference + I_PU)."""
    return ptx * pl0 * h2 * d ** (-alpha) / (noise + su_interf + pu_interf)


def capacity_bps(bandwidth: float, gamma: float) -> float:
    """Eq. (2): Shannon-Hartley capacity."""
    return bandwidth * math.log2(1.0 + gamma)


def pu_interference_at(world: World, su: int, channel: int, pu_state: np.ndarray, pu_tx_w: float = 1.0) -> float:
    """Mean interference at SU `su`'s receiver from the PU that owns `channel` when it is ON."""
    j = int(world.owner[channel])
    if not pu_state[j]:
        return 0.0
    d = max(float(np.linalg.norm(world.pu_pos[j] - world.su_pos[su])), 1.0)
    return pu_tx_w * world.pl0 * d ** (-world.cfg.path_loss_exp)
