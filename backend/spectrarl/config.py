"""Validated configuration objects. Defaults follow Tables 3 of the presentation where it gives a value;
everything the presentation leaves unspecified is listed in ASSUMPTIONS and documented in the README."""
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator

MAX_WORK_UNITS = 6_000_000_000  # episodes * slots * SUs * channels guard (keeps requests bounded)


class EnvConfig(BaseModel):
    # ---- Table 3 (presentation) ----
    K: int = Field(16, ge=2, le=32, description="orthogonal channels")
    N: int = Field(10, ge=1, le=64, description="secondary users")
    M: int = Field(8, ge=1, le=32, description="primary users")
    bandwidth_hz: float = Field(10e6, gt=0, description="B_k per channel")
    eta_pu: float = Field(0.4, ge=0.02, le=0.95, description="PU activity ratio (stationary ON probability)")
    ptx_dbm: float = Field(23.0, description="SU transmit power")
    noise_dbm_hz: float = Field(-174.0, description="noise PSD N0")
    path_loss_exp: float = Field(3.8, gt=1, lt=8)
    fc_hz: float = Field(2.4e9, gt=0)
    area_m: float = Field(1000.0, gt=10)
    # ---- not specified by the presentation (documented assumptions) ----
    link_dist_min_m: float = 20.0
    link_dist_max_m: float = 80.0
    pu_dynamics: float = Field(0.3, gt=0, le=4, description="scales PU ON/OFF switching speed")
    arrival_prob: float = Field(0.7, ge=0, le=1, description="packet arrival probability / slot / SU")
    queue_max: int = Field(10, ge=1, le=64)
    p_false_alarm: float = Field(0.05, ge=0, le=0.5)
    p_miss_detect: float = Field(0.05, ge=0, le=0.5)
    slot_ms: float = Field(100.0, gt=0, description="slot duration, used to convert handovers to per-minute")
    switch_overhead: float = Field(0.2, ge=0, le=1, description="fraction of a slot lost to RF re-tuning on a switch")
    su_contention: str = Field("domain", pattern="^(domain|sinr)$")
    sinr_threshold_db: float = 3.0
    world_seed: int = 7

    @property
    def ptx_w(self) -> float:
        return 10 ** (self.ptx_dbm / 10) / 1000.0

    @property
    def noise_w(self) -> float:
        return 10 ** (self.noise_dbm_hz / 10) / 1000.0 * self.bandwidth_hz


class QLConfig(BaseModel):
    alpha: float = Field(0.1, gt=0, le=1)
    gamma: float = Field(0.95, ge=0, le=1)
    eps0: float = Field(1.0, ge=0, le=1)
    eps_min: float = Field(0.01, ge=0, le=1)
    eps_decay: float = Field(0.995, gt=0, le=1)
    episodes: int = Field(1000, ge=1, le=20000)
    slots: int = Field(500, ge=10, le=5000)
    # reward weights (Eq. 3); the presentation does not publish its rho values
    w_throughput: float = Field(1.0, ge=0)
    rho_collision: float = Field(1.5, ge=0)
    rho_switch: float = Field(0.3, ge=0)
    rho_su_collision: float = Field(0.5, ge=0, description="EXTENSION: penalty when two SUs collide (0 = strict paper reward)")
    idle_reward: float = Field(0.1, ge=0, description="bonus for backing off when every channel is PU-occupied")
    idle_penalty: float = Field(0.2, ge=0, description="penalty for backing off while white space exists")
    seed: int = 1

    @model_validator(mode="after")
    def _chk(self):
        if self.eps_min > self.eps0:
            raise ValueError("eps_min must be <= eps0")
        return self


def check_budget(env: EnvConfig, ql: QLConfig, policies: int = 3) -> None:
    work = ql.episodes * ql.slots * env.N * env.K * policies
    if work > MAX_WORK_UNITS:
        raise ValueError(f"requested workload too large ({work:.2e} units > {MAX_WORK_UNITS:.1e}); reduce episodes/slots/N")


ASSUMPTIONS = [
    "Each PU owns channels {j, j+M, ...}; a PU is a two-state Markov ON/OFF source with stationary ON probability eta_PU. "
    "Per-PU switching speed s_j ~ U[0.06, 0.5] (P(OFF->ON)=s_j*eta, P(ON->OFF)=s_j*(1-eta)), so some channels are more predictable than others.",
    "Sensing: the SU observes PU occupancy of the previous slot (one-slot sensing latency) with false-alarm and miss-detection errors; "
    "channel SINR is estimated for the current slot (pilot) so Rayleigh fading is visible to the agent.",
    "Contention: in 'domain' mode two or more SUs on the same channel in a slot all fail (single collision domain); 'sinr' mode uses Eq. (1) "
    "with co-channel SU interference and a decoding threshold.",
    "The paper does not publish its reward weights rho_coll / rho_switch / idle terms; defaults here are chosen by us and are configurable.",
    "State compression for the tabular agent: S_t = (C_{t-1}, PU-occupancy sensed on C_{t-1}, best sensed-free channel by SINR, queue bucket, free-channel-count bucket). "
    "Each SU keeps its own Q-table (decentralised independent learners).",
    "SU traffic: Bernoulli packet arrivals into a finite queue; an SU with an empty queue stays silent (no penalty, no learning step).",
]
