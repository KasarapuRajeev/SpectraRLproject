# SpectraRL — RL-based Dynamic Spectrum Allocation

A working implementation of the project described in the deck *"RL-Based Dynamic Spectrum Allocation"*:
a Cognitive-Radio-Network simulator, per-user tabular Q-learning agents with the multi-objective reward of Eq. (3),
greedy/random baselines, a benchmark suite that recomputes the deck's results slides, and a full web application.

## Run
```bash
pip install -r backend/requirements.txt
cd frontend && npm install && npm run build && cd ..     # builds the UI into frontend/dist
python run.py --port 8000                                # http://127.0.0.1:8000
```
Docker: `docker build -t spectrarl . && docker run -p 8000:8000 -v spectrarl-data:/data spectrarl`
Dev UI: `cd frontend && npm run dev` (proxies /api to :8000). Optional write-protection: set `SPECTRARL_API_KEY`.

## Architecture
| Layer | What |
|---|---|
| `backend/spectrarl/core.py` | Numba-compiled simulation kernel: PU Markov ON/OFF, Rayleigh fading, Eq.(1) SINR, Eq.(2) capacity, Eq.(3) reward, Q-learning with Bellman updates, ε-greedy |
| `world.py`, `runner.py` | Geometry/channel model, training, frozen-policy evaluation, simulation traces, Bellman traces, edge `decide` |
| `experiments.py` | Benchmark suites: convergence, throughput vs N, collisions vs η, utilisation, handovers, α sweep, reward ablation, ρ trade-off, runtime |
| `db.py`, `jobs.py`, `api.py` | SQLite (WAL) persistence, background job manager, FastAPI REST API |
| `frontend/` | React + Vite + TypeScript; 11 pages (overview, system/MDP, reward lab, algorithm, training studio, live monitor, Q-table inspector, benchmarks, runs, setup) |

## Testing
* `cd backend && pytest` — 44 tests: state-encoding bijection, RNG uniformity, reward cases, Eq.(1)/(2) closed forms, simulator vs analytic expectations, PU Markov rate, common-random-numbers, determinism, exact Bellman-update check, ε-greedy/tie-break statistics, policy serialisation, "learning beats baselines on held-out seeds", API lifecycle/validation/auth. Mutation checks confirmed the Bellman and overhead tests catch injected bugs.
* `python e2e/e2e.py` — Playwright/Chrome: loads all 11 routes, trains an agent through the UI, opens monitor/inspector/runs/benchmarks, checks for console errors and mobile overflow.

## Measured results vs the deck (honest summary)
The deck's simulator and several parameters are unpublished, so **directions are comparable, magnitudes are not.**
| Claim | Measured here |
|---|---|
| QL beats greedy/random on throughput | Yes. At N=40 the gap is far larger than the deck's (+62.5%), because greedy herds. QL throughput is **not** monotone in N (peaks mid-range) — the deck's monotone growth is not reproduced |
| Lower PU collision | Only partially: vs greedy (which senses every slot) the advantage is small (6.3% vs 7.0% at η=0.4) and not uniform across η. Vs random it is large. ρ_coll trades it off against throughput (Trade-off tab) |
| Higher utilisation | Yes in direction (48.7% vs 36.1%), not in magnitude (deck: 89% vs 66%) |
| Fewer handovers | Yes, strongly (≈97% fewer at default PU dynamics) |
| Convergence ≈ 320–350 episodes | ≈550 episodes to 90% of final reward |
| α = 0.1 best | Yes on final reward |
| Run time 0.42 ms | Not comparable (different language/hardware) |

## Deviations & limitations
* Reward weights, PU dynamics (×0.3), traffic, sensing errors are **our choices** (see Setup page / `ASSUMPTIONS`).
* Added ρ_su (SU–SU collision penalty) — an extension to Eq.(3); without it SUs park in silent collisions.
* **Not implemented:** MADRL / federated learning, continuous power control, USRP/SDR hardware. The deck's Fig. 7 image was not in the supplied file.
* Single-machine job runner (2 workers); not horizontally scaled.
