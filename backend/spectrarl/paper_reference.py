"""Numbers REPORTED by the presentation (from the authors' own simulator, which we do not have).
They are served alongside our measured results for comparison only - they are never presented as our output."""
PAPER = {
    "source": "RL-Based Dynamic Spectrum Allocation presentation (Singh, Kumar, Singh, Rajeev, Singh; supervisor Dr. C. Dhasarathan)",
    "convergence": {"episodes": [100, 200, 350, 500, 750, 1000],
                    "ql": [41.25, 64.10, 74.80, 75.40, 75.60, 75.65], "greedy": [36.80, 38.20, 39.10, 39.40, 39.50, 39.60],
                    "random": [-5.40, -4.90, -5.20, -5.10, -5.00, -5.05]},
    "throughput": {"N": [5, 10, 15, 20, 25, 30, 35, 40], "ql": [18.2, 34.8, 49.5, 62.1, 72.8, 80.4, 85.1, 88.6],
                   "greedy": [16.5, 28.4, 38.2, 45.1, 49.8, 52.3, 53.9, 54.5], "random": [11.2, 19.5, 24.8, 28.2, 30.1, 31.4, 32.0, 32.2]},
    "collision": {"eta": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  "ql": [1.2, 2.1, 3.4, 4.8, 6.5, 8.8, 11.5, 14.8, 18.5], "greedy": [4.5, 8.2, 12.8, 17.9, 23.5, 29.8, 36.4, 43.5, 51.0],
                  "random": [9.5, 18.2, 26.8, 34.9, 42.8, 50.5, 58.2, 65.5, 72.5]},
    "utilisation": {"channels": 8, "ql": 89.36, "greedy": 65.66, "random": 43.36},
    "handovers_per_min": {"ql": 6.2, "greedy": 21.5, "random": 39.8},
    "alpha": [{"alpha": 0.01, "episodes": 480, "variance": 0.45, "reward": 72.40}, {"alpha": 0.1, "episodes": 320, "variance": 0.85, "reward": 75.65},
              {"alpha": 0.3, "episodes": 180, "variance": 3.20, "reward": 73.10}, {"alpha": 0.5, "episodes": 110, "variance": 8.90, "reward": 68.50}],
    "overall": {"throughput_n40": [88.6, 54.5, 32.2], "collision_eta04": [4.8, 17.9, 34.9], "utilisation": [89.36, 65.66, 43.36],
                "handovers": [6.2, 21.5, 39.8], "runtime_ms": [0.42, 0.12, 0.05], "convergence_episodes": 320},
    "literature": [
        ["Venkatesan (2026)", "Comput. Netw.", "Digital Twin + RL", "High network fidelity", "Large computation requirements"],
        ["Al-Habashna (2024)", "IEEE T-MC", "Deep Q-Networks", "Mobility handling", "Extensive pre-training data"],
        ["Zhang et al. (2024)", "IEEE T-WC", "Federated RL", "User privacy", "Policy drift (non-IID sensing)"],
        ["Sharma & Rawat (2023)", "Comput. Netw.", "Actor-Critic (DDPG)", "Continuous power control", "Unstable gradients in bursts"],
        ["Li et al. (2024)", "IEEE IoT-J", "Multi-Agent RL", "Scalable IoT framework", "Curse of dimensionality"],
        ["Nguyen & Le (2024)", "IEEE Access", "PPO", "Stable policy updates", "Too heavy for edge nodes"],
        ["Gupta & Roy (2023)", "Comput. Comm.", "Hidden Markov Model", "Predicts PU arrivals", "False alarms cause handovers"],
        ["Anand et al. (2023)", "Wireless Netw.", "Simple Q-Learning", "Easy to implement", "Ignores spectral switching"],
        ["Proposed (QL + MDP)", "This work", "Multi-objective QL", "Throughput / Interf. / HO", "Optimised for discrete sub-channels"]],
}
