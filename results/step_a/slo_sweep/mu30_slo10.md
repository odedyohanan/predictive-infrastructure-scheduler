# Step A tuning on development seeds

- code: git HEAD d1c3785602609858a3bb270e53b3ba52e5bfc9d6
- members: worldmodel:models/gru_mix_bounded.pt (sha256 aa1dea80d94832e4), models/ens_1.pt (sha256 de6f5c658c00aa37), models/ens_2.pt (sha256 f6134d88163b96b3), models/ens_3.pt (sha256 7b8ff07789452ca5), models/ens_4.pt (sha256 1c4893fa01673cbb)
- calibration seeds: 1100-1109
- evaluation seeds: 1110-1139
- grid: N 5; M 4; H 3,5; lam 1; tau percentile 90
- fixed: mu 30.0, change_cost 0.0, slo_lam 10.0, scenario 60 s
- slo margin (pp): 1.0
- disagreement: std across members of a.p99_ms, a.queue_work_ms, b.rate, c.work_core_ms (encoded), divided by member 0's training input std, mean over keys and horizon
- runtime: 717 s with 4 processes

Paired differences are (loop - base) per scenario, mean [95% CI]. Value in value units per second; SLO in percentage points of windows violating the SLO. Feasible = upper bound of the SLO v2 difference <= +1.0 pp (SLO v1 reported only), paired results on every evaluation seed, finite parameters, decision time p95 <= 100 ms in every scenario and max <= 1000 ms. **These are background-QoS limits, 2x the official 50 / 500 ms**: tuning runs at macOS background priority on efficiency cores with several processes, which inflates wall-clock time; the official limits are checked at normal priority by experiments/time_decisions.py.

## Base scheduler (R6 greedy)

- scenarios: 30; mean value per second 18991.2; SLO v2 6.56 %; SLO v1 6.53 %

## Calibrated trust thresholds (tau)

Percentiles of the ensemble disagreement of every candidate at base-scheduler states (calibration seeds).

| N | H | q90 |
|---|---|---|
| 5 | 3 | 0.1572 |
| 5 | 5 | 0.1731 |

## Grid

| N | M | H | lam | tau q | tau | value diff | SLO v2 diff | SLO v1 diff | fallback | greedy applied | decide p95 ms | n | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 4 | 5 | 1.00 | 90 | 0.1731 | 131.2 [15.9, 246.5] | -0.17 [-0.64, 0.31] | -0.17 [-0.64, 0.31] | 0.101 | 0.686 | 63.6 | 30 | yes **chosen** |
| 5 | 4 | 3 | 1.00 | 90 | 0.1572 | 129.2 [19.4, 239.0] | -0.17 [-0.74, 0.41] | -0.17 [-0.74, 0.41] | 0.104 | 0.703 | 49.2 | 30 | yes |

## Chosen configuration

`{"n_candidates_m": 4, "horizon_h": 5, "lam": 1.0, "mu": 30.0, "change_cost": 0.0, "tau": 0.17313824392675542, "slo_lam": 10.0}` with N = 5 members.
Measured decision time (background QoS): p95 mean 63.6 ms, worst p95 70.0 ms, max 279.0 ms.

Selected as the best of the grid on these seeds, so its paired value difference is optimistic (winner's curse); only the held-out evaluation measures it.
