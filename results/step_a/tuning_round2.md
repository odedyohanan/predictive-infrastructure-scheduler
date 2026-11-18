# Step A tuning on development seeds

- code: git HEAD c63da9ffedda5ffdd485cced249572e5155b4acc
- members: worldmodel:models/gru_mix_bounded.pt (sha256 aa1dea80d94832e4), models/ens_1.pt (sha256 de6f5c658c00aa37), models/ens_2.pt (sha256 f6134d88163b96b3), models/ens_3.pt (sha256 7b8ff07789452ca5), models/ens_4.pt (sha256 1c4893fa01673cbb)
- calibration seeds: 1100-1109
- evaluation seeds: 1110-1139
- grid: N 5; M 4; H 3; lam 1; tau percentile 90
- fixed: mu 8.0, change_cost 0.0, slo_lam 40.0, scenario 60 s
- slo margin (pp): 1.0
- disagreement: std across members of a.p99_ms, a.queue_work_ms, b.rate, c.work_core_ms (encoded), divided by member 0's training input std, mean over keys and horizon
- runtime: 2213 s with 4 processes

Paired differences are (loop - base) per scenario, mean [95% CI]. Value in value units per second; SLO in percentage points of windows violating the SLO. Feasible = upper bound of the SLO v2 difference <= +1.0 pp (SLO v1 reported only), paired results on every evaluation seed, finite parameters, decision time p95 <= 100 ms in every scenario and max <= 1000 ms. **These are background-QoS limits, 2x the official 50 / 500 ms**: tuning runs at macOS background priority on efficiency cores with several processes, which inflates wall-clock time; the official limits are checked at normal priority by experiments/time_decisions.py.

## Base scheduler (R6 greedy)

- scenarios: 30; mean value per second 18991.2; SLO v2 6.56 %; SLO v1 6.53 %

## Calibrated trust thresholds (tau)

Percentiles of the ensemble disagreement of every candidate at base-scheduler states (calibration seeds).

| N | H | q90 |
|---|---|---|
| 5 | 3 | 0.1572 |

## Grid

| N | M | H | lam | tau q | tau | value diff | SLO v2 diff | SLO v1 diff | fallback | greedy applied | decide p95 ms | n | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 4 | 3 | 1.00 | 90 | 0.1572 | 217.6 [88.8, 346.4] | -0.00 [-0.47, 0.47] | -0.00 [-0.47, 0.47] | 0.104 | 0.770 | 48.0 | 30 | yes **chosen** |

## Chosen configuration

`{"n_candidates_m": 4, "horizon_h": 3, "lam": 1.0, "mu": 8.0, "change_cost": 0.0, "tau": 0.1571656037280141, "slo_lam": 40.0}` with N = 5 members.
Measured decision time (background QoS): p95 mean 48.0 ms, worst p95 56.2 ms, max 289.9 ms.

Selected as the best of the grid on these seeds, so its paired value difference is optimistic (winner's curse); only the held-out evaluation measures it.

## Ablations at the chosen configuration

| arm | value diff vs base | SLO v2 diff vs base | fallback | greedy applied | decide p95 ms | n |
|---|---|---|---|---|---|---|
| loop (chosen) | 217.6 [88.8, 346.4] | -0.00 [-0.47, 0.47] | 0.104 | 0.770 | 48.0 | 30 |
| random | -80.8 [-209.2, 47.6] | 1.92 [1.36, 2.47] | 0.058 | 0.717 | 7.9 | 30 |
| oracle | 382.8 [233.7, 532.0] | 1.81 [-0.14, 3.75] | 0.058 | 0.840 | 3041.6 | 30 |
| no_uncertainty | 214.0 [83.1, 344.9] | -0.03 [-0.49, 0.43] | 0.104 | 0.768 | 48.9 | 30 |
| no_fallback | 288.7 [167.7, 409.8] | 0.64 [-0.03, 1.31] | 0.058 | 0.754 | 53.6 | 30 |

random = uniform choice among the same candidates; oracle = each candidate run in a copy of the real simulator and scored with the same reward terms (upper bound of the candidate set); no_uncertainty = lam 0 (tau gate kept); no_fallback = no tau gate.
