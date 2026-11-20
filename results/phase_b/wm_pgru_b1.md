# Probabilistic GRU world model pgru_b1

Checkpoint sha256 `b3785ac7511d671f31de5bb39a9315a59aa98ff72f1744aa704f400a59418241`; 41 epochs, best validation NLL -0.5249; trained on 240 history episodes (143760 steps), validation 30; 350 s.

Recipe: context 32 (+ room 32), burn-in 8, hidden 128 x 2 layers, lr 0.001, batch 64, patience 20, seed 0.

| epoch | train NLL | val NLL |
|---|---|---|
| 0 | 0.3478 | 0.1733 |
| 4 | -0.1471 | -0.2071 |
| 8 | -0.3378 | -0.3703 |
| 12 | -0.4656 | -0.4510 |
| 16 | -0.5623 | -0.5193 |
| 20 | -0.6424 | -0.5249 |
| 24 | -0.7003 | -0.5247 |
| 28 | -0.7531 | -0.4591 |
| 32 | -0.7953 | -0.1207 |
| 36 | -0.8310 | -0.1618 |
| 40 | -0.8337 | 0.0145 |

## Calibration on 50 episodes (1400 starts, 64 samples, H = 20)

Coverage of the central 80% interval and CRPS (encoded units). Overdue and late counts are 0 in most windows, where an interval containing 0 covers for free: they are shown on all windows and on the windows with a non-zero truth (the ladder judges the latter); p99 only on windows with completions. n = windows judged at H.

| output | windows | cov h=1 | cov h=5 | cov h=20 | CRPS h=1 | CRPS h=5 | CRPS h=20 | n |
|---|---|---|---|---|---|---|---|---|
| a.queue_overdue | all | 0.978 | 0.921 | 0.601 | 0.149 | 0.500 | 1.451 | 1400 |
| a.queue_overdue | non-zero | 0.691 | 0.517 | 0.761 | 1.324 | 2.234 | 2.159 | 67 |
| a.late_new | all | 0.848 | 0.916 | 0.745 | 0.429 | 0.761 | 1.311 | 1400 |
| a.late_new | non-zero | 0.843 | 0.933 | 0.915 | 0.551 | 0.840 | 1.259 | 729 |
| a.p99_ms | with completions | 0.919 | 0.946 | 0.960 | 0.100 | 0.237 | 0.511 | 1400 |

Brier of the rollout incident risk vs Monte Carlo labels: 0.0667 (base rate 0.049: 0.0318); skill -1.101 over 144 labelled states.

Intervention rank agreement (Spearman of predicted vs branched effect on mean a.queue_overdue at h=5, 8 reseeds): 0.221 over 150 effects on 50 states (gpu_to_B -0.244, isolate_A 0.182, take_from_B 0.340).

Ladder criterion (coverage at H within [0.70, 0.90] for the three outputs, non-zero windows for overdue and late, and Brier skill > 0): FAIL.
