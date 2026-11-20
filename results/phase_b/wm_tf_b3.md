# Causal transformer world model tf_b3

Checkpoint sha256 `cbac9449d272a82de33d2988a30eb3f164e758213990954fd72cfc0ba9db741f`; 37 epochs, best validation NLL -0.7712; trained on 240 history episodes (143760 steps), validation 30; 454 s.

Architecture: causal transformer (`pis.phase_b.wm_transformer`): the 79-number input projected to d_model 64 plus a learned position embedding over the 32-window context, 4 pre-norm blocks (causal self-attention with 4 heads, ReLU feed-forward of 128, dropout 0.0), a Gaussian head over the change of the 43 state numbers (standardised, log-variance clamped, p50/p99 masked without completions). Loss: masked Gaussian NLL; early stopping on the validation NLL. Rollouts: autoregressive over a sliding window of the last 32 steps (history, then sampled states), states clipped to the training range.

Recipe: context 32 (room 0), burn-in 8, d_model 64, heads 4, feed-forward 128, layers 4, dropout 0.0, lr 0.001, batch 64, patience 20, seed 0.

| epoch | train NLL | val NLL |
|---|---|---|
| 0 | 0.2230 | -0.0406 |
| 3 | -0.3385 | -0.4214 |
| 6 | -0.4776 | -0.6007 |
| 9 | -0.6238 | -0.7076 |
| 12 | -0.6750 | -0.4343 |
| 15 | -0.7487 | -0.7151 |
| 18 | -0.7649 | -0.4906 |
| 21 | -0.7271 | -0.6587 |
| 24 | -0.7598 | -0.5856 |
| 27 | -0.8697 | -0.6584 |
| 30 | -0.8952 | -0.5156 |
| 33 | -0.8553 | -0.7008 |
| 36 | -0.7681 | -0.7353 |

## Calibration on 50 episodes (1400 starts, 64 samples, H = 20)

Coverage of the central 80% interval and CRPS (encoded units). Overdue and late counts are 0 in most windows, where an interval containing 0 covers for free: they are shown on all windows and on the windows with a non-zero truth (the ladder judges the latter); p99 only on windows with completions. n = windows judged at H.

| output | windows | cov h=1 | cov h=5 | cov h=20 | CRPS h=1 | CRPS h=5 | CRPS h=20 | n |
|---|---|---|---|---|---|---|---|---|
| a.queue_overdue | all | 0.976 | 0.941 | 0.497 | 0.140 | 0.528 | 1.675 | 1400 |
| a.queue_overdue | non-zero | 0.632 | 0.433 | 0.791 | 1.249 | 2.453 | 2.140 | 67 |
| a.late_new | all | 0.850 | 0.901 | 0.800 | 0.425 | 0.782 | 1.195 | 1400 |
| a.late_new | non-zero | 0.843 | 0.906 | 0.897 | 0.539 | 0.868 | 1.210 | 729 |
| a.p99_ms | with completions | 0.925 | 0.946 | 0.960 | 0.102 | 0.260 | 0.540 | 1400 |

Brier of the rollout incident risk vs Monte Carlo labels: 0.0785 (base rate 0.049: 0.0318); skill -1.470 over 144 labelled states.

Intervention rank agreement (Spearman of predicted vs branched effect on mean a.queue_overdue at h=5, 8 reseeds): 0.193 over 150 effects on 50 states (gpu_to_B -0.331, isolate_A 0.336, take_from_B 0.005).

Ladder criterion (coverage at H within [0.70, 0.90] for the three outputs, non-zero windows for overdue and late, and Brier skill > 0): FAIL.
