# Recurrent state-space (RSSM) world model rssm_b2

Checkpoint sha256 `8ee702e513538b2c1c2ad1aca409c9d31ef81866f7ebaf1a36f03f9a17869228`; 275 epochs, best validation loss -2.6764; trained on 240 history episodes (143760 steps), validation 30; 4051 s.

Architecture: recurrent state-space model (`pis.phase_b.wm_rssm`): deterministic GRU state of 128, diagonal Gaussian latent of 16 with prior p(z | h) and posterior q(z | h, change), decoder to a Gaussian over the change of the 43 state numbers (standardised, p50/p99 masked without completions); prior, posterior and decoder networks with 1 hidden layer(s). Loss: masked Gaussian NLL + KL(q || p) floored at 1.0 nats (free bits), per target number; early stopping on the validation negative ELBO. Rollouts: history filtered through the posterior, futures sampled through the prior, states clipped to the training range.

Recipe: context 32 (+ room 32), burn-in 8, hidden 128, latent 16, layers 1, free bits 1.0, lr 0.001, batch 64, patience 20, seed 0.

| epoch | train loss | val loss |
|---|---|---|
| 0 | 0.2138 | -0.1953 |
| 27 | -1.9116 | -1.8144 |
| 54 | -2.1298 | -1.9837 |
| 81 | -2.1962 | -2.0865 |
| 108 | -2.3796 | -2.2309 |
| 135 | -2.4172 | -2.3522 |
| 162 | -2.4971 | -2.3936 |
| 189 | -2.5335 | -2.4689 |
| 216 | -2.6585 | -2.5418 |
| 243 | -2.6830 | -2.4886 |
| 270 | -2.7353 | -2.6699 |

## Calibration on 50 episodes (1400 starts, 64 samples, H = 20)

Coverage of the central 80% interval and CRPS (encoded units). Overdue and late counts are 0 in most windows, where an interval containing 0 covers for free: they are shown on all windows and on the windows with a non-zero truth (the ladder judges the latter); p99 only on windows with completions. n = windows judged at H.

| output | windows | cov h=1 | cov h=5 | cov h=20 | CRPS h=1 | CRPS h=5 | CRPS h=20 | n |
|---|---|---|---|---|---|---|---|---|
| a.queue_overdue | all | 0.984 | 0.757 | 0.549 | 0.111 | 0.274 | 0.577 | 1400 |
| a.queue_overdue | non-zero | 0.706 | 0.317 | 0.627 | 1.595 | 3.161 | 3.592 | 67 |
| a.late_new | all | 0.769 | 0.875 | 0.695 | 0.408 | 0.682 | 0.996 | 1400 |
| a.late_new | non-zero | 0.744 | 0.906 | 0.898 | 0.544 | 0.826 | 1.085 | 729 |
| a.p99_ms | with completions | 0.791 | 0.934 | 0.942 | 0.090 | 0.202 | 0.389 | 1400 |

Brier of the rollout incident risk vs Monte Carlo labels: 0.0331 (base rate 0.049: 0.0318); skill -0.043 over 144 labelled states.

Intervention rank agreement (Spearman of predicted vs branched effect on mean a.queue_overdue at h=5, 8 reseeds): 0.176 over 150 effects on 50 states (gpu_to_B -0.051, isolate_A 0.332, take_from_B -0.027).

Ladder criterion (coverage at H within [0.70, 0.90] for the three outputs, non-zero windows for overdue and late, and Brier skill > 0): FAIL.

Note: the KL term is divided by the 43 state numbers while the NLL is averaged over the unmasked numbers only, so with p50/p99 masked the KL weighs slightly less than in the per-number ELBO (an implicit beta just under 1).
