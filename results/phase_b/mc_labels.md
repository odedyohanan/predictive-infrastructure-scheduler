# Monte Carlo risk labels

Seeds: 3400-3499 (100 seeds), R6 episodes only. Rule k=19, interval=76, majority=0.5, warm-up 20; horizon H=20 windows.
States: 300 (shards of 20); 16 branches each, reseed 12_000_000 + seed*1000 + j; R6 greedy continues in every branch for H + k - 1 windows (onsets counted in (t, t + H]); reseeding redraws arrivals, service times and A's future bursts.
All states of one episode reuse the same reseed streams, so labels within an episode are slightly correlated.

## Plain branching

- risk = 0 in 85.3% of states; mean risk 0.044
- 192.3 ms per branch-window, replay 20.0 s per state

## Estimator

Multilevel splitting is used when the plain estimate is 0 in more than 80% of states: yes (final labels: restart).
- levels on A's overdue backlog [319, 1360, 3170], 16 roots, split 2, at most 128 paths; risk = 0 in 82.7% of states, mean risk 0.043

## Labels by stratum and kind

| stratum | kind | states | mean risk | risk 0 |
|---|---|---|---|---|
| pre_onset | ambient | 13 | 0.028 | 46% |
| pre_onset | cpu_loss | 24 | 0.089 | 67% |
| pre_onset | interference | 12 | 0.007 | 75% |
| pre_onset | leak | 101 | 0.067 | 80% |
| uniform | ambient | 11 | 0.004 | 82% |
| uniform | cpu_loss | 14 | 0.011 | 86% |
| uniform | gpu_loss | 9 | 0.000 | 100% |
| uniform | interference | 14 | 0.000 | 100% |
| uniform | leak | 12 | 0.236 | 67% |
| uniform | load_ramp | 9 | 0.000 | 100% |
| uniform | none | 81 | 0.009 | 93% |

## Replay check

Before a state is labelled, the replayed violation and a_queue_overdue series up to it must equal the stored rows; an episode that differs is left out. Refused episodes: 0.
