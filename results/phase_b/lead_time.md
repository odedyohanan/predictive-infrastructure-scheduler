# Minimum useful lead time

States: 24 (first window with a rising overdue backlog for 3 windows, one per episode, 50 R6 episodes of seeds 3400-3449). Interventions held for 40 windows vs R6 continuing, 8 paired reseeds; effect = first of 3 consecutive windows whose paired 95 % CI of A's overdue backlog (intervention - R6) is below 0.

| episode | window | take_from_B | gpu_to_B | isolate_A | earliest |
|---|---|---|---|---|---|
| 3401 | 470 | - | skipped | - | - |
| 3404 | 495 | - | - | - | - |
| 3409 | 114 | skipped | - | - | - |
| 3410 | 27 | skipped | - | - | - |
| 3411 | 188 | 1 | skipped | 1 | 1 |
| 3414 | 393 | skipped | - | - | - |
| 3416 | 213 | - | skipped | - | - |
| 3417 | 288 | skipped | - | - | - |
| 3421 | 375 | skipped | - | - | - |
| 3423 | 343 | - | skipped | - | - |
| 3424 | 52 | skipped | - | - | - |
| 3429 | 202 | skipped | - | - | - |
| 3431 | 387 | skipped | - | - | - |
| 3433 | 448 | - | skipped | - | - |
| 3435 | 371 | skipped | - | - | - |
| 3438 | 305 | skipped | - | - | - |
| 3439 | 29 | 1 | - | - | 1 |
| 3440 | 319 | skipped | - | - | - |
| 3441 | 90 | 1 | - | - | 1 |
| 3443 | 48 | - | skipped | - | - |
| 3444 | 32 | skipped | - | - | - |
| 3447 | 22 | skipped | - | - | - |
| 3448 | 43 | skipped | 26 | - | 26 |
| 3449 | 33 | skipped | - | - | - |

States without any effect: 20 (reported, not used).
min_lead_windows = 1 (median over 4 states, rounded up); H = max(20, 2 x 1) = 20 windows.
Timing: 212.3 ms per branch-window, replay 35.8 s per state.
