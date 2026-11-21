# FastSimulator speed

Phase B scenarios driven by R6 (greedy). *episode*: the closed-loop episode (simulator + R6 + bookkeeping); *sim only*: the same policies replayed on the simulator alone. Wall seconds per episode; speed-up = Python / fast.

## background priority (macOS background QoS)

Darwin arm64, commit 492793e, episode length 300 s, kernel warm-up 0.72 s (once per process).

| seed | kind | windows | max A queue | episode Python s | episode fast s | x | sim only Python s | sim only fast s | x |
|---|---|---|---|---|---|---|---|---|---|
| 3100 | none | 600 | 15 | 87.2 | 3.071 | 28 | 90.8 | 0.494 | 184 |
| 3101 | gpu_loss | 600 | 27 | 99.5 | 3.590 | 28 | 94.4 | 1.278 | 74 |
| 3102 | none | 600 | 290 | 95.6 | 3.449 | 28 | 93.4 | 1.372 | 68 |
| mean | | | | 94.1 | 3.370 | 28 | 92.9 | 1.048 | 89 |

## normal priority (nice 0)

Darwin arm64, commit d5d2509, episode length 300 s, kernel warm-up 0.10 s (once per process).

| seed | kind | windows | max A queue | episode Python s | episode fast s | x | sim only Python s | sim only fast s | x |
|---|---|---|---|---|---|---|---|---|---|
| 3100 | none | 600 | 15 | 13.6 | 0.466 | 29 | 12.9 | 0.181 | 71 |
| 3101 | gpu_loss | 600 | 27 | 13.4 | 0.497 | 27 | 13.2 | 0.196 | 68 |
| 3102 | none | 600 | 290 | 13.1 | 0.457 | 29 | 12.9 | 0.202 | 64 |
| mean | | | | 13.4 | 0.474 | 28 | 13.0 | 0.193 | 67 |
