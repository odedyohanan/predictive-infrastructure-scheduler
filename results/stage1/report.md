# Stage 1: forecast vs fact

- dev seeds: 1100-1109
- scenario: 60 s, base scheduler R6 greedy
- base checkpoint: sched:models/rl_r6/final.pt (sha256 ca02c87cf70d40a7)
- world model: worldmodel:models/gru_mix_bounded.pt (sha256 aa1dea80d94832e4)
- R6 decide time: 0.58 ms mean

Mean absolute error in real units; persistence = state stays as at the start.

## Horizon 1 windows (n = 1190 starts)

| target | model MAE | persistence MAE | model / persistence |
|---|---|---|---|
| a.p99_ms | 11.72 | 8.313 | 1.41 |
| a.queue_work_ms | 59.44 | 51.9 | 1.15 |
| a.cpu_cores | 0.1786 | 0.2043 | 0.87 |
| b.rate | 20.05 | 35.29 | 0.57 |
| b.buffer | 1.942 | 2.762 | 0.70 |
| c.work_core_ms | 55.47 | 127.9 | 0.43 |
| gpu_temp_0 | 0.03063 | 0.2942 | 0.10 |
| gpu_temp_1 | 0.06634 | 0.3407 | 0.19 |
| reward | 0.4224 | 0.6453 | 0.65 |

## Horizon 5 windows (n = 1150 starts)

| target | model MAE | persistence MAE | model / persistence |
|---|---|---|---|
| a.p99_ms | 34.41 | 34.46 | 1.00 |
| a.queue_work_ms | 169.9 | 200 | 0.85 |
| a.cpu_cores | 0.6089 | 0.5609 | 1.09 |
| b.rate | 44.98 | 130.8 | 0.34 |
| b.buffer | 3.351 | 3.848 | 0.87 |
| c.work_core_ms | 154.3 | 415.5 | 0.37 |
| gpu_temp_0 | 0.1585 | 1.36 | 0.12 |
| gpu_temp_1 | 0.4273 | 1.493 | 0.29 |
| reward | 1.079 | 2.173 | 0.50 |

## Horizon 10 windows (n = 1100 starts)

| target | model MAE | persistence MAE | model / persistence |
|---|---|---|---|
| a.p99_ms | 34.03 | 51.19 | 0.66 |
| a.queue_work_ms | 180.2 | 284.2 | 0.63 |
| a.cpu_cores | 0.9477 | 0.8982 | 1.06 |
| b.rate | 66.77 | 206.9 | 0.32 |
| b.buffer | 4.667 | 4.558 | 1.02 |
| c.work_core_ms | 243.5 | 644.3 | 0.38 |
| gpu_temp_0 | 0.3402 | 2.478 | 0.14 |
| gpu_temp_1 | 1.091 | 2.703 | 0.40 |
| reward | 1.747 | 3.487 | 0.50 |
