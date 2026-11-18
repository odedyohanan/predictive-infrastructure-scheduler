# Decision time, Step A at normal priority

- code: git HEAD 1b8799d852bd6b4819ea55db6e358aae35c1a102
- config: configs/step_a.json
- members: worldmodel:models/gru_mix_bounded.pt (sha256 aa1dea80d94832e4), models/ens_1.pt (sha256 de6f5c658c00aa37), models/ens_2.pt (sha256 f6134d88163b96b3), models/ens_3.pt (sha256 7b8ff07789452ca5), models/ens_4.pt (sha256 1c4893fa01673cbb)
- controller params: {"n_candidates_m": 4, "horizon_h": 3, "lam": 1.0, "mu": 8.0, "change_cost": 0.0, "tau": 0.1571656037280141, "slo_lam": 40.0}
- seeds: 1100-1102; scenario 60 s; 1 process, 1 torch thread, nice 0
- runtime: 10 s

Official limits: p95 <= 50 ms and max <= 500 ms in every scenario.

| seed | decisions | mean ms | p95 ms | max ms |
|---|---|---|---|---|
| 1100 | 120 | 4.0 | 5.5 | 5.8 |
| 1101 | 120 | 2.9 | 3.8 | 3.9 |
| 1102 | 120 | 3.6 | 4.8 | 5.4 |
| all | 360 | 3.5 | 5.2 | 5.8 |

- worst per-scenario p95 5.5 ms: within the 50 ms limit
- worst max 5.8 ms: within the 500 ms limit
