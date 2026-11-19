# Decision time, Step B at normal priority

- code: git HEAD a582002c680ffd9621e596630225d9d84dfe6879
- config: configs/step_b.json
- members: worldmodel:models/gru_mix_bounded.pt (sha256 aa1dea80d94832e4), models/ens_1.pt (sha256 de6f5c658c00aa37), models/ens_2.pt (sha256 f6134d88163b96b3), models/ens_3.pt (sha256 7b8ff07789452ca5), models/ens_4.pt (sha256 1c4893fa01673cbb)
- controller params: {"n_candidates_m": 4, "horizon_h": 3, "lam": 1.0, "mu": 8.0, "change_cost": 0.0, "tau": 0.1571656037280141, "slo_lam": 40.0}; reviser {"mechanism": "gradient", "rev_steps": 4, "rev_lam": 4.0, "rev_lr": 0.6}
- seeds: 1100-1102; scenario 60 s; 1 process, 1 torch thread, nice 0
- runtime: 11 s

Official limits: p95 <= 50 ms and max <= 500 ms in every scenario.

| seed | decisions | mean ms | p95 ms | max ms | revised |
|---|---|---|---|---|---|
| 1100 | 120 | 4.9 | 5.4 | 40.7 | 0 |
| 1101 | 120 | 3.0 | 3.9 | 4.6 | 0 |
| 1102 | 120 | 12.8 | 41.0 | 42.8 | 0 |
| all | 360 | 6.9 | 35.8 | 42.8 | 0 |

- worst per-scenario p95 41.0 ms: within the 50 ms limit
- worst max 42.8 ms: within the 500 ms limit
