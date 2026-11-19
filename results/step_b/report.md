# Step B tuning on development seeds

- code: git HEAD 1011970aef15a7141c0e819e9bb36bd86fb870d6
- step A config: configs/step_a.json (Step A beats base: True)
- members: worldmodel:models/gru_mix_bounded.pt (sha256 aa1dea80d94832e4), models/ens_1.pt (sha256 de6f5c658c00aa37), models/ens_2.pt (sha256 f6134d88163b96b3), models/ens_3.pt (sha256 7b8ff07789452ca5), models/ens_4.pt (sha256 1c4893fa01673cbb)
- step A params: {"n_candidates_m": 4, "horizon_h": 3, "lam": 1.0, "mu": 8.0, "change_cost": 0.0, "tau": 0.1571656037280141, "slo_lam": 40.0}
- evaluation seeds: 1140-1199
- grid: mechanism gradient,neighbourhood; gradient steps 2,4; neighbourhood steps 1,3; revision lam 1,4
- fixed: gradient learning rate 0.6, scenario 60 s
- slo margin (pp): 1.0
- runtime: 4746 s with 4 processes

Paired differences are per scenario, mean [95% CI]. Value in value units per second; SLO in percentage points of windows violating the SLO. Feasible = paired results on every evaluation seed, upper bound of the (reviser - base) SLO v2 difference <= +1.0 pp (SLO v1 reported only), decision time p95 <= 100 ms in every scenario and max <= 1000 ms. **These are background-QoS limits, 2x the official 50 / 500 ms**: tuning runs at macOS background priority on efficiency cores with several processes, which inflates wall-clock time; the official limits are checked at normal priority by experiments/time_decisions.py.

## Reference arms

- base scheduler (R6 greedy): 60 scenarios; mean value per second 19012.4; SLO v2 5.76 %; SLO v1 5.74 %
- Step A loop vs base: value 144.2 [48.9, 239.5]; SLO v2 0.26 [0.03, 0.50]; fallback 0.102; decide p95 49.6 ms (worst 64.9); n 60

## Grid

| mechanism | steps | lam | value vs base | SLO v2 vs base | value vs Step A | SLO v2 vs Step A | revised / triggers | exploit rate | real gain / revision | real gain vs greedy (untrusted) | decide p95 ms (worst) | decide max ms | n | feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neighbourhood | 3 | 4.00 | 244.9 [29.4, 460.4] | 1.93 [-0.11, 3.97] | 100.7 [-82.6, 284.0] | 1.67 [-0.24, 3.57] | 104 / 516 | 0.250 | 0.658 | 3.307 (28) | 123.8 (373.4) | 482.9 | 60 | no |
| gradient | 2 | 4.00 | 181.4 [70.7, 292.2] | 1.11 [-0.23, 2.45] | 37.2 [-10.4, 84.8] | 0.85 [-0.37, 2.06] | 12 / 475 | 0.167 | 5.735 | 0.552 (7) | 100.1 (243.0) | 534.0 | 60 | no |
| gradient | 2 | 1.00 | 180.7 [70.0, 291.4] | 1.11 [-0.23, 2.45] | 36.5 [-10.9, 83.9] | 0.85 [-0.37, 2.06] | 12 / 474 | 0.167 | 5.635 | 0.373 (7) | 99.9 (255.2) | 341.9 | 60 | no |
| neighbourhood | 1 | 1.00 | 169.7 [60.0, 279.3] | 0.65 [0.20, 1.10] | 25.4 [-18.8, 69.7] | 0.39 [-0.00, 0.78] | 53 / 447 | 0.245 | 1.799 | 1.265 (29) | 75.6 (157.3) | 236.1 | 60 | no |
| neighbourhood | 3 | 1.00 | 163.9 [69.4, 258.4] | 0.40 [0.15, 0.65] | 19.7 [2.3, 37.2] | 0.14 [-0.03, 0.30] | 39 / 408 | 0.308 | -0.051 | 3.000 (20) | 164.8 (585.2) | 1040.5 | 60 | no |
| gradient | 4 | 4.00 | 154.6 [57.0, 252.2] | 0.26 [0.05, 0.48] | 10.4 [-2.8, 23.7] | 0.00 [-0.09, 0.09] | 24 / 407 | 0.417 | 0.386 | 0.413 (15) | 145.4 (408.3) | 481.2 | 60 | no |
| gradient | 4 | 1.00 | 151.9 [55.1, 248.7] | 0.25 [0.05, 0.45] | 7.7 [-2.6, 17.9] | -0.01 [-0.11, 0.09] | 23 / 411 | 0.435 | 4.590 | 0.888 (12) | 144.6 (395.7) | 547.2 | 60 | no |
| neighbourhood | 1 | 4.00 | 130.7 [9.7, 251.8] | 1.69 [0.20, 3.18] | -13.5 [-86.3, 59.4] | 1.43 [-0.06, 2.92] | 115 / 521 | 0.235 | 0.729 | 1.739 (42) | 78.8 (313.5) | 650.3 | 60 | no |

Triggers: Step A's best candidate untrusted (disagreement above tau) or predicted to violate A's SLO. Revised: the revision was applied (better revision score and, with the trust gate, disagreement within tau). Exploit rate: share of applied revisions whose predicted reward rose while the real simulator scored the revised policy below the start policy over the same horizon (same random state); real gain / revision: mean real score difference (revised - start) per applied revision; real gain vs greedy: the same against the greedy fallback Step A would have applied, for revisions of untrusted candidates (count in brackets).

## Model exploitation

No config's exploit rate exceeds 0.5.

## Gradient vs neighbourhood search

Best of each mechanism: gradient = gradient steps 2 lam 4.00; neighbourhood = neighbourhood steps 3 lam 4.00. Paired (gradient - neighbourhood): value -63.5 [-204.6, 77.7], SLO v2 -0.82 [-2.13, 0.49], n 60.

## Chosen configuration

No configuration met the SLO non-inferiority and decision-time constraints; none is recommended.
