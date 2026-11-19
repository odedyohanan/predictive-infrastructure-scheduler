# Step 0: counterfactual gate results

One section per attempt; never edited.

## Attempt 1: **PASS**

- code: git HEAD 7e4287b254e9a8a6669e4bf10f2dac10bc28d01a
- protocol sha256: 6a3aa1d1fbe0c3ec54be124d6e4be90534a1506215ae36c80ac23d9c13e4ebab
- model: worldmodel:models/gru_mix_bounded.pt (sha256 aa1dea80d94832e4)
- states: 500 evaluated, 0 dropped (requested 500; seeds 1100-1149, sampling seed 0)
- K / n_seeds / horizons: 10 / 8 / 1, 4
- gate thresholds: {"min_resolvable_states": 200, "min_intervention_cases": 100, "median_spearman": 0.5, "margin_over_random_null_p95": 0.25, "pairwise_sign_acc": 0.7, "intervention_sign_acc": 0.7}
- runtime: 10436 s with 5 processes

### Gate (h = 1)

| criterion | value | pass |
|---|---|---|
| G0 enough evidence | 499 resolvable states; cases take_from_B 1085, gpu_to_B 1077, isolate_A 1218 | yes |
| G1 median Spearman >= 0.50 | 0.794 | yes |
| G2 median lower bound >= random-null p95 + 0.25 | lower bound 0.782, random-null p95 0.030, needed 0.280 | yes |
| G3 pairwise sign accuracy >= 0.70 | 0.832 | yes |
| G4 interventions sign accuracy >= 0.70 | take_from_B 0.826, gpu_to_B 0.974, isolate_A 0.756 | yes |

Bootstrap 95% lower bound of (model - control) median Spearman, reported, not gated: heuristic 0.040, logprob 0.000, action_blind 0.782, persistence 0.782

Controls: heuristic = resource-ratio heuristic (sees the action); logprob = RL scheduler log-probability ranking (score only, no interventions); action_blind and persistence are leak checks that cannot see the action (0 by construction).

### Horizon 1

| predictor | median Spearman (resolvable) | median Spearman (all) | resolvable states | pairwise sign acc | resolvable pairs | reward MAE |
|---|---|---|---|---|---|---|
| model | 0.794 | 0.794 | 499/500 | 0.832 | 19100 | 2.0541 |
| heuristic | 0.721 | 0.721 | 499/500 | 0.786 | 19100 | 2.5523 |
| logprob | 0.770 | 0.770 | 499/500 | 0.817 | 19100 | n/a |
| action_blind | 0.000 | 0.000 | 499/500 | 0.000 | 19100 | 4.0833 |
| persistence | 0.000 | 0.000 | 499/500 | 0.000 | 19100 | 4.4009 |

### Horizon 4

| predictor | median Spearman (resolvable) | median Spearman (all) | resolvable states | pairwise sign acc | resolvable pairs | reward MAE |
|---|---|---|---|---|---|---|
| model | 0.794 | 0.794 | 499/500 | 0.823 | 20052 | 2.2112 |
| heuristic | 0.721 | 0.721 | 499/500 | 0.772 | 20052 | 2.7379 |
| logprob | 0.758 | 0.758 | 499/500 | 0.800 | 20052 | n/a |
| action_blind | 0.000 | 0.000 | 499/500 | 0.000 | 20052 | 4.1220 |
| persistence | 0.000 | 0.000 | 499/500 | 0.000 | 20052 | 4.5328 |

### Named interventions (h = 1)

| predictor | intervention | sign acc | cases | state majority acc | A p99 | B rate | C work |
|---|---|---|---|---|---|---|---|
| model | take_from_B | 0.826 | 1085 | 0.752 | 0.709 (141) | 0.981 (474) | 0.704 (470) |
| model | gpu_to_B | 0.974 | 1077 | 1.000 | 0.696 (92) | 1.000 (486) | 1.000 (499) |
| model | isolate_A | 0.756 | 1218 | 0.749 | 0.748 (318) | 0.880 (409) | 0.658 (491) |
| heuristic | take_from_B | 0.533 | 1085 | 0.242 | 0.738 (141) | 1.000 (474) | 0.000 (470) |
| heuristic | gpu_to_B | 0.894 | 1077 | 0.956 | 0.000 (92) | 0.967 (486) | 0.988 (499) |
| heuristic | isolate_A | 0.715 | 1218 | 0.661 | 0.761 (318) | 0.919 (409) | 0.515 (491) |
| action_blind | take_from_B | 0.000 | 1085 | 0.000 | 0.000 (141) | 0.000 (474) | 0.000 (470) |
| action_blind | gpu_to_B | 0.000 | 1077 | 0.000 | 0.000 (92) | 0.000 (486) | 0.000 (499) |
| action_blind | isolate_A | 0.000 | 1218 | 0.000 | 0.000 (318) | 0.000 (409) | 0.000 (491) |
| persistence | take_from_B | 0.000 | 1085 | 0.000 | 0.000 (141) | 0.000 (474) | 0.000 (470) |
| persistence | gpu_to_B | 0.000 | 1077 | 0.000 | 0.000 (92) | 0.000 (486) | 0.000 (499) |
| persistence | isolate_A | 0.000 | 1218 | 0.000 | 0.000 (318) | 0.000 (409) | 0.000 (491) |

