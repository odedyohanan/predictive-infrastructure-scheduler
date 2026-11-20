# Incident advisor: final paired evaluation

Protocol: configs/phase_b/final_eval.json, frozen in its own commit before the run. One section per attempt, written when the attempt starts and completed when it finishes; never edited afterwards.

## Attempt 1

- code: git HEAD 9a30ff0c1533b5eb89828c0249a43edbcf510b72
- protocol: configs/phase_b/final_eval.json (sha256 a2a526866ccee196, variant shadow_only, frozen in commit 9a30ff0c1533b5eb89828c0249a43edbcf510b72)
- frozen files: incident configs/phase_b/incident.json (1a455ffb10469434), alarms configs/phase_b/alarms.json (bee5e521a8c59095), world_model models/tf_b3.pt (cbac9449d272a82d), shadow_note results/phase_b/shadow_dev.md (8163b8203885cc70), step_a configs/step_a.json (2afb3198e4314fff), phase_a_protocol configs/final_eval.json (26b75314c5226a21), fitted results/raw/phase_b/shadow_tf_b3/predictors.pkl (882f23e5682034b0), shadow_config results/raw/phase_b/shadow_tf_b3/config.json (b5953668410a4259), shadow_risks results/raw/phase_b/shadow_tf_b3 (1796cfdde9cb36c4)
- seeds: 3000-3099 (100 episodes, 300 s each)
- arms: r6
- raw: results/raw/phase_b/final/attempt_1/
- runtime: 11777 s with 4 process(es) (macOS background QoS); seconds per episode by arm (mean): r6 120.2; predictor scoring 349.6 s per R6 episode
- projection: Projected official run (100 seeds x 300 s, all arms): 13.0 h on one process, 3.3 h with 4 process(es), 3.3 h with 4; linear in the episode length, so a rough estimate from short episodes
- finished (UTC): 2026-11-20 15:08:20

### Question 2: advisor vs R6, time in incident

R6: 100 episodes, 1 chronic (more than half the windows in incident); mean value per second 17629.11, SLO v2 8.4 %, mean time in incident over the non-chronic episodes 9.39 s, 30 counted incidents.

Question 2: not attempted (the shadow-mode gate stopped the phase before prevention: results/phase_b/shadow_dev.md).

### Question 1: learned vs not learned predictors on the held-out R6 episodes

Thresholds: conformal, from the shadow run's calibration episodes (never refitted here). Recall at lead >= min lead; Brier of the window risk against the realised outcome (new counted onset within H).

| predictor | threshold | recall building-up | recall sudden | median lead bu | false-alarm rate | alarms per quiet episode | Brier all / bu / sudden (n) |
|---|---|---|---|---|---|---|---|
| world_model | 0.7812 | 0.29 (6/21) | 0.11 (1/9) | 7.0 | 0.667 | 0.10 | 0.074 / 0.088 / 0.067 (54200/16867/37333) |
| hazard | 0.0593 | 0.00 (0/21) | 0.00 (0/9) | - | 1.000 | 0.04 | 0.010 / 0.021 / 0.005 (54200/16867/37333) |
| analogue | 0.3800 | 0.05 (1/21) | 0.00 (0/9) | 2.0 | 0.968 | 0.21 | 0.011 / 0.023 / 0.005 (54200/16867/37333) |
| queueing | 0.8556 | 0.00 (0/21) | 0.00 (0/9) | - | 1.000 | 0.12 | 0.210 / 0.229 / 0.201 (54200/16867/37333) |
| change_point | none feasible | 0.00 (0/21) | 0.00 (0/9) | - | - | 0.00 | 0.070 / 0.130 / 0.043 (54200/16867/37333) |

Better non-learned predictor (held-out building-up recall): change_point. Recall difference, 95 % bootstrap interval over episodes (2000 resamples, shared stream):

| predictor | role | building-up | sudden | above zero |
|---|---|---|---|---|
| world_model | secondary (no success claim) | +0.29 [+0.08, +0.54] | +0.11 [+0.00, +0.30] | yes |
| hazard | secondary (no success claim) | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | no |
| analogue | primary | +0.05 [+0.00, +0.18] | +0.00 [+0.00, +0.00] | no |

The protocol fixed one primary learned predictor before the run (analogue: best building-up recall on the shadow run's selection episodes); the success rule is judged on it alone. The other learned predictors are secondary and reported without a success claim.

**Question 1: not met (analogue not above the better non-learned predictor)**

