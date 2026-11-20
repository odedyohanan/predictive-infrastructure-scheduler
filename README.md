# Predictive infrastructure scheduler

Can a scheduler imagine the consequences of an action before it takes it, and get better for it?

This repository closes a loop between three pieces that were built separately:

- a reinforcement-learning **scheduler** (R6) that proposes how to split CPU cores and a GPU between three
  workloads every 500 ms (from the sibling repo `priority-scheduler`);
- a learned **world model**, a GRU that predicts the next telemetry of the cluster from the recent history and
  the chosen policy (from the sibling repo `scheduler-simulation-model`);
- an **evaluator** that scores candidate policies by their imagined consequences, and either accepts the best
  one or falls back to what the scheduler would have done anyway.

"Reality" is the mathematical simulator the world model was trained on. The world model never sees it at decision
time; it only imagines.

```
          telemetry history
                 |
                 v
     +-----------------------+   M candidate policies   +------------------------------+
     |  RL scheduler (R6)    | -----------------------> |  world-model ensemble (5 GRU) |
     |  greedy + samples     |                          |  imagine H windows ahead      |
     +-----------------------+                          +------------------------------+
                 ^                                                   |
                 | greedy policy (fallback)                          v
                 |                                    +-------------------------------+
                 +----------------------------------- |  evaluator                    |
                                                      |  predicted value - lam*disagr |
                                                      |  - mu*queue penalty; trust tau |
                                                      +-------------------------------+
                                                                     |
                                  Step B (optional): revise the chosen policy by gradient
                                  through the GRU or neighbourhood search, then re-check
                                                                     |
                                                                     v
                                                           policy applied to the simulator
```

## Results (Phase A)

All numbers are paired per scenario against the base scheduler R6 (same scenario, same simulator randomness),
mean [95% CI]. Value per second is the RL scheduler's own reward (batches of B and core-milliseconds of C);
the SLO metric is the share of 500 ms windows in which workload A violates its latency SLO. Tuning used only
development seeds 1100-1199; the final comparison used held-out seeds 2000-2099 once, with a protocol frozen in
its own commit before the run.

### Step 0: does the world model understand actions?

A gate before anything else: for 500 states, rank 10 candidate policies by their real outcome (simulator
branches) and by the GRU's prediction. The GRU passed every criterion on the first attempt (median Spearman
0.79, pairwise sign accuracy 0.83, intervention sign accuracy 0.76-0.97). Its ranking is, however, only
marginally better than the scheduler's own log-probability ranking (bootstrap lower bound of the difference
0.000), so large gains were not expected. [results/step0/step0.md](results/step0/step0.md)

### Final evaluation (100 held-out scenarios)

| arm | value / s vs R6 | SLO v2 pp vs R6 | success rule |
|---|---|---|---|
| **Step A** (best-of-N with ensemble and fallback) | **+125.0 [69.5, 180.5]** (+0.66 %) | +0.08 [-0.13, 0.30] | **met** |
| **Step B** (Step A + revision by gradient) | **+130.6 [74.9, 186.3]** (+0.69 %) | +0.11 [-0.11, 0.33] | **met** |
| random choice among the same candidates | -140.3 [-198.5, -82.1] | +1.65 [1.28, 2.02] | |
| oracle choice (candidates scored in the real simulator) | +269.8 [192.4, 347.1] | +0.71 [-0.40, 1.81] | |
| Step A without the uncertainty penalty | +126.5 [71.0, 182.1] | +0.06 [-0.16, 0.28] | |
| Step A without the trust gate / fallback | +163.3 [104.9, 221.7] | +0.17 [-0.07, 0.42] | |
| R3 (earlier scheduler) | -62.1 [-92.3, -31.9] | +0.15 [-0.01, 0.31] | |
| non-ML feedback controller | -545.2 [-721.9, -368.5] | +0.67 [0.20, 1.13] | |

Success rule: value CI above zero, SLO v2 upper bound at most +1 pp, decision time p95 <= 50 ms and max <= 500 ms
at normal priority. Step A decides in 5.5 ms (worst per-scenario p95), Step B in 41 ms.
[results/final/report.md](results/final/report.md)

What this says, plainly:

- **Imagining before acting helps, a little.** Step A gains about 0.7 % value per second without a detectable
  SLO cost. Better in 74 of 100 scenarios, worse in 25; the median gain (+47) is much smaller than the mean, a
  few scenarios gain a lot.
- **The gain comes from the world model's ranking.** Choosing at random among the same candidates is clearly
  worse than the base scheduler; the model's choice is +265 value/s better than random.
- **Revision (Step B) adds nothing detectable:** +5.6 [-1.8, 13.0] over Step A, at seven times the decision time.
- **The ensemble's uncertainty penalty adds nothing detectable** (-1.5 [-7.5, 4.5]).
- **The trust gate and fallback cost value** (-38 value/s versus no gate) and buy a smaller, not significant,
  SLO risk.
- **There is headroom.** An oracle that scores the same candidates in the real simulator gains another +145
  value/s over Step A (at a much higher SLO risk), so a better world model, not a better search, is the
  bottleneck.
  This oracle is an inflated upper bound: its simulator branches (and those of Step 0's ground truth) shared
  the same future random bursts, so the oracle saw the future the chosen policy would meet.

### How we got there (development seeds)

- **Step A, round 1** ([results/step_a/tuning.md](results/step_a/tuning.md)): a grid over the number of
  ensemble members, candidates, horizon, uncertainty weight and trust threshold. Every configuration gained
  value but raised SLO violations by more than the +1 pp margin: no configuration qualified.
- **Step A, round 2** ([results/step_a/slo_sweep/](results/step_a/slo_sweep/),
  [results/step_a/tuning_round2.md](results/step_a/tuning_round2.md)): the evaluator's queue penalty `mu` and
  SLO weight `slo_lam` had been fixed in round 1. Raising them traded some value for SLO and every pass
  qualified. Chosen: N 5, M 4, H 3, `mu` 8, `slo_lam` 40, trust threshold at the 90th percentile of the ensemble
  disagreement. The highest-value configuration was not chosen: it sat at the SLO and time limits for a gain
  within noise.
- **Step B** ([results/step_b/report.md](results/step_b/report.md),
  [results/step_b/report_round2.md](results/step_b/report_round2.md)): gradient through the GRU versus
  neighbourhood search. Round 1 accepted no configuration because the decision-time gate for the slow
  background-priority runs was set at 2x the official limit, a guess; the measured background/normal ratio of
  Step A was about 9x. Re-selecting from the same runs with that ratio chose gradient revision with 4 steps;
  its official decision time was then measured at normal priority. Neighbourhood search earned more value on
  development seeds but was too slow or broke the SLO margin.

Selection on development seeds makes those numbers optimistic; only the held-out evaluation measures the
chosen configurations.

## Results (Phase B): incident advisor

Phase B asked two questions on a harder world: scenarios of 300 s with injected degradations (load ramps,
memory leaks, interference, CPU loss, and distractors that should not raise alarms). **Question 1:** can a
learned predictor warn about an SLO incident early enough, better than predictors that learn nothing?
**Question 2:** if so, can an advisor that switches the scheduler into a careful mode on the alarm cut the time
spent in incidents? A gate between them: if no predictor catches building-up incidents (recall above 0.2 at
the calibrated false-alarm rate), prevention is not attempted.

**Answer: no on question 1, so question 2 was not attempted.** On 100 held-out scenarios (seeds 3000-3099,
frozen protocol, one attempt, R6 only), the pre-registered primary learned predictor (nearest-neighbour
analogues, chosen on development seeds) caught 1 of 21 building-up incidents: +0.05 [0.00, 0.18] over the
better non-learned predictor, which caught none. [results/phase_b/final.md](results/phase_b/final.md)

| predictor | building-up recall | sudden recall | false-alarm rate | Brier (window risk) |
|---|---|---|---|---|
| world model (transformer, rollouts) | 0.29 (6/21) | 0.11 (1/9) | 0.67 | 0.074 |
| analogue kNN (primary) | 0.05 (1/21) | 0.00 (0/9) | 0.97 | 0.011 |
| hazard (logistic) | 0.00 | 0.00 | 1.00 | 0.010 |
| queueing model (no learning) | 0.00 | 0.00 | 1.00 | 0.210 |
| change point (no learning) | 0.00 | 0.00 | no feasible threshold | 0.070 |

- **The world model is the only predictor that caught a useful share**, +0.29 [+0.08, +0.54] over the better
  non-learned predictor on held-out scenarios. It was a secondary predictor in the protocol (it tied with the
  analogue on development seeds, 2 of 10, and lost the tie on false alarms), so this is a lead, not a result.
- **Alarms are mostly false.** With thresholds calibrated by conformal risk control (alpha 0.1) on development
  episodes, two thirds or more of the alarm events of every predictor were not followed by an incident within
  H = 20 windows on held-out scenarios: the calibration did not transfer, and few incidents (10 building-up in
  the selection episodes) make every rate noisy.
- **Low Brier is not detection.** Hazard and analogue have the best Brier scores because they predict the base
  rate almost everywhere.
- **No world model met the calibration bar** ([results/phase_b/wm_pgru_b1.md](results/phase_b/wm_pgru_b1.md),
  [wm_rssm_b2.md](results/phase_b/wm_rssm_b2.md), [wm_tf_b3.md](results/phase_b/wm_tf_b3.md)): 80 % intervals
  covering 70-90 % at H for queue overdue, late requests and p99 latency, and incident-risk Brier skill above
  zero against Monte Carlo labels. All three rungs (probabilistic GRU, RSSM, causal transformer) failed; the
  transformer failed fewest criteria and was used by the rule fixed before the rungs were compared. The RSSM
  had by far the best risk Brier (skill -0.04 against -1.1 and -1.5).
- **Preventive interventions rarely matter** ([results/phase_b/lead_time.md](results/phase_b/lead_time.md)):
  in 4 of 24 sampled pre-incident states did any intervention change the outcome measurably, so even a good
  alarm would have had little to act on.

### How we got there (development seeds)

400 history episodes from varied policies ([history.md](results/phase_b/history.md)); a frozen incident rule
(19 of 76 windows in violation, new onsets only, 20 warm-up windows; [incident_rule.md](results/phase_b/incident_rule.md));
Monte Carlo risk labels for 300 states with multilevel splitting, because the plain estimate was zero in 85 %
of them ([mc_labels.md](results/phase_b/mc_labels.md)); shadow mode of five predictors on 50 calibration and
50 selection episodes, rerun once per world-model rung ([shadow_dev.md](results/phase_b/shadow_dev.md) and its
git history). Every rung stopped at the gate.

## Caveats

- One laptop, one simulator family, 60 s scenarios. The simulator is "reality" by construction; nothing here
  touches a real cluster.
- The official decision-time check uses 3 development scenarios at normal priority. Under background priority
  in the final run (several processes on efficiency cores) Step B exceeded 500 ms in 8 decisions.
- Phase B: 300 s scenarios with one injected degradation each; 10 building-up incidents in the selection
  episodes and 21 in the held-out ones, so recalls and false-alarm rates are coarse. The Monte Carlo labels use
  300 states x 16 branches (a full-size run did not fit a night). The rollout world-model predictor is slow
  (about 0.5 s per window for the transformer on one efficiency core).
- The base scheduler R6 is the reference throughout and never changed. The world-model ensemble members 1-4
  were trained here with the sibling repo's recipe on different seeds; their checkpoints are not in git.

## Running it

Python 3.11+, CPU only. The sibling repositories must be checked out next to this one, or pointed to by
environment variables:

| variable | default |
|---|---|
| `SCHED_REPO` | `../priority-scheduler` |
| `WORLDMODEL_REPO` | `../scheduler-simulation-model` |
| `OLD_DATA` | `../priority_scheduler_model/data` |

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Long jobs run at low priority and can be stopped together:

```bash
scripts/gentle.sh <name> .venv/bin/python experiments/<script>.py --procs 4   # log in logs/<name>.log
scripts/stop.sh
```

| script | what it does |
|---|---|
| `experiments/train_ensemble.py` | trains the extra GRU ensemble members |
| `experiments/step0_run.py` | Step 0 counterfactual gate |
| `experiments/tune_step_a.py`, `scripts/step_a_slo_sweep.sh` | Step A tuning on development seeds |
| `experiments/tune_step_b.py` | Step B tuning (and re-selection from saved runs) |
| `experiments/time_decisions.py` | decision time at normal priority |
| `experiments/final_eval.py` | the frozen final paired evaluation |
| `experiments/b_generate_history.py` | Phase B history episodes (degraded scenarios, varied policies) |
| `experiments/b_incident_rule.py`, `b_lead_time.py` | incident rule and minimum useful lead time |
| `experiments/b_mc_labels.py` | Monte Carlo incident-risk labels |
| `experiments/b_train_world_model.py` | probabilistic world models (`--arch` pgru, rssm or transformer) |
| `experiments/b_shadow.py` | shadow mode: five predictors, conformal alarms, the gate |
| `experiments/b_tune_advisor.py`, `b_time_advisor.py` | advisor tuning and decision time (not reached) |
| `experiments/b_final_eval.py` | the frozen Phase B final evaluation |

## Layout

`pis/` the package (adapters to the sibling repos, harness, ensemble, evaluator, loop, reviser, runner,
statistics); `experiments/` one script per step; `configs/` chosen configurations and the frozen protocol;
`results/` the notes every number above comes from; `tests/` unit tests.

## Next

Phase A asked whether imagination helps at all; the oracle gap pointed at the world model. Phase B found that
better calibrated world models are still not enough to warn about incidents reliably on this simulator, while
the rollout world model was the only predictor with a useful catch rate. The next step is a faster simulator
(to afford far more history, labels and incidents) and then a world model trained and judged on the incident
outputs directly.

## License

MIT
