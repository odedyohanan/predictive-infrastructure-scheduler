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

## Caveats

- One laptop, one simulator family, 60 s scenarios. The simulator is "reality" by construction; nothing here
  touches a real cluster.
- The official decision-time check uses 3 development scenarios at normal priority. Under background priority
  in the final run (several processes on efficiency cores) Step B exceeded 500 ms in 8 decisions.
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

## Layout

`pis/` the package (adapters to the sibling repos, harness, ensemble, evaluator, loop, reviser, runner,
statistics); `experiments/` one script per step; `configs/` chosen configurations and the frozen protocol;
`results/` the notes every number above comes from; `tests/` unit tests.

## Next

Phase A asked whether imagination helps at all. The oracle gap points at the world model as the next thing to
improve (training on the loop's own decisions, other architectures), before more elaborate search.

## License

MIT
