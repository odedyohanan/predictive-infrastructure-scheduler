# Shadow mode on development seeds

Label: real. Calibration episodes 50, selection episodes 50 (R6, from the store); predictors fitted on 240 history episodes. Rule k=19, interval=76, majority=0.5, warm-up 20; H=20 windows, min lead 1 windows. World model: b3785ac7511d671f31de5bb39a9315a59aa98ff72f1744aa704f400a59418241 (n=32 futures).

Alarm: on at risk >= threshold, off below 0.7 x threshold; conformal risk control of the per-episode false-alarm share on the calibration episodes, alpha = 0.10. An alarm event is false when no new incident starts within H after it; events whose H windows run past the episode end are not judged.

The queueing model's scale and the conformal thresholds are both set on the CALIBRATION episodes. The scale is a monotone map of the queueing statistic, so the alarm-on threshold selects the same windows whatever the scale; only `low` (0.7 x threshold, in risk units) depends on it.

## Thresholds (calibration)

| predictor | threshold | low | adjusted risk | false-alarm rate | events |
|---|---|---|---|---|---|
| world_model | 0.6562 | 0.4594 | 0.054 | 0.438 | 18 |
| hazard | 0.0593 | 0.0415 | 0.039 | 1.000 | 1 |
| analogue | 0.3800 | 0.2660 | 0.078 | 1.000 | 6 |
| queueing | 0.8556 | 0.5989 | 0.098 | 1.000 | 14 |
| change_point | no feasible threshold | - | - | - | 0 |

## Selection episodes

| predictor | recall building-up | recall sudden | median lead bu / sudden | false-alarm rate | alarms per quiet episode all / judged | Brier all / bu / sudden (n states) |
|---|---|---|---|---|---|---|
| world_model | 0.20 (2/10) | 0.00 (0/4) | 12.5 / - | 0.818 | 0.05 / 0.00 | 0.078 / 0.095 / 0.062 (n 144/69/75) |
| hazard | 0.00 (0/10) | 0.00 (0/4) | - / - | 1.000 | 0.02 / 0.02 | 0.032 / 0.066 / 0.000 (n 144/69/75) |
| analogue | 0.20 (2/10) | 0.00 (0/4) | 16.5 / - | 0.625 | 0.00 / 0.00 | 0.021 / 0.044 / 0.001 (n 144/69/75) |
| queueing | 0.00 (0/10) | 0.00 (0/4) | - / - | 1.000 | 0.21 / 0.19 | 0.198 / 0.174 / 0.221 (n 144/69/75) |
| change_point | 0.00 (0/10) | 0.00 (0/4) | - / - | - | 0.00 / 0.00 | 0.076 / 0.133 / 0.023 (n 144/69/75) |

Brier: Monte Carlo labelled selection states with a finite risk of that predictor (n states per predictor: all / building-up / sudden). Quiet episode = no counted incident. Alarms per quiet episode: "all" includes the censored events near the episode's end (neither true nor false), "judged" counts only the true and false ones.

## Question 1: learned vs not learned

Better non-learned predictor (selection building-up recall): change_point. Difference of recall (lead >= min lead), 95 % bootstrap interval over selection episodes (2000 resamples):

| predictor | building-up | sudden |
|---|---|---|
| world_model | +0.20 [+0.00, +0.33] | +0.00 [+0.00, +0.00] |
| hazard | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| analogue | +0.20 [+0.00, +0.57] | +0.00 [+0.00, +0.00] |

## Timing

185.4 ms per window for all predictors together (world_model 122.0, hazard 5.8, analogue 14.2, queueing 5.2, change_point 38.3); 100 episodes x 600 windows projected 3.1 h on one process.

## Decision

Alarm predictor: analogue (threshold 0.3800, low 0.2660).

GATE: stop before prevention

Reason: the best building-up recall at lead >= min lead is 0.20 (<= 0.2): no predictor catches building-up incidents early enough at the calibrated false-alarm rate.
