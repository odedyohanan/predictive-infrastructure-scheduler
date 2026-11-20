# Shadow mode on development seeds

Label: real. Calibration episodes 50, selection episodes 50 (R6, from the store); predictors fitted on 240 history episodes. Rule k=19, interval=76, majority=0.5, warm-up 20; H=20 windows, min lead 1 windows. World model: 8ee702e513538b2c1c2ad1aca409c9d31ef81866f7ebaf1a36f03f9a17869228 (n=32 futures).

Alarm: on at risk >= threshold, off below 0.7 x threshold; conformal risk control of the per-episode false-alarm share on the calibration episodes, alpha = 0.10. An alarm event is false when no new incident starts within H after it; events whose H windows run past the episode end are not judged.

The queueing model's scale and the conformal thresholds are both set on the CALIBRATION episodes. The scale is a monotone map of the queueing statistic, so the alarm-on threshold selects the same windows whatever the scale; only `low` (0.7 x threshold, in risk units) depends on it.

## Thresholds (calibration)

| predictor | threshold | low | adjusted risk | false-alarm rate | events |
|---|---|---|---|---|---|
| world_model | 0.5938 | 0.4156 | 0.078 | 0.400 | 12 |
| hazard | 0.0593 | 0.0415 | 0.039 | 1.000 | 1 |
| analogue | 0.3800 | 0.2660 | 0.078 | 1.000 | 6 |
| queueing | 0.8556 | 0.5989 | 0.098 | 1.000 | 14 |
| change_point | no feasible threshold | - | - | - | 0 |

## Selection episodes

| predictor | recall building-up | recall sudden | median lead bu / sudden | false-alarm rate | alarms per quiet episode all / judged | Brier all / bu / sudden (n states) |
|---|---|---|---|---|---|---|
| world_model | 0.20 (2/10) | 0.00 (0/4) | 3.5 / - | 0.667 | 0.00 / 0.00 | 0.030 / 0.046 / 0.016 (n 144/69/75) |
| hazard | 0.00 (0/10) | 0.00 (0/4) | - / - | 1.000 | 0.02 / 0.02 | 0.032 / 0.066 / 0.000 (n 144/69/75) |
| analogue | 0.20 (2/10) | 0.00 (0/4) | 16.5 / - | 0.625 | 0.00 / 0.00 | 0.021 / 0.044 / 0.001 (n 144/69/75) |
| queueing | 0.00 (0/10) | 0.00 (0/4) | - / - | 1.000 | 0.21 / 0.19 | 0.198 / 0.174 / 0.221 (n 144/69/75) |
| change_point | 0.00 (0/10) | 0.00 (0/4) | - / - | - | 0.00 / 0.00 | 0.076 / 0.133 / 0.023 (n 144/69/75) |

Brier: Monte Carlo labelled selection states with a finite risk of that predictor (n states per predictor: all / building-up / sudden). Quiet episode = no counted incident. Alarms per quiet episode: "all" includes the censored events near the episode's end (neither true nor false), "judged" counts only the true and false ones.

## Question 1: learned vs not learned

Better non-learned predictor (selection building-up recall): change_point. Difference of recall (lead >= min lead), 95 % bootstrap interval over selection episodes (2000 resamples):

| predictor | building-up | sudden |
|---|---|---|
| world_model | +0.20 [+0.00, +0.57] | +0.00 [+0.00, +0.00] |
| hazard | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| analogue | +0.20 [+0.00, +0.57] | +0.00 [+0.00, +0.00] |

## Timing

222.6 ms per window for all predictors together (world_model 156.0, hazard 6.1, analogue 14.7, queueing 5.4, change_point 40.5); 100 episodes x 600 windows projected 3.7 h on one process.

## Decision

Alarm predictor: analogue (threshold 0.3800, low 0.2660).

GATE: stop before prevention

Reason: the best building-up recall at lead >= min lead is 0.20 (<= 0.2): no predictor catches building-up incidents early enough at the calibrated false-alarm rate.
