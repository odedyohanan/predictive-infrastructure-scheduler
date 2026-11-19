# Phase B history

Generated with `experiments/b_generate_history.py --seeds 3100-3499 --procs 4` through scripts/gentle.sh (background QoS),
40 shards, 0 failed, 11802 s wall time; 400 episodes of 300 s (240000 windows).

Seeds 3100-3399 (history) use varied policy sources; 3400-3499 (calibration, selection) are all R6.

```
shards done 40, skipped 0, failed 0  in 11802 s
400 episodes, 240000 windows
  adaptive            47 episodes, wall s/episode mean 131.2 max 176.6
  loop_a              29 episodes, wall s/episode mean 149.0 max 198.5
  r3                  31 episodes, wall s/episode mean 97.5 max 140.4
  r6                 194 episodes, wall s/episode mean 101.1 max 143.6
  r6_interventions    55 episodes, wall s/episode mean 108.2 max 164.6
  random              44 episodes, wall s/episode mean 160.9 max 312.1
  kind ambient         33
  kind cpu_loss        32
  kind gpu_loss        33
  kind interference    32
  kind leak            34
  kind load_ramp       34
  kind none           202

```
