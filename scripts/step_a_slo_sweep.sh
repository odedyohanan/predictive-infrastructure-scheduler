#!/bin/zsh
# Step A round 2 on development seeds: sweep the SLO levers of the evaluator (queue penalty mu, SLO weight
# slo_lam) at the structural settings with the lowest SLO cost in round 1. One note and config per pass.
#
#   scripts/gentle.sh step_a_sweep scripts/step_a_slo_sweep.sh
set -e
out=results/step_a/slo_sweep
mkdir -p $out
for pair in "8 10" "30 10" "2 40" "8 40" "30 40"; do
  set -- ${=pair}
  mu=$1; sl=$2
  echo "=== pass mu $mu slo_lam $sl"
  .venv/bin/python experiments/tune_step_a.py --procs 4 --no-ablations \
      --grid-n 5 --grid-m 4 --grid-h 3,5 --grid-lam 1 --grid-tau-q 90 --mu $mu --slo-lam $sl \
      --out $out/mu${mu}_slo${sl}.md --config-out $out/mu${mu}_slo${sl}.json \
      --raw results/raw/step_a/sweep_mu${mu}_slo${sl}.json
done
echo "=== sweep done"
