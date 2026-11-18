#!/bin/zsh
# Run a long job without cooking the laptop.
#
#   scripts/gentle.sh <log-name> <command...>
#
# - taskpolicy -b : macOS "background" QoS. On Apple Silicon this keeps the
#                   process (and its child processes) on the efficiency cores:
#                   low power, quiet fan. Your apps keep the performance cores.
# - caffeinate    : no idle sleep while the job runs (closing the lid still sleeps).
# - single-threaded numpy / BLAS, so each process really uses one core.
# - nohup + log   : survives closing the terminal.
#
# Example:
#   scripts/gentle.sh step0 .venv/bin/python experiments/step0_run.py \
#       --model worldmodel:models/gru_mix_bounded.pt --attempt 1 --procs 5
#
# Follow:  tail -f logs/<log-name>.log
# Stop:    scripts/stop.sh      (kills the job and its worker processes)
set -e
name=$1; shift
mkdir -p logs
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup taskpolicy -b "$@" > "logs/$name.log" 2>&1 &
pid=$!
nohup caffeinate -i -w $pid > /dev/null 2>&1 &
echo $pid > "logs/$name.pid"
echo "started pid $pid -> logs/$name.log"
echo "follow: tail -f logs/$name.log    stop: scripts/stop.sh"
