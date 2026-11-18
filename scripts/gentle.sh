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
# - own process group (setsid): the job and all its worker processes
#   (multiprocessing pool workers included) share one group that stop.sh kills.
#
# Example:
#   scripts/gentle.sh step0 .venv/bin/python experiments/step0_run.py \
#       --model worldmodel:models/gru_mix_bounded.pt --attempt 1 --procs 5
#
# Follow:  tail -f logs/<log-name>.log
# Stop:    scripts/stop.sh      (kills every job started here, with its worker processes)
set -e
name=$1; shift
mkdir -p logs
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# perl's setsid: new session and process group whose id is the job's pid (exec keeps the pid)
nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV or die "exec: $!"' taskpolicy -b "$@" \
    > "logs/$name.log" 2>&1 &
pid=$!
nohup caffeinate -i -w $pid > /dev/null 2>&1 &
echo $pid > "logs/$name.pid"
echo "started pid $pid (process group $pid) -> logs/$name.log"
echo "follow: tail -f logs/$name.log    stop: scripts/stop.sh"
