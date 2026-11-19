#!/bin/zsh
# Stop every long job started with scripts/gentle.sh, including multiprocessing pool workers:
# each job runs in its own process group (pgid = the pid in logs/<name>.pid), and the whole group is killed.
setopt null_glob
stopped=0
for f in logs/*.pid; do
  pid=$(cat "$f")
  if kill -0 -- -$pid 2>/dev/null; then
    kill -TERM -- -$pid 2>/dev/null && echo "stopped $(basename "$f" .pid) (process group $pid)" && stopped=1
    sleep 1
    kill -KILL -- -$pid 2>/dev/null || true
  fi
  rm -f "$f"
done
# Fallback for jobs started without gentle.sh: the main scripts and their direct children.
for pid in $(pgrep -f "experiments/(step0_run|make_counterfactual_data|train_world_model).py"); do
  pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null && echo "stopped pid $pid" && stopped=1
done
[ $stopped = 1 ] || echo "nothing running"
