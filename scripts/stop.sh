#!/bin/zsh
# Stop every long job of this project (Step 0 run, counterfactual data, training), including pool workers.
pkill -f "experiments/(step0_run|make_counterfactual_data|train_world_model).py" && echo "stopped"
sleep 1
pgrep -fl "experiments/(step0_run|make_counterfactual_data|train_world_model).py" || echo "nothing running"
