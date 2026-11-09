#!/bin/bash
# 5-seed for the 2x2 fidelity ablation + temperature Pareto (seeds 1-4).
# 4 chains: 2 per GPU (2 concurrent per GPU), 2 seeds per chain.
cd /root/RSIR
PY=/root/anaconda3/bin/python

launch() {
  local log=$1; shift
  nohup bash -c "$*" > "$log" 2>&1 &
}

launch exp/run5_sport_12.log "$PY exp/ablation2x2.py --dataset amazon-sport --seed 1 --device 0 && $PY exp/ablation2x2.py --dataset amazon-sport --seed 2 --device 0"
launch exp/run5_sport_34.log "$PY exp/ablation2x2.py --dataset amazon-sport --seed 3 --device 0 && $PY exp/ablation2x2.py --dataset amazon-sport --seed 4 --device 0"
launch exp/run5_yelp_12.log "$PY exp/ablation2x2.py --dataset yelp --seed 1 --device 1 && $PY exp/ablation2x2.py --dataset yelp --seed 2 --device 1"
launch exp/run5_yelp_34.log "$PY exp/ablation2x2.py --dataset yelp --seed 3 --device 1 && $PY exp/ablation2x2.py --dataset yelp --seed 4 --device 1"
echo "launched 4 chains"
