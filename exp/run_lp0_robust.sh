#!/bin/bash
# local_prob=0 control (5 seeds x 2 datasets), detached so it survives SSH drops.
cd /root/RSIR
PY=/root/anaconda3/bin/python

setsid nohup bash -c "
  for s in 0 1 2 3 4; do
    $PY exp/local_prob0_control.py --dataset amazon-sport --seed \$s --device 0
  done
" > exp/lp0_sport.log 2>&1 &

setsid nohup bash -c "
  for s in 0 1 2 3 4; do
    $PY exp/local_prob0_control.py --dataset yelp --seed \$s --device 1
  done
" > exp/lp0_yelp.log 2>&1 &

echo "lp0 launched (detached)"
