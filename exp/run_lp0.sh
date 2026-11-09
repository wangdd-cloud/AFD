#!/bin/bash
cd /root/RSIR
PY=/root/anaconda3/bin/python
(
  for s in 0 1 2 3 4; do
    $PY exp/local_prob0_control.py --dataset amazon-sport --seed $s --device 0
  done
) > exp/lp0_sport.log 2>&1 &
(
  for s in 0 1 2 3 4; do
    $PY exp/local_prob0_control.py --dataset yelp --seed $s --device 1
  done
) > exp/lp0_yelp.log 2>&1 &
wait
