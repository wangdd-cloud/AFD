#!/bin/bash
# Recommendation-list diversity for the 2x2 ablation (5 seeds x 2 datasets).
# Runs serially per GPU; nohup+setsid so the jobs survive an SSH disconnect.
cd /root/RSIR
PY=/root/anaconda3/bin/python

setsid nohup bash -c "
  for s in 0 1 2 3 4; do
    $PY exp/rec_div.py --dataset amazon-sport --seed \$s --device 0
  done
" > exp/recdiv_sport.log 2>&1 &

setsid nohup bash -c "
  for s in 0 1 2 3 4; do
    $PY exp/rec_div.py --dataset yelp --seed \$s --device 1
  done
" > exp/recdiv_yelp.log 2>&1 &

echo "rec_div launched"
