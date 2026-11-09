#!/bin/bash
DS=$1
DEV=$2
cd /root/RSIR
for s in 0 1 2 3 4; do
  /root/anaconda3/bin/python -u exp/run_temp5.py --dataset $DS --seed $s --device $DEV \
    > exp/temp5_${DS}_seed${s}.log 2>&1
  echo "[done] $DS seed$s rc=$?" | tee -a exp/temp5_master.log
done
echo "[ALL DONE] $DS" | tee -a exp/temp5_master.log
