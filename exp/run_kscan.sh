#!/bin/bash
# 5-seed fidelity-gate k-scan (k in 20/50/100/200/off), 2 concurrent per GPU.
cd /root/RSIR
PY=/root/anaconda3/bin/python

J0=exp/kscan_gpu0.txt
J1=exp/kscan_gpu1.txt
: > "$J0"; : > "$J1"

i=0
for ds in amazon-sport yelp; do
  for s in 0 1 2 3 4; do
    gpu=$(( i % 2 ))
    cmd="$PY -u exp/sweep_k.py --dataset $ds --seed $s --device $gpu > exp/kscan_${ds}_s${s}.log 2>&1"
    if [ "$gpu" -eq 0 ]; then printf '%s\n' "$cmd" >> "$J0"; else printf '%s\n' "$cmd" >> "$J1"; fi
    i=$(( i + 1 ))
  done
done

echo "kscan GPU0 jobs: $(wc -l < "$J0")"
echo "kscan GPU1 jobs: $(wc -l < "$J1")"

nohup xargs -P 2 -I '{}' bash -c '{}' < "$J0" > exp/kscan_gpu0.log 2>&1 &
nohup xargs -P 2 -I '{}' bash -c '{}' < "$J1" > exp/kscan_gpu1.log 2>&1 &
echo "kscan launched $(date +%H:%M:%S)"
