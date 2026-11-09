#!/bin/bash
# 5-seed for penalty / adaptive / down-weighting (seeds 1-4). 2 concurrent per GPU.
cd /root/RSIR
PY=/root/anaconda3/bin/python

J0=exp/neg_jobs_gpu0.txt
J1=exp/neg_jobs_gpu1.txt
: > "$J0"
: > "$J1"

i=0
addjob() {
  if [ "$1" -eq 0 ]; then printf '%s\n' "$2" >> "$J0"; else printf '%s\n' "$2" >> "$J1"; fi
}

# --- penalty sweep (sweep_pen.py), seeds 1-4 ---
for ds in amazon-sport yelp; do
  for s in 1 2 3 4; do
    gpu=$(( i % 2 ))
    addjob "$gpu" "$PY -u exp/sweep_pen.py --dataset $ds --seed $s --device $gpu > exp/pen_${ds}_s${s}.log 2>&1"
    i=$(( i + 1 ))
  done
done

# --- adaptive / down-weighting (afd_pipeline.py A/B/C/D), seeds 1-4 ---
for ds in amazon-sport yelp; do
  for tag in A B C D; do
    for s in 1 2 3 4; do
      gpu=$(( i % 2 ))
      extra=""
      [ "$tag" = B ] && extra="--adaptive"
      [ "$tag" = C ] && extra="--synthetic_weight 0.5"
      [ "$tag" = D ] && extra="--adaptive --synthetic_weight 0.5"
      addjob "$gpu" "$PY -u exp/afd_pipeline.py --dataset $ds --seed $s --device $gpu --outdir exp/abl_${ds}_${tag} --temperature 2.0 --pop_penalty 0 --methods RSIR,AFD_RSIR $extra > exp/abl5_${ds}_${tag}_s${s}.log 2>&1"
      i=$(( i + 1 ))
    done
  done
done

echo "GPU0 jobs: $(wc -l < "$J0")"
echo "GPU1 jobs: $(wc -l < "$J1")"

nohup xargs -P 2 -I '{}' bash -c '{}' < "$J0" > exp/neg_gpu0.log 2>&1 &
nohup xargs -P 2 -I '{}' bash -c '{}' < "$J1" > exp/neg_gpu1.log 2>&1 &
echo "launched $(date +%H:%M:%S)"
