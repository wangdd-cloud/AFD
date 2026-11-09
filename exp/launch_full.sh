#!/bin/bash
cd /root/RSIR
run_series() {
  local dev="$1"; shift
  local ds="$1"; shift
  for s in "$@"; do
    echo "[$(date +%T)] START $ds seed $s dev $dev"
    /root/anaconda3/bin/python -u exp/afd_pipeline.py --dataset "$ds" --seed "$s" --device "$dev" --outdir exp/afd_results >> "exp/log_${ds}_s${s}.log" 2>&1
    echo "[$(date +%T)] END   $ds seed $s dev $dev (exit $?)"
  done
}
run_series 0 amazon-sport 0 1 2 3 4 > exp/series0.out 2>&1 &
run_series 1 yelp 0 1 2 3 4 > exp/series1.out 2>&1 &
wait
echo "ALL DONE $(date)"
