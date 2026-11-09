#!/bin/bash
DS=$1
DEV=$2
cd /root/RSIR
run() {
  tag=$1; shift
  /root/anaconda3/bin/python -u exp/afd_pipeline.py --dataset $DS --seed 0 --device $DEV     --outdir exp/abl_${DS}_${tag} --temperature 2.0 --pop_penalty 0 --methods RSIR,AFD_RSIR "$@"     > exp/abl_${DS}_${tag}.log 2>&1
  echo "[done] $DS $tag rc=$?"
}
run A
run B --adaptive
run C --synthetic_weight 0.5
run D --adaptive --synthetic_weight 0.5
echo "[ALL DONE] $DS"
