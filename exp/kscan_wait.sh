#!/bin/bash
# Wait for the 5-seed negative run (penalty + adaptive/down-weighting) to finish,
# then launch the k-scan. Signal = no xargs / afd_pipeline / sweep_pen processes.
cd /root/RSIR
while pgrep -x xargs >/dev/null 2>&1 || pgrep -f 'exp/afd_pipeline.py' >/dev/null 2>&1 || pgrep -f 'exp/sweep_pen.py' >/dev/null 2>&1; do
  sleep 120
done
sleep 30
bash /root/RSIR/exp/run_kscan.sh
