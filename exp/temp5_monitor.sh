#!/bin/bash
cd /root/RSIR
for i in $(seq 1 240); do
  a=$(grep -c "ALL DONE.*amazon-sport" exp/temp5_master.log 2>/dev/null || true)
  y=$(grep -c "ALL DONE.*yelp" exp/temp5_master.log 2>/dev/null || true)
  if [ "$a" -ge 1 ] && [ "$y" -ge 1 ]; then
    break
  fi
  sleep 30
done
echo "=== amazon-sport results ==="
cat exp/temp5_amazon-sport.jsonl 2>/dev/null
echo "=== yelp results ==="
cat exp/temp5_yelp.jsonl 2>/dev/null
echo "=== master ==="
cat exp/temp5_master.log 2>/dev/null
