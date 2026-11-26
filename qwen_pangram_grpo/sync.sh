#!/bin/sh
# Copy adapters, logs, results and ledger from the box to ./pulled (no base model, no optimizer state).
cd "$(dirname "$0")"; mkdir -p pulled
rsync -az -e "ssh -i $HOME/.ssh/qwen_verl_pilot_20261101 -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p 16732" \
  --exclude='optimizer.pt' --exclude='__pycache__' --include='runs/***' --exclude='init/***' --include='logs/***' --include='results/***' \
  --include='code/***' --exclude='*' root@ssh8.vast.ai:/workspace/ pulled/ 2>/dev/null
