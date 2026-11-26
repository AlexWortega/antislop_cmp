#!/bin/sh
# Keep localhost:30001 -> builder:30001 open.
while true; do ssh -i ~/.ssh/qwen_verl_pilot_20261101 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -N -L 30001:127.0.0.1:30001 -p 36738 root@ssh5.vast.ai; sleep 5; done
