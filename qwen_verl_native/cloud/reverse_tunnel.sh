#!/usr/bin/env bash
# Runs on the GPU. Vast's ssh proxy admits only account keys, so the GPU dials the agent host and
# exposes its own sshd there; the agent host then reaches the GPU at 127.0.0.1:2222.
HOST=$1; PORT=$2
while [ ! -f /workspace/native/STOP ]; do
  ssh -N -i /root/.ssh/to_agents -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -p "$PORT" -R 127.0.0.1:2222:127.0.0.1:2223 "root@$HOST"
  sleep 5
done
