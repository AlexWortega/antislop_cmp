#!/usr/bin/env bash
# One-time setup of the agent host (a Vast VM with Docker): Python env, agent images, GPU SSH key.
set -euo pipefail
R=/root/antislop_cmp; L=$R/qwen_scaffold_longrun; N=$R/qwen_verl_native
export DEBIAN_FRONTEND=noninteractive
command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync git curl python3; }
docker info >/dev/null
# Drop a dead reverse tunnel within 45 s, so the GPU can rebind 127.0.0.1:2222 when it reconnects.
# A drop-in, because lines appended after a Match block in sshd_config would apply only inside it.
mkdir -p /etc/ssh/sshd_config.d
printf 'ClientAliveInterval 15\nClientAliveCountMax 3\n' > /etc/ssh/sshd_config.d/10-antislop-keepalive.conf
grep -q '^Include /etc/ssh/sshd_config.d' /etc/ssh/sshd_config || sed -i '1i Include /etc/ssh/sshd_config.d/*.conf' /etc/ssh/sshd_config
sshd -t && { systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true; }
# Ubuntu's docker.io ships without the compose and buildx plugins, which Harbor needs. Sources in order:
# the copies the launcher uploads, Docker's apt pool, then GitHub releases (which has returned 504 for hours).
P=/usr/local/lib/docker/cli-plugins; mkdir -p $P
POOL=https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/amd64
plugin() {  # name, deb, github url
  docker $1 version >/dev/null 2>&1 && return 0
  if [ -x $N/cloud/bin/docker-$1 ]; then install -m 755 $N/cloud/bin/docker-$1 $P/docker-$1
  elif curl -fsSL --retry 5 --retry-all-errors $POOL/$2 -o /tmp/$2; then
    rm -rf /tmp/deb-$1 && dpkg-deb -x /tmp/$2 /tmp/deb-$1 && install -m 755 /tmp/deb-$1/usr/libexec/docker/cli-plugins/docker-$1 $P/docker-$1
  else curl -fsSL --retry 5 --retry-all-errors $3 -o $P/docker-$1 && chmod +x $P/docker-$1; fi
  docker $1 version
}
plugin compose docker-compose-plugin_2.29.7-1~ubuntu.22.04~jammy_amd64.deb https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64
plugin buildx docker-buildx-plugin_0.17.1-1~ubuntu.22.04~jammy_amd64.deb https://github.com/docker/buildx/releases/download/v0.17.1/buildx-v0.17.1.linux-amd64
# 32 parallel trials each create a compose network; the default pools run out at about 30.
if ! grep -q default-address-pools /etc/docker/daemon.json 2>/dev/null; then
  echo '{"default-address-pools":[{"base":"10.210.0.0/16","size":24}]}' > /etc/docker/daemon.json
  systemctl restart docker; sleep 3; docker info >/dev/null
fi
command -v python3 >/dev/null && { python3 -m venv --help >/dev/null 2>&1 || apt-get install -y -qq python3-venv; }
[ -x /root/.local/bin/uv ] || (curl -LsSf https://astral.sh/uv/0.11.16/install.sh | sh) || { python3 -m venv /opt/uvpip && /opt/uvpip/bin/pip install -q uv==0.11.16 && mkdir -p /root/.local/bin && ln -sf /opt/uvpip/bin/uv /root/.local/bin/uv; }
export PATH=/root/.local/bin:$PATH
cd $L
# The lock was frozen on macOS; skip local-path lines, take CPU torch for the embedding judge.
grep -vE '^-e |@ file:' local_environment.txt > /tmp/requirements.txt
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
docker build -q -t antislop-pilot-hermes:local -f $N/cloud/agent_images/hermes.Dockerfile $N/cloud/agent_images
docker build -q -t antislop-pilot-pi:local -f $N/cloud/agent_images/pi.Dockerfile $N/cloud/agent_images
mkdir -p /tmp/antislop-longrun-docker-config && echo '{}' > /tmp/antislop-longrun-docker-config/config.json
[ -f /root/.ssh/antislop_gpu ] || ssh-keygen -q -t ed25519 -N '' -f /root/.ssh/antislop_gpu
touch $N/cloud/SETUP_DONE
