# Rebuild of antislop-pilot-hermes:local for x86 Linux hosts, pinned to the release the Mac image used.
FROM python:3.12-slim
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y curl git ripgrep xz-utils ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/bf0203a794a07aac3ff764972ab9087baa4e4f4e/scripts/install.sh | bash -s -- --commit bf0203a794a07aac3ff764972ab9087baa4e4f4e --skip-setup
ENV PATH=/root/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
RUN hermes --version | grep -q 'v0.22.5'
RUN ln -s /root/.hermes /tmp/hermes
WORKDIR /app
CMD ["bash"]
