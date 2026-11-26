# Rebuild of antislop-pilot-pi:local for x86 Linux hosts.
FROM python:3.12-slim
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y curl git ripgrep xz-utils ca-certificates && rm -rf /var/lib/apt/lists/*
ENV NVM_DIR=/root/.nvm
RUN curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && source "$NVM_DIR/nvm.sh" && nvm install 22 && nvm alias default 22 && npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.4.0 && pi --version
WORKDIR /app
CMD ["bash"]
