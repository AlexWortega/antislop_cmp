#!/usr/bin/env bash
# Runs on the playground GPU: install Hermes (the release used in training) and point it at the local step50 model.
# Secrets arrive in /workspace/.hermes_env (OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git ripgrep xz-utils > /dev/null)
if ! command -v hermes >/dev/null && [ ! -x /root/.local/bin/hermes ]; then
  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/bf0203a794a07aac3ff764972ab9087baa4e4f4e/scripts/install.sh \
    | bash -s -- --commit bf0203a794a07aac3ff764972ab9087baa4e4f4e --skip-setup
fi
export PATH=/root/.local/bin:$PATH
hermes --version
mkdir -p /root/.hermes
cat > /root/.hermes/config.yaml <<'EOF'
model: step50
provider: custom:runtime
custom_providers:
- name: runtime
  model: step50
  base_url: http://127.0.0.1:8000/v1
  key_env: OPENAI_API_KEY
  api_mode: chat_completions
toolsets:
- hermes-cli
terminal:
  backend: local
  timeout: 180
EOF
install -m 600 /workspace/.hermes_env /root/.hermes/.env
rm -f /workspace/.hermes_env
echo HERMES_READY
