#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs checkpoints results

if [[ ! -d upstream/.git ]]; then
  git clone https://github.com/microsoft/ClimaX.git upstream
fi
git -C upstream checkout 6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c

module load mamba/1.4.9
if [[ ! -x climax-env/bin/python ]]; then
  rm -rf climax-env
  mamba env create -p "$PWD/climax-env" -f upstream/docker/environment.yml \
    2>&1 | tee logs/environment-install.log
fi

if ! echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  checkpoints/5.625deg.ckpt' | sha256sum -c --status 2>/dev/null; then
  curl -fL --retry 3 \
    'https://huggingface.co/tungnd/climax/resolve/main/5.625deg.ckpt?download=true' \
    -o checkpoints/5.625deg.ckpt
fi

echo '6393876948b8244e65f8b341474a5a14e4242dd4ff4c1db811700935be347c43  checkpoints/5.625deg.ckpt' | sha256sum -c
