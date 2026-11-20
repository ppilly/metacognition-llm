#!/usr/bin/env bash
# One-time setup of a rented Linux GPU machine (e.g. RunPod / Lambda / EC2) for the MUSE experiments.
#
#   HF_TOKEN=hf_xxx bash scripts/setup_cloud.sh
#
# Installs Miniforge + a Python 3.11 env (vLLM >= 0.32 needs >= 3.11) under $WORKDIR, builds ALFWorld,
# downloads the ALFWorld games and the models, and runs the unit tests. Safe to re-run.
# Everything lives under $WORKDIR (default /workspace, RunPod's persistent volume).
set -euo pipefail

WORKDIR=${WORKDIR:-/workspace}
CONDA_DIR=$WORKDIR/miniforge3
ENV_DIR=$CONDA_DIR/envs/muse
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${MODELS:-"mistralai/Mistral-7B-Instruct-v0.2 Open-Orca/Mistral-7B-OpenOrca apple/OpenELM-3B-Instruct meta-llama/Llama-2-7b-hf sentence-transformers/all-MiniLM-L6-v2"}

: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face read token (with the Mistral and Llama-2 licences accepted)}"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== Miniforge"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$CONDA_DIR/bin/conda" create -y -q -p "$ENV_DIR" -c conda-forge python=3.11 make cmake cxx-compiler
fi
export PATH=$ENV_DIR/bin:$PATH        # compilers must be on PATH to build jericho / fast-downward
PIP="$ENV_DIR/bin/python -m pip"

echo "== Python packages"
$PIP install -q --upgrade pip
$PIP install -q "vllm==0.32.0"        # brings a matching torch / transformers
$PIP install -q peft bitsandbytes sentence-transformers scikit-learn pyyaml matplotlib pytest
$PIP install -q "alfworld==0.4.2"
"$ENV_DIR/bin/python" -c "import vllm, peft, textworld, alfworld; print('vllm', vllm.__version__)"

echo "== ALFWorld games"
export ALFWORLD_DATA=$WORKDIR/alfworld_data
if [ ! -d "$ALFWORLD_DATA/json_2.1.1/valid_unseen" ]; then
  "$ENV_DIR/bin/alfworld-download"
fi

echo "== Models"
export HF_HOME=$WORKDIR/hf_cache
HF_TOKEN=$HF_TOKEN MODELS=$MODELS "$ENV_DIR/bin/python" - <<'EOF'
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
for m in os.environ["MODELS"].split():
    # skip duplicate checkpoint formats (e.g. Mistral's 14 GB consolidated.safetensors)
    snapshot_download(m, ignore_patterns=["consolidated*", "original/*", "*.pth", "*.onnx", "*.msgpack", "*.h5"])
    print("ok", m)
EOF

echo "== Tests"
cd "$REPO" && "$ENV_DIR/bin/python" -m pytest -q tests

cat > "$WORKDIR/muse_env.sh" <<EOF
export PATH=$ENV_DIR/bin:\$PATH
export ALFWORLD_DATA=$ALFWORLD_DATA
export HF_HOME=$HF_HOME
export HF_HUB_OFFLINE=1          # models are already downloaded; runs never need the Hub
export TOKENIZERS_PARALLELISM=false
cd $REPO
EOF
echo "Setup complete. In new shells run: source $WORKDIR/muse_env.sh"
