#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential swig
python -m pip install --upgrade pip
pip install -e .           # usa o pyproject; NÃO use requirements.txt (TA-Lib quebra)
mkdir -p data/raw results trained_models
echo "Ambiente FinRL pronto."