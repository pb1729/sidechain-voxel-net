#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ubuntu/cath_s40}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/env}"
DATA_ROOT="${DATA_ROOT:-/lambda/nfs/research/datasets}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/cath-cif}"
DATA_URL="${DATA_URL:-https://zenodo.org/records/18506092/files/cath-cif.tar.gz?download=1}"
RUN_ROOT="${RUN_ROOT:-/lambda/nfs/research/runs/cath_s40_flow}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

cd "$PROJECT_DIR"

python3 -m venv --system-site-packages "$VENV_DIR"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -r requirements-remote.txt

if ! REQUIRE_CUDA="$REQUIRE_CUDA" python - <<'PY'
import os
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
elif os.environ.get("REQUIRE_CUDA") != "0":
    raise SystemExit("CUDA is not available")
PY
then
  python -m pip install torch
  REQUIRE_CUDA="$REQUIRE_CUDA" python - <<'PY'
import os
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
elif os.environ.get("REQUIRE_CUDA") != "0":
    raise SystemExit("CUDA is not available")
PY
fi

mkdir -p "$DATA_ROOT"
sudo mkdir -p "$RUN_ROOT"
sudo chown -R "$(id -u):$(id -g)" "$RUN_ROOT"

if [ ! -d "$DATA_DIR" ] || [ -z "$(find "$DATA_DIR" -maxdepth 1 -type f -name '*.cif' -print -quit)" ]; then
  tmp_dir="${DATA_ROOT}/cath-cif-download.$$"
  extract_dir="${tmp_dir}/extract"
  archive="${tmp_dir}/cath-cif.tar.gz"
  rm -rf "$tmp_dir"
  mkdir -p "$extract_dir"

  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 5 -o "$archive" "$DATA_URL"
  else
    wget -O "$archive" "$DATA_URL"
  fi

  tar -xzf "$archive" -C "$extract_dir"
  if [ ! -d "${extract_dir}/cath-cif" ]; then
    echo "expected extracted cath-cif directory under ${extract_dir}" >&2
    exit 1
  fi

  rm -rf "${DATA_DIR}.partial"
  mv "${extract_dir}/cath-cif" "${DATA_DIR}.partial"
  rm -rf "$DATA_DIR"
  mv "${DATA_DIR}.partial" "$DATA_DIR"
  rm -rf "$tmp_dir"
fi

python - <<PY
from pathlib import Path
data_dir = Path(${DATA_DIR@Q})
count = sum(1 for _ in data_dir.glob("*.cif"))
print(f"dataset: {data_dir} ({count} cif files)")
if count == 0:
    raise SystemExit("dataset is empty")
PY
