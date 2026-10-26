#!/bin/sh
set -eu
LAB_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2 NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES='' PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONDONTWRITEBYTECODE=1
if [ -n "${LAB_PYTHON:-}" ]; then
  LAB_INTERPRETER=$LAB_PYTHON
elif [ -f "$LAB_DIR/.local_python" ]; then
  LAB_INTERPRETER=$(cat "$LAB_DIR/.local_python")
else
  LAB_INTERPRETER="$LAB_DIR/.venv/bin/python"
fi
if [ ! -x "$LAB_INTERPRETER" ]; then LAB_INTERPRETER=python3; fi
cd "$LAB_DIR"
case "${1:-demo}" in
  data) exec "$LAB_INTERPRETER" -B generate_data.py ;;
  train) exec "$LAB_INTERPRETER" -B train.py ;;
  test) exec "$LAB_INTERPRETER" -B -m unittest -v test_mechanisms ;;
  evaluate) exec "$LAB_INTERPRETER" -B evaluate.py ;;
  demo) shift 2>/dev/null || true; exec "$LAB_INTERPRETER" -B demo.py "$@" ;;
  all)
    "$LAB_INTERPRETER" -B generate_data.py
    "$LAB_INTERPRETER" -B train.py
    "$LAB_INTERPRETER" -B -m unittest -v test_mechanisms
    "$LAB_INTERPRETER" -B evaluate.py
    exec "$LAB_INTERPRETER" -B demo.py ;;
  *) echo 'Usage: sh run.sh [data|train|test|evaluate|demo|all]'; exit 2 ;;
esac
