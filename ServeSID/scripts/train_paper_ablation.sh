#!/usr/bin/env bash
# Run exactly one paper variant with the same canonical Full trainer, NOT old checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARM="${1:?Provide absres, posuni, shuffled, or collapsed}"
case "$ARM" in
  absres|posuni|shuffled|collapsed) ;;
  *) echo "Unknown paper ablation: $ARM" >&2; exit 2 ;;
esac
CATEGORY="${CATEGORY:-Beauty}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
CONFIG="$ROOT/configs/abl_hcda_${ARM}.yaml" \
RUN_TAG_PREFIX="paper_${ARM}" \
bash "$ROOT/scripts/train.sh.servesid" sidmlp "$CATEGORY" "$CUDA_DEVICE"
