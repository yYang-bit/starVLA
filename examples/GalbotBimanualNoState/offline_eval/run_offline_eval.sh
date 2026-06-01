#!/bin/bash
# Offline eval for QwenGR00T galbot bimanual NO-STATE policy.
#
# Usage:
#   bash run_offline_eval.sh [EPISODE_IDX] [FRAME_IDX]

set -e

###########################################################################################
# === Edit these two absolute paths ===
checkpoint=/mnt/home/liuyi/project/starVLA/Checkpoints/galbot_delta_action/galbot_bimanual_no_state_0528/checkpoints/steps_80000_pytorch_model.pt
dataset_path=/mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526
# Position unit scale: 1.0 if pos is already in cm, 100.0 if in meters
unit_scale=1.0
###########################################################################################

episode_idx=${1:-0}
frame_idx=${2:-0}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd ${repo_root}

python examples/GalbotBimanualNoState/offline_eval/offline_eval_galbot_no_state.py \
  --checkpoint   ${checkpoint} \
  --dataset-path ${dataset_path} \
  --episode-idx  ${episode_idx} \
  --frame-idx    ${frame_idx} \
  --unit-scale   ${unit_scale} \
  --device cuda
