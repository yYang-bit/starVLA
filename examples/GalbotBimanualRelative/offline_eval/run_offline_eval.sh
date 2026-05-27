#!/bin/bash
# Offline eval for QwenGR00T galbot bimanual self-mode.
#
# Minimal interface — just edit the 2 abs paths below, then run.
# Output: <checkpoint without .pt>/ep{NNNN}_frame{NNNN}/
#
# Usage:
#   bash run_offline_eval.sh [EPISODE_IDX] [FRAME_IDX]

set -e

###########################################################################################
# === Edit these two absolute paths ===
checkpoint=/mnt/home/liuyi/project/starVLA/Checkpoints/galbot_delta_action/galbot_bimanual_self_0526/checkpoints/steps_80000_pytorch_model.pt
dataset_path=/mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526
# Must match the self_mode used during training (run_galbot_bimanual_self.sh: self_mode=delta)
self_mode=delta
###########################################################################################

episode_idx=${1:-0}
frame_idx=${2:-0}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd ${repo_root}

python examples/GalbotBimanualRelative/offline_eval/offline_eval_galbot_bimanual_self.py \
  --checkpoint   ${checkpoint} \
  --dataset-path ${dataset_path} \
  --episode-idx  ${episode_idx} \
  --frame-idx    ${frame_idx} \
  --self-mode    ${self_mode} \
  --device cuda
