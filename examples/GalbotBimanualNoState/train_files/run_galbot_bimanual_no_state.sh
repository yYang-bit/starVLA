#!/bin/bash
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000
# export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Please modify the following paths according to your environment ===
data_root=/mnt/project/human_action_data/liuyi/umi_lerobot_data
data_name=galbot_lerobot_dual_cup_0529_359piece   #lerobot v3数据集路径
data_mix=galbot_bimanual_self_no_state_mix
config_yaml=examples/GalbotBimanualNoState/train_files/starvla_qwengroot_galbot_no_state.yaml
run_root_dir=/mnt/home/liuyi/project/starVLA/Checkpoints/galbot_delta_action
run_id=galbot_bimanual_no_state_$(date +%m%d)

# === Self-mode config (change self_mode to switch experiment) ===
self_mode=delta           # abs | delta | chunk_relative

# === Stats path (default: <data_root>/<data_name>/meta/stats.json) ===
# stats_path=${data_root}/${data_name}/meta/stats_delta_chunk30.json

# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# Build optional stats_path argument
stats_path_arg=""
if [ -n "${stats_path:-}" ]; then
  stats_path_arg="--datasets.vla_data.stats_path ${stats_path}"
fi

# Auto-log to checkpoint dir
train_log=${output_dir}/train.log
echo "=== Training log: ${train_log} ==="

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_name ${data_name} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 12 \
  --datasets.vla_data.self_mode ${self_mode} \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.gripper_normalization binary \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  ${stats_path_arg} \
  2>&1 | tee ${train_log}


##### Multi-Server Multi-GPU training script #####
# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   starVLA/training/train_starvla.py \
#   --config_yaml ${config_yaml} \
#   --datasets.vla_data.data_root_dir ${data_root} \
#   --datasets.vla_data.data_mix ${data_mix} \
#   --datasets.vla_data.self_mode ${self_mode} \
#   --datasets.vla_data.include_state false \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project starVLA_galbot \
#   --wandb_entity your_wandb_entity
##### Multi-Server Multi-GPU training script #####
