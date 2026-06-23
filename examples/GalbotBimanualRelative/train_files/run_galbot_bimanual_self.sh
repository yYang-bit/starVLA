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
data_mix=galbot_bimanual_self_mix
config_yaml=examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative_self.yaml
run_root_dir=/mnt/home/liuyi/project/starVLA/Checkpoints/galbot_delta_action
run_id=galbot_bimanual_self_$(date +%m%d)

# === Action mode config (统一: abs | delta | relative_pose) ===
action_mode=delta              # abs | delta | relative_pose
action_chunk_size=30           # action horizon, 需与 stats 计算时一致

# === Gripper config ===
# binary_independent: 本任务专用, gripper 不走归一化, 末尾用阈值二值化 (x > threshold -> {0,1})
# binary: 走框架 StateActionTransform binary 归一化 (需 stats 校验)
# min_max: min_max 归一化
# none: 不处理, 保持真值
gripper_normalization=binary_independent
gripper_binary_threshold=100.0   # mm, 二值化阈值

# === Stats path (与 action_mode / action_chunk_size 对应) ===
stats_file="stats_${action_mode}_chunk${action_chunk_size}.json"
stats_path=${data_root}/${data_name}/meta/${stats_file}

# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# Check stats file
if [ ! -f "${stats_path}" ]; then
  echo "⚠️  Stats file not found: ${stats_path}"
  echo "   Please run first:"
  echo "   python scripts/compute_dataset_stats.py \\"
  echo "       --dataset_path ${data_root}/${data_name} \\"
  echo "       --robot_type galbot_bimanual_self \\"
  echo "       --data_registry examples.GalbotBimanualRelative.train_files.data_registry.data_config \\"
  echo "       --output ${stats_path} \\"
  echo "       --action_mode ${action_mode} \\"
  echo "       --action_chunk_size ${action_chunk_size} \\"
  echo "       --gripper_normalization none"
  exit 1
fi

# Auto-log to checkpoint dir
train_log=${output_dir}/train.log
echo "=== Training log: ${train_log} ==="

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_name ${data_name} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 10 \
  --datasets.vla_data.action_mode ${action_mode} \
  --datasets.vla_data.action_chunk_size ${action_chunk_size} \
  --datasets.vla_data.gripper_normalization ${gripper_normalization} \
  --datasets.vla_data.gripper_binary_threshold ${gripper_binary_threshold} \
  --datasets.vla_data.stats_path ${stats_path} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
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
#   --datasets.vla_data.action_mode ${action_mode} \
#   --datasets.vla_data.action_chunk_size ${action_chunk_size} \
#   --datasets.vla_data.gripper_normalization ${gripper_normalization} \
#   --datasets.vla_data.stats_path ${stats_path} \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project starVLA_galbot \
#   --wandb_entity your_wandb_entity
##### Multi-Server Multi-GPU training script #####