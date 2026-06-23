#!/bin/bash
# FastUMI 训练脚本
# 支持 abs / delta / relative_pose 三种 action 模式

###########################################################################################
# === Please modify the following paths according to your environment ===
data_root=/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm
data_name=Add_Rice_to_Rice_Cooker   # FastUMI 数据集名称
data_mix=fastumi_dual_arm_mix
config_yaml=examples/FastUMI/train_files/starvla_qwengroot_fastumi.yaml
run_root_dir=/mnt/home/liuyi/project/starVLA/Checkpoints/fastumi_relative_pose
run_id=fastumi_dual_arm_$(date +%m%d)

# === Action mode config ===
action_mode=relative_pose     # abs | delta | relative_pose
action_chunk_size=16          # Action horizon (FastUMI 默认 16)
gripper_normalization=none    # none | binary | min_max

# === Stats path (auto-generated based on mode) ===
# Format: meta/stats_{action_mode}_chunk{action_chunk_size}.json
stats_file="stats_${action_mode}_chunk${action_chunk_size}.json"
stats_path="${data_root}/${data_name}/meta/${stats_file}"

if [ -f "${stats_path}" ]; then
    echo "✓ Using stats file: ${stats_path}"
    stats_path_arg="--datasets.vla_data.stats_path=${stats_path}"
else
    echo "⚠️  Stats file not found: ${stats_path}"
    echo "   Please run compute_dataset_stats.py first:"
    echo "   python scripts/compute_dataset_stats.py \\"
    echo "       --dataset_path ${data_root}/${data_name} \\"
    echo "       --robot_type fastumi_dual_arm \\"
    echo "       --data_registry examples.FastUMI.train_files.data_registry.data_config \\"
    echo "       --output ${stats_path} \\"
    echo "       --action_mode ${action_mode} \\"
    echo "       --action_chunk_size ${action_chunk_size} \\"
    echo "       --sample_ratio 0.1"
    exit 1
fi

# === Training log ===
train_log="${run_root_dir}/${run_id}/train.log"
mkdir -p "${run_root_dir}/${run_id}"

###########################################################################################
# === Single-GPU training script ===
python starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.action_mode ${action_mode} \
  --datasets.vla_data.action_chunk_size ${action_chunk_size} \
  --datasets.vla_data.gripper_normalization ${gripper_normalization} \
  --trainer.max_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  ${stats_path_arg} \
  2>&1 | tee ${train_log}


##### Multi-GPU training script (if needed) #####
# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --num_processes=4 \
#   starVLA/training/train_starvla.py \
#   --config_yaml ${config_yaml} \
#   --datasets.vla_data.data_root_dir ${data_root} \
#   --datasets.vla_data.data_mix ${data_mix} \
#   --datasets.vla_data.action_mode ${action_mode} \
#   --datasets.vla_data.action_chunk_size ${action_chunk_size} \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project starVLA_fastumi \
#   --wandb_entity your_wandb_entity
##### Multi-GPU training script #####
