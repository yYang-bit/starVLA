# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000
# export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Please modify the following paths according to your environment ===
data_root=/path/to/your/lerobot/root
data_mix=galbot_bimanual_mix
config_yaml=examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative.yaml
run_root_dir=./playground/Checkpoints
run_id=galbot_bimanual_qwengroot_$(date +%m%d)
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.gripper_normalization binary \
  --trainer.max_train_steps 40000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_galbot \
  --wandb_entity your_wandb_entity
  # --is_debug True


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
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project starVLA_galbot \
#   --wandb_entity your_wandb_entity
##### Multi-Server Multi-GPU training script #####
