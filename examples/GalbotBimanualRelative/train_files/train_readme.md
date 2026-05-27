# GalbotBimanualRelative 训练指南

## 整体流程

```
原始 h5 数据
  ↓ (convert_galbot_bimanual_relative.py)
LeRobot v3 数据集（只存 abs 原始值）
  ↓ (compute_galbot_stats_self_mode.py，按 self_mode 分别计算)
meta/stats.json（离线预计算，与 self_mode 对齐）
  ↓ (训练脚本 + self_mode 配置)
StarVLA/QwenGR00T 模型训练
```

---

## 1. 数据生成

将原始 UMI 双臂 h5 数据转换为 LeRobot v2.1 数据集。

**环境**：
```bash
source /mnt/home/liuyi/anaconda3/bin/activate lerobot_v2_zbl
cd /mnt/home/liuyi/project/UMI/umi_data_process
```

**运行**：
```bash
python convert_galbot_bimanual_relative.py \
    --input_dir /mnt/project/public/yangzhibo/umidata/processed_h5/20260521/20260520_zhibo_evaluation_100tasks_wo_virtual_extrinsic/ \
    --output_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526 \
    --task "Hold the cups in your hand, and finally stack the two cups" \
    --repo_id galbot/bimanual_relative_cup \
    --resume
```

**输出列说明**：

| parquet 列 | dim | 含义 |
|---|---|---|
| `action` | 20 | **t+1 时刻**双臂绝对 eef pose + gripper |
| `observation.state` | 20 | **t 时刻**双臂绝对 eef pose + gripper |
| `observation.dual_relative_pose` | 9 | t 时刻左臂在右臂坐标系下的相对位姿（输入模型的 state）|

> 只存 abs 原始值，所有衍生值（delta / chunk_relative action）在训练时在线计算。

---

## 2. 离线统计值计算

在训练前，**根据所选 self_mode 运行一次** stats 脚本，生成 `meta/stats.json`。

**环境**：
```bash
source /mnt/home/liuyi/anaconda3/bin/activate starVLA
cd /mnt/home/liuyi/project/starVLA/examples/GalbotBimanualRelative/train_files
```

### 三种 self_mode 对应的 action 形式

| self_mode | 训练时 action 含义 | 计算方式 |
|---|---|---|
| `abs` | 绝对目标 eef pose | `action[t]` 原值不变 |
| `delta` | 逐帧增量（位移）| `action[t] = action[t] - action[t-1]`，`action[0] = 0` |
| `chunk_relative` | chunk 内相对位姿 | `action[t] = action[t] - action[0]`，以 chunk 第 0 帧为 base |

> **重要**：stats 计算时会模拟训练时的 chunk 采样 + action 变换逻辑，因此每种 mode 需要单独计算一次 stats。

### 按 mode 运行指令

**abs 模式（绝对目标位姿）**：
```bash
python compute_galbot_stats_self_mode.py \
    --dataset_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526 \
    --self_mode abs \
    --chunk_size 30 \
    --gripper_threshold 100.0 \
    --output_path /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526/meta/stats_abs.json
```

**delta 模式（逐帧差分）**：
```bash
python compute_galbot_stats_self_mode.py \
    --dataset_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526 \
    --self_mode delta \
    --chunk_size 30 \
    --gripper_threshold 100.0 \
```

**chunk_relative 模式（chunk 内相对）**：
```bash
python compute_galbot_stats_self_mode.py \
    --dataset_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526 \
    --self_mode chunk_relative \
    --chunk_size 30 \
    --gripper_threshold 100.0 \
    --output_path /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526/meta/stats_chunk_relative.json
```

**训练前将对应 mode 的 stats 复制为 `stats.json`**：
```bash
# 以 delta 为例
cp /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526/meta/stats_delta.json \
   /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526/meta/stats.json
```

---

## 3. 模型训练

### 3.1 修改数据路径

在 `data_registry/data_config.py` 中将 `<your_dataset_dir_name>` 替换为实际数据集目录名：

```python
DATASET_NAMED_MIXTURES = {
    "galbot_bimanual_self_mix": [
        ("galbot_lerobot_dual_cup_0526", 1.0, "galbot_bimanual_self"),  # ← 替换这里
    ],
}
```

### 3.2 启动训练

修改 `run_galbot_bimanual_self.sh` 中的变量后运行：

```bash
# === 按需修改这两个变量 ===
data_root=/mnt/project/human_action_data/liuyi/umi_lerobot_data
self_mode=abs   # abs | delta | chunk_relative
# ==========================

# 修改 run_galbot_bimanual_self.sh 中对应的 data_root 和 self_mode，然后：
bash examples/GalbotBimanualRelative/train_files/run_galbot_bimanual_self.sh
```

### 3.3 三种 mode 的训练指令（单卡 debug）

**abs 模式**：
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative_self.yaml \
  --datasets.vla_data.data_root_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data \
  --datasets.vla_data.data_mix galbot_bimanual_self_mix \
  --datasets.vla_data.self_mode abs \
  --datasets.vla_data.gripper_normalization binary \
  --trainer.max_train_steps 10 \
  --run_root_dir ./playground/Checkpoints \
  --run_id debug_abs
```

**delta 模式**：
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative_self.yaml \
  --datasets.vla_data.data_root_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data \
  --datasets.vla_data.data_mix galbot_bimanual_self_mix \
  --datasets.vla_data.self_mode delta \
  --datasets.vla_data.gripper_normalization binary \
  --trainer.max_train_steps 10 \
  --run_root_dir ./playground/Checkpoints \
  --run_id debug_delta
```

**chunk_relative 模式**：
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative_self.yaml \
  --datasets.vla_data.data_root_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data \
  --datasets.vla_data.data_mix galbot_bimanual_self_mix \
  --datasets.vla_data.self_mode chunk_relative \
  --datasets.vla_data.gripper_normalization binary \
  --trainer.max_train_steps 10 \
  --run_root_dir ./playground/Checkpoints \
  --run_id debug_chunk_relative
```

---

## 4. 关键参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `self_mode` | `abs` | action 变换模式：`abs` / `delta` / `chunk_relative` |
| `gripper_normalization` | `binary` | gripper 归一化：`binary`（二值化）/ `min_max` |
| `gripper_binary_threshold` | `100.0` | gripper 二值化阈值（mm），> threshold → 1.0 |
| `chunk_size` | `30` | action chunk 步数 |
| `action_horizon` | `30` | 模型 action 预测步数 |

---

## 5. 推理侧说明

模型输出的 action 经过反归一化后：

- **abs 模式**：直接是目标 eef pose，发给机器人执行
- **delta 模式**：逐帧增量，需要累加到当前 eef pose 上得到绝对目标
- **chunk_relative 模式**：相对 chunk 第 0 帧的偏移，需要加上当前 eef pose 得到绝对目标
- **gripper**：模型输出 0 → 46mm（关闭），1 → 124mm（打开）

---

## 6. 文件结构

```
examples/GalbotBimanualRelative/train_files/
├── train_readme.md                              ← 本文件
├── data_registry/
│   └── data_config.py                           ← GalbotBimanualSelfDataConfig
├── compute_galbot_stats_self_mode.py            ← 离线 stats 计算脚本
├── starvla_qwengroot_galbot_relative_self.yaml  ← 训练配置
├── run_galbot_bimanual_self.sh                  ← 多卡训练启动脚本
└── modality.json                                ← 字段映射模板
```
