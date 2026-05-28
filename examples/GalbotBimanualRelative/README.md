# GalbotBimanualRelative — StarVLA/QwenGR00T 双臂训练说明

## 概述

在 StarVLA/QwenGR00T 框架下训练 Galbot 双臂机器人 VLA 模型：

| 项目 | 说明 |
|---|---|
| 视觉输入 | 双臂腕部相机（`left_wrist` + `right_wrist`） |
| state | 左臂在右臂坐标系下的相对位姿，9 dim（xyz + rot6d） |
| action | next-step 双臂绝对 eef pose + gripper，20 dim（每臂 xyz + rot6d + gripper = 10 dim） |
| action chunk | 30 步（训练时） |
| action 变换 | 支持 `abs` / `delta` / `chunk_relative` 三种模式（通过 yaml 配置切换） |

---

## 目录结构

```
examples/GalbotBimanualRelative/
├── README.md                          # 本文件
├── offline_eval/
│   ├── README.md                      # 离线评估说明
│   ├── offline_eval_galbot_bimanual_self.py
│   └── run_offline_eval.sh
└── train_files/
    ├── train_readme.md                # 训练详细说明
    ├── starvla_qwengroot_galbot_relative_self.yaml   # 训练配置
    ├── run_galbot_bimanual_self.sh    # 启动脚本
    ├── modality.json                  # 字段映射模板
    ├── compute_galbot_stats_self_mode.py             # 离线计算 stats
    ├── convert_v21_to_v30.py          # 数据格式转换（v2.1→v3.0）
    └── data_registry/
        └── data_config.py             # 数据策略注册
```

---

## 数据层设计

LeRobot parquet 只存原始 abs 值，所有 action 变换（delta/chunk_relative）在训练时由 `SelfModeActionTransform` 在线完成。

| 列名 | dim | 含义 |
|---|---|---|
| `action` | 20 | next-step 双臂绝对 eef pose + gripper |
| `observation.state` | 20 | t 时刻双臂绝对 eef pose（delta 模式 t=0 的 base） |
| `observation.dual_relative_pose` | 9 | 左臂在右臂坐标系下的相对位姿（模型输入） |
| `video.left_wrist` | video | 左腕相机 |
| `video.right_wrist` | video | 右腕相机 |

action 维度说明（每臂 10 dim，共 20 dim）：

```
[0:3]   L_pos      左臂 xyz
[3:9]   L_ori_6d   左臂旋转 6d
[9]     L_gripper  左夹爪（abs: mm；训练时二值化 >100mm→1）
[10:13] R_pos      右臂 xyz
[13:19] R_ori_6d   右臂旋转 6d
[19]    R_gripper  右夹爪
```

---

## 数据转换

数据转换脚本在 `/mnt/home/liuyi/project/UMI/umi_data_process/convert_galbot_bimanual_relative.py`，需要激活 `lerobot_v2_zbl` 环境：

```bash
source /mnt/home/liuyi/anaconda3/bin/activate lerobot_v2_zbl
cd /mnt/home/liuyi/project/UMI/umi_data_process

python convert_galbot_bimanual_relative.py \
    --input_dir /mnt/project/public/yangzhibo/umidata/processed_h5/20260521/20260520_zhibo_evaluation_100tasks_wo_virtual_extrinsic/ \
    --output_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526 \
    --task "Hold the cups in your hand, and finally stack the two cups" \
    --repo_id galbot/bimanual_relative_cup \
    --resume
```

---

## 离线计算 Stats

Stats 必须在训练前离线计算，与 `SelfModeActionTransform` 逻辑对齐（模拟 chunk 采样 + action 变换 + gripper 二值化）：

```bash
cd /mnt/home/liuyi/project/starVLA/examples/GalbotBimanualRelative/train_files

# delta mode（当前推荐）
python compute_galbot_stats_self_mode.py \
    --dataset_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526 \
    --self_mode delta \
    --chunk_size 30 \
    --gripper_threshold 100.0

# abs mode
python compute_galbot_stats_self_mode.py \
    --dataset_dir /path/to/dataset \
    --self_mode abs \
    --chunk_size 30 \
    --gripper_threshold 100.0

# chunk_relative mode
python compute_galbot_stats_self_mode.py \
    --dataset_dir /path/to/dataset \
    --self_mode chunk_relative \
    --chunk_size 30 \
    --gripper_threshold 100.0
```

Stats 默认写到 `<dataset_dir>/meta/stats.json`，可通过 `--output_path` 指定路径。

**注意**：`stats.json` 中的 `__cache_config` 固定写为 `{"mode": "abs"}`（框架 cache 校验只看 action_mode，与 self_mode 解耦）。在 yaml 中设置 `self_mode: delta`（或其他模式）控制 transform 行为，不影响 stats 校验。

---

## 训练

### 环境

```bash
source /mnt/home/liuyi/anaconda3/bin/activate starvla
cd /mnt/home/liuyi/project/starVLA
```

### 启动训练

```bash
bash examples/GalbotBimanualRelative/train_files/run_galbot_bimanual_self.sh \
    2>&1 | tee /path/to/train.log
```

或在 tmux 中运行以保持会话：

```bash
tmux new -s train
bash examples/GalbotBimanualRelative/train_files/run_galbot_bimanual_self.sh \
    2>&1 | tee Checkpoints/galbot_delta_action/galbot_bimanual_self_$(date +%m%d)/train.log
```

### 关键配置（`starvla_qwengroot_galbot_relative_self.yaml`）

```yaml
datasets:
  vla_data:
    self_mode: delta          # abs / delta / chunk_relative
    gripper_binary_threshold: 100.0
    gripper_normalization: binary

training:
  action_chunk_size: 30
  action_dim: 20
  state_dim: 9
```

### Wandb 配置

首次使用需登录（key 保存后永久生效）：

```bash
wandb login
# 在 https://wandb.ai/authorize 获取 API key 并粘贴
```

---

## Transform 流程

训练时 action 的处理顺序：

```
parquet abs action
    → StateActionToTensor          # 转 tensor
    → SelfModeActionTransform      # delta/chunk_relative 变换（gripper 不参与差分）
    → StateActionTransform         # 归一化
        - pos: q99 → [-1, 1]
        - ori_6d: 不归一化（本身 [-1,1]）
        - gripper: binary（>100mm → 1）
```

实现文件：`starVLA/dataloader/gr00t_lerobot/transform/self_mode_action.py`

---

## 离线评估

见 `offline_eval/README.md`。

快速运行：

```bash
bash examples/GalbotBimanualRelative/offline_eval/run_offline_eval.sh
```

---

## 关键路径速查

| 用途 | 路径 |
|---|---|
| 训练框架代码 | `/mnt/home/liuyi/project/starVLA` |
| 训练配置目录 | `examples/GalbotBimanualRelative/train_files/` |
| 数据转换脚本 | `/mnt/home/liuyi/project/UMI/umi_data_process/convert_galbot_bimanual_relative.py` |
| 自定义 action transform | `starVLA/dataloader/gr00t_lerobot/transform/self_mode_action.py` |
| 数据集目录 | `/mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526` |
| Checkpoints | `Checkpoints/galbot_delta_action/` |

## 环境

| 用途 | conda 环境 |
|---|---|
| 数据转换 | `lerobot_v2_zbl` |
| 训练 / 评估 | `starvla` |

## Git

- 当前分支：`liuyi_dev`（追踪 `origin/liuyi_dev`）
- 多机协作：开始工作前 `git pull --rebase`，结束前 `git push`
