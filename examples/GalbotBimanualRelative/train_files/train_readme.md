# GalbotBimanualRelative — QwenGR00T 双臂相对状态训练

## 目标

在 StarVLA/QwenGR00T 框架下训练双臂机器人 VLA 模型：

- **视觉输入**：双臂腕部相机（left_wrist + right_wrist）
- **语言输入**：任务描述文本
- **状态输入（state）**：双臂末端执行器之间的相对位姿，即左臂在右臂坐标系下的表示，9 dim（rel_xyz + rel_rot6d，不含 gripper）
- **动作输出（action）**：双臂 eef 绝对位姿 chunk，20 dim（每臂：xyz + rotation_6d + gripper = 10 dim）
- **动作长度**：16 步 chunk

---

## 目录结构

```
examples/GalbotBimanualRelative/
└── train_files/
    ├── train_readme.md                              ← 本文件
    ├── run_galbot_bimanual_relative.sh              ← 训练启动脚本
    ├── starvla_qwengroot_galbot_relative.yaml       ← 训练配置
    ├── modality.json                                ← 数据字段映射模板（需拷入数据集）
    └── data_registry/
        └── data_config.py                           ← 数据策略注册（transform / mix / robot_type）
```

---

## 各文件职责

| 文件 | 职责 | 被谁读取 |
|---|---|---|
| `run_..._.sh` | 启动入口；设路径变量；拼接 `accelerate launch` 命令 | 用户手动 `bash` |
| `starvla_..._.yaml` | 所有超参的真理源：`action_dim=20`、`state_dim=9`、`include_state=true`、freeze 策略、lr、max_steps | `train_starvla.py` 在启动时 `OmegaConf.load` |
| `modality.json` | 定义 StarVLA 字段名 → LeRobot 原始列名 的映射及维度切片 | `LeRobotSingleDataset` 启动时读 `<dataset>/meta/modality.json` |
| `data_config.py` | ① 注册 robot_type / embodiment / data_mix；② 定义加载哪些 video/state/action/language keys；③ 配置 transform 链（归一化策略）| `registry.py` 在包首次 import 时自动扫描 |

---

## 数据集要求

### LeRobot 目录结构

```
<data_root>/
└── <your_dataset_dir_name>/
    ├── meta/
    │   ├── info.json
    │   ├── tasks.parquet
    │   ├── modality.json          ← 从 train_files/modality.json 拷贝过来
    │   └── stats.json             ← 离线计算后生成（见下方）
    ├── data/
    │   └── chunk-*/
    │       └── episode_*.parquet
    └── videos/
        └── chunk-*/
            ├── observation.images.left_wrist/
            └── observation.images.right_wrist/
```

### parquet 必须包含的列

| 列名 | 类型 | 说明 |
|---|---|---|
| `observation.bimanual_relative` | `float32[9]` | **核心**：左臂在右臂坐标系下的相对位姿（见下方计算公式） |
| `action` | `float32[20]` | 双臂绝对 eef pose chunk，左右各 10 dim |
| `observation.state` | `float32[any]` | 可保留原始双臂绝对 state，供调试 |
| `task_index` | `int` | 任务 ID，映射到语言描述 |
| `episode_index` | `int` | episode 编号 |
| `frame_index` | `int` | 帧编号 |

### observation.bimanual_relative 计算公式

**定义**：左臂末端执行器位姿在右臂末端执行器坐标系下的表示。

```
维度布局（9 dim）：
  [0:3]   rel_xyz   = R_right^T @ (p_left - p_right)
  [3:9]   rel_rot6d = matrix_to_rotation_6d(R_right^T @ R_left)
```

其中 `R_left`、`R_right` 均从绝对 rot6d 转换而来（`rotation_6d_to_matrix`）。

**不含 gripper**：gripper 是独立离散控制量，与双臂空间关系无语义关联。

### action 维度布局（20 dim）

```
[0:3]   left  xyz        (绝对位置，单位：m)
[3:9]   left  rotation_6d
[9:10]  left  gripper    (0=open, 1=close)
[10:13] right xyz
[13:19] right rotation_6d
[19:20] right gripper
```

---

## 离线数据处理步骤

### 第一步：生成 observation.bimanual_relative 列

在数据集生成脚本中（采集/转换为 LeRobot 格式时）直接计算并存入 parquet：

```python
import numpy as np
import pytorch3d.transforms as pt
import torch

def compute_bimanual_relative(abs_state: np.ndarray) -> np.ndarray:
    """
    abs_state: [T, 20], 布局：left_xyz(3) + left_rot6d(6) + left_gripper(1)
                                + right_xyz(3) + right_rot6d(6) + right_gripper(1)
    返回: [T, 9]  rel_xyz(3) + rel_rot6d(6)
    """
    p_left  = torch.from_numpy(abs_state[:, 0:3]).float()
    R_left  = pt.rotation_6d_to_matrix(torch.from_numpy(abs_state[:, 3:9]).float())
    p_right = torch.from_numpy(abs_state[:, 10:13]).float()
    R_right = pt.rotation_6d_to_matrix(torch.from_numpy(abs_state[:, 13:19]).float())

    R_right_inv = R_right.transpose(-1, -2)
    rel_pos  = torch.einsum('tij,tj->ti', R_right_inv, p_left - p_right)   # [T, 3]
    rel_rot  = R_right_inv @ R_left                                          # [T, 3, 3]
    rel_rot6d = pt.matrix_to_rotation_6d(rel_rot)                           # [T, 6]

    return torch.cat([rel_pos, rel_rot6d], dim=-1).numpy().astype(np.float32)
```

> **注意**：请在数据集生成时确认你的 `observation.state` 的双臂绝对位姿维度顺序，
> 并与上方代码中的 slice 下标对齐。如果原始数据使用四元数或欧拉角，
> 需先转换为 rotation_6d 再调用本函数。

### 第二步：拷贝 modality.json 到数据集

```bash
cp examples/GalbotBimanualRelative/train_files/modality.json \
   <data_root>/<your_dataset_dir_name>/meta/modality.json
```

`modality.json` 告诉 dataloader：
- `state.bimanual_relative_pos` → `observation.bimanual_relative` 的 [0:3]
- `state.bimanual_relative_ori_6d` → `observation.bimanual_relative` 的 [3:9]
- `action.left_pos / left_ori_6d / left_gripper / ...` → `action` 的对应切片
- 视频来源：`observation.images.left_wrist` / `observation.images.right_wrist`

### 第三步：离线计算 stats.json

删除旧 stats 缓存（若有），在首次训练启动时框架会自动用 Welford 流式算法计算并写入 `meta/stats.json`：

```bash
rm -f <data_root>/<your_dataset_dir_name>/meta/stats.json
```

或者手动提前计算（推荐大数据集使用，避免训练前等待）：

```bash
python - <<'PY'
from pathlib import Path
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from omegaconf import OmegaConf

cfg = OmegaConf.load("examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative.yaml")
cfg.datasets.vla_data.data_root_dir = "/path/to/your/lerobot/root"
cfg.output_dir = "/tmp/stats_debug"

Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data)
dataset.save_dataset_statistics(Path(cfg.output_dir) / "dataset_statistics.json")
print("Done. stats saved.")
PY
```

### 第四步：更新 data_config.py 中的数据集名

编辑 `data_registry/data_config.py`，将占位符替换为实际目录名：

```python
DATASET_NAMED_MIXTURES = {
    "galbot_bimanual_mix": [
        ("your_actual_dataset_dir_name", 1.0, "galbot_bimanual_relative"),
    ],
}
```

---

## 归一化策略

| 维度 | 归一化方式 | 原因 |
|---|---|---|
| `action.{left,right}_pos` | `min_max` → [-1, 1] | 位置量纲差异大，需归一 |
| `action.{left,right}_ori_6d` | **不归一化** | rotation_6d 天然有界 [-1, 1]，min_max 会破坏正交性 |
| `action.{left,right}_gripper` | `binary`（默认）或 `min_max` | 可通过 `gripper_normalization` 参数切换，支持对比实验 |
| `state.bimanual_relative_pos` | `min_max` → [-1, 1] | 相对位置受机械限位约束，归一后与 rot6d 量级对齐 |
| `state.bimanual_relative_ori_6d` | **不归一化** | 同 action rot6d |

切换 gripper 归一化方式（对比实验）：

```bash
# 方案 A：二值化（默认）
--datasets.vla_data.gripper_normalization binary

# 方案 B：连续 min_max
--datasets.vla_data.gripper_normalization min_max
```

---

## 训练启动

### 前置检查

```bash
# 1. 确认 registry 发现新 data_mix
python - <<'PY'
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
print("mix found:", "galbot_bimanual_mix" in DATASET_NAMED_MIXTURES)
print("robot found:", "galbot_bimanual_relative" in ROBOT_TYPE_CONFIG_MAP)
PY

# 2. dataloader smoke test（确认 action.shape=(16,20), state.shape=(1,9)）
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative.yaml

# 3. 模型 smoke test（确认 forward 不报 shape 错误）
python starVLA/model/framework/VLM4A/QwenGR00T.py \
  --config_yaml examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative.yaml
```

### 单卡 Debug（10 步）

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative.yaml \
  --datasets.vla_data.data_root_dir /path/to/your/lerobot/root \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 10 \
  --trainer.eval_interval 5 \
  --run_root_dir /tmp/debug \
  --run_id debug_$(date +%H%M)
```

期望：action_loss 为有限正数，无 shape mismatch，checkpoint 正常保存。

### 正式多卡训练

```bash
bash examples/GalbotBimanualRelative/train_files/run_galbot_bimanual_relative.sh
```

---

## 训练调用链（简述）

```
run_galbot_bimanual_relative.sh
  └── accelerate launch train_starvla.py --config_yaml starvla_..._.yaml
        ├── OmegaConf.load(yaml) + CLI dotlist merge
        ├── build_framework(cfg)          → QwenGR00T
        │     ├── Qwen2.5-VL-3B         (VLM backbone, 冻结)
        │     └── FlowmatchingActionHead
        │           ├── state_encoder: MLP(9 → 1024)
        │           ├── action_encoder: MLP(20 → 768)
        │           └── DiT-B (cross-attn with VLM hidden)
        ├── build_dataloader(cfg)
        │     ├── registry 自动发现 data_registry/data_config.py
        │     ├── 加载 meta/modality.json（字段映射）
        │     ├── 加载/计算 meta/stats.json（归一化统计量）
        │     └── __getitem__: raw → transform chain → _pack_sample
        │           sample = {image:[PIL,PIL], lang:str, action:[16,20], state:[1,9]}
        └── VLATrainer.train()
              model.forward(batch)
                ├── qwen_vl(images, lang) → hidden [B, L, H]
                └── action_head(hidden, action_target, state) → flow-matching loss
```

---

## 关键配置参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `framework.action_model.action_dim` | `20` | 双臂 (xyz+rot6d+gripper)×2 |
| `framework.action_model.state_dim` | `9` | 双臂相对位姿 rel_xyz+rel_rot6d |
| `framework.action_model.action_horizon` | `16` | action chunk 长度 |
| `framework.action_model.future_action_window_size` | `15` | = action_horizon - 1 |
| `datasets.vla_data.include_state` | `true` | **必须为 true**，否则 state 不进模型 |
| `datasets.vla_data.gripper_normalization` | `binary` | `binary` 或 `min_max`，对比实验用 |
| `trainer.freeze_modules` | `qwen_vl_interface` | 冻结 VLM，只训 action expert |

---

## 风险点与注意事项

### 1. `include_state` 必须显式开启

若 `include_state` 未设为 `true`，`_pack_sample` 不会生成 `sample["state"]`，
`QwenGR00T.forward` 的 `state=None` 分支静默通过——loss 可以正常下降，
但模型从未看过 state，相当于没有训练 state_encoder。

### 2. observation.bimanual_relative 维度顺序

数据集生成时必须保证列的维度顺序与 `modality.json` 中的 `start/end` 一致：

```
[0:3]  rel_xyz
[3:9]  rel_rot6d
```

如果原始 state 的双臂字段顺序不同（如右臂在前），需在生成脚本中调整切片。

### 3. stats.json 必须先于训练存在

框架在 `is_main()` 进程上自动计算 stats，其他进程等 `dist.barrier()` 后从文件读取。
若文件不存在且分布式进程数 > 1，非主进程会在 barrier 处阻塞失败。
**建议在正式多卡训练前，先用单进程跑一次 dataloader smoke test，触发 stats 生成。**

### 4. rot6d 不归一化是正确的

rotation_6d 的每个分量是旋转矩阵列向量的分量，天然有界于 [-1, 1]。
若强行做 min_max，会破坏两列向量的正交约束，导致 decode 回旋转矩阵时误差增大。
当前策略（xyz min_max + rot6d 不归 + gripper binary）与 GR00T / π0 官方实现一致。

### 5. action chunk 的 base 定义

当前 action 在 lerobot 中存的是**绝对 eef pose**，不是 relative action chunk。
模型直接学绝对位姿序列，inference 时输出的也是绝对位姿，部署端需根据机器人 API 选择发送绝对还是 delta 指令。
若后续想改为 relative action chunk，在 `data_config.py` 的 transform 链中加入
`RelativePoseActionTransform` 并重新计算 `meta/relative_stats.json` 即可，不影响模型架构。
