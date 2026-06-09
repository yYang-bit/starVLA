# GalbotBimanualRelative 训练开发进度

## 当前任务

在 StarVLA/QwenGR00T 框架下训练双臂机器人 VLA 模型：
- **视觉输入**：双臂腕部相机（left_wrist + right_wrist）
- **state**：双臂 eef 相对位姿，9 dim（左臂在右臂坐标系下）
- **action**：next-step 双臂绝对 eef pose + gripper，20 dim（每臂 xyz+rot6d+gripper = 10 dim）
- **action chunk**：16 步
- **action 变换**：支持 abs/delta/chunk_relative 三种模式（通过配置切换）

训练相关文件在 `examples/GalbotBimanualRelative/train_files/`。

---

## 已完成

### 1. 训练框架文件
位于 `examples/GalbotBimanualRelative/train_files/`：
- `train_readme.md` - 完整训练说明
- `starvla_qwengroot_galbot_relative.yaml` - 训练配置（action_dim=20, state_dim=9, include_state=true）
- `run_galbot_bimanual_relative.sh` - 启动脚本
- `modality.json` - 字段映射模板
- `data_registry/data_config.py` - 数据策略注册

### 2. 数据转换脚本 ✓（已重构）

位于 `/mnt/home/liuyi/project/UMI/umi_data_process/convert_galbot_bimanual_relative.py`

**数据层原则**：只存 abs 原始值，所有衍生值（delta/relative action）在训练时在线计算。

**输出 parquet 列**：
- `action`：float32[20]，**next-step** 双臂绝对 eef pose + gripper
- `observation.state`：float32[20]，**t 时刻**双臂绝对 eef pose + gripper（与 action 同格式，用于 delta 模式 t=0 的 base）
- `observation.dual_relative_pose`：float32[9]，左臂在右臂坐标系下的相对位姿

**视频列**：
- `observation.images.left_wrist`
- `observation.images.right_wrist`

**用法**：
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

### 3. Smoke test 验证 ✓
输出在 `/mnt/home/liuyi/project/UMI/umi_data_process/galbot_smoke_test`，跑了 2 个 episode（329+336 帧）。验证项：
- 目录结构符合 LeRobot v3.0 ✓
- `action` shape=(T, 20), `observation.state` shape=(T, 20), `observation.dual_relative_pose` shape=(T, 9) ✓
- `action[t] == observation.state[t+1]` 验证通过 ✓

### 4. 项目清单 ✓
- `0525.md` — 模块化改造项目清单，作为后续开发的指南

### 5. SelfModeActionTransform ✓
- 新增文件：`starVLA/dataloader/gr00t_lerobot/transform/self_mode_action.py`
- 支持 `abs` / `delta` / `chunk_relative` 三种模式
- 已注册到 `transform/__init__.py`

### 6. GalbotBimanualSelfDataConfig ✓
- 唯一保留的 config：`GalbotBimanualSelfDataConfig`（注册 `galbot_bimanual_self`）
- 注册 `galbot_bimanual_self_mix` 到 `DATASET_NAMED_MIXTURES`
- 流程：`StateActionToTensor → SelfModeActionTransform → StateActionTransform`
- 已删除旧的 `GalbotBimanualRelativeDataConfig`

### 7. Self-mode 离线 stats 脚本 ✓
- 文件：`examples/GalbotBimanualRelative/train_files/compute_galbot_stats_self_mode.py`
- 支持 `abs`/`delta`/`chunk_relative` 三种模式
- 与训练时 `StateActionTransform` 逻辑对齐：模拟 chunk 采样 + action 变换 + gripper 二值化

### 8. 字段命名统一 ✓
- modality.json 字段名与 parquet 列名一致：`dual_relative_pose_pos` / `dual_relative_pose_ori_6d`
- data_config.py state_keys 同步使用 `dual_relative_pose_pos`
- 删除旧 yaml/sh 文件（`starvla_qwengroot_galbot_relative.yaml`、`run_galbot_bimanual_relative.sh`）

**用法**：
```bash
source /mnt/home/liuyi/anaconda3/bin/activate lerobot_v2_zbl
cd /mnt/home/liuyi/project/starVLA/examples/GalbotBimanualRelative/train_files

# abs mode
python compute_galbot_stats_self_mode.py \
    --dataset_dir /path/to/lerobot_dataset \
    --self_mode abs \
    --chunk_size 16 \
    --gripper_threshold 100.0

# delta mode
python compute_galbot_stats_self_mode.py \
    --dataset_dir /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526 \
    --self_mode delta \
    --chunk_size 30 \
    --gripper_threshold 100.0

# chunk_relative mode
python compute_galbot_stats_self_mode.py \
    --dataset_dir /path/to/lerobot_dataset \
    --self_mode chunk_relative \
    --chunk_size 16 \
    --gripper_threshold 100.0
```
stats 默认写到 `<dataset_dir>/meta/stats.json`，可通过 `--output_path` 指定。

### 8. 训练配置 ✓
- 新增 `starvla_qwengroot_galbot_relative_self.yaml`（配置 `self_mode` 参数）
- 新增 `run_galbot_bimanual_self.sh`（通过 `self_mode` 参数切换实验）

---

## 待完成

### 1. Dataloader smoke test
验证三种 self_mode 下 action/state shape 和值正确。

### 2. 模型 forward smoke test
### 3. 单卡 debug 训练
### 4. 正式多卡训练

---

## 关键路径速查

| 用途 | 路径 |
|---|---|
| 训练框架代码 | `/mnt/home/liuyi/project/starVLA` |
| 训练配置目录 | `examples/GalbotBimanualRelative/train_files/` |
| 数据转换脚本 | `/mnt/home/liuyi/project/UMI/umi_data_process/convert_galbot_bimanual_relative.py` |
| 自定义 action transform | `starVLA/dataloader/gr00t_lerobot/transform/self_mode_action.py` |
| 转换脚本依赖的 lerobot | `/mnt/home/liuyi/project/UMI/umi_data_process/lerobot_zbl/src` |
| Smoke test 输出 | `/mnt/home/liuyi/project/UMI/umi_data_process/galbot_smoke_test` |
| 项目清单 | `/mnt/home/liuyi/project/starVLA/0525.md` |

## 环境

- 转换脚本环境：`source /mnt/home/liuyi/anaconda3/bin/activate lerobot_v2_zbl`
- 训练环境：见 starVLA 项目 README

## Git

- 当前分支：`liuyi_dev`（追踪 `origin/liuyi_dev`，已开启 rebase 模式）
- 协作场景：两台电脑共用 liuyi_dev；开始工作前 `git pull`，结束前 `git push`