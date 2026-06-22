# Action Chunk Transform - 完整使用指南

本文档提供 ActionChunkTransform 改造后的完整训练流程使用方法。

---

## 📋 目录

1. [核心功能](#核心功能)
2. [完整训练流程](#完整训练流程)
3. [单数据集训练](#单数据集训练)
4. [多数据集混训](#多数据集混训)
5. [常见问题](#常见问题)

---

## 核心功能

### 支持的 Action 模式

- **`abs`** - 绝对坐标（原始数据）
- **`delta`** - 帧间差分（相对前一帧）
- **`relative_pose`** - 相对 base（相对第一帧）

### 已支持的数据集

- ✅ **Galbot Bimanual** (`examples/GalbotBimanualRelative`)
- ✅ **FastUMI Dual Arm** (`examples/FastUMI`)

---

## 完整训练流程

### 步骤 1: 准备数据集

确保数据集格式为 LeRobot v2.1 或 v3.0 格式：

```
dataset_path/
  ├── meta/
  │   ├── episodes.jsonl (v2.1) 或 episodes/*/*.parquet (v3.0)
  │   └── (stats.json 将在步骤 2 生成)
  └── data/
      └── chunk-*/
```

---

### 步骤 2: 计算数据集统计信息

**为每个 action_mode 生成对应的 stats 文件。**

#### 示例 1: Galbot - Delta 模式

```bash
python scripts/compute_dataset_stats.py \
    --dataset_path /path/to/galbot_data \
    --robot_type galbot_bimanual_self \
    --data_registry examples.GalbotBimanualRelative.train_files.data_registry.data_config \
    --output /path/to/galbot_data/meta/stats_delta_chunk30.json \
    --action_mode delta \
    --action_chunk_size 30
```

#### 示例 2: FastUMI - Relative Pose 模式

```bash
python scripts/compute_dataset_stats.py \
    --dataset_path /mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker \
    --robot_type fastumi_dual_arm \
    --data_registry examples.FastUMI.train_files.data_registry.data_config \
    --output /mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker/meta/stats_relative_pose_chunk16.json \
    --action_mode relative_pose \
    --action_chunk_size 16 \
    --sample_ratio 0.1  # 大数据集可用采样加速
```

#### 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--dataset_path` | ✅ | 数据集路径 |
| `--robot_type` | ✅ | 机器人类型（见下表） |
| `--data_registry` | ✅ | 数据配置模块路径 |
| `--output` | ✅ | 输出 stats 文件路径 |
| `--action_mode` | ⭐ | abs / delta / relative_pose |
| `--action_chunk_size` | ⭐ | Action horizon 大小 |
| `--gripper_normalization` | ⚪ | none / binary / min_max |
| `--sample_ratio` | ⚪ | 采样比例（0.0-1.0，默认 1.0） |

**机器人类型映射：**

| robot_type | 数据集 | data_registry 路径 |
|------------|--------|-------------------|
| `galbot_bimanual_self` | Galbot | `examples.GalbotBimanualRelative.train_files.data_registry.data_config` |
| `fastumi_dual_arm` | FastUMI | `examples.FastUMI.train_files.data_registry.data_config` |

---

### 步骤 3: 单数据集训练

#### Galbot 训练

```bash
cd /mnt/home/liuyi/project/starVLA
bash examples/GalbotBimanualRelative/train_files/run_galbot_bimanual_self.sh
```

**修改训练脚本配置：**

打开 `run_galbot_bimanual_self.sh`，修改：

```bash
# === Action mode config ===
action_mode=relative_pose     # 修改为你想要的模式
action_chunk_size=30          # 修改为对应的 chunk size

# === 数据路径 ===
data_root=/path/to/your/data
data_name=your_dataset_name
```

#### FastUMI 训练

```bash
cd /mnt/home/liuyi/project/starVLA
bash examples/FastUMI/train_files/run_fastumi_dual_arm.sh
```

**修改训练脚本配置：**

打开 `run_fastumi_dual_arm.sh`，修改：

```bash
# === Action mode config ===
action_mode=relative_pose     # abs | delta | relative_pose
action_chunk_size=16          # 对应 stats 文件的 chunk size

# === 数据路径 ===
data_root=/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm
data_name=Add_Rice_to_Rice_Cooker
```

---

### 步骤 4: 多数据集混训（跨本体学习）

**关键：** 混训时所有数据集必须使用**相同的 action_mode**！

#### 4.1 为每个数据集生成 stats

```bash
# Galbot - relative_pose
python scripts/compute_dataset_stats.py \
    --dataset_path /path/to/galbot_data \
    --robot_type galbot_bimanual_self \
    --data_registry examples.GalbotBimanualRelative.train_files.data_registry.data_config \
    --output /path/to/galbot_data/meta/stats_relative_pose_chunk30.json \
    --action_mode relative_pose \
    --action_chunk_size 30

# FastUMI - relative_pose
python scripts/compute_dataset_stats.py \
    --dataset_path /path/to/fastumi_data \
    --robot_type fastumi_dual_arm \
    --data_registry examples.FastUMI.train_files.data_registry.data_config \
    --output /path/to/fastumi_data/meta/stats_relative_pose_chunk30.json \
    --action_mode relative_pose \
    --action_chunk_size 30
```

#### 4.2 修改训练配置

创建混训配置文件 `data_mix_config.yaml`：

```yaml
# 数据集混合配置
datasets:
  - name: galbot_bimanual
    path: /path/to/galbot_data
    robot_type: galbot_bimanual_self
    weight: 1.0
    
  - name: fastumi_dual_arm
    path: /path/to/fastumi_data
    robot_type: fastumi_dual_arm
    weight: 1.0

# 统一配置（所有数据集共享）
action_mode: relative_pose      # 必须相同！
action_chunk_size: 30           # 必须相同！
gripper_normalization: none
```

#### 4.3 启动混训

```bash
python starVLA/training/train_starvla.py \
    --config_yaml examples/Mixed/train_files/mixed_training_config.yaml \
    --datasets.vla_data.action_mode relative_pose \
    --datasets.vla_data.action_chunk_size 30 \
    --run_root_dir /path/to/checkpoints \
    --run_id mixed_training_$(date +%m%d)
```

---

## 配置说明

### Action Mode 选择建议

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| `abs` | 绝对坐标 | 单一机器人，工作空间固定 |
| `delta` | 帧间差分 | 精细控制，短期预测 |
| `relative_pose` | 相对 base | **跨本体学习，混训推荐** ⭐ |

### Action Chunk Size 选择

- **Galbot**: 默认 30（轨迹较长）
- **FastUMI**: 默认 16（2-step interval，每步 8 帧）
- **混训**: 建议统一为较小值（如 16）

### Gripper Normalization

- `none` - 不归一化（推荐，保持原始值）
- `binary` - 二值化（开/关，threshold=100.0mm）
- `min_max` - 最小最大归一化

---

## 验证测试

### 测试 Transform 逻辑

```bash
cd /mnt/home/liuyi/project/starVLA
source /mnt/home/liuyi/anaconda3/bin/activate starvla_qwen35
python scripts/test_action_chunk_transform_quick.py
```

**预期输出：**
```
abs            : ✅ PASSED
delta          : ✅ PASSED
relative_pose  : ✅ PASSED
gripper        : ✅ PASSED
```

---

## 常见问题

### Q1: Stats 文件命名规则？

**A:** 格式为 `stats_{action_mode}_chunk{action_chunk_size}.json`

示例：
- `stats_delta_chunk30.json`
- `stats_relative_pose_chunk16.json`

### Q2: 混训时 action_chunk_size 不同怎么办？

**A:** 必须统一！建议选择较小的值（如 16），然后为所有数据集重新生成 stats。

### Q3: 如何验证 stats 文件是否正确？

**A:** 检查文件内容：

```bash
python -c "import json; print(json.load(open('path/to/stats.json')).keys())"
```

应该包含所有 action keys（如 `action.left_arm`, `action.left_ori_6d` 等）。

### Q4: compute_dataset_stats.py 运行太慢？

**A:** 使用 `--sample_ratio` 采样加速：

```bash
--sample_ratio 0.1  # 只使用 10% 数据计算 stats
```

### Q5: 训练时提示找不到 stats 文件？

**A:** 检查：
1. stats 文件路径是否正确
2. 文件名是否匹配 `action_mode` 和 `action_chunk_size`
3. 训练脚本中的 `stats_path` 配置是否正确

---

## 文件结构

```
starVLA/
├── starVLA/dataloader/gr00t_lerobot/transform/
│   └── action_chunk_mode.py          # ActionChunkTransform 核心实现
├── scripts/
│   ├── compute_dataset_stats.py      # Stats 计算脚本
│   └── test_action_chunk_transform_quick.py  # Transform 测试
├── examples/
│   ├── GalbotBimanualRelative/
│   │   └── train_files/
│   │       ├── data_registry/data_config.py  # Galbot 配置
│   │       └── run_galbot_bimanual_self.sh   # Galbot 训练脚本
│   └── FastUMI/
│       └── train_files/
│           ├── data_registry/data_config.py  # FastUMI 配置
│           └── run_fastumi_dual_arm.sh       # FastUMI 训练脚本
```

---

## 技术细节

### ActionChunkTransform 公式

**Delta 模式：**
```python
delta_xyz[t] = R[t-1].T @ (p[t] - p[t-1])  # 位置在前一帧局部坐标系
delta_R[t]   = R[t-1].T @ R[t]              # SO(3) 相对旋转
delta[0]     = 0                            # 首帧为零
```

**Relative Pose 模式：**
```python
rel_xyz[t] = R[0].T @ (p[t] - p[0])  # 位置在 base 局部坐标系
rel_R[t]   = R[0].T @ R[t]            # 相对 base 的旋转
rel[0]     = 0                        # 首帧为零（相对自身）
```

### 跨本体学习原理

局部坐标系消除了不同机器人的绝对坐标差异：

```python
# 问题：不同机器人的绝对坐标不同
Robot A: action_abs = [x=1.1, y=0.5, z=0.3]
Robot B: action_abs = [x=0.7, y=0.3, z=0.2]  # 无法共享

# 解决：转换到局部坐标系
Robot A: action_rel = [Δx=+0.1, Δy=0, Δz=0]  # "向前 10cm"
Robot B: action_rel = [Δx=+0.1, Δy=0, Δz=0]  # 相同的局部运动！
# ✅ 可以共享，支持跨本体学习
```

---

## 更新日志

- **2026-06-10**: 初始版本
  - 支持 abs/delta/relative_pose 三种模式
  - Galbot 和 FastUMI 完整支持
  - compute_dataset_stats.py 支持 action_mode 参数

---

## 联系方式

如有问题，请查看：
- 代码仓库：`umi_pretrain` 分支
- 技术文档：`ACTION_CHUNK_TRANSFORM_SUMMARY.md`
- 完整报告：`ACTION_CHUNK_TRANSFORM_FINAL_REPORT.md`
