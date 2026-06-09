# Action Chunk Transform 改造总结

## 📋 改造目标

统一 Galbot 和 FastUMI 的 action 表示，支持跨本体学习（embodiment-agnostic learning）。

## ✅ 已完成工作

### Phase 1: 创建统一 ActionChunkTransform ✅

**文件：** `starVLA/dataloader/gr00t_lerobot/transform/action_chunk_mode.py`

**功能：**
- 支持 3 种模式：`abs` / `delta` / `relative_pose`
- 局部坐标系表示（支持跨本体）
- 纯 action chunk 处理（无 state 依赖）
- Gripper 保持绝对值

**关键公式（局部帧）：**
```python
# Delta mode (逐帧差分)
delta_xyz[t] = R[t-1].T @ (p[t] - p[t-1])
delta_R[t]   = R[t-1].T @ R[t]

# Relative pose mode (相对 base)
rel_xyz[t] = R[0].T @ (p[t] - p[0])
rel_R[t]   = R[0].T @ R[t]
```

### Phase 2: 更新 Galbot data_config.py ✅

**文件：** `examples/GalbotBimanualRelative/train_files/data_registry/data_config.py`

**更新：**
- 动态 `action_chunk_size` 在 `modality_config(data_cfg)`
- 集成 `ActionChunkTransform`（替代 `SelfModeActionTransform`）
- 添加 `transform_for_stats(data_cfg)` - 用于 stats 计算（无 normalization）
- 添加 `stats_path(data_cfg)` - 自动生成 stats 文件名
- Gripper 归一化可选（`None` = 使用原始值）
- 参数统一：`self_mode` → `action_mode`

**配置示例：**
```python
data_cfg = {
    "action_mode": "relative_pose",       # abs | delta | relative_pose
    "action_chunk_size": 30,              # chunk horizon
    "gripper_normalization": None,        # None | binary | min_max
    "gripper_binary_threshold": 100.0,    # 阈值（mm）
}
```

### Phase 3: 更新 FastUMI data_config.py ✅

**文件：** `examples/FastUMI/train_files/data_registry/data_config.py`

**更新：**
- 与 Galbot 相同的改造
- 集成 `ActionChunkTransform`（替代 `RelativePoseActionTransform`）
- 默认 `action_chunk_size=16`（2-step interval）
- 参数统一：`action_chunk_representation` → `action_mode`

### Phase 4: 测试验证 🔄

**测试脚本：** `scripts/test_action_chunk_transform.py`

**测试内容：**
- FastUMI abs/delta/relative_pose 3 种模式
- 验证输出 shape 正确
- 验证首帧行为（delta[0]=0, rel[0]=0）
- 轻量化（只测 1 个 episode, 4 帧 chunk）

## 🎯 核心成果

### 1. 统一的 Action 表示

所有数据集使用相同的接口：
```python
ActionChunkTransform(
    mode="relative_pose",
    action_keys=[...],
    position_suffix="_arm",
    rotation_suffix="_ori_6d",
    gripper_suffix="_gripper",
)
```

### 2. 跨本体学习支持

局部坐标系表示消除了本体差异：
```python
# 不同机器人的绝对坐标不同
Robot A: action = [x=1.1, y=0.5, z=0.3]
Robot B: action = [x=0.7, y=0.3, z=0.2]

# 但局部偏移相同！
Robot A: rel_action = [Δx=+0.1, Δy=0, Δz=0]  
Robot B: rel_action = [Δx=+0.1, Δy=0, Δz=0]  ← 可迁移！
```

### 3. 配置透传架构

从训练脚本到 transform 的清晰分层：
```bash
# .sh script
ACTION_MODE="relative_pose"
ACTION_CHUNK_SIZE=30

# → train.py
--data_cfg.action_mode=$ACTION_MODE
--data_cfg.action_chunk_size=$ACTION_CHUNK_SIZE

# → data_config.py
modality_config(data_cfg)  # 动态 chunk size
transform(data_cfg)        # 添加 ActionChunkTransform

# → ActionChunkTransform
# 纯执行层，无硬编码配置
```

### 4. Stats 计算一致性

```python
# Stats 计算使用相同的 transform
transforms = config.transform_for_stats(data_cfg)

# 保证 stats 和训练时的数据分布一致
# stats_delta_chunk30.json ← delta mode, chunk 30
# stats_relative_chunk16.json ← relative mode, chunk 16
```

## 📁 文件变更

### 新增文件
- `starVLA/dataloader/gr00t_lerobot/transform/action_chunk_mode.py` - 统一 transform
- `scripts/test_action_chunk_transform.py` - 测试脚本

### 修改文件
- `examples/GalbotBimanualRelative/train_files/data_registry/data_config.py`
- `examples/FastUMI/train_files/data_registry/data_config.py`

### 待标记为 Deprecated
- `starVLA/dataloader/gr00t_lerobot/transform/self_mode_action.py` - 旧版
- `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` 中的 `RelativePoseActionTransform`

## 🔧 使用示例

### 训练配置（.sh 脚本）

```bash
# Galbot relative_pose mode
ACTION_MODE="relative_pose"
ACTION_CHUNK_SIZE=30
GRIPPER_NORM="binary"

python train_starvla.py \
    --data_cfg.action_mode=$ACTION_MODE \
    --data_cfg.action_chunk_size=$ACTION_CHUNK_SIZE \
    --data_cfg.gripper_normalization=$GRIPPER_NORM \
    ...

# FastUMI delta mode
ACTION_MODE="delta"
ACTION_CHUNK_SIZE=16

python train_starvla.py \
    --data_cfg.action_mode=$ACTION_MODE \
    --data_cfg.action_chunk_size=$ACTION_CHUNK_SIZE \
    ...
```

### Stats 计算

```bash
python scripts/compute_dataset_stats.py \
    --dataset_path=/path/to/data \
    --robot_type=galbot_bimanual_self \
    --action_mode=relative_pose \
    --action_chunk_size=30

# 输出：/path/to/data/meta/stats_relative_chunk30.json
```

### 混训配置

```python
# 两个数据集用相同的局部坐标系表示
data_cfg = {
    "action_mode": "relative_pose",
    "action_chunk_size": 30,
}

galbot_dataset = LeRobotSingleDataset(
    transforms=galbot_config.transform(data_cfg),
)

fastumi_dataset = LeRobotSingleDataset(
    transforms=fastumi_config.transform(data_cfg),
)

# 混训支持跨本体学习！
```

## 📝 待完成工作

### 高优先级
- [ ] Phase 4: 完成测试验证（进行中）
- [ ] 标记旧 transform 为 deprecated
- [ ] 更新主 README.md

### 中优先级
- [ ] 创建通用 stats 计算脚本
- [ ] 添加配置文档
- [ ] Galbot 回归测试（可选）

### 低优先级
- [ ] 提取 DerivedKeysTransform 到共享模块
- [ ] 文档完善
- [ ] 性能优化

## 🎉 总结

**已完成进度：** ~85%

**核心架构完成：** ✅
- 统一 ActionChunkTransform
- 局部坐标系表示
- 配置透传机制
- Stats 计算支持

**可用性：**
- ✅ FastUMI 数据加载
- ✅ 配置系统
- ⏳ 测试验证中
- ⏳ 文档待完善

**关键成果：**
1. 支持跨本体学习的局部坐标系表示
2. 统一的配置接口（Galbot + FastUMI）
3. 可扩展架构（易于添加新数据集）
