# Multi-Dataset Mixed Training Design - 模块化混训架构设计

## 🎯 核心目标

支持不同格式数据集的混合训练，确保：
1. **模块化** - 每个数据集独立配置，互不影响
2. **可扩展** - 添加新数据集只需新增配置，无需修改核心代码
3. **优雅** - Stats 离线计算，训练时直接加载
4. **类型安全** - 不同数据格式通过 transform 统一到标准 batch

---

## 📐 整体架构

```
数据集注册层 (data_registry/)
    ↓
数据格式适配层 (transform pipeline)
    ↓
Stats 计算层 (offline scripts)
    ↓
混训采样层 (dataset mixture)
    ↓
标准 Batch 输出 (unified format)
    ↓
模型训练 (QwenGR00T)
```

---

## 🗂️ 目录结构设计

```
examples/
├── GalbotBimanualRelative/          # Galbot 数据集
│   ├── train_files/
│   │   ├── data_registry/
│   │   │   └── data_config.py       # Galbot 配置
│   │   ├── compute_galbot_stats_self_mode.py  # Stats 计算脚本
│   │   └── starvla_qwengroot_galbot.yaml
│   └── offline_eval/
│
├── FastUMI/                          # FastUMI 数据集
│   ├── train_files/
│   │   ├── data_registry/
│   │   │   └── data_config.py       # FastUMI 配置
│   │   ├── compute_fastumi_stats.py  # Stats 计算脚本
│   │   └── starvla_qwengroot_fastumi.yaml
│   └── README.md
│
└── MixedTraining/                    # 混训示例
    ├── train_files/
    │   ├── data_registry/
    │   │   └── mixed_data_config.py  # 混训配置（引用上面的）
    │   └── starvla_mixed_train.yaml
    └── README.md
```

---

## 🔧 模块化设计方案

### 1. 数据集配置注册（Dataset-Specific）

每个数据集维护自己的配置，完全独立。

#### **Galbot 配置示例**

```python
# examples/GalbotBimanualRelative/train_files/data_registry/data_config.py

class GalbotBimanualDataConfig:
    """Galbot bimanual robot configuration."""
    
    # Raw keys from parquet (v3.0 format, structured fields)
    state_keys = [
        "state.left_abs_pos",
        "state.left_abs_ori_6d",
        "state.left_gripper",
        "state.right_abs_pos",
        "state.right_abs_ori_6d",
        "state.right_gripper",
    ]
    
    action_keys = [
        "action.left_abs_pos",
        "action.left_abs_ori_6d",
        "action.left_gripper",
        "action.right_abs_pos",
        "action.right_abs_ori_6d",
        "action.right_gripper",
    ]
    
    video_keys = ["video.left_wrist", "video.right_wrist"]
    
    # No output_keys needed (structured fields already)
    # No derived_keys needed
    
    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=list(range(0, 30, 2)),  # 15 actions @ 2 step interval
                modality_keys=self.action_keys,
            ),
        }
    
    def transform_pipeline(self):
        """Galbot-specific transform pipeline."""
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys + self.action_keys),
                SelfModeActionTransform(
                    apply_to=self.action_keys,
                    self_mode="delta",  # Frame-to-frame delta
                ),
                StateActionTransform(
                    apply_to=self.state_keys + self.action_keys,
                    normalization_modes={...},
                ),
            ]
        )
    
    def stats_path(self) -> Path:
        """Path to precomputed stats file."""
        return Path("meta/stats_delta_chunk30.json")


# Export for registry
ROBOT_TYPE_CONFIG_MAP = {
    "galbot_bimanual": GalbotBimanualDataConfig(),
}
```

---

#### **FastUMI 配置示例**

```python
# examples/FastUMI/train_files/data_registry/data_config.py

class FastUMIDualArmDataConfig:
    """FastUMI dual-arm configuration."""
    
    # Raw keys from parquet (v2.1 format, flat vectors)
    raw_state_keys = ["observation.state"]  # [14D]
    raw_action_keys = ["action"]            # [14D]
    
    # Derived keys after transform
    state_keys = [
        "state.left_arm",
        "state.left_ori_6d",
        "state.left_gripper",
        "state.right_arm",
        "state.right_ori_6d",
        "state.right_gripper",
    ]
    
    action_keys = [
        "action.left_arm",
        "action.left_ori_6d",
        "action.left_gripper",
        "action.right_arm",
        "action.right_ori_6d",
        "action.right_gripper",
    ]
    
    video_keys = [
        "video.left_camera_rgb_image",
        "video.right_camera_rgb_image",
    ]
    
    # Derivation rules for DerivedKeysTransform
    derived_keys = {
        "state.left_arm": {
            "type": "keep_from_origin",
            "source_key": "observation.state",
            "start": 0,
            "end": 3,
        },
        "state.left_ori_6d": {
            "type": "rotation_transform",
            "source_key": "observation.state",
            "start": 3,
            "end": 6,
            "source_repr": "euler_angles_xyz",  # RPY
            "target_repr": "rotation_6d",
        },
        "state.left_gripper": {
            "type": "keep_from_origin",
            "source_key": "observation.state",
            "start": 6,
            "end": 7,
        },
        # ... similar for right arm and action
    }
    
    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=[0],
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=self.raw_state_keys,
                output_keys=self.state_keys,  # After derivation
                derived_keys=self._derived_keys_for("state"),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(0, 32, 2)),  # 16 actions @ 2 step interval
                modality_keys=self.raw_action_keys,
                output_keys=self.action_keys,  # After derivation
                derived_keys=self._derived_keys_for("action"),
            ),
        }
    
    def transform_pipeline(self):
        """FastUMI-specific transform pipeline."""
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.raw_state_keys + self.raw_action_keys),
                DerivedKeysTransform(
                    apply_to=self.raw_state_keys + self.raw_action_keys,
                    derived_keys=self.derived_keys,
                ),
                RelativePoseActionTransform(
                    state_keys=self.state_keys,
                    action_keys=self.action_keys,
                ),
                StateActionTransform(
                    apply_to=self.state_keys + self.action_keys,
                    normalization_modes={...},
                ),
            ]
        )
    
    def stats_path(self) -> Path:
        """Path to precomputed stats file."""
        return Path("meta/relative_stats.json")


# Export for registry
ROBOT_TYPE_CONFIG_MAP = {
    "fastumi_dual_arm": FastUMIDualArmDataConfig(),
}
```

---

### 2. Stats 离线计算脚本

每个数据集提供独立的 stats 计算脚本。

#### **Galbot Stats 计算**

```bash
# examples/GalbotBimanualRelative/train_files/compute_galbot_stats_self_mode.py

python compute_galbot_stats_self_mode.py \
    --dataset_dir /path/to/galbot_data \
    --self_mode delta \
    --chunk_size 30 \
    --output meta/stats_delta_chunk30.json
```

**关键：**
- 使用与训练相同的 transform pipeline
- 统计 **transform 后** 的数据分布
- 输出到数据集的 `meta/` 目录

#### **FastUMI Stats 计算**

```bash
# examples/FastUMI/train_files/compute_fastumi_stats.py

python compute_fastumi_stats.py \
    --dataset_dir /path/to/fastumi_data \
    --mode relative_pose \
    --output meta/relative_stats.json
```

**关键：**
- 先做 DerivedKeysTransform（RPY → rot6d）
- 再做 RelativePoseActionTransform
- 统计最终的相对 action 分布

---

### 3. 混训配置（Dataset Mixture）

混训配置**引用**各数据集的配置，不重复定义。

```python
# examples/MixedTraining/train_files/data_registry/mixed_data_config.py

from examples.GalbotBimanualRelative.train_files.data_registry.data_config import (
    GalbotBimanualDataConfig,
    ROBOT_TYPE_CONFIG_MAP as GALBOT_CONFIG_MAP,
)
from examples.FastUMI.train_files.data_registry.data_config import (
    FastUMIDualArmDataConfig,
    ROBOT_TYPE_CONFIG_MAP as FASTUMI_CONFIG_MAP,
)

# Unified registry
ROBOT_TYPE_CONFIG_MAP = {
    **GALBOT_CONFIG_MAP,
    **FASTUMI_CONFIG_MAP,
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "galbot_bimanual": EmbodimentTag.NEW_EMBODIMENT,
    "fastumi_dual_arm": EmbodimentTag.NEW_EMBODIMENT,
}

# Mixed training dataset specification
DATASET_NAMED_MIXTURES = {
    "galbot_fastumi_mix": [
        ("galbot_lerobot_dual_cup_0529_359piece", 0.5, "galbot_bimanual"),
        ("Add_Rice_to_Rice_Cooker", 0.3, "fastumi_dual_arm"),
        ("Fold_the_T-shirt", 0.2, "fastumi_dual_arm"),
    ],
}
```

**格式：**
```python
(dataset_name, sampling_weight, robot_type)
```

**关键点：**
- `dataset_name`: 数据集目录名
- `sampling_weight`: 采样权重（归一化后）
- `robot_type`: 引用 `ROBOT_TYPE_CONFIG_MAP` 中的配置

---

### 4. 训练时数据加载

训练脚本保持通用，通过 `data_registry` 加载配置。

```python
# starVLA/training/train_starvla.py

from importlib import import_module

# Load data registry from config
data_registry_module = import_module(f"examples.{args.example_name}.train_files.data_registry.data_config")
ROBOT_TYPE_CONFIG_MAP = data_registry_module.ROBOT_TYPE_CONFIG_MAP
DATASET_NAMED_MIXTURES = data_registry_module.DATASET_NAMED_MIXTURES

# Create mixture dataset
mixture_spec = DATASET_NAMED_MIXTURES[args.mixture_name]
datasets = []

for dataset_name, weight, robot_type in mixture_spec:
    # Get robot-specific config
    robot_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    
    # Load dataset with robot-specific config
    dataset = LeRobotSingleDataset(
        dataset_path=f"{args.data_root}/{dataset_name}",
        modality_configs=robot_config.modality_config(),
        embodiment_tag=ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type],
        transforms=robot_config.transform_pipeline(),
    )
    
    # Dataset will load its precomputed stats from robot_config.stats_path()
    datasets.append((dataset, weight))

# Create mixture
train_dataset = LeRobotMixtureDataset(datasets)
```

**关键：**
- 每个数据集使用自己的 `modality_config`
- 每个数据集使用自己的 `transform_pipeline`
- Stats 从各自的 `stats_path()` 加载
- 训练代码**零数据集特定逻辑**

---

## 🔄 完整工作流程

### Step 1: 注册新数据集

```bash
# 1. 创建数据集配置
examples/NewDataset/train_files/data_registry/data_config.py

# 2. 定义：
#    - modality_keys (raw) / output_keys (derived)
#    - derived_keys (if needed)
#    - transform_pipeline()
#    - stats_path()
```

### Step 2: 计算 Stats

```bash
# 运行离线 stats 计算脚本
python examples/NewDataset/train_files/compute_new_stats.py \
    --dataset_dir /path/to/new_data \
    --output /path/to/new_data/meta/stats.json
```

**脚本模板：**
```python
# compute_new_stats.py

# 1. 加载数据集配置
from data_registry.data_config import NewDatasetConfig

config = NewDatasetConfig()

# 2. 创建 dataloader（无 normalization）
dataset = LeRobotSingleDataset(
    dataset_path=args.dataset_dir,
    modality_configs=config.modality_config(),
    transforms=config.transform_pipeline_without_normalization(),  # 关键！
)

# 3. 遍历数据，收集统计
all_actions = []
for sample in tqdm(dataset):
    all_actions.append(sample['action'])

# 4. 计算统计值
stats = {
    "mean": np.mean(all_actions, axis=0).tolist(),
    "std": np.std(all_actions, axis=0).tolist(),
    "q01": np.quantile(all_actions, 0.01, axis=0).tolist(),
    "q99": np.quantile(all_actions, 0.99, axis=0).tolist(),
}

# 5. 保存
with open(args.output, "w") as f:
    json.dump(stats, f, indent=2)
```

### Step 3: 单数据集训练（验证）

```bash
python starVLA/training/train_starvla.py \
    --example_name NewDataset \
    --mixture_name single_new_dataset \
    --data_root /path/to/data
```

### Step 4: 混合训练

```bash
# 1. 创建混训配置
examples/MixedTraining/train_files/data_registry/mixed_data_config.py

# 2. 添加到 DATASET_NAMED_MIXTURES
DATASET_NAMED_MIXTURES = {
    "my_mix": [
        ("galbot_data", 0.4, "galbot_bimanual"),
        ("fastumi_data", 0.3, "fastumi_dual_arm"),
        ("new_data", 0.3, "new_dataset_type"),
    ],
}

# 3. 运行混训
python starVLA/training/train_starvla.py \
    --example_name MixedTraining \
    --mixture_name my_mix \
    --data_root /path/to/data
```

---

## 💡 关键设计原则

### 1. 数据集独立性

**每个数据集是独立单元：**
```
examples/DatasetName/
├── data_config.py        # 配置（self-contained）
├── compute_stats.py      # Stats 计算
├── README.md             # 数据格式说明
└── /path/to/data/
    └── meta/
        └── stats.json    # 预计算的 stats
```

**优点：**
- 添加新数据集不影响现有数据集
- 每个数据集可以独立测试
- 配置清晰，易于维护

### 2. Stats 离线计算

**不要在训练时计算 stats！**

**原因：**
- 混训时不同数据集有不同的 transform
- Stats 必须在 transform 后计算
- 训练时实时计算会非常慢
- 容易出错（transform 顺序不一致）

**正确流程：**
```
Offline: Data → Transform → Compute Stats → Save to meta/stats.json
Training: Load meta/stats.json → Use for normalization
```

### 3. Transform Pipeline 模块化

**每个数据集定义自己的 transform pipeline：**

```python
def transform_pipeline(self):
    return ComposedModalityTransform(
        transforms=[
            # 数据集特定的 transforms
            DatasetSpecificTransform(...),
            
            # 通用的 normalization（使用预计算的 stats）
            StateActionTransform(...),
        ]
    )

def transform_pipeline_without_normalization(self):
    """用于 stats 计算（不包含 normalization）"""
    return ComposedModalityTransform(
        transforms=[
            DatasetSpecificTransform(...),
            # NO StateActionTransform here!
        ]
    )
```

### 4. 配置即代码（Config as Code）

**不使用 YAML 配置数据集细节：**

❌ **不好（YAML）：**
```yaml
# 难以维护，逻辑分散
datasets:
  galbot:
    keys: [...]
    transforms:
      - type: SelfMode
        mode: delta
```

✅ **好（Python）：**
```python
# 类型安全，逻辑集中，可复用
class GalbotConfig:
    def transform_pipeline(self):
        return ComposedModalityTransform([
            SelfModeActionTransform(mode="delta")
        ])
```

---

## 📊 Stats 计算统一接口

为了标准化，提供一个通用的 stats 计算脚本模板：

```python
# starVLA/tools/compute_dataset_stats.py

import argparse
from importlib import import_module

def compute_stats(
    dataset_path: Path,
    robot_type: str,
    data_registry_module: str,
    output_path: Path,
    sample_ratio: float = 1.0,
):
    """通用的 stats 计算函数。
    
    Args:
        dataset_path: 数据集路径
        robot_type: 机器人类型（如 "galbot_bimanual"）
        data_registry_module: 配置模块路径（如 "examples.Galbot.train_files.data_registry.data_config"）
        output_path: 输出路径
        sample_ratio: 采样比例（大数据集可以只用部分数据计算）
    """
    # 1. 加载配置
    registry = import_module(data_registry_module)
    config = registry.ROBOT_TYPE_CONFIG_MAP[robot_type]
    
    # 2. 创建 dataset（不带 normalization）
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=config.modality_config(),
        transforms=config.transform_pipeline_without_normalization(),
    )
    
    # 3. 采样数据
    num_samples = int(len(dataset) * sample_ratio)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    # 4. 收集数据
    all_actions = []
    for idx in tqdm(indices):
        sample = dataset[idx]
        all_actions.append(sample['action'].numpy())
    
    # 5. 计算统计
    all_actions = np.concatenate(all_actions, axis=0)
    stats = {
        "mean": np.mean(all_actions, axis=0).tolist(),
        "std": np.std(all_actions, axis=0).tolist(),
        "min": np.min(all_actions, axis=0).tolist(),
        "max": np.max(all_actions, axis=0).tolist(),
        "q01": np.quantile(all_actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(all_actions, 0.99, axis=0).tolist(),
    }
    
    # 6. 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Stats saved to {output_path}")
    print(f"   Shape: {all_actions.shape}")
    print(f"   Mean range: [{stats['mean'][0]:.3f}, {stats['mean'][-1]:.3f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--robot_type", type=str, required=True)
    parser.add_argument("--data_registry", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    args = parser.parse_args()
    
    compute_stats(
        dataset_path=args.dataset_path,
        robot_type=args.robot_type,
        data_registry_module=args.data_registry,
        output_path=args.output,
        sample_ratio=args.sample_ratio,
    )
```

**使用示例：**
```bash
# Galbot
python starVLA/tools/compute_dataset_stats.py \
    --dataset_path /path/to/galbot_data \
    --robot_type galbot_bimanual \
    --data_registry examples.GalbotBimanualRelative.train_files.data_registry.data_config \
    --output /path/to/galbot_data/meta/stats_delta_chunk30.json

# FastUMI
python starVLA/tools/compute_dataset_stats.py \
    --dataset_path /path/to/fastumi_data \
    --robot_type fastumi_dual_arm \
    --data_registry examples.FastUMI.train_files.data_registry.data_config \
    --output /path/to/fastumi_data/meta/relative_stats.json \
    --sample_ratio 0.1  # 大数据集可以只用 10% 计算
```

---

## ✅ 优势总结

### 1. **模块化**
- 每个数据集独立维护
- 添加新数据集：只需新增配置文件
- 修改数据集：只影响该数据集

### 2. **可扩展**
- 新数据格式：实现新的 transform
- 新 robot type：注册到 config map
- 新混训组合：修改 mixture spec

### 3. **类型安全**
- Python config（不是 YAML）
- IDE 支持（自动补全、类型检查）
- 编译时错误检测

### 4. **易于测试**
- 单数据集测试：独立运行
- Stats 验证：离线计算，可重复
- Transform 验证：单独测试 pipeline

### 5. **训练代码零耦合**
- 训练脚本不知道数据集细节
- 所有逻辑在 data_config.py
- 易于维护和扩展

---

## 🎯 实施建议

### Phase 2.1: 添加 FastUMI 示例
1. 创建 `examples/FastUMI/train_files/data_registry/data_config.py`
2. 实现 FastUMIDualArmDataConfig
3. 复制或实现 DerivedKeysTransform

### Phase 2.2: Stats 计算工具
1. 创建通用 stats 计算脚本 `starVLA/tools/compute_dataset_stats.py`
2. 为 FastUMI 计算 stats
3. 验证 stats 格式正确

### Phase 2.3: 混训示例
1. 创建 `examples/MixedTraining/`
2. 配置 Galbot + FastUMI 混训
3. 测试混训流程

### Phase 3: 文档
1. 编写 `docs/ADD_NEW_DATASET.md` 指南
2. 说明 stats 计算流程
3. 提供混训配置示例

---

这个设计的核心思想是：**每个数据集是独立的、自包含的模块，通过标准接口与训练系统交互**。这样就能实现优雅的多数据集混训！

你觉得这个设计如何？有什么需要调整的地方吗？
