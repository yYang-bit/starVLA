# umi_pretrain Try Feature Integration Plan

## Goal

Use `umi_pretrain` as the stable base and preserve all existing `liuyi_dev` / `umi_pretrain` behavior, especially the `GalbotBimanual*` training examples and naming conventions. Extract only the new, useful functionality from `try`, mainly the FastUMI data path and its minimal dataloader support.

## Must Preserve

- `examples/GalbotBimanualNoState/**`
- `examples/GalbotBimanualRelative/**`
- Existing Galbot training script names, config names, run IDs, and README guidance
- Existing `offline_eval` tools under `GalbotBimanual*`
- Existing `compute_galbot_stats_self_mode.py` stats tools under `GalbotBimanual*`
- `SelfModeActionTransform`
- `rotation_utils.py`
- `data_name` override in `starVLA/dataloader/lerobot_datasets.py`
- `stats_path` override and current relative/self-mode behavior
- Current `.gitignore` behavior for Markdown files
- Existing wandb tracker guard in `train_starvla.py`

## Do Not Import From try

- `galbot/validate.py`: `umi_pretrain` already has Galbot-specific `offline_eval`.
- `galbot/compute_relative_stats.py`: Galbot self-mode already has dedicated stats scripts. The try script is FastUMI/relative-stats oriented and should not replace Galbot tools.
- Deletions of `examples/GalbotBimanualNoState/**` or `examples/GalbotBimanualRelative/**`
- Deletion of `self_mode_action.py` or `rotation_utils.py`
- Removal of `data_name` override
- Changes that re-ignore `*.md` / nested Markdown files
- Temporary training script edits from `try`, including duplicated `freeze_module_list` and hard-coded experiment paths
- Removal of wandb `trackers` guard in `train_starvla.py`

## Features To Extract From try

### 1. FastUMI Example

Add a new FastUMI example without modifying Galbot examples:

```text
examples/FastUMI/train_files/data_registry/data_config.py
```

The FastUMI config should include:

- `FastUMIDualArmDataConfig`
- `FASTUMI_DUAL_ARM_TASKS`
- Dual camera video keys
- Raw low-dimensional keys:
  - `observation.state`
  - `action`
- Derived state/action keys:
  - `state.left_arm`
  - `state.left_ori_6d`
  - `state.left_gripper`
  - `state.right_arm`
  - `state.right_ori_6d`
  - `state.right_gripper`
  - corresponding `action.*` keys
- RPY to `rotation_6d`
- `action_indices = list(range(0, 32, 2))`
- `q99` normalization for arm position keys
- `min_max` normalization for gripper keys

Keep `DerivedKeysTransform` local to the FastUMI config at first. Once FastUMI runs, consider moving it into a shared transform module.

### 2. `ModalityConfig.output_keys` And `derived_keys`

Extend `ModalityConfig` in `starVLA/dataloader/gr00t_lerobot/datasets.py`:

```python
output_keys: list[str] | None = None
derived_keys: dict[str, dict] = Field(default_factory=dict)
```

Semantics:

- `modality_keys`: raw keys read from parquet / dataset files
- `output_keys`: keys produced by transforms and packed into model samples
- fallback: if `output_keys` is unset, use `modality_keys`

This must be backward-compatible with all existing Galbot configs.

### 3. Pack Samples Using Output Keys

Update sample packing in `LeRobotSingleDataset` so action/state packing uses:

```python
config.output_keys or config.modality_keys
```

This is required because FastUMI reads raw vectors but trains on derived keys.

Do not remove existing branches for:

- `action.relative_pose`
- `relative_pose`
- `self_mode`
- `stats_path`
- Galbot action/state packing behavior

### 4. Derived Metadata And Statistics