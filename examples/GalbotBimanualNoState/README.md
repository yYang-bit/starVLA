# GalbotBimanualNoState - Vision-Only Baseline

This example is a **contrast experiment** for `GalbotBimanualRelative`. It trains the same bimanual robot control task but **without state input** (vision-only baseline).

## Key Differences from GalbotBimanualRelative

| Feature | GalbotBimanualRelative | GalbotBimanualNoState |
|---------|------------------------|------------------------|
| State input | ✅ 9-dim dual_relative_pose | ❌ None (vision-only) |
| Vision input | ✅ Dual wrist cameras | ✅ Dual wrist cameras |
| Action output | ✅ 20-dim (2×10) | ✅ 20-dim (2×10) |
| Self-mode | ✅ abs/delta/chunk_relative | ✅ abs/delta/chunk_relative |
| Dataset | galbot_lerobot_dual_cup_0526 | galbot_lerobot_dual_cup_0526 |
| Stats file | Shared | Shared |

## Quick Start

```bash
cd /mnt/home/liuyi/project/starVLA
bash examples/GalbotBimanualNoState/train_files/run_galbot_bimanual_no_state.sh
```

## Configuration

- **Config YAML**: `train_files/starvla_qwengroot_galbot_no_state.yaml`
  - `state_dim: 0`
  - `include_state: false`
  - `data_mix: galbot_bimanual_self_no_state_mix`

- **Data Registry**: `train_files/data_registry/data_config.py`
  - `state_keys = []` (empty)
  - Registers `galbot_bimanual_self_no_state` and `galbot_bimanual_self_no_state_mix`

- **Run Script**: `train_files/run_galbot_bimanual_no_state.sh`
  - Sets `--datasets.vla_data.include_state false`
  - Default `self_mode=delta`

## Experiment Purpose

Compare vision-only performance vs. vision+state to understand:
- How much proprioceptive state helps bimanual coordination
- Whether vision alone is sufficient for the dual-cup task
- Impact on sample efficiency and final performance

## Output

Checkpoints saved to: `/mnt/home/liuyi/project/starVLA/Checkpoints/galbot_delta_action/galbot_bimanual_no_state_MMDD/`

## Notes

- Stats file (`stats.json`) is shared with the original example
- All other training hyperparameters match the original
- The model architecture automatically handles `state=None` input
