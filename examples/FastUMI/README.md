# FastUMI Dataset Configuration

⚠️ **Status:** Work in Progress - Configuration complete, needs full testing

## Overview

FastUMI uses a **flat vector format** where robot state and actions are stored as single 14-dimensional vectors. This differs from Galbot's structured field format.

## Data Format

### LeRobot v2.1 Structure
```
meta/
  ├── info.json           ✅ (version, features, fps)
  ├── episodes.jsonl      ✅ (episode list)
  ├── tasks.jsonl         ✅ (task descriptions)
  └── stats.json          ❌ (needs computation)

data/
  └── chunk-*/episode_*.parquet  ✅ (raw data)
```

### Raw Data Format
```python
observation.state: [14D] float32
  [0:3]   left_xyz
  [3:6]   left_rpy      ← Euler angles (XYZ convention)
  [6:7]   left_gripper
  [7:10]  right_xyz
  [10:13] right_rpy
  [13:14] right_gripper

action: [14D] float32  (same structure)
```

### Derived Keys (After Transform)
```python
state.left_arm: [3D]        # Sliced from observation.state[0:3]
state.left_ori_6d: [6D]     # RPY→rotation_6d from observation.state[3:6]
state.left_gripper: [1D]    # Sliced from observation.state[6:7]
# ... same for right arm and action
```

---

## Transform Pipeline

### 1. DerivedKeysTransform
**Purpose:** Split flat vectors + convert rotations

```python
observation.state [14D]
    ↓
DerivedKeysTransform
    ├─ Slice [0:3] → state.left_arm [3D]
    ├─ RPY→rot6d [3:6] → state.left_ori_6d [6D]
    ├─ Slice [6:7] → state.left_gripper [1D]
    └─ ... (right arm)
```

**Key Features:**
- Handles flat vector slicing
- Converts RPY (Euler angles) to rotation_6d
- Supports quaternion → rotation_6d
- Drops source keys after derivation

### 2. RelativePoseActionTransform (Optional)
**Purpose:** Compute actions relative to current state

```python
action.left_arm (absolute)
    ↓
RelativePoseActionTransform
    ↓
action.left_arm (relative to state.left_arm)
```

**Use when:** `action_chunk_representation = "relative_pose"`

### 3. StateActionTransform
**Purpose:** Normalize to [-1, 1]

Uses precomputed stats from `meta/stats.json`

---

## Key Differences from Galbot

| Aspect | Galbot | FastUMI |
|--------|--------|---------|
| **Data Format** | Structured fields | Flat vectors |
| **Version** | LeRobot v3.0 | LeRobot v2.1 |
| **Raw Keys** | `state.left_abs_pos` | `observation.state` |
| **Rotation Format** | rotation_6d (stored) | RPY → rotation_6d (derived) |
| **Transform** | SelfModeActionTransform | DerivedKeysTransform |
| **Action Mode** | delta (frame-to-frame) | relative_pose (vs state) |
| **Stats File** | `stats_delta_chunk30.json` | `stats.json` or `relative_stats.json` |

---

## Usage

### Step 1: Compute Stats (Required)

```bash
# Using the generic stats computation tool
python starVLA/tools/compute_dataset_stats.py \
    --dataset_path /path/to/fastumi_data/Add_Rice_to_Rice_Cooker \
    --robot_type fastumi_dual_arm \
    --data_registry examples.FastUMI.train_files.data_registry.data_config \
    --output /path/to/fastumi_data/Add_Rice_to_Rice_Cooker/meta/stats.json \
    --sample_ratio 0.1  # Use 10% of data for large datasets
```

**What it does:**
1. Loads raw `observation.state` and `action`
2. Applies DerivedKeysTransform (split + RPY→rot6d)
3. Applies RelativePoseActionTransform (if configured)
4. Computes statistics (mean, std, q01, q99)
5. Saves to `meta/stats.json`

### Step 2: Training (Single Dataset)

```bash
python starVLA/training/train_starvla.py \
    --example_name FastUMI \
    --mixture_name single_task \
    --data_root /path/to/fastumi_data
```

### Step 3: Mixed Training (with Galbot)

See `examples/MixedTraining/` (to be created)

---

## Configuration Details

### modality_keys vs output_keys

```python
ModalityConfig(
    modality_keys=["observation.state"],  # What to load from parquet
    output_keys=["state.left_arm", ...],  # What to pack into batch
    derived_keys={...},                    # How to transform
)
```

**Why separate?**
- `modality_keys`: Dataloader reads raw flat vector
- `derived_keys`: Transform splits it into structured keys
- `output_keys`: Sample packing uses structured keys

### derived_keys Format

```python
derived_keys = {
    "output_key_name": {
        "type": "keep_from_origin",      # or "transform_from_origin"
        "source_key": "observation.state",
        "start": 0,
        "end": 3,
        # For rotation transform:
        "from": "rpy",                   # or "quaternion"
        "to": "rotation_6d",
        "convention": "XYZ",
    }
}
```

---

## Available Tasks

```python
FASTUMI_DUAL_ARM_TASKS = [
    "Add_Rice_to_Rice_Cooker",
    "Arrange_Toothbrush_and_Toothpaste",
    "Clean_Desktop",
    "Dispose_of_Desktop_Debris",
    "Fold_the_Jeans",
    "Fold_the_Suit",
    "Fold_the_T-shirt",
    "Open_Double_Door_Cabinet",
]
```

---

## Troubleshooting

### "Missing stats.json"
**Solution:** Run stats computation script (Step 1 above)

### "Key observation.state not found"
**Issue:** Dataset might be v3.0 format (not v2.1)  
**Solution:** Check `meta/` for `episodes.jsonl` (v2.1) vs `episodes/*/*.parquet` (v3.0)

### "RPY conversion failed"
**Issue:** rotation_6d dimension mismatch  
**Solution:** Verify slice ranges in `derived_keys` ([3:6] for RPY)

### "output_keys not matching derived_keys"
**Issue:** Configuration mismatch  
**Solution:** Ensure all `output_keys` have corresponding `derived_keys` entries

---

## Next Steps

1. ⏳ **Compute stats** for target FastUMI datasets
2. ⏳ **Test dataloader** with single task
3. ⏳ **Validate transforms** produce correct shapes
4. ⏳ **Run training** for 100 steps to verify
5. ⏳ **Add to mixed training** with Galbot

---

## References

- **Design Doc:** `MIXED_TRAINING_DESIGN.md`
- **Integration Plan:** `INTEGRATION_PLAN.md`
- **Phase 1.2 Changes:** `PHASE1_2_CHANGES.md` (output_keys support)
- **Phase 1.3 Changes:** `PHASE1_3_CHANGES.md` (metadata fallback)

---

**Note:** This configuration is complete but has NOT been fully tested. Stats computation and dataloader testing are required before production use.
