# try → umi_pretrain Integration Plan

**Goal:** Merge try branch features into umi_pretrain while preserving all existing Galbot functionality and maintaining modular design principles.

**Core Design Principle:** Different data formats → Dataset-specific transforms → Unified batch format → Unified model training

---

## Phase 1: Universal Infrastructure (Must Debug & Test)

### 1.1 LeRobot v2.1 Support ✅ MUST DEBUG

**File:** `starVLA/dataloader/gr00t_lerobot/datasets.py`

**Changes:**
```python
# Add constants
LE_ROBOT2_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT2_EPISODES_FILENAME = "meta/episodes.jsonl"
LE_ROBOT2_DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LE_ROBOT2_DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

# Add helper functions
def _normalize_lerobot_version(version: str | None) -> str | None:
    """Normalize version string to v2.1 or v3.0"""
    
def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL format used in v2.1"""
```

**Backward Compatibility:**
- Auto-detect version: check for `meta/episodes.jsonl` (v2.1) vs `meta/episodes/*/*.parquet` (v3.0)
- Fallback to v3.0 if not specified
- All existing v3.0 paths remain unchanged

**Testing:**
- Load a v2.1 dataset
- Load a v3.0 dataset (Galbot)
- Verify both work correctly

---

### 1.2 Extend ModalityConfig ✅ MUST DEBUG

**File:** `starVLA/dataloader/gr00t_lerobot/datasets.py`

**Changes:**
```python
class ModalityConfig(BaseModel):
    modality_keys: list[str]  # Existing: raw keys from parquet
    output_keys: list[str] | None = None  # NEW: keys after transform
    derived_keys: dict[str, dict] = Field(default_factory=dict)  # NEW: derivation rules
    # ... existing fields ...
```

**Semantics:**
- `modality_keys`: Raw keys read from parquet files
- `output_keys`: Keys produced by transforms and packed into batch
- `derived_keys`: Specifications for deriving output_keys from modality_keys
- **Fallback:** If `output_keys` is None, use `modality_keys` (backward compatible)

**Update Sample Packing:**
```python
# In LeRobotSingleDataset.__getitem__()
# OLD:
action_keys = config.modality_keys

# NEW:
action_keys = config.output_keys if config.output_keys is not None else config.modality_keys
```

**Preserve Existing Branches:**
```python
# Must keep all existing logic:
if hasattr(config, 'self_mode') and config.self_mode != 'abs':
    # Existing Galbot self-mode handling
    ...

if 'relative_pose' in action_keys or 'action.relative_pose' in action_keys:
    # Existing relative pose handling
    ...
```

**Testing:**
- Galbot configs (no output_keys): should fallback to modality_keys
- FastUMI configs (with output_keys): should use output_keys
- Verify both pack samples correctly

---

### 1.3 Add Metadata Helpers ✅ MUST DEBUG

**File:** `starVLA/dataloader/gr00t_lerobot/datasets.py`

**Add Functions:**
```python
def _infer_feature_width(feature: dict, key: str) -> int:
    """Infer dimension from feature metadata"""

def _load_lerobot_modality_metadata(
    modality_meta_path: Path,
    info_meta_path: Path | None = None,
) -> LeRobotModalityMetadata:
    """Load modality.json with fallback to info.json"""

def _build_direct_lerobot_modality_metadata(
    modality_configs: dict,
    info_meta: dict,
) -> LeRobotModalityMetadata:
    """Build metadata directly from info.json when modality.json missing"""

def _fixed_rotation_6d_stats() -> dict:
    """Return fixed stats for rotation_6d (always [-1, 1])"""
```

**Use Cases:**
- Galbot: Load existing `meta/modality.json`
- FastUMI: Build from `meta/info.json` if `modality.json` missing
- Both paths must work independently

**Testing:**
- Galbot dataset with modality.json: use existing path
- FastUMI dataset without modality.json: use fallback path
- Verify metadata loaded correctly in both cases

---

### 1.4 Test Galbot Compatibility ✅ MUST PASS

**Critical Regression Tests:**

```bash
# 1. GalbotBimanualRelative - delta mode
cd /mnt/home/liuyi/project/starVLA
bash examples/GalbotBimanualRelative/train_files/run_galbot_bimanual_self.sh

# 2. GalbotBimanualNoState - vision-only
bash examples/GalbotBimanualNoState/train_files/run_galbot_bimanual_no_state.sh

# 3. Stats computation
python examples/GalbotBimanualRelative/train_files/compute_galbot_stats_self_mode.py \
    --dataset_dir /path/to/galbot_data \
    --self_mode delta \
    --chunk_size 30

# 4. Offline eval
bash examples/GalbotBimanualRelative/offline_eval/run_offline_eval.sh
```

**Success Criteria:**
- All scripts run without errors
- Training starts and logs metrics
- Stats file generated correctly
- Eval loads checkpoint and runs inference

**If ANY test fails:** Debug Phase 1 before proceeding to Phase 2

---

## Phase 2: FastUMI Support (Add, But Not Fully Debug)

### 2.1 Add FastUMI Example 🟡 MINIMAL INTEGRATION

**Goal:** Add FastUMI as a reference example, mark as "needs debugging"

**Directory Structure:**
```
examples/FastUMI/
├── README.md  # ⚠️ IMPORTANT: Mark as "Work in Progress"
└── train_files/
    └── data_registry/
        └── data_config.py  # Copy from try branch
```

**README.md Template:**
```markdown
# FastUMI Example (Work in Progress)

⚠️ **Status:** This example is added for reference but has NOT been fully tested yet.

## Data Format

FastUMI uses a flat vector format:
- `observation.state`: [14D] = [left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper]
- `action`: [14D] = same structure

This differs from Galbot's structured fields (state.left_abs_pos, etc.)

## Key Features

1. **DerivedKeysTransform**: Splits flat vectors into structured keys
2. **RPY → rotation_6d**: Converts Euler angles to continuous representation
3. **RelativePoseActionTransform**: Computes actions relative to current state

## Usage

⚠️ This example requires further debugging before production use.

For tested examples, see:
- `examples/GalbotBimanualRelative/`
- `examples/GalbotBimanualNoState/`
```

**Action:**
- Copy `data_config.py` from try branch as-is
- Keep `DerivedKeysTransform` embedded in the config (don't extract yet)
- No training scripts yet (user will add later)

---

### 2.2 Extract DerivedKeysTransform (Optional) 🟡 OPTIONAL

**Only do this if Phase 1 is stable and you have time**

**File:** `starVLA/dataloader/gr00t_lerobot/transform/derived_keys.py`

**Extract from:** `examples/FastUMI/train_files/data_registry/data_config.py`

**Rationale:**
- If another dataset needs similar flat-vector splitting, it's reusable
- But since only FastUMI uses it now, keeping it local is fine

**Decision:** Skip for now, revisit when second dataset needs it

---

## Phase 3: Documentation & Validation

### 3.1 Document Modular Design ✅ CRITICAL

**File:** `docs/DATA_INTEGRATION_GUIDE.md`

**Content:**
```markdown
# Data Integration Guide

## Philosophy: Modular Transform Design

StarVLA separates concerns:

1. **Dataset-specific layer** (`examples/*/data_config.py`)
   - Handles raw data format
   - Defines transforms
   - Outputs standardized batch

2. **Unified training layer** (`train_starvla.py`)
   - Receives standardized batch
   - Trains QwenGR00T model
   - No dataset-specific code

## Two Reference Implementations

### Galbot: Structured Fields + Self-Mode

**Data format:**
```
state.left_abs_pos: [3]
state.left_abs_ori_6d: [6]  # Already rotation_6d
action.left_abs_pos: [T, 3]
...
```

**Transform pipeline:**
```
Raw fields → SelfModeActionTransform (delta/chunk_relative) 
          → StateActionTransform (normalize) 
          → Batch [B, T, 20]
```

**Use case:** Action sequences relative to previous frames

---

### FastUMI: Flat Vectors + Relative Pose

**Data format:**
```
observation.state: [14]  # [xyz, rpy, gripper] * 2
action: [14]
```

**Transform pipeline:**
```
Flat vectors → DerivedKeysTransform (split + RPY→rot6d)
            → RelativePoseActionTransform (relative to state)
            → StateActionTransform (normalize)
            → Batch [B, T, 14]
```

**Use case:** Actions relative to current robot state

---

## Adding a New Dataset

### Step 1: Create data_config.py

```python
# examples/MyDataset/train_files/data_registry/data_config.py

class MyDatasetConfig:
    # Define your raw keys
    raw_state_keys = ["observation.robot_state"]
    raw_action_keys = ["action.commands"]
    
    # Define your output keys
    state_keys = ["state.joint_pos", "state.joint_vel"]
    action_keys = ["action.joint_pos", "action.joint_vel"]
    
    # Define how to derive output from raw
    derived_keys = {
        "state.joint_pos": {
            "type": "keep_from_origin",
            "source_key": "observation.robot_state",
            "start": 0,
            "end": 7,
        },
        # ... more derivations
    }
    
    def modality_config(self):
        return {
            "state": ModalityConfig(
                modality_keys=self.raw_state_keys,
                output_keys=self.state_keys,  # NEW!
                derived_keys=self._derived_keys_for("state"),
            ),
            # ...
        }
```

### Step 2: Choose Transform Strategy

**Option A: Self-Mode (like Galbot)**
- Use if: Actions should be relative to previous timesteps
- Use: `SelfModeActionTransform`

**Option B: Relative Pose (like FastUMI)**
- Use if: Actions should be relative to current state
- Use: `RelativePoseActionTransform`

**Option C: Custom**
- Use if: Neither fits
- Implement: Custom transform in your data_config.py

### Step 3: Compute Stats

Choose the appropriate stats script:
- Self-mode: Use `compute_galbot_stats_self_mode.py` as reference
- Relative-pose: Use `compute_relative_stats.py` as reference

### Step 4: Test

```bash
# Smoke test the dataloader
python starVLA/dataloader/lerobot_datasets.py \
    --config_yaml examples/MyDataset/train_files/config.yaml

# Run training
bash examples/MyDataset/train_files/run_train.sh
```
```

---

### 3.2 Full Regression Test ✅ MUST PASS

**Run all tests from Phase 1.4 again:**
- Galbot relative training
- Galbot no-state training
- Stats computation
- Offline eval

**Success Criteria:**
- Zero regressions
- All original functionality intact
- New features accessible but don't interfere

---

## Implementation Checklist

### Phase 1: Universal Infrastructure (Must Complete)
- [ ] 1.1: Add LeRobot v2.1 support + version detection
- [ ] 1.2: Add output_keys/derived_keys to ModalityConfig
- [ ] 1.3: Add metadata helper functions
- [ ] 1.4: Test Galbot examples (ALL MUST PASS)

### Phase 2: FastUMI Support (Minimal)
- [ ] 2.1: Copy FastUMI data_config.py with "WIP" README
- [ ] 2.2: (Optional) Extract DerivedKeysTransform to shared module

### Phase 3: Documentation (Critical)
- [ ] 3.1: Write DATA_INTEGRATION_GUIDE.md
- [ ] 3.2: Full regression test suite
- [ ] 3.3: Update main README with integration notes

---

## Key Design Decisions

### ✅ Modular Separation
```
Dataset-specific transforms → Unified batch format → Model training
```

### ✅ Backward Compatibility
```python
# All new fields have fallback
output_keys = config.output_keys or config.modality_keys  # Fallback to old behavior
```

### ✅ Independent Paths
```python
# Galbot path (unchanged)
if self_mode == 'delta':
    SelfModeActionTransform()

# FastUMI path (new, isolated)
if action_repr == 'relative_pose':
    DerivedKeysTransform() + RelativePoseActionTransform()
```

### ✅ Clear Responsibility
- **datasets.py**: Universal infrastructure
- **data_config.py**: Dataset-specific logic
- **train_starvla.py**: Zero dataset-specific code

---

## Testing Strategy

### Unit Tests (Per Phase)
- Phase 1.1: Load v2.1 dataset, load v3.0 dataset
- Phase 1.2: Pack samples with/without output_keys
- Phase 1.3: Load metadata via different paths

### Integration Tests (End of Phase 1)
- Run Galbot training for 100 steps
- Compute stats and verify format
- Run offline eval on checkpoint

### Regression Tests (Phase 3)
- All Galbot scripts must produce identical results
- No change in checkpoint format
- No change in stats format

---

## Rollback Plan

If Phase 1 breaks Galbot:
1. Revert datasets.py changes
2. Debug in isolation with unit tests
3. Re-apply changes incrementally

If Phase 2 interferes with Galbot:
1. Remove FastUMI example
2. Phase 1 should still work independently

---

## Success Criteria

### Must Have (Phase 1)
- ✅ Galbot training works identically to before
- ✅ LeRobot v2.1 datasets can be loaded
- ✅ output_keys mechanism works with fallback

### Nice to Have (Phase 2)
- ✅ FastUMI example present as reference
- ⚠️ FastUMI training not required to work yet

### Critical (Phase 3)
- ✅ Documentation explains modular design
- ✅ Zero regressions in all tests
- ✅ Clear path for adding new datasets

---

## Timeline Estimate

- **Phase 1**: 4-6 hours (careful debugging required)
- **Phase 2**: 1-2 hours (minimal integration)
- **Phase 3**: 2-3 hours (documentation + full test)
- **Total**: ~8-11 hours for complete integration

---

## Notes

- Focus debugging effort on Phase 1 (universal infrastructure)
- Phase 2 (FastUMI) is added "as-is" for future use
- Phase 3 (docs) is critical for maintainability
- Always test Galbot after each change in Phase 1
