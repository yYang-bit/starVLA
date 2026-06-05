# Lightweight Testing Strategy

## 🎯 Goal: Test code changes without consuming GPU/CPU resources

## 📋 Testing Approach

### ✅ DO (Lightweight):
- Load 1-3 episodes from dataset
- Check metadata files exist
- Verify sample shapes and dtypes
- Print tensor dimensions
- Run smoke tests (instantiate classes, call methods once)

### ❌ DON'T (Heavy):
- Full training runs
- Loading entire datasets into memory
- Multi-GPU operations
- Long-running loops
- Model forward passes with large batches

---

## 🧪 Test Datasets

### v2.1 (FastUMI - Primary Focus)
```bash
/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker
```
- Format: episodes.jsonl
- Features: observation.state [14D], action [14D]
- This is the **main target** for v2.1 support

### v3.0 (Galbot - Regression Test)
```bash
/mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0529_359piece
```
- Format: episodes/*/*.parquet
- Features: state.left_abs_pos, action.left_abs_pos, etc.
- Must continue working after changes

---

## 🔧 Test Script Usage

### Run lightweight dataloader test:
```bash
cd /mnt/home/liuyi/project/starVLA

# Test both datasets (loads 3 episodes each)
python scripts/test_dataloader_lightweight.py

# Test only v2.1
python scripts/test_dataloader_lightweight.py --skip-v30

# Test only v3.0 (regression)
python scripts/test_dataloader_lightweight.py --skip-v21

# Custom paths
python scripts/test_dataloader_lightweight.py \
    --v21-fastumi /path/to/fastumi \
    --v30-galbot /path/to/galbot
```

---

## 📊 Test Checklist (Per Phase)

### Phase 1.1: LeRobot v2.1 Support
- [ ] Script detects v2.1 format (episodes.jsonl)
- [ ] Script detects v3.0 format (episodes parquet)
- [ ] Both datasets load without errors
- [ ] Sample shapes are correct

### Phase 1.2: output_keys Support
- [ ] Galbot (no output_keys): uses modality_keys fallback
- [ ] FastUMI (with output_keys): uses output_keys
- [ ] Sample packing works for both

### Phase 1.3: Metadata Helpers
- [ ] Load metadata from modality.json (Galbot)
- [ ] Build metadata from info.json (FastUMI)
- [ ] Feature widths inferred correctly

### Phase 1.4: Galbot Regression
- [ ] All 3 Galbot samples load correctly
- [ ] Action shapes match expected [T, 20]
- [ ] State shapes match expected [9] or [0]

---

## ⚡ Quick Smoke Tests (No GPU)

### Test 1: Import modules
```python
python -c "from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset; print('✅ Import OK')"
```

### Test 2: Load info.json
```python
python -c "
import json
with open('/mnt/project/public/umi_data_from_web/fastumi_data/dual_arm/Add_Rice_to_Rice_Cooker/meta/info.json') as f:
    info = json.load(f)
print(f'✅ Episodes: {info[\"total_episodes\"]}')
"
```

### Test 3: Check version detection
```python
python -c "
from pathlib import Path
from starVLA.dataloader.gr00t_lerobot.datasets import _normalize_lerobot_version
print(_normalize_lerobot_version('v2.1'))
print(_normalize_lerobot_version('v3.0'))
print('✅ Version detection OK')
"
```

---

## 🎬 Incremental Testing Workflow

### After each code change:

1. **Quick check:**
   ```bash
   python -c "from starVLA.dataloader.gr00t_lerobot.datasets import *; print('✅')"
   ```

2. **Light test:**
   ```bash
   python scripts/test_dataloader_lightweight.py --skip-v21  # Test v3.0 only
   ```

3. **Full test:**
   ```bash
   python scripts/test_dataloader_lightweight.py  # Test both
   ```

4. **If all pass:** Proceed to next change

5. **If any fail:** Debug immediately before moving on

---

## 🛡️ Safety Guards

The test script:
- ✅ Only loads 3 episodes max
- ✅ No model instantiation
- ✅ No GPU operations
- ✅ No training loops
- ✅ Exits quickly (<30 seconds)
- ✅ Prints progress clearly

---

## 📝 Expected Output

```
============================================================
  Testing: FastUMI (LeRobot v2.1)
============================================================

📁 Meta files:
  ✅ info.json
  ✅ episodes.jsonl
  ✅ tasks.jsonl
  ❌ modality.json
  ❌ stats.json

📊 Dataset info:
  Total episodes: 1104
  Total frames: 476548
  FPS: 15
  Version: LeRobot v2.1 (episodes.jsonl)

🔑 Available features (6):
  - observation.state: shape=[14], dtype=float32
  - action: shape=[14], dtype=float32
  - observation.images.left_camera_rgb_image: ...
  - observation.images.right_camera_rgb_image: ...
  - episode_index: shape=[], dtype=int64
  - frame_index: shape=[], dtype=int64

🧪 Testing dataset instantiation...
  Video keys: ['observation.images.left_camera_rgb_image', ...]
  State keys: ['observation.state']
  Action keys: ['action']
  ✅ Dataset created successfully
  Total samples: 11040

🎯 Loading first sample...
  Sample keys: ['action', 'state', 'video', ...]
    action: torch.Size([15, 14]) torch.float32
    state: torch.Size([14]) torch.float32
    video: list[2]

📦 Loading 3 samples...
  Sample 0: action shape = torch.Size([15, 14])
  Sample 1: action shape = torch.Size([15, 14])
  Sample 2: action shape = torch.Size([15, 14])

✅ Dataset test PASSED
```

---

## 🚨 What to Watch For

### Phase 1.1 (v2.1 Support)
- ⚠️ `episodes.jsonl` vs `episodes/*/*.parquet` detection
- ⚠️ Episode length field name (`length` vs `num_frames`)
- ⚠️ Data path format strings

### Phase 1.2 (output_keys)
- ⚠️ Backward compatibility: Galbot should still use modality_keys
- ⚠️ Sample packing logic: check both paths work

### Phase 1.3 (Metadata)
- ⚠️ Missing modality.json should fallback to info.json
- ⚠️ Feature width inference for different shapes

### Phase 1.4 (Regression)
- ⚠️ Galbot sample shapes must match exactly
- ⚠️ No changes in action/state dimensions
- ⚠️ No changes in tensor dtypes

---

## 💾 Resource Usage Estimate

Per test run:
- **Memory:** <2GB RAM
- **CPU:** 1-2 cores for <30 seconds
- **GPU:** None
- **Disk I/O:** Minimal (reads 3 episodes)
- **Network:** None

Safe to run on shared server! ✅
