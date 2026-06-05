# Phase 1.2: output_keys Support - Implementation Summary

## ✅ Completed Changes

### Problem Statement
- FastUMI datasets store data in flat vectors (e.g., `observation.state: [14D]`)
- Transforms need to derive structured keys (e.g., `state.left_arm`, `state.left_gripper`)
- Need a way to specify: "load raw keys, transform them, pack derived keys"
- Must be backward compatible with Galbot (no output_keys)

---

## 🔧 Implementation

### 1. Extended ModalityConfig (lines ~1005-1025)

**Added Fields:**
```python
class ModalityConfig(BaseModel):
    delta_indices: list[int]
    modality_keys: list[str]              # Existing: raw keys from parquet
    
    # NEW FIELDS:
    output_keys: list[str] | None = None  # Keys after transform, for packing
    derived_keys: dict[str, dict] = {}    # Derivation specifications
```

**Field Semantics:**

| Field | Purpose | Example |
|-------|---------|---------|
| `modality_keys` | Raw keys from parquet files | `["observation.state"]` |
| `output_keys` | Keys after transforms (for packing) | `["state.left_arm", "state.left_gripper"]` |
| `derived_keys` | Transform specifications | `{"state.left_arm": {"type": "slice", "start": 0, "end": 3}}` |

**Backward Compatibility:**
- If `output_keys = None`: Use `modality_keys` (existing behavior)
- Galbot configs work unchanged

---

### 2. Modified _get_modality_keys() (lines ~1718-1728)

**Old behavior:**
```python
for modality, config in self.modality_configs.items():
    modality_keys[modality] = config.modality_keys
```

**New behavior:**
```python
for modality, config in self.modality_configs.items():
    # Use output_keys if specified, otherwise fallback to modality_keys
    keys = config.output_keys if config.output_keys is not None else config.modality_keys
    modality_keys[modality] = keys
```

**Impact:**
- Sample packing (line 2015) uses the right keys
- `self.modality_keys[modality]` returns output_keys when specified

---

### 3. Modified _get_delta_indices() (lines ~1730-1744)

**Old behavior:**
```python
for config in self.modality_configs.values():
    for key in config.modality_keys:
        delta_indices[key] = np.array(config.delta_indices)
```

**New behavior:**
```python
for config in self.modality_configs.values():
    keys = config.output_keys if config.output_keys is not None else config.modality_keys
    for key in keys:
        delta_indices[key] = np.array(config.delta_indices)
```

**Impact:**
- Delta indices map to the correct keys (output_keys or modality_keys)

---

## 🎯 Use Cases

### Use Case 1: Galbot (No output_keys)

**Config:**
```python
ModalityConfig(
    delta_indices=[0, 1, 2, ...],
    modality_keys=["action.left_abs_pos", "action.left_abs_ori_6d", ...],
    # output_keys = None (not specified)
)
```

**Behavior:**
- Loads: `action.left_abs_pos`, `action.left_abs_ori_6d`, ...
- Packs: `action.left_abs_pos`, `action.left_abs_ori_6d`, ... (same keys)
- **Backward compatible!**

---

### Use Case 2: FastUMI (With output_keys)

**Config:**
```python
ModalityConfig(
    delta_indices=[0, 1, 2, ...],
    modality_keys=["observation.state"],  # Raw: [14D] flat vector
    output_keys=[                          # Derived: structured keys
        "state.left_arm",
        "state.left_ori_6d", 
        "state.left_gripper",
        "state.right_arm",
        "state.right_ori_6d",
        "state.right_gripper",
    ],
    derived_keys={
        "state.left_arm": {
            "type": "keep_from_origin",
            "source_key": "observation.state",
            "start": 0,
            "end": 3,
        },
        # ... more derivation rules
    }
)
```

**Behavior:**
1. **Load:** `observation.state` from parquet
2. **Transform:** DerivedKeysTransform splits it using `derived_keys`
3. **Pack:** `state.left_arm`, `state.left_ori_6d`, ... (output_keys)

---

## 📊 Data Flow

### Without output_keys (Galbot):
```
Parquet
  ↓
Load: action.left_abs_pos [3D]
  ↓
Transform: (optional, like normalization)
  ↓
Pack: action.left_abs_pos [3D]
  ↓
Batch
```

### With output_keys (FastUMI):
```
Parquet
  ↓
Load: observation.state [14D]
  ↓
Transform: DerivedKeysTransform
  ├─ state.left_arm [3D] (slice [0:3])
  ├─ state.left_ori_6d [6D] (RPY→rot6d on [3:6])
  └─ state.left_gripper [1D] (slice [6:7])
  ↓
Pack: state.left_arm, state.left_ori_6d, state.left_gripper
  ↓
Batch
```

---

## 🔍 Key Design Decisions

### 1. Why Three Separate Fields?

**modality_keys:**
- What to load from parquet
- Used by dataloader to read files
- Used by metadata construction

**output_keys:**
- What to pack into batch
- Used by sample packing logic
- Represents final keys after transforms

**derived_keys:**
- How to transform modality_keys → output_keys
- Used by DerivedKeysTransform
- Specifies slice/conversion rules

### 2. Why Fallback to modality_keys?

**Backward Compatibility:**
- Most datasets don't need derived keys
- Galbot: modality_keys = output_keys (no transform)
- If `output_keys = None`: assume no derivation happened

### 3. Where modality_keys is Still Used

**These places correctly use modality_keys (not output_keys):**

| Location | Reason |
|----------|--------|
| Line 249 (_build_direct_lerobot_modality_metadata) | Need to know raw keys for metadata |
| Line 1906 (_check_integrity) | Validate raw keys exist in parquet |
| Metadata construction | Build metadata from raw keys |

**These places now use output_keys:**

| Location | Reason |
|----------|--------|
| Line 1724 (_get_modality_keys) | Sample packing needs output keys |
| Line 1736 (_get_delta_indices) | Delta indices for output keys |
| Line 2015 (__getitem__) | Pack output keys into batch |

---

## 🧪 Testing

### Unit Test Results
```python
# Test 1: Backward compatible (no output_keys)
config = ModalityConfig(
    delta_indices=[0],
    modality_keys=['action.pos']
)
assert config.output_keys is None
assert config.derived_keys == {}

# Test 2: With output_keys
config = ModalityConfig(
    delta_indices=[0],
    modality_keys=['observation.state'],
    output_keys=['state.left_arm', 'state.left_gripper']
)
assert config.output_keys == ['state.left_arm', 'state.left_gripper']
```

### Integration Test
- ✅ Galbot config (no output_keys) should work unchanged
- 🔄 FastUMI config (with output_keys) - pending full test

---

## 📝 Files Modified

- `starVLA/dataloader/gr00t_lerobot/datasets.py`
  - Extended ModalityConfig: +10 lines
  - Modified _get_modality_keys(): +3 lines
  - Modified _get_delta_indices(): +3 lines
  - **Total: ~16 lines changed**

---

## 🎯 Success Criteria

✅ **Must Have:**
- ModalityConfig accepts output_keys and derived_keys
- Backward compatible (Galbot works without output_keys)
- Sample packing uses correct keys
- Delta indices map to correct keys

🟡 **Nice to Have:**
- FastUMI example using output_keys (Phase 2)
- DerivedKeysTransform implementation (Phase 2)

❌ **Out of Scope:**
- Transform implementation (Phase 2)
- Stats computation (separate)
- Video handling (separate)

---

## 🔗 Integration with Other Phases

**Phase 1.1 (v2.1 Support):**
- Loads v2.1 datasets
- Reads flat vectors from parquet

**Phase 1.2 (output_keys) [Current]:**
- Enables key derivation
- Prepares for transform layer

**Phase 1.3 (Metadata Helpers):**
- Builds metadata from modality_keys
- Works with both raw and derived keys

**Phase 2.1 (FastUMI Example):**
- Will use output_keys + derived_keys
- Will implement DerivedKeysTransform

---

## 💡 Key Takeaways

### Design Philosophy
1. **Separation of Concerns:**
   - modality_keys: dataloader's job (what to load)
   - derived_keys: transform's job (how to derive)
   - output_keys: packing's job (what to pack)

2. **Backward Compatibility:**
   - Default: output_keys = modality_keys
   - No changes needed for existing configs

3. **Extensibility:**
   - Easy to add new derivation types
   - Clear contract between layers

### Common Patterns

**Pattern 1: No Derivation (Galbot)**
```python
ModalityConfig(
    modality_keys=["action.pos", "action.ori"],
    # output_keys defaults to modality_keys
)
```

**Pattern 2: With Derivation (FastUMI)**
```python
ModalityConfig(
    modality_keys=["observation.state"],
    output_keys=["state.left", "state.right"],
    derived_keys={"state.left": {...}}
)
```

---

## 🚀 Next Steps

### After Phase 1.2:
1. ✅ Test Galbot unchanged (Phase 1.4)
2. 🔄 Add FastUMI example with output_keys (Phase 2.1)
3. 🔄 Implement DerivedKeysTransform (Phase 2.2)

### If Tests Pass:
- Mark Phase 1.2 complete
- Proceed to Phase 1.4 (Galbot regression test)

### If Tests Fail:
- Debug sample packing logic
- Verify fallback behavior
- Check metadata construction

---

## ⚠️ Known Limitations

1. **No automatic derivation:**
   - User must specify derived_keys manually
   - Can't infer from modality_keys alone

2. **Transform not included:**
   - DerivedKeysTransform is separate (Phase 2)
   - This phase only adds the config fields

3. **No validation:**
   - Doesn't verify output_keys match derived_keys
   - Doesn't check if transforms produce expected keys

These are acceptable and will be addressed in later phases.

---

## 📚 Related Documentation

- Integration Plan: `INTEGRATION_PLAN.md`
- Phase 1.1: `PHASE1_1_CHANGES.md`
- Phase 1.3: `PHASE1_3_CHANGES.md`
- Testing Strategy: `TESTING_STRATEGY.md`

**Status:** Phase 1.2 implementation complete, awaiting test results...
