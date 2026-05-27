# Offline Eval — Galbot Bimanual (self-mode)

Runs the trained QwenGR00T policy on a single `(episode_idx, frame_idx)` sample
drawn through the **exact same** training-time LeRobot dataset/transform
pipeline, then compares GT vs. predicted actions in normalized space.

## Files

```
offline_eval/
├── offline_eval_galbot_bimanual_self.py   # the eval script
├── run_offline_eval.sh                    # thin shell launcher
└── README.md
```

## Minimal CLI

You only need **4 inputs**: an absolute checkpoint path, an absolute dataset
path, and the `(episode_idx, frame_idx)` you want to inspect.

```bash
python examples/GalbotBimanualRelative/offline_eval/offline_eval_galbot_bimanual_self.py \
  --checkpoint   /mnt/home/liuyi/project/starVLA/Checkpoints/galbot_delta_action/galbot_bimanual_self_0526/checkpoints/steps_80000_pytorch_model.pt \
  --dataset-path /mnt/project/human_action_data/liuyi/umi_lerobot_data/galbot_lerobot_dual_cup_0526 \
  --episode-idx 0 --frame-idx 50
```

## Output path rule

The output dir is derived automatically from the checkpoint path:

```
<checkpoint without .pt>/ep{NNNN}_frame{NNNN}/
```

Example:

```
ckpt:  /.../checkpoints/steps_80000_pytorch_model.pt
out :  /.../checkpoints/steps_80000_pytorch_model/ep0000_frame0000/
```

Override with `--out-dir <root>` if needed; the `ep{NNNN}_frame{NNNN}`
sub-folder is always appended.

## Shell launcher

`run_offline_eval.sh` hard-codes `checkpoint` and `dataset_path` at the top —
edit those two lines, then:

```bash
bash examples/GalbotBimanualRelative/offline_eval/run_offline_eval.sh 0 0
#                                                                     ^ ^
#                                                                     ep frame
```

## All arguments

| Arg | Required | Default | Notes |
|---|---|---|---|
| `--checkpoint`   | ✅ | — | Absolute path to the `.pt` file |
| `--dataset-path` | ✅ | — | Absolute path to the LeRobot dataset dir (parent is treated as `data_root_dir`, basename as `data_name`) |
| `--episode-idx`  | ✅ | — | Integer; must exist in `dataset.trajectory_ids` |
| `--frame-idx`    | ✅ | — | Frame index within the chosen episode (0-based) |
| `--config-yaml`  | ❌ | `examples/GalbotBimanualRelative/train_files/starvla_qwengroot_galbot_relative_self.yaml` | Used to build model + transforms; `robot_type` is read from its `data_mix` |
| `--self-mode`    | ❌ | from YAML | One of `abs`, `delta`, `chunk_relative` |
| `--out-dir`      | ❌ | derived from checkpoint | Sub-folder `ep{NNNN}_frame{NNNN}` is always appended |
| `--device`       | ❌ | `cuda` | Falls back to CPU if CUDA is unavailable |

## Output files (per sample)

```
ep{NNNN}_frame{NNNN}/
├── input_left_wrist.png
├── input_right_wrist.png
├── prompt.txt
├── gt_actions.npy                # [T, 20] normalized GT
├── pred_actions.npy              # [T, 20] normalized prediction
├── traj3d_xyz.png                      # 3D xyz traj, left & right arms (combined)
├── raw_action_curves_left.png          # 10D left-arm raw curves, GT vs Pred
├── raw_action_curves_right.png         # 10D right-arm raw curves
├── cumulative_action_curves_left.png   # left arm: cumsum on pos/ori (if delta), raw gripper
├── cumulative_action_curves_right.png  # right arm: cumsum on pos/ori (if delta), raw gripper
└── report.txt                          # per-dim MAE / RMSE / Corr / SignMatch
```

## 20D action layout

| Dim | Field |
|---|---|
| 0:3   | left_pos (xyz) |
| 3:9   | left_ori_6d |
| 9     | left_gripper |
| 10:13 | right_pos (xyz) |
| 13:19 | right_ori_6d |
| 19    | right_gripper |

## Notes

- **`--self-mode` must match the training run.** The training launcher
  `run_galbot_bimanual_self.sh` overrides the YAML default via
  `--datasets.vla_data.self_mode ${self_mode}` (currently `delta`). If you
  forget to pass `--self-mode delta` here, the dataset transform falls back
  to the YAML default (`abs`) and the resulting GT actions get clipped to
  ±1 by the normalization stats — the 3D GT trajectory then collapses to a
  single point. The shell launcher hard-codes `--self-mode delta` for this
  reason.
- GT and Pred are both compared **in normalized space** — they pass through the
  identical training transform stack, so no inverse-normalization is needed.
- For `self_mode=delta`, `traj3d_xyz.png` and `cumulative_action_curves_*.png`
  apply `cumsum` on pos/ori dims to recover trajectory-space curves; for
  `abs` / `chunk_relative` the values are plotted as-is.
- Gripper dims are always plotted raw.
- A warning about missing/unexpected state-dict keys is expected when the
  checkpoint was saved by a wrapper (e.g. Lightning / DeepSpeed) — only worry
  if model-core keys are missing.

## FAQ

**`episode_idx=X not found`** — the printed list shows available ids; use one
of those.

**`frame_idx out of range`** — pick a value less than the printed frame count
for that episode.

**Want to sweep many frames?** Wrap in a shell loop:

```bash
for f in 0 10 20 30; do
  bash run_offline_eval.sh 0 $f
done
```
