#!/usr/bin/env python3
"""Offline action validation for StarVLA checkpoints.

The script reuses StarVLA's own VLA dataloader, feeds dataset samples to a
checkpointed model, and writes per-timestep error tables/plots for left and
right hand action dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import read_mode_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


@dataclass
class ErrorAccumulator:
    """Accumulate per-timestep vector errors for one hand."""

    count: list[int] = field(default_factory=list)
    sum_l2: list[float] = field(default_factory=list)
    sum_mae: list[float] = field(default_factory=list)
    sum_rmse: list[float] = field(default_factory=list)
    sum_max_abs: list[float] = field(default_factory=list)

    def _ensure(self, length: int) -> None:
        while len(self.count) < length:
            self.count.append(0)
            self.sum_l2.append(0.0)
            self.sum_mae.append(0.0)
            self.sum_rmse.append(0.0)
            self.sum_max_abs.append(0.0)

    def update(self, diff: np.ndarray) -> None:
        """Update with diff shaped [B, T, D_hand]."""
        if diff.ndim != 3:
            raise ValueError(f"Expected diff [B, T, D], got {diff.shape}")
        if diff.shape[-1] == 0:
            raise ValueError("Hand slice is empty; check --left-slice/--right-slice.")

        batch_size, horizon, _ = diff.shape
        self._ensure(horizon)

        l2 = np.linalg.norm(diff, axis=-1)
        mae = np.mean(np.abs(diff), axis=-1)
        rmse = np.sqrt(np.mean(np.square(diff), axis=-1))
        max_abs = np.max(np.abs(diff), axis=-1)

        for timestep in range(horizon):
            self.count[timestep] += batch_size
            self.sum_l2[timestep] += float(l2[:, timestep].sum())
            self.sum_mae[timestep] += float(mae[:, timestep].sum())
            self.sum_rmse[timestep] += float(rmse[:, timestep].sum())
            self.sum_max_abs[timestep] += float(max_abs[:, timestep].sum())

    def rows(self) -> list[dict[str, float | int]]:
        rows = []
        for timestep, count in enumerate(self.count):
            denom = max(count, 1)
            rows.append(
                {
                    "timestep": timestep,
                    "count": count,
                    "mean_l2": self.sum_l2[timestep] / denom,
                    "mae": self.sum_mae[timestep] / denom,
                    "rmse": self.sum_rmse[timestep] / denom,
                    "mean_max_abs": self.sum_max_abs[timestep] / denom,
                }
            )
        return rows

    def overall(self) -> dict[str, float]:
        total = sum(self.count)
        if total == 0:
            return {"mean_l2": float("nan"), "mae": float("nan"), "rmse": float("nan"), "mean_max_abs": float("nan")}
        return {
            "mean_l2": sum(self.sum_l2) / total,
            "mae": sum(self.sum_mae) / total,
            "rmse": sum(self.sum_rmse) / total,
            "mean_max_abs": sum(self.sum_max_abs) / total,
        }


def parse_slice_spec(spec: str, action_dim: int) -> np.ndarray:
    """Parse 'start:end' or comma-separated indices into an index array."""
    spec = spec.strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid slice spec `{spec}`. Use start:end or comma indices.")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else action_dim
        indices = np.arange(start, end)
    else:
        indices = np.array([int(part.strip()) for part in spec.split(",") if part.strip()], dtype=np.int64)

    if indices.size == 0:
        raise ValueError(f"Slice spec `{spec}` selects no dimensions.")
    if indices.min() < 0 or indices.max() >= action_dim:
        raise ValueError(f"Slice spec `{spec}` is out of bounds for action_dim={action_dim}.")
    return indices


def infer_hand_slices(action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Infer left/right dimensions from common StarVLA dual-arm layouts."""
    if action_dim == 20:
        return np.arange(0, 10), np.arange(10, 20)
    if action_dim == 14:
        return np.arange(0, 7), np.arange(7, 14)
    if action_dim % 2 == 0:
        half = action_dim // 2
        return np.arange(0, half), np.arange(half, action_dim)
    raise ValueError(
        f"Cannot infer left/right slices for odd action_dim={action_dim}. "
        "Please pass --left-slice and --right-slice."
    )


def ndarray_from_action(action) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32)


def stack_gt_actions(examples: list[dict]) -> np.ndarray:
    actions = [ndarray_from_action(example["action"]) for example in examples]
    return np.stack(actions, axis=0)


def align_prediction_and_gt(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align model output and dataset action the same way training uses GT chunks."""
    if pred.ndim == 2:
        pred = pred[None, ...]
    if gt.ndim == 2:
        gt = gt[None, ...]
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"Expected pred/gt as [B, T, D], got pred={pred.shape}, gt={gt.shape}")

    horizon = min(pred.shape[1], gt.shape[1])
    action_dim = min(pred.shape[2], gt.shape[2])
    if horizon <= 0 or action_dim <= 0:
        raise ValueError(f"Invalid aligned shape from pred={pred.shape}, gt={gt.shape}")

    pred_aligned = pred[:, :horizon, :action_dim]
    gt_aligned = gt[:, -horizon:, :action_dim]
    return pred_aligned.astype(np.float32), gt_aligned.astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = ["timestep", "count", "mean_l2", "mae", "rmse", "mean_max_abs"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, float | int]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timesteps = [int(row["timestep"]) for row in rows]
    mean_l2 = [float(row["mean_l2"]) for row in rows]
    rmse = [float(row["rmse"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]

    plt.figure(figsize=(8, 4.8))
    plt.plot(timesteps, mean_l2, marker="o", label="mean L2")
    plt.plot(timesteps, rmse, marker="s", label="RMSE")
    plt.plot(timesteps, mae, marker="^", label="MAE")
    plt.xlabel("Action timestep")
    plt.ylabel("Error")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_value_trace_plot(
    path: Path,
    pred: np.ndarray,
    gt: np.ndarray,
    indices: np.ndarray,
    title: str,
) -> None:
    """Write per-action-dimension GT vs model-output curves for one sample."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timesteps = np.arange(pred.shape[0])
    num_dims = len(indices)
    ncols = 2
    nrows = int(np.ceil(num_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.0, 2.2 * nrows)), squeeze=False)
    flat_axes = axes.reshape(-1)

    for axis, dim in zip(flat_axes, indices):
        axis.plot(timesteps, gt[:, dim], marker="o", linewidth=1.8, label="GT")
        axis.plot(timesteps, pred[:, dim], marker="x", linewidth=1.6, label="model output")
        axis.set_title(f"dim {int(dim)}")
        axis.set_xlabel("action timestep")
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.3)

    for axis in flat_axes[num_dims:]:
        axis.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_input_images(output_dir: Path, sample_id: int, example: dict) -> None:
    """Save the images passed to predict_action for one validation sample."""
    images = example.get("image", [])
    if not isinstance(images, (list, tuple)):
        images = [images]

    for view_idx, image in enumerate(images):
        if hasattr(image, "save"):
            image.save(output_dir / f"sample_{sample_id:04d}_input_view_{view_idx}.png")
        else:
            from PIL import Image

            arr = np.asarray(image)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(output_dir / f"sample_{sample_id:04d}_input_view_{view_idx}.png")


def load_validation_config(args: argparse.Namespace, dotlist: list[str]):
    if args.config_yaml:
        cfg = OmegaConf.load(args.config_yaml)
    else:
        cfg_dict, _ = read_mode_config(args.pretrained_checkpoint)
        cfg = OmegaConf.create(cfg_dict)

    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if args.batch_size is not None:
        cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    if args.data_root_dir is not None:
        cfg.datasets.vla_data.data_root_dir = args.data_root_dir
    if args.data_mix is not None:
        cfg.datasets.vla_data.data_mix = args.data_mix

    cfg.output_dir = str(args.output_dir)
    return cfg


def load_model(pretrained_checkpoint: str, cfg, strict: bool = True):
    """Load a model from checkpoint, allowing harmless config path overrides."""
    _, norm_stats = read_mode_config(pretrained_checkpoint)
    model = build_framework(cfg)

    checkpoint_path = Path(pretrained_checkpoint)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_path))
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Warning: missing checkpoint keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected checkpoint keys: {unexpected}")
    model.norm_stats = norm_stats
    return model


def examples_for_model(examples: Iterable[dict]) -> list[dict]:
    """Shallow-copy examples so validation never mutates dataloader samples."""
    return [dict(example) for example in examples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate StarVLA action predictions against dataset GT.")
    parser.add_argument("--pretrained_checkpoint", "--checkpoint", required=True, help="Path to model .pt/.safetensors.")
    parser.add_argument("--config_yaml", default=None, help="Optional validation config. Defaults to checkpoint config.")
    parser.add_argument("--output_dir", default="validation_outputs", type=Path, help="Directory for CSV/PNG outputs.")
    parser.add_argument("--max_batches", type=int, default=100, help="Maximum dataloader batches to evaluate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override datasets.vla_data.per_device_batch_size.")
    parser.add_argument("--data_root_dir", default=None, help="Override datasets.vla_data.data_root_dir.")
    parser.add_argument("--data_mix", default=None, help="Override datasets.vla_data.data_mix.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--left_slice", default=None, help="Left hand dims, e.g. 0:10 or 0,1,2,3.")
    parser.add_argument("--right_slice", default=None, help="Right hand dims, e.g. 10:20 or 10,11,12.")
    parser.add_argument("--strict_load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--num_value_plots",
        type=int,
        default=1,
        help="Save GT-vs-model-output plots for the first N dataset examples.",
    )
    parser.add_argument(
        "--save_input_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the observation images used for the GT-vs-output sample plots.",
    )
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    cfg = load_validation_config(args, dotlist)
    model = load_model(args.pretrained_checkpoint, cfg=cfg, strict=args.strict_load)
    model = model.to(args.device).eval()

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    left_acc = ErrorAccumulator()
    right_acc = ErrorAccumulator()
    left_indices = None
    right_indices = None
    num_examples = 0
    num_saved_value_plots = 0

    with torch.inference_mode():
        iterator = tqdm(enumerate(dataloader), total=args.max_batches, desc="Validating")
        for batch_idx, examples in iterator:
            if batch_idx >= args.max_batches:
                break
            if not examples:
                continue

            output = model.predict_action(examples=examples_for_model(examples))
            pred = np.asarray(output["normalized_actions"], dtype=np.float32)
            gt = stack_gt_actions(examples)
            pred, gt = align_prediction_and_gt(pred, gt)

            action_dim = pred.shape[-1]
            if left_indices is None or right_indices is None:
                if args.left_slice and args.right_slice:
                    left_indices = parse_slice_spec(args.left_slice, action_dim)
                    right_indices = parse_slice_spec(args.right_slice, action_dim)
                elif args.left_slice or args.right_slice:
                    raise ValueError("Please pass both --left-slice and --right-slice, or neither.")
                else:
                    left_indices, right_indices = infer_hand_slices(action_dim)

                print(f"Using left dims: {left_indices.tolist()}")
                print(f"Using right dims: {right_indices.tolist()}")

            diff = pred - gt
            left_acc.update(diff[:, :, left_indices])
            right_acc.update(diff[:, :, right_indices])

            if args.plot and num_saved_value_plots < args.num_value_plots:
                remaining = args.num_value_plots - num_saved_value_plots
                for local_idx in range(min(pred.shape[0], remaining)):
                    sample_id = num_saved_value_plots
                    write_value_trace_plot(
                        args.output_dir / f"sample_{sample_id:04d}_left_gt_vs_output.png",
                        pred[local_idx],
                        gt[local_idx],
                        left_indices,
                        f"Sample {sample_id} Left Hand GT vs Model Output",
                    )
                    write_value_trace_plot(
                        args.output_dir / f"sample_{sample_id:04d}_right_gt_vs_output.png",
                        pred[local_idx],
                        gt[local_idx],
                        right_indices,
                        f"Sample {sample_id} Right Hand GT vs Model Output",
                    )
                    if args.save_input_images:
                        save_input_images(args.output_dir, sample_id, examples[local_idx])
                    num_saved_value_plots += 1

            num_examples += pred.shape[0]

    left_rows = left_acc.rows()
    right_rows = right_acc.rows()

    left_csv = args.output_dir / "left_hand_error_curve.csv"
    right_csv = args.output_dir / "right_hand_error_curve.csv"
    write_csv(left_csv, left_rows)
    write_csv(right_csv, right_rows)

    if args.plot:
        write_plot(args.output_dir / "left_hand_error_curve.png", left_rows, "Left Hand Output vs GT Error")
        write_plot(args.output_dir / "right_hand_error_curve.png", right_rows, "Right Hand Output vs GT Error")

    summary = {
        "checkpoint": str(args.pretrained_checkpoint),
        "config_yaml": str(args.config_yaml) if args.config_yaml else "checkpoint config.yaml",
        "data_mix": str(cfg.datasets.vla_data.data_mix),
        "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
        "num_examples": num_examples,
        "left_dims": left_indices.tolist() if left_indices is not None else [],
        "right_dims": right_indices.tolist() if right_indices is not None else [],
        "num_saved_value_plots": num_saved_value_plots,
        "input_images_saved": bool(args.save_input_images and num_saved_value_plots),
        "left_overall": left_acc.overall(),
        "right_overall": right_acc.overall(),
        "note": "Errors compare normalized model outputs with normalized dataset GT actions.",
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {left_csv}")
    print(f"Wrote {right_csv}")
    if args.plot:
        print(f"Wrote {args.output_dir / 'left_hand_error_curve.png'}")
        print(f"Wrote {args.output_dir / 'right_hand_error_curve.png'}")
        if num_saved_value_plots:
            print(f"Wrote {num_saved_value_plots} GT-vs-output sample plot pair(s) under {args.output_dir}")
            if args.save_input_images:
                print(f"Wrote input image(s) as sample_XXXX_input_view_*.png under {args.output_dir}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

# 如果想看前 10 个训练集 observation：
# python validate.py \
#   --pretrained_checkpoint playground/Checkpoints/starvla_qwengroot_0521_full/checkpoints/steps_10000_pytorch_model.pt \
#   --data_root_dir /mnt/project/public/yangzhibo/umidata/lerobot \
#   --data_mix my_mix \
#   --batch_size 1 \
#   --max_batches 10 \
#   --num_value_plots 10 \
#   --output_dir validation_outputs
