#!/usr/bin/env python3
"""Compute per-dataset relative action stats for UMI absolute-pose LeRobot data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytorch3d.transforms as pt
import torch
from tqdm import tqdm

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LE_ROBOT2_DEFAULT_DATA_PATH,
    LE_ROBOT2_EPISODES_FILENAME,
    LE_ROBOT3_DEFAULT_DATA_PATH,
    LE_ROBOT3_EPISODE_FILENAME,
    LE_ROBOT_INFO_FILENAME,
    LE_ROBOT_RELATIVE_POSE_STATS_FILENAME,
    _get_state_action_meta_from_info,
)
from starVLA.dataloader.gr00t_lerobot.transform.state_action import RelativePoseActionTransform


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_false_like(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"false", "0", "no"}
    return value is False or bool(value) is False and value is not None


def _episode_passes_wbc_threshold(episode: dict[str, Any]) -> bool:
    if "wbc_threshold_passed" not in episode:
        return True
    return not _is_false_like(episode.get("wbc_threshold_passed"))


def _load_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"data_config.py not found: {path}")
    module_name = f"_starvla_relative_stats_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import data_config.py from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_lerobot_version(version: str | None) -> str | None:
    if version is None:
        return None
    normalized = str(version).strip().lower()
    if normalized in {"", "auto", "none", "null"}:
        return None
    normalized = normalized.removeprefix("lerobot-").removeprefix("lerobot_").removeprefix("v")
    if normalized.startswith("2.1"):
        return "v2.1"
    if normalized.startswith("3"):
        return "v3.0"
    raise ValueError(f"Unsupported LeRobot version `{version}`. Expected v2.1, v3.0, or auto.")


def _detect_lerobot_version(dataset_path: Path, configured_version: str | None) -> str:
    configured = _normalize_lerobot_version(configured_version)
    if configured is not None:
        return configured
    if (dataset_path / LE_ROBOT2_EPISODES_FILENAME).exists():
        return "v2.1"
    if list(dataset_path.glob(LE_ROBOT3_EPISODE_FILENAME)):
        return "v3.0"
    raise FileNotFoundError(f"Unable to detect LeRobot version from {dataset_path}")


def _episode_records(
    dataset_path: Path,
    info: dict[str, Any],
    version: str,
    skip_failed_wbc: bool,
) -> tuple[str, list[dict[str, Any]]]:
    chunk_size = int(info.get("chunks_size", info.get("chunk_size", 1000)))
    if version == "v2.1":
        data_pattern = info.get("data_path", LE_ROBOT2_DEFAULT_DATA_PATH)
        episodes_path = dataset_path / LE_ROBOT2_EPISODES_FILENAME
        if not episodes_path.exists():
            raise FileNotFoundError(f"Missing {episodes_path}")
        records = []
        skipped_failed_wbc = 0
        for index, episode in enumerate(_read_jsonl(episodes_path)):
            if skip_failed_wbc and not _episode_passes_wbc_threshold(episode):
                skipped_failed_wbc += 1
                continue
            episode_index = int(episode.get("episode_index", index))
            length = episode.get("length", episode.get("num_frames", episode.get("episode_length")))
            if length is None:
                raise KeyError(f"Episode {episode_index} is missing length/num_frames/episode_length in {episodes_path}")
            episode_chunk = int(episode.get("episode_chunk", episode.get("data/chunk_index", episode_index // chunk_size)))
            records.append(
                {
                    "episode_index": episode_index,
                    "length": int(length),
                    "episode_chunk": episode_chunk,
                    "chunk_index": int(episode.get("data/chunk_index", episode_chunk)),
                    "file_index": int(episode.get("data/file_index", episode_index)),
                    "file_from_index": int(episode.get("data/file_from_index", 0)),
                }
            )
        if skipped_failed_wbc:
            print(f"Skipped {skipped_failed_wbc} episodes with wbc_threshold_passed=false from {dataset_path}")
        if not records:
            raise ValueError(f"No episodes remain after filtering {episodes_path}")
        return data_pattern, records

    if version == "v3.0":
        data_pattern = info.get("data_path", LE_ROBOT3_DEFAULT_DATA_PATH)
        records = []
        skipped_failed_wbc = 0
        for episodes_path in sorted(dataset_path.glob(LE_ROBOT3_EPISODE_FILENAME)):
            episodes = pd.read_parquet(episodes_path)
            for row_index, episode in episodes.iterrows():
                if skip_failed_wbc and not _episode_passes_wbc_threshold(episode):
                    skipped_failed_wbc += 1
                    continue
                records.append(
                    {
                        "episode_index": int(episode["episode_index"]),
                        "length": int(episode["length"]),
                        "episode_chunk": int(episode.get("data/chunk_index", 0)),
                        "chunk_index": int(episode["data/chunk_index"]),
                        "file_index": int(episode["data/file_index"]),
                        "file_from_index": int(episode.get("data/file_from_index", row_index)),
                    }
                )
        if skipped_failed_wbc:
            print(f"Skipped {skipped_failed_wbc} episodes with wbc_threshold_passed=false from {dataset_path}")
        if not records:
            raise FileNotFoundError(f"No LeRobot v3 episode files found under {dataset_path / 'meta' / 'episodes'}")
        return data_pattern, records

    raise ValueError(f"Unsupported LeRobot version: {version}")


def _format_data_path(dataset_path: Path, pattern: str, episode: dict[str, Any]) -> Path:
    values = {
        "episode_index": int(episode["episode_index"]),
        "episode_chunk": int(episode["episode_chunk"]),
        "chunk_index": int(episode["chunk_index"]),
        "file_index": int(episode["file_index"]),
    }
    return dataset_path / pattern.format(**values)


def _load_episode_dataframe(dataset_path: Path, data_pattern: str, episode: dict[str, Any]) -> pd.DataFrame:
    parquet_path = _format_data_path(dataset_path, data_pattern, episode)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    data = pd.read_parquet(parquet_path)
    if "episode_index" in data.columns:
        data = data.loc[data["episode_index"] == int(episode["episode_index"])].copy()
    data = data.reset_index(drop=True)
    if len(data) != int(episode["length"]):
        raise ValueError(
            f"Episode {episode['episode_index']} length mismatch: meta={episode['length']}, parquet={len(data)}"
        )
    return data


def _read_low_dim_array(df: pd.DataFrame, info_meta: dict[str, Any], modality: str, key: str) -> np.ndarray:
    field_meta = _get_state_action_meta_from_info(info_meta, key)
    original_key = field_meta.original_key
    if original_key not in df.columns:
        raise KeyError(f"Column `{original_key}` for `{key}` not found in parquet")
    values = np.stack(df[original_key])
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected 2D low-dim array for `{key}`/`{original_key}`, got {values.shape}")
    return values[:, field_meta.start : field_meta.end].astype(np.float32, copy=False)


def _sample_chunk(array: np.ndarray, delta_indices: list[int], base_index: int) -> np.ndarray:
    step_indices = np.asarray(delta_indices, dtype=np.int64) + int(base_index)
    step_indices = np.clip(step_indices, 0, len(array) - 1)
    return array[step_indices].astype(np.float32, copy=False)


def _iter_transforms(transform) -> list[Any]:
    transforms = []
    for child in getattr(transform, "transforms", []):
        transforms.extend(_iter_transforms(child))
    transforms.append(transform)
    return transforms


def _relative_transform_from_config(data_config) -> RelativePoseActionTransform:
    composed = data_config.transform(data_cfg={"action_chunk_representation": "relative_pose"})
    matches = [t for t in _iter_transforms(composed) if isinstance(t, RelativePoseActionTransform)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one RelativePoseActionTransform, found {len(matches)}")
    transform = matches[0]
    if not transform.arm_prefixes:
        raise ValueError("This stats script requires explicit arm_prefixes in RelativePoseActionTransform")
    return transform


def _normalization_modes(data_config) -> dict[str, str]:
    modes = getattr(data_config, "normalization_modes", None)
    if not isinstance(modes, dict):
        raise ValueError("data_config must define normalization_modes as a dict")
    normalized_modes = {}
    for key, mode in modes.items():
        if not key.startswith("action."):
            raise ValueError(f"This script writes only action_stats; remove non-action normalization key `{key}`")
        normalized_mode = str(mode)
        if normalized_mode not in {"min_max", "q99"}:
            raise ValueError(
                f"Only min_max and q99 normalization are supported for relative_stats, got {key}: {mode}"
            )
        lowered = key.lower()
        if "ori_6d" in lowered or "rotation_6d" in lowered:
            raise ValueError(f"Rotation key `{key}` must not be normalized")
        normalized_modes[key] = normalized_mode
    if not normalized_modes:
        raise ValueError("No action keys found in normalization_modes")
    return normalized_modes


def _validate_normalization_modes(
    normalization_modes: dict[str, str],
    arm_keys: dict[str, dict[str, str]],
) -> None:
    position_keys = {spec["action_pos"] for spec in arm_keys.values()}
    gripper_keys = {spec["action_gripper"] for spec in arm_keys.values()}

    for key, mode in normalization_modes.items():
        if key in position_keys:
            continue
        if key in gripper_keys:
            if mode != "min_max":
                raise ValueError(f"Gripper key `{key}` must use min_max normalization, got {mode}")
            continue
        raise KeyError(
            f"Normalization key `{key}` is neither an action position key nor an action gripper key "
            f"from RelativePoseActionTransform"
        )


def _required_arm_keys(data_config, rel_transform: RelativePoseActionTransform) -> dict[str, dict[str, str]]:
    modality_config = data_config.modality_config()
    state_cfg = modality_config["state"]
    action_cfg = modality_config["action"]
    state_output_keys = set(state_cfg.output_keys or state_cfg.modality_keys)
    action_output_keys = set(action_cfg.output_keys or action_cfg.modality_keys)

    arm_keys = {}
    for arm_prefix in rel_transform.arm_prefixes:
        spec = {
            "state_pos": f"state.{arm_prefix}{rel_transform.state_position_suffix}",
            "state_rot": f"state.{arm_prefix}{rel_transform.state_rotation_suffix}",
            "action_pos": f"action.{arm_prefix}{rel_transform.action_position_suffix}",
            "action_rot": f"action.{arm_prefix}{rel_transform.action_rotation_suffix}",
            "action_gripper": f"action.{arm_prefix}{rel_transform.action_gripper_suffix}",
        }
        for name in ("state_pos", "state_rot"):
            if spec[name] not in state_output_keys:
                raise KeyError(f"{spec[name]} is required by RelativePoseActionTransform but missing from state output keys")
        for name in ("action_pos", "action_rot", "action_gripper"):
            if spec[name] not in action_output_keys:
                raise KeyError(f"{spec[name]} is required by RelativePoseActionTransform but missing from action output keys")
        arm_keys[arm_prefix] = spec
    return arm_keys


def _expected_dims(
    data_config,
    info_meta: dict[str, Any],
    arm_keys: dict[str, dict[str, str]],
    normalization_modes: dict[str, str],
) -> dict[str, int]:
    position_keys = {spec["action_pos"] for spec in arm_keys.values()}
    gripper_keys = {spec["action_gripper"] for spec in arm_keys.values()}
    derived_keys = data_config.modality_config()["action"].derived_keys
    expected = {}
    for key in normalization_modes:
        if key in position_keys:
            expected[key] = 3
            continue
        if key in gripper_keys:
            if key not in derived_keys:
                raise KeyError(f"Cannot infer gripper width because `{key}` is missing from action derived_keys")
            derived_spec = derived_keys[key]
            source_key = derived_spec.get("source_key")
            if not source_key:
                raise KeyError(f"Cannot infer gripper width because `{key}` has no source_key in derived_keys")
            source_meta = _get_state_action_meta_from_info(info_meta, str(source_key))
            if "start" in derived_spec or "end" in derived_spec:
                start = int(derived_spec.get("start", 0))
                end = int(derived_spec.get("end", source_meta.end - source_meta.start))
                expected[key] = end - start
            else:
                expected[key] = int(source_meta.end - source_meta.start)
            continue
    return expected


def _validate_existing_stats(
    stats_path: Path,
    expected_dims: dict[str, int],
    normalization_modes: dict[str, str],
) -> None:
    if not stats_path.exists():
        raise FileNotFoundError(f"relative stats file not found: {stats_path}")
    payload = _read_json(stats_path)
    if "action_stats" not in payload:
        raise KeyError(f"`action_stats` not found in {stats_path}")
    action_stats = payload["action_stats"]
    expected_subkeys = {key.split(".", 1)[1] for key in expected_dims}
    actual_subkeys = set(action_stats)
    if actual_subkeys != expected_subkeys:
        raise KeyError(
            f"action_stats keys mismatch in {stats_path}: expected {sorted(expected_subkeys)}, "
            f"got {sorted(actual_subkeys)}"
        )
    for full_key, dim in expected_dims.items():
        subkey = full_key.split(".", 1)[1]
        stats = action_stats[subkey]
        required_stat_names = ["min", "max"]
        if normalization_modes[full_key] == "q99":
            required_stat_names.extend(["q01", "q99"])

        for stat_name in required_stat_names:
            if stat_name not in stats:
                raise KeyError(f"`action_stats.{subkey}.{stat_name}` missing in {stats_path}")
            values = np.asarray(stats[stat_name], dtype=np.float32).reshape(-1)
            if values.size != dim:
                raise ValueError(
                    f"`action_stats.{subkey}.{stat_name}` shape mismatch in {stats_path}: "
                    f"expected {dim}, got {values.size}"
                )


def _ensure_2d(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected `{name}` to be 2D, got {array.shape}")
    return array


def _rotation_6d_to_matrix(name: str, value: Any) -> np.ndarray:
    rotation_6d = np.asarray(value, dtype=np.float32)
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected `{name}` to have last dim 6, got {rotation_6d.shape}")
    matrix = pt.rotation_6d_to_matrix(torch.from_numpy(rotation_6d))
    return matrix.detach().cpu().numpy().astype(np.float32, copy=False)


def _update_stats(
    accumulator: dict[str, dict[str, Any]],
    key: str,
    values: np.ndarray,
    collect_quantiles: bool = False,
) -> None:
    values = _ensure_2d(key, values)
    current_min = values.min(axis=0)
    current_max = values.max(axis=0)
    if key not in accumulator:
        accumulator[key] = {"min": current_min, "max": current_max}
    else:
        accumulator[key]["min"] = np.minimum(accumulator[key]["min"], current_min)
        accumulator[key]["max"] = np.maximum(accumulator[key]["max"], current_max)

    if collect_quantiles:
        accumulator[key].setdefault("values", []).append(values.astype(np.float32, copy=True))


def _build_payload(
    accumulator: dict[str, dict[str, Any]],
    expected_dims: dict[str, int],
    normalization_modes: dict[str, str],
) -> dict[str, Any]:
    missing = [key for key in expected_dims if key not in accumulator]
    if missing:
        raise KeyError(f"No stats were collected for keys: {missing}")
    action_stats = {}
    for full_key in expected_dims:
        subkey = full_key.split(".", 1)[1]
        stats = accumulator[full_key]
        if stats["min"].size != expected_dims[full_key] or stats["max"].size != expected_dims[full_key]:
            raise ValueError(f"Stats shape mismatch for `{full_key}`")
        key_payload = {
            "min": stats["min"].astype(float).tolist(),
            "max": stats["max"].astype(float).tolist(),
        }
        if normalization_modes[full_key] == "q99":
            if "values" not in stats:
                raise KeyError(f"No quantile values were collected for `{full_key}`")
            values = np.concatenate(stats["values"], axis=0)
            q01 = np.quantile(values, 0.01, axis=0)
            q99 = np.quantile(values, 0.99, axis=0)
            key_payload["q01"] = q01.astype(float).tolist()
            key_payload["q99"] = q99.astype(float).tolist()
        action_stats[subkey] = key_payload
    return {"action_stats": action_stats}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
    os.replace(tmp_path, path)


def _select_base_indices(
    episodes: list[dict[str, Any]],
    sample_num: int | None,
    sample_seed: int,
) -> dict[int, np.ndarray]:
    lengths = np.asarray([int(episode["length"]) for episode in episodes], dtype=np.int64)
    total_steps = int(lengths.sum())
    if total_steps <= 0:
        raise ValueError("Dataset has no steps to sample")

    if sample_num is None:
        return {index: np.arange(length, dtype=np.int64) for index, length in enumerate(lengths)}

    if sample_num <= 0:
        raise ValueError(f"sample_num must be positive when provided, got {sample_num}")
    if sample_num > total_steps:
        raise ValueError(f"sample_num={sample_num} exceeds dataset steps={total_steps}")

    offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(lengths)])
    rng = np.random.default_rng(sample_seed)
    global_indices = np.sort(rng.choice(total_steps, size=sample_num, replace=False))

    selected: dict[int, list[int]] = {}
    episode_indices = np.searchsorted(offsets[1:], global_indices, side="right")
    for episode_index, global_index in zip(episode_indices, global_indices):
        base_index = int(global_index - offsets[episode_index])
        selected.setdefault(int(episode_index), []).append(base_index)

    return {index: np.asarray(base_indices, dtype=np.int64) for index, base_indices in selected.items()}


def _compute_dataset_stats(
    dataset_path: Path,
    data_config,
    lerobot_version: str | None,
    sample_num: int | None,
    sample_seed: int,
    skip_failed_wbc: bool,
) -> dict[str, Any]:
    info_path = dataset_path / LE_ROBOT_INFO_FILENAME
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    info = _read_json(info_path)
    version = _detect_lerobot_version(dataset_path, lerobot_version)
    data_pattern, episodes = _episode_records(dataset_path, info, version, skip_failed_wbc)

    modality_config = data_config.modality_config()
    state_cfg = modality_config["state"]
    action_cfg = modality_config["action"]
    if list(state_cfg.delta_indices) != [0]:
        raise ValueError(f"This UMI relative stats script requires state_indices=[0], got {state_cfg.delta_indices}")
    if len(action_cfg.delta_indices) == 0:
        raise ValueError("action_indices must not be empty")

    rel_transform = _relative_transform_from_config(data_config)
    arm_keys = _required_arm_keys(data_config, rel_transform)
    normalization_modes = _normalization_modes(data_config)
    _validate_normalization_modes(normalization_modes, arm_keys)
    expected_dims = _expected_dims(data_config, info, arm_keys, normalization_modes)
    normalization_key_set = set(normalization_modes)

    pose_transforms = data_config.pose_transforms()
    accumulator: dict[str, dict[str, Any]] = {}
    selected_base_indices = _select_base_indices(episodes, sample_num, sample_seed)
    if sample_num is not None:
        total_steps = sum(int(episode["length"]) for episode in episodes)
        print(f"Sampling {sample_num} / {total_steps} base steps from {dataset_path}")

    for episode_index, episode in enumerate(tqdm(episodes, desc=f"Computing {dataset_path.name}", unit="episode")):
        if episode_index not in selected_base_indices:
            continue
        df = _load_episode_dataframe(dataset_path, data_pattern, episode)
        raw_arrays = {}
        for key in state_cfg.modality_keys:
            raw_arrays[key] = _read_low_dim_array(df, info, "state", key)
        for key in action_cfg.modality_keys:
            raw_arrays[key] = _read_low_dim_array(df, info, "action", key)

        for base_index in selected_base_indices[episode_index]:
            sample = {}
            for key in state_cfg.modality_keys:
                sample[key] = _sample_chunk(raw_arrays[key], list(state_cfg.delta_indices), int(base_index))
            for key in action_cfg.modality_keys:
                sample[key] = _sample_chunk(raw_arrays[key], list(action_cfg.delta_indices), int(base_index))

            for transform in pose_transforms:
                sample = transform(sample)

            for arm_prefix, spec in arm_keys.items():
                state_pos = _ensure_2d(spec["state_pos"], sample[spec["state_pos"]])
                action_pos = _ensure_2d(spec["action_pos"], sample[spec["action_pos"]])
                state_rot = _rotation_6d_to_matrix(spec["state_rot"], sample[spec["state_rot"]])
                if state_rot.ndim != 3 or len(state_rot) != len(state_pos):
                    raise ValueError(
                        f"State rotation/position shape mismatch for `{arm_prefix}`: {state_rot.shape}, {state_pos.shape}"
                    )

                if spec["action_pos"] in normalization_key_set:
                    base_pos = state_pos[-1]
                    base_rot = state_rot[-1]
                    rel_pos_world = action_pos - base_pos[None, :]
                    rel_pos = rel_pos_world @ base_rot
                    _update_stats(
                        accumulator,
                        spec["action_pos"],
                        rel_pos,
                        collect_quantiles=normalization_modes[spec["action_pos"]] == "q99",
                    )

                if spec["action_gripper"] in normalization_key_set:
                    _update_stats(accumulator, spec["action_gripper"], sample[spec["action_gripper"]])

    return _build_payload(accumulator, expected_dims, normalization_modes)


def _process_dataset(
    dataset_path: Path,
    data_config,
    lerobot_version: str | None,
    regenerate: bool,
    sample_num: int | None,
    sample_seed: int,
    skip_failed_wbc: bool,
) -> None:
    stats_path = dataset_path / LE_ROBOT_RELATIVE_POSE_STATS_FILENAME
    info_path = dataset_path / LE_ROBOT_INFO_FILENAME
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    info = _read_json(info_path)
    rel_transform = _relative_transform_from_config(data_config)
    arm_keys = _required_arm_keys(data_config, rel_transform)
    normalization_modes = _normalization_modes(data_config)
    _validate_normalization_modes(normalization_modes, arm_keys)
    expected_dims = _expected_dims(data_config, info, arm_keys, normalization_modes)

    if not regenerate:
        _validate_existing_stats(stats_path, expected_dims, normalization_modes)
        print(f"[OK] Read existing relative stats: {stats_path}")
        return

    payload = _compute_dataset_stats(dataset_path, data_config, lerobot_version, sample_num, sample_seed, skip_failed_wbc)
    _write_json(stats_path, payload)
    print(f"[OK] Wrote relative stats: {stats_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", required=True, type=Path, help="Path to the data_config.py to read.")
    parser.add_argument("--data-root-dir", required=True, type=Path, help="Root directory containing dataset folders.")
    parser.add_argument("--data-mix", required=True, help="Mixture name from DATASET_NAMED_MIXTURES.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Recompute and overwrite meta/relative_stats.json. If omitted, only read and validate existing files.",
    )
    parser.add_argument(
        "--lerobot-version",
        default=None,
        help="Optional LeRobot version override: v2.1, v3.0, or auto. Defaults to auto detection.",
    )
    parser.add_argument(
        "--sample-num",
        "--sample_num",
        dest="sample_num",
        type=int,
        default=None,
        help="Number of base steps to sample per dataset when regenerating. Defaults to all steps.",
    )
    parser.add_argument(
        "--sample-seed",
        "--sample_seed",
        dest="sample_seed",
        type=int,
        default=42,
        help="Random seed used with --sample-num. Defaults to 42.",
    )
    parser.add_argument(
        "--include-failed-wbc",
        action="store_true",
        help="Include episodes whose metadata has wbc_threshold_passed=false. Defaults to skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = _load_module(args.data_config)
    if not hasattr(module, "DATASET_NAMED_MIXTURES"):
        raise AttributeError(f"{args.data_config} does not define DATASET_NAMED_MIXTURES")
    if not hasattr(module, "ROBOT_TYPE_CONFIG_MAP"):
        raise AttributeError(f"{args.data_config} does not define ROBOT_TYPE_CONFIG_MAP")
    if args.data_mix not in module.DATASET_NAMED_MIXTURES:
        raise KeyError(f"data_mix `{args.data_mix}` not found in DATASET_NAMED_MIXTURES")
    if not args.regenerate and args.sample_num is not None:
        raise ValueError("--sample-num only applies when --regenerate is set")

    seen = set()
    for dataset_name, _, robot_type in module.DATASET_NAMED_MIXTURES[args.data_mix]:
        dataset_key = (dataset_name, robot_type)
        if dataset_key in seen:
            continue
        seen.add(dataset_key)
        if robot_type not in module.ROBOT_TYPE_CONFIG_MAP:
            raise KeyError(f"robot_type `{robot_type}` not found in ROBOT_TYPE_CONFIG_MAP")
        dataset_path = args.data_root_dir / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        _process_dataset(
            dataset_path=dataset_path,
            data_config=module.ROBOT_TYPE_CONFIG_MAP[robot_type],
            lerobot_version=args.lerobot_version,
            regenerate=args.regenerate,
            sample_num=args.sample_num,
            sample_seed=args.sample_seed,
            skip_failed_wbc=not args.include_failed_wbc,
        )


if __name__ == "__main__":
    main()

# python galbot/compute_relative_stats.py --data-config examples/MyData/train_files/data_registry/data_config.py  --data-root-dir /  --data-mix my_mix  --regenerate  --sample-num 1000
