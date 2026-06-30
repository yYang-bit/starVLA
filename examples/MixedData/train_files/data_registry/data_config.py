import importlib.util
from pathlib import Path


def _load_data_config(example_name: str):
    data_config_path = (
        Path(__file__).resolve().parents[3]
        / example_name
        / "train_files"
        / "data_registry"
        / "data_config.py"
    )
    spec = importlib.util.spec_from_file_location(f"_{example_name}_data_config", data_config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load data config from {data_config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_AGIBOT = _load_data_config("AgiBotWolrd")
_FASTUMI = _load_data_config("FastUMI")


ROBOT_TYPE_CONFIG_MAP = {
    **_AGIBOT.ROBOT_TYPE_CONFIG_MAP,
    **_FASTUMI.ROBOT_TYPE_CONFIG_MAP,
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    **_AGIBOT.ROBOT_TYPE_TO_EMBODIMENT_TAG,
    **_FASTUMI.ROBOT_TYPE_TO_EMBODIMENT_TAG,
}

_MIXED_DATA = (
    _AGIBOT.DATASET_NAMED_MIXTURES["agi_data"]
    + _FASTUMI.DATASET_NAMED_MIXTURES["fastumi_data"]
)

DATASET_NAMED_MIXTURES = {
    "mixed_data": _MIXED_DATA,
}
