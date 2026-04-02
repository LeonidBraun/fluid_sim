from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import h5py
except ImportError:  # pragma: no cover - depends on local Python environment
    h5py = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on local Python environment
    np = None


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "grid": {
        "nx": 256,
        "ny": 128,
        "dx": 0.02,
        "dy": 0.02,
    },
    "simulation": {
        "end_time": 10.0,
        "cfl": 0.05,
        "reference_density": 1.225,
        "kinematic_viscosity": 1.5e-5,
        "density_diffusivity": 1.0e-5,
        "pressure_iterations": 60,
    },
    "output": {
        "output_interval": 1.0,
    },
}

GRID_KEYS = {"nx", "ny", "dx", "dy"}
SIMULATION_KEYS = {
    "end_time",
    "cfl",
    "reference_density",
    "kinematic_viscosity",
    "density_diffusivity",
    "pressure_iterations",
}
OUTPUT_KEYS = {"output_interval"}
LEGACY_KEYS = {"steps", "dt", "output_every", "viscosity", "density_diffusion", "density_decay"}


@dataclass(frozen=True)
class FrameReference:
    index: int
    time: float
    file_path: Path


@dataclass
class FrameData:
    index: int
    time: float
    time_step: float
    file_path: Path
    attributes: Dict[str, Any]
    density_offset: Any
    velocity: Any


class FluidSimIO:
    """Create solver configs and load simulation outputs for pre/postprocessing."""

    def __init__(self, config_path: str | Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.config_path = Path(config_path).expanduser()
        self.config = self._normalize_config(config or deepcopy(DEFAULT_CONFIG))
        self.validate()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "FluidSimIO":
        config_path = Path(config_path).expanduser()
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return cls(config_path=config_path, config=config)

    @property
    def output_root(self) -> Path:
        return self.config_path.parent / "outputs"

    @property
    def data_dir(self) -> Path:
        return self.output_root / "data"

    @property
    def series_path(self) -> Path:
        return self.output_root / "series.xdmf"

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy(self.config)

    def set_grid(
        self,
        *,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
    ) -> None:
        updates = {"nx": nx, "ny": ny, "dx": dx, "dy": dy}
        self._apply_updates("grid", updates)

    def set_simulation(
        self,
        *,
        end_time: Optional[float] = None,
        cfl: Optional[float] = None,
        reference_density: Optional[float] = None,
        kinematic_viscosity: Optional[float] = None,
        density_diffusivity: Optional[float] = None,
        pressure_iterations: Optional[int] = None,
    ) -> None:
        updates = {
            "end_time": end_time,
            "cfl": cfl,
            "reference_density": reference_density,
            "kinematic_viscosity": kinematic_viscosity,
            "density_diffusivity": density_diffusivity,
            "pressure_iterations": pressure_iterations,
        }
        self._apply_updates("simulation", updates)

    def set_output(self, *, output_interval: Optional[float] = None) -> None:
        self._apply_updates("output", {"output_interval": output_interval})

    def write_config(self, path: str | Path | None = None, indent: int = 2) -> Path:
        self.validate()
        destination = Path(path).expanduser() if path is not None else self.config_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=indent)
            handle.write("\n")
        return destination

    def cell_centers(self) -> tuple[Any, Any]:
        self._require_numpy()
        nx = int(self.config["grid"]["nx"])
        ny = int(self.config["grid"]["ny"])
        dx = float(self.config["grid"]["dx"])
        dy = float(self.config["grid"]["dy"])
        x = (np.arange(nx, dtype=float) + 0.5) * dx
        y = (np.arange(ny, dtype=float) + 0.5) * dy
        return np.meshgrid(x, y, indexing="xy")

    def list_frames(self) -> list[FrameReference]:
        if self.series_path.exists():
            return self._list_frames_from_xdmf(self.series_path)
        return self._list_frames_from_hdf5(self.data_dir)

    def read_frame(self, index: int) -> FrameData:
        frames = self.list_frames()
        try:
            reference = frames[index]
        except IndexError as exc:
            raise IndexError(f"Frame index {index} is out of range for {len(frames)} frame(s).") from exc
        return self.read_frame_file(reference.file_path, index=reference.index, time_hint=reference.time)

    def read_last_frame(self) -> FrameData:
        frames = self.list_frames()
        if not frames:
            raise FileNotFoundError(f"No frame files found under {self.output_root}.")
        return self.read_frame(frames[-1].index)

    def read_all_frames(self) -> list[FrameData]:
        return [self.read_frame(frame.index) for frame in self.list_frames()]

    def read_frame_file(
        self,
        frame_path: str | Path,
        *,
        index: int = 0,
        time_hint: Optional[float] = None,
    ) -> FrameData:
        self._require_hdf5()
        frame_path = Path(frame_path).expanduser()
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame file does not exist: {frame_path}")

        with h5py.File(frame_path, "r") as handle:
            attributes = {key: self._to_python_scalar(value) for key, value in handle.attrs.items()}
            density_offset = np.asarray(handle["density_offset"])
            velocity = np.asarray(handle["velocity"])

        density_offset = self._drop_leading_time_axis(density_offset)
        velocity = self._drop_leading_time_axis(velocity)

        frame_time = float(attributes.get("time", time_hint if time_hint is not None else 0.0))
        time_step = float(attributes.get("time_step", 0.0))

        return FrameData(
            index=index,
            time=frame_time,
            time_step=time_step,
            file_path=frame_path,
            attributes=attributes,
            density_offset=density_offset,
            velocity=velocity,
        )

    def validate(self) -> None:
        grid = self.config["grid"]
        simulation = self.config["simulation"]
        output = self.config["output"]

        if int(grid["nx"]) < 8 or int(grid["ny"]) < 8:
            raise ValueError("Grid must be at least 8x8.")
        if int(simulation["pressure_iterations"]) < 1:
            raise ValueError("pressure_iterations must be positive.")
        if float(simulation["end_time"]) <= 0.0:
            raise ValueError("end_time must be positive.")
        if float(simulation["cfl"]) <= 0.0:
            raise ValueError("cfl must be positive.")
        if float(output["output_interval"]) <= 0.0:
            raise ValueError("output_interval must be positive.")
        if float(grid["dx"]) <= 0.0 or float(grid["dy"]) <= 0.0:
            raise ValueError("dx and dy must be positive.")
        if float(simulation["reference_density"]) <= 0.0:
            raise ValueError("reference_density must be positive.")
        if float(simulation["kinematic_viscosity"]) < 0.0:
            raise ValueError("kinematic_viscosity must be non-negative.")
        if float(simulation["density_diffusivity"]) < 0.0:
            raise ValueError("density_diffusivity must be non-negative.")

    def _apply_updates(self, section: str, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if value is not None:
                self.config[section][key] = value
        self.validate()

    @classmethod
    def _normalize_config(cls, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(config, dict):
            raise TypeError("Config root must be a JSON object.")

        cls._reject_legacy_keys(config, scope="root")

        normalized = deepcopy(DEFAULT_CONFIG)

        root_updates = {
            "grid": {key: config[key] for key in GRID_KEYS if key in config},
            "simulation": {key: config[key] for key in SIMULATION_KEYS if key in config},
            "output": {key: config[key] for key in OUTPUT_KEYS if key in config},
        }
        cls._merge_sections(normalized, root_updates)

        for section_name, allowed_keys in (
            ("grid", GRID_KEYS),
            ("simulation", SIMULATION_KEYS),
            ("output", OUTPUT_KEYS),
        ):
            section_value = config.get(section_name)
            if section_value is None:
                continue
            if not isinstance(section_value, dict):
                raise TypeError(f"Config field '{section_name}' must be a JSON object.")
            cls._reject_legacy_keys(section_value, scope=section_name)
            unknown_keys = set(section_value) - allowed_keys
            if unknown_keys:
                raise KeyError(
                    f"Unknown config key(s) in {section_name}: {', '.join(sorted(unknown_keys))}"
                )
            normalized[section_name].update(section_value)

        unknown_root_keys = set(config) - {"grid", "simulation", "output"} - GRID_KEYS - SIMULATION_KEYS - OUTPUT_KEYS
        if unknown_root_keys:
            raise KeyError(f"Unknown top-level config key(s): {', '.join(sorted(unknown_root_keys))}")

        return normalized

    @staticmethod
    def _merge_sections(target: Dict[str, Dict[str, Any]], updates: Dict[str, Dict[str, Any]]) -> None:
        for section, section_updates in updates.items():
            target[section].update(section_updates)

    @staticmethod
    def _reject_legacy_keys(values: Dict[str, Any], *, scope: str) -> None:
        found = LEGACY_KEYS.intersection(values)
        if found:
            key = sorted(found)[0]
            raise ValueError(f"Legacy config key '{key}' found in {scope}.")
        if scope == "output" and "directory" in values:
            raise ValueError("Legacy config key 'directory' found in output.")

    @classmethod
    def _list_frames_from_xdmf(cls, series_path: Path) -> list[FrameReference]:
        tree = ET.parse(series_path)
        root = tree.getroot()
        collection = root.find("./Domain/Grid")
        if collection is None:
            raise ValueError(f"Could not find temporal grid collection in {series_path}.")

        frames: list[FrameReference] = []
        for grid in collection.findall("Grid"):
            time_node = grid.find("Time")
            density_item = grid.find("./Attribute[@Name='density_offset']/DataItem")
            if time_node is None or density_item is None or density_item.text is None:
                continue

            relative_path = density_item.text.strip().split(":/", 1)[0]
            frame_path = (series_path.parent / relative_path).resolve()
            frames.append(
                FrameReference(
                    index=len(frames),
                    time=float(time_node.attrib["Value"]),
                    file_path=frame_path,
                )
            )

        return frames

    @classmethod
    def _list_frames_from_hdf5(cls, data_dir: Path) -> list[FrameReference]:
        cls._require_hdf5()
        if not data_dir.exists():
            return []

        frames: list[FrameReference] = []
        for frame_path in sorted(data_dir.glob("frame_*.h5")):
            with h5py.File(frame_path, "r") as handle:
                time_value = float(cls._to_python_scalar(handle.attrs.get("time", 0.0)))
            frames.append(FrameReference(index=len(frames), time=time_value, file_path=frame_path.resolve()))
        return frames

    @staticmethod
    def _drop_leading_time_axis(values: Any) -> Any:
        if getattr(values, "ndim", 0) > 0 and values.shape[0] == 1:
            return values[0]
        return values

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return value
        return value

    @staticmethod
    def _require_numpy() -> None:
        if np is None:
            raise ImportError("numpy is required for grid helpers and frame loading.")

    @classmethod
    def _require_hdf5(cls) -> None:
        cls._require_numpy()
        if h5py is None:
            raise ImportError("h5py is required to read simulation output files.")


__all__ = ["DEFAULT_CONFIG", "FluidSimIO", "FrameData", "FrameReference"]
