from __future__ import annotations

import json
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


DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "solver_settings": {
        "cfl": 0.05,
        "pressure_iterations": 60,
    },
    "output_settings": {
        "end_time": 10.0,
        "output_interval": 1.0,
    },
    "init_state": "init_state.json",
    "outputs": [],
}

DEFAULT_INIT_STATE: Dict[str, Any] = {
    "time": 0.0,
    "grid": {
        "nx": 256,
        "ny": 128,
        "h": 1.0,
        "initial_density": 1.225,
    },
    "material": {
        "reference_density": 1.225,
        "kinematic_viscosity": 1.5e-5,
        "density_diffusivity": 1.0e-5,
    },
    "frame": "init_frame.h5",
}

SOLVER_KEYS = {"cfl", "pressure_iterations"}
OUTPUT_KEYS = {"end_time", "output_interval"}
GRID_KEYS = {"nx", "ny", "h", "initial_density"}
MATERIAL_KEYS = {"reference_density", "kinematic_viscosity", "density_diffusivity"}


@dataclass(frozen=True)
class StateReference:
    index: int
    time: float
    state_path: Path
    frame_path: Path


@dataclass
class StateData:
    index: int
    time: float
    state_path: Path
    frame_path: Path
    grid: Dict[str, Any]
    material: Dict[str, Any]
    attributes: Dict[str, Any]
    density_offset: Any
    velocity: Any


FrameReference = StateReference
FrameData = StateData


class FluidSimIO:
    """Create run configs and load simulation states for preprocessing/postprocessing."""

    def __init__(
        self,
        run_config_path: str | Path,
        run_config: Optional[Dict[str, Any]] = None,
        init_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_config_path = Path(run_config_path).expanduser()
        self.run_config = self._normalize_run_config(run_config or deepcopy(DEFAULT_RUN_CONFIG))
        self.init_state = self._normalize_state_dict(init_state or deepcopy(DEFAULT_INIT_STATE))
        self.validate()

    @classmethod
    def from_run_config(cls, run_config_path: str | Path) -> "FluidSimIO":
        run_config_path = Path(run_config_path).expanduser()
        run_config = cls._load_json_file(run_config_path)
        if not isinstance(run_config, dict):
            raise TypeError("Run config root must be a JSON object.")
        init_state_path = cls._resolve_reference_path(run_config_path, run_config["init_state"])
        init_state = cls._load_json_file(init_state_path)
        if not isinstance(init_state, dict):
            raise TypeError("Init state root must be a JSON object.")
        return cls(run_config_path=run_config_path, run_config=run_config, init_state=init_state)

    from_config = from_run_config

    @property
    def run_root(self) -> Path:
        return self.run_config_path.parent

    @property
    def init_state_path(self) -> Path:
        return self._resolve_reference_path(self.run_config_path, self.run_config["init_state"])

    @property
    def output_root(self) -> Path:
        return self.run_root / "outputs"

    @property
    def data_dir(self) -> Path:
        return self.output_root / "data"

    def to_dict(self) -> Dict[str, Any]:
        return self.to_run_dict()

    def to_run_dict(self) -> Dict[str, Any]:
        return deepcopy(self.run_config)

    def to_init_state_dict(self) -> Dict[str, Any]:
        return deepcopy(self.init_state)

    def set_solver(self, *, cfl: Optional[float] = None, pressure_iterations: Optional[int] = None) -> None:
        self._apply_updates("solver_settings", {"cfl": cfl, "pressure_iterations": pressure_iterations})

    def set_output(self, *, end_time: Optional[float] = None, output_interval: Optional[float] = None) -> None:
        self._apply_updates("output_settings", {"end_time": end_time, "output_interval": output_interval})

    def set_grid(
        self,
        *,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        h: Optional[float] = None,
        initial_density: Optional[float] = None,
    ) -> None:
        self._apply_state_updates("grid", {"nx": nx, "ny": ny, "h": h, "initial_density": initial_density})

    def set_material(
        self,
        *,
        reference_density: Optional[float] = None,
        kinematic_viscosity: Optional[float] = None,
        density_diffusivity: Optional[float] = None,
    ) -> None:
        self._apply_state_updates(
            "material",
            {
                "reference_density": reference_density,
                "kinematic_viscosity": kinematic_viscosity,
                "density_diffusivity": density_diffusivity,
            },
        )

    def set_init_state(self, *, time: Optional[float] = None, frame: Optional[str | Path] = None) -> None:
        if time is not None:
            self.init_state["time"] = float(time)
        if frame is not None:
            self.init_state["frame"] = self._serialize_reference(frame)
        self.validate()

    def write_run_config(self, path: str | Path | None = None, indent: int = 2) -> Path:
        self.validate()
        destination = Path(path).expanduser() if path is not None else self.run_config_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.run_config, handle, indent=indent)
            handle.write("\n")
        return destination

    def write_init_state(self, path: str | Path | None = None, indent: int = 2) -> Path:
        self.validate()
        destination = Path(path).expanduser() if path is not None else self.init_state_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.init_state, handle, indent=indent)
            handle.write("\n")
        return destination

    def write_case(
        self,
        run_config_path: str | Path | None = None,
        init_state_path: str | Path | None = None,
        indent: int = 2,
    ) -> tuple[Path, Path]:
        if run_config_path is not None:
            self.run_config_path = Path(run_config_path).expanduser()
        if init_state_path is not None:
            self.run_config["init_state"] = self._reference_from_base(self.run_config_path, Path(init_state_path))
        run_path = self.write_run_config(indent=indent)
        state_path = self.write_init_state(indent=indent)
        return run_path, state_path

    def write_config(
        self,
        path: str | Path | None = None,
        init_state_path: str | Path | None = None,
        indent: int = 2,
    ) -> tuple[Path, Path]:
        return self.write_case(run_config_path=path, init_state_path=init_state_path, indent=indent)

    def cell_centers(self) -> tuple[Any, Any]:
        self._require_numpy()
        nx = int(self.init_state["grid"]["nx"])
        ny = int(self.init_state["grid"]["ny"])
        h = float(self.init_state["grid"]["h"])
        x = (np.arange(nx, dtype=float) + 0.5) * h
        y = (np.arange(ny, dtype=float) + 0.5) * h
        return np.meshgrid(x, y, indexing="xy")

    def list_state_files(self) -> list[StateReference]:
        references: list[StateReference] = []
        for index, state_path in enumerate(self._output_state_paths()):
            state_dict = self._normalize_state_dict(self._load_json_file(state_path))
            frame_path = self._resolve_reference_path(state_path, state_dict["frame"])
            references.append(
                StateReference(
                    index=index,
                    time=float(state_dict["time"]),
                    state_path=state_path,
                    frame_path=frame_path,
                )
            )
        return references

    def list_frames(self) -> list[FrameReference]:
        return self.list_state_files()

    def read_initial_state(self) -> StateData:
        return self.read_state_file(self.init_state_path, index=0)

    def read_state(self, index: int) -> StateData:
        references = self.list_state_files()
        try:
            reference = references[index]
        except IndexError as exc:
            raise IndexError(f"State index {index} is out of range for {len(references)} output state(s).") from exc
        return self.read_state_file(reference.state_path, index=reference.index)

    def read_frame(self, index: int) -> FrameData:
        return self.read_state(index)

    def read_last_state(self) -> StateData:
        references = self.list_state_files()
        if not references:
            raise FileNotFoundError(f"No output state files found for {self.run_config_path}.")
        return self.read_state(references[-1].index)

    def read_last_frame(self) -> FrameData:
        return self.read_last_state()

    def read_all_states(self) -> list[StateData]:
        return [self.read_state(reference.index) for reference in self.list_state_files()]

    def read_all_frames(self) -> list[FrameData]:
        return self.read_all_states()

    def read_state_file(self, state_path: str | Path, *, index: int = 0) -> StateData:
        self._require_hdf5()
        state_path = Path(state_path).expanduser()
        state_dict = self._normalize_state_dict(self._load_json_file(state_path))
        frame_path = self._resolve_reference_path(state_path, state_dict["frame"])

        with h5py.File(frame_path, "r") as handle:
            attributes = {key: self._to_python_scalar(value) for key, value in handle.attrs.items()}
            density_offset = np.asarray(handle["density_offset"])
            velocity = np.asarray(handle["velocity"])

        density_offset = self._drop_leading_time_axis(density_offset)
        velocity = self._drop_leading_time_axis(velocity)

        return StateData(
            index=index,
            time=float(state_dict["time"]),
            state_path=state_path,
            frame_path=frame_path,
            grid=deepcopy(state_dict["grid"]),
            material=deepcopy(state_dict["material"]),
            attributes=attributes,
            density_offset=density_offset,
            velocity=velocity,
        )

    def validate(self) -> None:
        solver = self.run_config["solver_settings"]
        output = self.run_config["output_settings"]
        state = self.init_state
        grid = state["grid"]
        material = state["material"]

        if not self.run_config["init_state"]:
            raise ValueError("init_state must not be empty.")
        if float(solver["cfl"]) <= 0.0:
            raise ValueError("solver_settings.cfl must be positive.")
        if int(solver["pressure_iterations"]) < 1:
            raise ValueError("solver_settings.pressure_iterations must be positive.")
        if float(output["output_interval"]) <= 0.0:
            raise ValueError("output_settings.output_interval must be positive.")
        if float(output["end_time"]) < float(state["time"]):
            raise ValueError("output_settings.end_time must be greater than or equal to init_state.time.")
        if not state["frame"]:
            raise ValueError("init_state.frame must not be empty.")
        if int(grid["nx"]) < 8 or int(grid["ny"]) < 8:
            raise ValueError("grid must be at least 8x8.")
        if float(grid["h"]) <= 0.0:
            raise ValueError("grid.h must be positive.")
        if float(grid["initial_density"]) <= 0.0:
            raise ValueError("grid.initial_density must be positive.")
        if float(material["reference_density"]) <= 0.0:
            raise ValueError("material.reference_density must be positive.")
        if float(material["kinematic_viscosity"]) < 0.0:
            raise ValueError("material.kinematic_viscosity must be non-negative.")
        if float(material["density_diffusivity"]) < 0.0:
            raise ValueError("material.density_diffusivity must be non-negative.")

    def _apply_updates(self, section: str, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if value is not None:
                self.run_config[section][key] = value
        self.validate()

    def _apply_state_updates(self, section: str, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if value is not None:
                self.init_state[section][key] = value
        self.validate()

    def _output_state_paths(self) -> list[Path]:
        outputs = list(self.run_config.get("outputs", []))
        if outputs:
            return [self._resolve_reference_path(self.run_config_path, output) for output in outputs]
        if not self.data_dir.exists():
            return []
        return sorted(self.data_dir.glob("state_*.json"))

    @classmethod
    def _normalize_run_config(cls, run_config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(run_config, dict):
            raise TypeError("Run config root must be a JSON object.")

        normalized = deepcopy(DEFAULT_RUN_CONFIG)

        unknown_root = set(run_config) - {"solver_settings", "output_settings", "init_state", "outputs"}
        if unknown_root:
            raise KeyError(f"Unknown run config key(s): {', '.join(sorted(unknown_root))}")

        if "solver_settings" in run_config:
            solver = run_config["solver_settings"]
            if not isinstance(solver, dict):
                raise TypeError("Run config field 'solver_settings' must be a JSON object.")
            unknown_solver = set(solver) - SOLVER_KEYS
            if unknown_solver:
                raise KeyError(f"Unknown solver_settings key(s): {', '.join(sorted(unknown_solver))}")
            normalized["solver_settings"].update(solver)

        if "output_settings" in run_config:
            output = run_config["output_settings"]
            if not isinstance(output, dict):
                raise TypeError("Run config field 'output_settings' must be a JSON object.")
            unknown_output = set(output) - OUTPUT_KEYS
            if unknown_output:
                raise KeyError(f"Unknown output_settings key(s): {', '.join(sorted(unknown_output))}")
            normalized["output_settings"].update(output)

        if "init_state" in run_config:
            normalized["init_state"] = cls._serialize_reference(run_config["init_state"])

        if "outputs" in run_config:
            outputs = run_config["outputs"]
            if not isinstance(outputs, list):
                raise TypeError("Run config field 'outputs' must be an array.")
            normalized["outputs"] = [cls._serialize_reference(output) for output in outputs]

        return normalized

    @classmethod
    def _normalize_state_dict(cls, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state_dict, dict):
            raise TypeError("State root must be a JSON object.")

        normalized = deepcopy(DEFAULT_INIT_STATE)
        unknown_root = set(state_dict) - {"time", "grid", "material", "frame"}
        if unknown_root:
            raise KeyError(f"Unknown state key(s): {', '.join(sorted(unknown_root))}")

        if "time" in state_dict:
            normalized["time"] = state_dict["time"]

        if "grid" in state_dict:
            grid = state_dict["grid"]
            if not isinstance(grid, dict):
                raise TypeError("State field 'grid' must be a JSON object.")
            unknown_grid = set(grid) - GRID_KEYS
            if unknown_grid:
                raise KeyError(f"Unknown grid key(s): {', '.join(sorted(unknown_grid))}")
            normalized["grid"].update(grid)

        if "material" in state_dict:
            material = state_dict["material"]
            if not isinstance(material, dict):
                raise TypeError("State field 'material' must be a JSON object.")
            unknown_material = set(material) - MATERIAL_KEYS
            if unknown_material:
                raise KeyError(f"Unknown material key(s): {', '.join(sorted(unknown_material))}")
            normalized["material"].update(material)

        if "frame" in state_dict:
            normalized["frame"] = cls._serialize_reference(state_dict["frame"])

        return normalized

    @classmethod
    def _load_json_file(cls, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read()
        return json.loads(cls._strip_trailing_commas(cls._strip_line_comments(text)))

    @staticmethod
    def _strip_line_comments(text: str) -> str:
        result: list[str] = []
        in_string = False
        escaping = False

        index = 0
        while index < len(text):
            char = text[index]

            if in_string:
                result.append(char)
                if escaping:
                    escaping = False
                elif char == "\\":
                    escaping = True
                elif char == '"':
                    in_string = False
                index += 1
                continue

            if char == '"':
                in_string = True
                result.append(char)
                index += 1
                continue

            if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
                while index < len(text) and text[index] != "\n":
                    index += 1
                continue

            result.append(char)
            index += 1

        return "".join(result)

    @staticmethod
    def _strip_trailing_commas(text: str) -> str:
        result: list[str] = []
        in_string = False
        escaping = False
        index = 0

        while index < len(text):
            char = text[index]

            if in_string:
                result.append(char)
                if escaping:
                    escaping = False
                elif char == "\\":
                    escaping = True
                elif char == '"':
                    in_string = False
                index += 1
                continue

            if char == '"':
                in_string = True
                result.append(char)
                index += 1
                continue

            if char == ",":
                lookahead = index + 1
                while lookahead < len(text) and text[lookahead].isspace():
                    lookahead += 1
                if lookahead < len(text) and text[lookahead] in "}]":
                    index += 1
                    continue

            result.append(char)
            index += 1

        return "".join(result)

    @classmethod
    def _resolve_reference_path(cls, base_path: Path, reference: str | Path) -> Path:
        reference_path = Path(reference).expanduser()
        if cls._is_absolute_reference(str(reference)):
            return reference_path

        candidate = (base_path.parent / reference_path).resolve()
        if candidate.exists():
            return candidate

        normalized_reference = Path(*[part for part in reference_path.parts if part not in {"output", "outputs"}])
        if reference_path.parts and reference_path.parts[0] in {"output", "outputs"}:
            fallback = (base_path.parent / "outputs" / normalized_reference).resolve()
            if fallback.exists():
                return fallback

        return candidate

    @staticmethod
    def _reference_from_base(base_path: Path, target_path: Path) -> str:
        target_path = target_path.expanduser()
        if FluidSimIO._is_absolute_reference(str(target_path)):
            return target_path.as_posix()
        return Path(target_path).as_posix()

    @staticmethod
    def _serialize_reference(value: str | Path) -> str:
        return Path(value).as_posix()

    @staticmethod
    def _is_absolute_reference(value: str) -> bool:
        return Path(value).is_absolute() or (len(value) >= 3 and value[1] == ":" and value[2] in "\\/")

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
            raise ImportError("h5py is required to read HDF5 frame files.")


__all__ = [
    "DEFAULT_INIT_STATE",
    "DEFAULT_RUN_CONFIG",
    "FluidSimIO",
    "FrameData",
    "FrameReference",
    "StateData",
    "StateReference",
]
