from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from ._deps import np, require_numpy, to_float32_vector
from .hdf5_io import read_frame_payload, write_frame_payload
from .json_io import read_relaxed_json, write_json
from .paths import resolve_sibling_path

T = TypeVar("T")


@dataclass(frozen=True)
class Filed(Generic[T]):
    file: str
    data: T

    def with_file(self, file: str | Path) -> "Filed[T]":
        return replace(self, file=Path(file).as_posix())

    def with_data(self, data: T) -> "Filed[T]":
        return replace(self, data=data)


@dataclass(frozen=True)
class Frame:
    density_offset: Any = field(default_factory=tuple)
    momentum: Any = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "density_offset", to_float32_vector(self.density_offset, name="density_offset"))
        object.__setattr__(self, "momentum", to_float32_vector(self.momentum, name="momentum"))

    @classmethod
    def zeros(cls, nx: int, ny: int, nz: int = 1) -> "Frame":
        require_numpy()
        cell_count = int(nx) * int(ny) * int(nz)
        return cls(
            density_offset=np.zeros(cell_count, dtype=np.float32),
            momentum=np.zeros(cell_count * 3, dtype=np.float32),
        )

    @classmethod
    def read_hdf5(cls, path: str | Path, nx: int, ny: int, nz: int = 1) -> "Frame":
        density_offset, momentum = read_frame_payload(path)
        frame = cls(density_offset=density_offset, momentum=momentum)
        frame.validate(nx, ny, nz)
        return frame

    def write_hdf5(self, path: str | Path, nx: int, ny: int, nz: int = 1) -> Path:
        self.validate(nx, ny, nz)
        return write_frame_payload(path, self.density_offset, self.momentum, nx, ny, nz)

    def validate(self, nx: int, ny: int, nz: int = 1) -> None:
        cell_count = int(nx) * int(ny) * int(nz)
        if self.density_offset.size != cell_count:
            raise ValueError(
                f"density_offset size {self.density_offset.size} does not match nx*ny*nz={cell_count}."
            )
        if self.momentum.size != cell_count * 3:
            raise ValueError(
                f"momentum size {self.momentum.size} does not match 3*nx*ny*nz={cell_count * 3}."
            )

    def density_offset_grid(self, nx: int, ny: int, nz: int = 1) -> Any:
        self.validate(nx, ny, nz)
        return self.density_offset.reshape(int(nz), int(ny), int(nx))

    def momentum_grid(self, nx: int, ny: int, nz: int = 1) -> Any:
        self.validate(nx, ny, nz)
        return self.momentum.reshape(int(nz), int(ny), int(nx), 3)


@dataclass(frozen=True)
class StateGrid:
    frame: Optional[Filed[Frame]] = None
    nx: int = 200
    ny: int = 100
    nz: int = 1
    h: float = 1.0

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any], state_path: str | Path) -> "StateGrid":
        grid = cls(
            nx=int(payload["nx"]),
            ny=int(payload["ny"]),
            nz=int(payload.get("nz", 1)),
            h=float(payload["h"]),
        )
        frame_name = payload.get("frame")
        if frame_name is None:
            return grid
        frame_path = resolve_sibling_path(state_path, frame_name)
        frame = Frame.read_hdf5(frame_path, grid.nx, grid.ny, grid.nz)
        return replace(grid, frame=Filed(file=str(frame_name), data=frame))

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"nx": self.nx, "ny": self.ny, "nz": self.nz, "h": self.h}
        if self.frame is not None and self.frame.file:
            payload["frame"] = self.frame.file
        return payload


@dataclass(frozen=True)
class StateMaterialProperties:
    speed_of_sound: float = 10.0
    reference_density: float = 1.225
    kinematic_viscosity: float = 1.5e-5
    density_diffusivity: float = 1.0e-5

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "StateMaterialProperties":
        return cls(
            speed_of_sound=float(payload["speed_of_sound"]),
            reference_density=float(payload["reference_density"]),
            kinematic_viscosity=float(payload["kinematic_viscosity"]),
            density_diffusivity=float(payload["density_diffusivity"]),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "speed_of_sound": self.speed_of_sound,
            "reference_density": self.reference_density,
            "kinematic_viscosity": self.kinematic_viscosity,
            "density_diffusivity": self.density_diffusivity,
        }


@dataclass(frozen=True)
class State:
    time: float = 0.0
    grid: StateGrid = field(default_factory=StateGrid)
    material: StateMaterialProperties = field(default_factory=StateMaterialProperties)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any], state_path: str | Path) -> "State":
        return cls(
            time=float(payload["time"]),
            grid=StateGrid.from_json_dict(payload["grid"], state_path),
            material=StateMaterialProperties.from_json_dict(payload["material"]),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "State":
        source = Path(path)
        payload = read_relaxed_json(source)
        if not isinstance(payload, dict):
            raise TypeError("State file must contain a JSON object.")
        return cls.from_json_dict(payload, source)

    def to_json_dict(self) -> dict[str, Any]:
        return {"time": self.time, "grid": self.grid.to_json_dict(), "material": self.material.to_json_dict()}

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_json_dict(), indent=indent)


@dataclass(frozen=True)
class SolverSettings:
    cfl: float = 0.05
    pressure_iterations: int = 60

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "SolverSettings":
        return cls(cfl=float(payload["cfl"]), pressure_iterations=int(payload["pressure_iterations"]))

    def to_json_dict(self) -> dict[str, Any]:
        return {"cfl": self.cfl, "pressure_iterations": self.pressure_iterations}


@dataclass(frozen=True)
class OutputSettings:
    end_time: float = 10.0
    output_interval: float = 1.0

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "OutputSettings":
        return cls(end_time=float(payload["end_time"]), output_interval=float(payload["output_interval"]))

    def to_json_dict(self) -> dict[str, Any]:
        return {"end_time": self.end_time, "output_interval": self.output_interval}


@dataclass(frozen=True)
class RunConfig:
    solver_settings: SolverSettings = field(default_factory=SolverSettings)
    output_settings: OutputSettings = field(default_factory=OutputSettings)
    init_state: Filed[State] = field(default_factory=lambda: Filed("default_state.json", State()))
    outputs: tuple[str, ...] = ()

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any], config_path: str | Path) -> "RunConfig":
        source = Path(config_path)
        init_state_file = str(payload["init_state"])
        init_state_path = resolve_sibling_path(source, init_state_file)
        return cls(
            solver_settings=SolverSettings.from_json_dict(payload["solver_settings"]),
            output_settings=OutputSettings.from_json_dict(payload["output_settings"]),
            init_state=Filed(file=init_state_file, data=State.read_json(init_state_path)),
            outputs=tuple(str(output) for output in payload.get("outputs", [])),
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "RunConfig":
        source = Path(path)
        payload = read_relaxed_json(source)
        if not isinstance(payload, dict):
            raise TypeError("Run config file must contain a JSON object.")
        return cls.from_json_dict(payload, source)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "solver_settings": self.solver_settings.to_json_dict(),
            "output_settings": self.output_settings.to_json_dict(),
            "init_state": self.init_state.file,
            "outputs": list(self.outputs),
        }

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_json_dict(), indent=indent)

    def write_case(self, path: str | Path, *, indent: int = 2) -> tuple[Path, Path]:
        config_path = Path(path)
        state_path = resolve_sibling_path(config_path, self.init_state.file)
        self.write_json(config_path, indent=indent)
        self.init_state.data.write_json(state_path, indent=indent)
        frame = self.init_state.data.grid.frame
        if frame is not None and frame.file:
            frame_path = resolve_sibling_path(state_path, frame.file)
            frame.data.write_hdf5(
                frame_path,
                self.init_state.data.grid.nx,
                self.init_state.data.grid.ny,
                self.init_state.data.grid.nz,
            )
        return config_path, state_path

    def load_output_state(self, config_path: str | Path, index: int) -> Filed[State]:
        state_file = self.outputs[index]
        state_path = resolve_sibling_path(config_path, state_file)
        return Filed(file=state_file, data=State.read_json(state_path))

    def load_output_states(self, config_path: str | Path) -> tuple[Filed[State], ...]:
        return tuple(self.load_output_state(config_path, index) for index in range(len(self.outputs)))



__all__ = [
    "Filed",
    "Frame",
    "OutputSettings",
    "RunConfig",
    "SolverSettings",
    "State",
    "StateGrid",
    "StateMaterialProperties",
]
