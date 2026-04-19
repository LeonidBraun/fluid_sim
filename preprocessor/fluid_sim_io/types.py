from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar
import numpy as np

from .hdf5_io import read_frame_payload, write_frame_payload
from .grid import Grid
from .json_io import read_relaxed_json, write_json
from .paths import resolve_sibling_path

T = TypeVar("T")


@dataclass(frozen=True)
class Filed(Generic[T]):
    data: T
    file: str = ""


@dataclass(frozen=True)
class Frame:
    density_offset: Any = field(default_factory=tuple)
    momentum: Any = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "density_offset",
            np.asarray(self.density_offset, dtype=np.float32).reshape(-1),
        )
        object.__setattr__(
            self,
            "momentum",
            np.asarray(self.momentum, dtype=np.float32).reshape(-1),
        )

    @classmethod
    def from_fields(cls, density_offset: Any, momentum: Any) -> "Frame":
        density = np.ascontiguousarray(np.asarray(density_offset, dtype=np.float32))
        if density.ndim == 2:
            density = density[:, :, np.newaxis]
        if density.ndim != 3:
            raise ValueError(
                "density_offset field must have shape (nx, ny, nz) or (nx, ny)."
            )

        momentum_field = np.ascontiguousarray(np.asarray(momentum, dtype=np.float32))
        if momentum_field.ndim == 3 and momentum_field.shape[-1] == 3:
            momentum_field = momentum_field[:, :, np.newaxis, :]
        if momentum_field.ndim != 4 or momentum_field.shape[-1] != 3:
            raise ValueError(
                "momentum field must have shape (nx, ny, nz, 3) or (nx, ny, 3)."
            )
        if momentum_field.shape[:3] != density.shape:
            raise ValueError(
                "density_offset and momentum fields must agree on (nx, ny, nz)."
            )

        # Solver storage order is (nz, ny, nx[, 3]); Python-facing field order is (nx, ny, nz[, 3]).
        density_solver = np.transpose(density, (2, 1, 0))
        momentum_solver = np.transpose(momentum_field, (2, 1, 0, 3))
        return cls(
            density_offset=density_solver.reshape(-1),
            momentum=momentum_solver.reshape(-1),
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
    def from_json_dict(
        cls, payload: dict[str, Any], state_path: str | Path
    ) -> "StateGrid":
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
        return replace(grid, frame=Filed(data=frame, file=str(frame_name)))

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "h": self.h,
        }
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
        return {
            "time": self.time,
            "grid": self.grid.to_json_dict(),
            "material": self.material.to_json_dict(),
        }

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        return write_json(path, self.to_json_dict(), indent=indent)


@dataclass(frozen=True)
class SolverSettings:
    cfl: float = 1.5
    pressure_iterations: int = 6

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "SolverSettings":
        return cls(
            cfl=float(payload["cfl"]),
            pressure_iterations=int(payload["pressure_iterations"]),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {"cfl": self.cfl, "pressure_iterations": self.pressure_iterations}


@dataclass(frozen=True)
class OutputSettings:
    end_time: float = 10.0
    output_interval: float = 1.0

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "OutputSettings":
        return cls(
            end_time=float(payload["end_time"]),
            output_interval=float(payload["output_interval"]),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {"end_time": self.end_time, "output_interval": self.output_interval}


@dataclass(frozen=True)
class RunConfig:
    solver_settings: SolverSettings = field(default_factory=SolverSettings)
    output_settings: OutputSettings = field(default_factory=OutputSettings)
    init_state: Filed[State] = field(
        default_factory=lambda: Filed(data=State(), file="init_state.json")
    )
    outputs: tuple[str, ...] = ()

    @classmethod
    def from_json_dict(
        cls, payload: dict[str, Any], config_path: str | Path
    ) -> "RunConfig":
        source = Path(config_path)
        init_state_file = str(payload["init_state"])
        init_state_path = resolve_sibling_path(source, init_state_file)
        return cls(
            solver_settings=SolverSettings.from_json_dict(payload["solver_settings"]),
            output_settings=OutputSettings.from_json_dict(payload["output_settings"]),
            init_state=Filed(
                data=State.read_json(init_state_path), file=init_state_file
            ),
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

    def write_case(self, path: str | Path, *, indent: int = 2) -> tuple[Path, Path]:
        config_path = Path(path)
        init_state_file = self.init_state.file or "init_state.json"
        state_path = resolve_sibling_path(config_path, init_state_file)
        frame = self.init_state.data.grid.frame
        if frame is not None and not frame.file:
            frame = replace(frame, file="init_frame.h5")
        init_state = replace(
            self.init_state.data,
            grid=replace(self.init_state.data.grid, frame=frame),
        )
        materialized = replace(
            self,
            init_state=Filed(data=init_state, file=init_state_file),
        )

        write_json(config_path, materialized.to_json_dict(), indent=indent)
        materialized.init_state.data.write_json(state_path, indent=indent)
        frame = materialized.init_state.data.grid.frame
        if frame is not None and frame.file:
            frame_path = resolve_sibling_path(state_path, frame.file)
            frame.data.write_hdf5(
                frame_path,
                materialized.init_state.data.grid.nx,
                materialized.init_state.data.grid.ny,
                materialized.init_state.data.grid.nz,
            )
        return config_path, state_path


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
