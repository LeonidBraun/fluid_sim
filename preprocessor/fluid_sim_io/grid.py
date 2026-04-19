from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import h5py

from .hdf5_io import read_grid_payload, write_grid_payload


@dataclass
class Grid:
    shape: tuple[int, int, int] = (200, 100, 50)
    h: float = 1.0
    origin: Any = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    cell_attributes: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)
    point_attributes: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shape = tuple(int(axis) for axis in self.shape)
        if len(self.shape) != 3:
            raise ValueError("shape must contain exactly three dimensions.")

        origin = np.asarray(self.origin, dtype=np.float32)
        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,).")
        self.origin = origin
        self.cell_attributes = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in self.cell_attributes.items()
        }
        self.point_attributes = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in self.point_attributes.items()
        }
        self._validate_attributes()

    @classmethod
    def read_hdf5(cls, path: str | Path) -> "Grid":
        source = Path(path)
        with h5py.File(source, "r") as handle:
            shape = tuple(
                int(axis) for axis in np.asarray(handle["shape"], dtype=np.int32)
            )
            if len(shape) != 3:
                raise ValueError(
                    "Grid shape stored in HDF5 must have exactly three entries."
                )
            h = float(np.asarray(handle["h"], dtype=np.float32))
            origin = np.asarray(handle["origin"], dtype=np.float32)

            cell_attributes: dict[str, object] = {}
            if "cell_attributes" in handle:
                for name, dataset in handle["cell_attributes"].items():
                    cell_attributes[name] = _from_storage_order(
                        np.asarray(dataset, dtype=np.float32)
                    )

            point_attributes: dict[str, object] = {}
            if "point_attributes" in handle:
                for name, dataset in handle["point_attributes"].items():
                    point_attributes[name] = _from_storage_order(
                        np.asarray(dataset, dtype=np.float32)
                    )

        return cls(
            shape=shape,
            h=h,
            origin=origin,
            cell_attributes=cell_attributes,
            point_attributes=point_attributes,
        )

    def write_hdf5(self, path: str | Path) -> Path:
        self._validate_attributes()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(destination, "w") as handle:
            handle.create_dataset(
                "shape", data=np.asarray(self.shape, dtype=np.int32), dtype="i4"
            )
            handle.create_dataset(
                "h", data=np.asarray(self.h, dtype=np.float32), dtype="f4"
            )
            handle.create_dataset(
                "origin", data=np.asarray(self.origin, dtype=np.float32), dtype="f4"
            )

            cell_group = handle.create_group("cell_attributes")
            for name, values in self.cell_attributes.items():
                cell_group.create_dataset(
                    name,
                    data=_to_storage_order(np.asarray(values, dtype=np.float32)),
                    dtype="f4",
                )

            point_group = handle.create_group("point_attributes")
            for name, values in self.point_attributes.items():
                point_group.create_dataset(
                    name,
                    data=_to_storage_order(np.asarray(values, dtype=np.float32)),
                    dtype="f4",
                )

        return destination
    def _validate_attributes(self) -> None:
        cell_shape = self.shape
        point_shape = tuple(axis + 1 for axis in self.shape)

        for name, values in self.cell_attributes.items():
            if values.ndim < 3 or tuple(values.shape[:3]) != cell_shape:
                raise ValueError(
                    f"cell attribute '{name}' must have shape {cell_shape} or "
                    f"{cell_shape} + trailing component dimensions."
                )

        for name, values in self.point_attributes.items():
            if values.ndim < 3 or tuple(values.shape[:3]) != point_shape:
                raise ValueError(
                    f"point attribute '{name}' must have shape {point_shape} or "
                    f"{point_shape} + trailing component dimensions."
                )


def _to_storage_order(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if values.ndim < 3:
        raise ValueError("grid attributes must have rank at least 3.")
    axes = (2, 1, 0) + tuple(range(3, values.ndim))
    return np.transpose(values, axes)


def _from_storage_order(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if values.ndim < 3:
        raise ValueError("grid attributes must have rank at least 3.")
    axes = (2, 1, 0) + tuple(range(3, values.ndim))
    return np.transpose(values, axes)

__all__ = ["Grid"]
