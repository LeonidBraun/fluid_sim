from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np


def read_frame_payload(path: str | Path, nx: int, ny: int, nz: int) -> tuple[object, object]:
    source = Path(path)
    with h5py.File(source, "r") as handle:
        density_dataset = np.asarray(handle["density_offset"], dtype=np.float32)
        momentum_dataset = np.asarray(handle["momentum"], dtype=np.float32)

    if density_dataset.shape != (int(nx), int(ny), int(nz)):
        raise ValueError(
            "density_offset dataset shape does not match (nx, ny, nz)."
        )
    if momentum_dataset.shape != (int(nx), int(ny), int(nz), 3):
        raise ValueError(
            "momentum dataset shape does not match (nx, ny, nz, 3)."
        )
    return density_dataset.reshape(-1), momentum_dataset.reshape(-1)


def write_frame_payload(path: str | Path, density_offset, momentum, nx: int, ny: int, nz: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        handle.create_dataset(
            "density_offset",
            data=density_offset.reshape(int(nx), int(ny), int(nz)),
            dtype="f4",
        )
        handle.create_dataset(
            "momentum",
            data=momentum.reshape(int(nx), int(ny), int(nz), 3),
            dtype="f4",
        )
    return destination


def read_grid_payload(path: str | Path) -> tuple[tuple[int, int, int], float, object, dict[str, object], dict[str, object]]:
    source = Path(path)
    with h5py.File(source, "r") as handle:
        shape = tuple(int(axis) for axis in np.asarray(handle["shape"], dtype=np.int32))
        if len(shape) != 3:
            raise ValueError("Grid shape stored in HDF5 must have exactly three entries.")
        h = float(np.asarray(handle["h"], dtype=np.float32))
        origin = np.asarray(handle["origin"], dtype=np.float32)

        cell_attributes: dict[str, object] = {}
        if "cell_attributes" in handle:
            for name, dataset in handle["cell_attributes"].items():
                cell_attributes[name] = np.asarray(dataset, dtype=np.float32)

        point_attributes: dict[str, object] = {}
        if "point_attributes" in handle:
            for name, dataset in handle["point_attributes"].items():
                point_attributes[name] = np.asarray(dataset, dtype=np.float32)

    return shape, h, origin, cell_attributes, point_attributes


def write_grid_payload(
    path: str | Path,
    *,
    shape: tuple[int, int, int],
    h: float,
    origin,
    cell_attributes: dict[str, object],
    point_attributes: dict[str, object],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        handle.create_dataset("shape", data=np.asarray(shape, dtype=np.int32), dtype="i4")
        handle.create_dataset("h", data=np.asarray(h, dtype=np.float32), dtype="f4")
        handle.create_dataset("origin", data=np.asarray(origin, dtype=np.float32), dtype="f4")

        cell_group = handle.create_group("cell_attributes")
        for name, values in cell_attributes.items():
            cell_group.create_dataset(name, data=np.asarray(values, dtype=np.float32), dtype="f4")

        point_group = handle.create_group("point_attributes")
        for name, values in point_attributes.items():
            point_group.create_dataset(name, data=np.asarray(values, dtype=np.float32), dtype="f4")

    return destination


__all__ = [
    "read_frame_payload",
    "write_frame_payload",
    "read_grid_payload",
    "write_grid_payload",
]
