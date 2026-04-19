from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import h5py
from xml.sax.saxutils import escape

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

    @classmethod
    def write_xdmf_for_hdf5(
        cls, hdf5_path: str | Path, *, xdmf_path: str | Path | None = None
    ) -> Path:
        hdf5_path = Path(hdf5_path)
        if xdmf_path is None:
            xdmf_path = hdf5_path.with_suffix(".xdmf")
        xdmf_path = Path(xdmf_path)

        with h5py.File(hdf5_path, "r") as handle:
            shape = tuple(int(axis) for axis in np.asarray(handle["shape"], dtype=np.int32))
            if len(shape) != 3:
                raise ValueError("Grid shape stored in HDF5 must have exactly three entries.")
            h = float(np.asarray(handle["h"], dtype=np.float32))
            origin = np.asarray(handle["origin"], dtype=np.float32)

            cell_attributes = {
                name: tuple(int(axis) for axis in dataset.shape)
                for name, dataset in handle.get("cell_attributes", {}).items()
            }
            point_attributes = {
                name: tuple(int(axis) for axis in dataset.shape)
                for name, dataset in handle.get("point_attributes", {}).items()
            }

        return _write_xdmf(
            xdmf_path=xdmf_path,
            hdf5_path=hdf5_path,
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

    def write_xdmf(
        self, path: str | Path, *, hdf5_path: str | Path | None = None
    ) -> Path:
        self._validate_attributes()
        requested_path = Path(path)
        if hdf5_path is None and requested_path.suffix.lower() == ".h5":
            hdf5_path = requested_path
            xdmf_path = requested_path.with_suffix(".xdmf")
        else:
            xdmf_path = requested_path
            if hdf5_path is None:
                hdf5_path = xdmf_path.with_suffix(".h5")
        hdf5_path = Path(hdf5_path)

        self.write_hdf5(hdf5_path)
        return _write_xdmf(
            xdmf_path=xdmf_path,
            hdf5_path=hdf5_path,
            shape=self.shape,
            h=self.h,
            origin=self.origin,
            cell_attributes={name: values.shape for name, values in self.cell_attributes.items()},
            point_attributes={name: values.shape for name, values in self.point_attributes.items()},
        )

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


def _xdmf_attribute_type(shape: tuple[int, ...]) -> str:
    if len(shape) == 3:
        return "Scalar"
    if len(shape) == 4:
        return "Vector" if shape[-1] == 3 else "Matrix"
    return "Tensor"


def _xdmf_dimensions(shape: tuple[int, ...]) -> str:
    leading = shape[:3][::-1]
    trailing = shape[3:]
    return " ".join(str(int(axis)) for axis in (*leading, *trailing))


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


def _write_xdmf_attribute(
    *,
    stream,
    hdf5_ref: str,
    name: str,
    shape: tuple[int, ...],
    center: str,
    dataset_path: str,
) -> None:
    escaped_name = escape(name)
    stream.write(
        f'      <Attribute Name="{escaped_name}" AttributeType="{_xdmf_attribute_type(shape)}" Center="{center}">\n'
    )
    stream.write(
        f'        <DataItem Dimensions="{_xdmf_dimensions(shape)}" NumberType="Float" Precision="4" Format="HDF">{escape(hdf5_ref)}:{dataset_path}</DataItem>\n'
    )
    stream.write("      </Attribute>\n")


def _write_xdmf(
    *,
    xdmf_path: Path,
    hdf5_path: Path,
    shape: tuple[int, int, int],
    h: float,
    origin,
    cell_attributes: dict[str, tuple[int, ...]],
    point_attributes: dict[str, tuple[int, ...]],
) -> Path:
    xdmf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        hdf5_ref = hdf5_path.relative_to(xdmf_path.parent).as_posix()
    except ValueError:
        if hdf5_path.parent.resolve() == xdmf_path.parent.resolve():
            hdf5_ref = hdf5_path.name
        else:
            hdf5_ref = hdf5_path.as_posix()
    nx, ny, nz = shape
    ox, oy, oz = (float(value) for value in origin)

    with xdmf_path.open("w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0" ?>\n')
        stream.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        stream.write('<Xdmf Version="3.0">\n')
        stream.write("  <Domain>\n")
        stream.write('    <Grid Name="grid" GridType="Uniform">\n')
        stream.write(
            f'      <Topology TopologyType="3DCoRectMesh" Dimensions="{nz + 1} {ny + 1} {nx + 1}"/>\n'
        )
        stream.write('      <Geometry GeometryType="ORIGIN_DXDYDZ">\n')
        stream.write(
            f'        <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">{oz} {oy} {ox}</DataItem>\n'
        )
        stream.write(
            f'        <DataItem Dimensions="3" NumberType="Float" Precision="4" Format="XML">{h} {h} {h}</DataItem>\n'
        )
        stream.write("      </Geometry>\n")

        for name, attribute_shape in cell_attributes.items():
            _write_xdmf_attribute(
                stream=stream,
                hdf5_ref=hdf5_ref,
                name=name,
                shape=attribute_shape,
                center="Cell",
                dataset_path=f"/cell_attributes/{name}",
            )

        for name, attribute_shape in point_attributes.items():
            _write_xdmf_attribute(
                stream=stream,
                hdf5_ref=hdf5_ref,
                name=name,
                shape=attribute_shape,
                center="Node",
                dataset_path=f"/point_attributes/{name}",
            )

        stream.write("    </Grid>\n")
        stream.write("  </Domain>\n")
        stream.write("</Xdmf>\n")

    return xdmf_path


__all__ = ["Grid"]
