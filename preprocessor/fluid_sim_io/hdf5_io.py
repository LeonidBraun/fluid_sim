from __future__ import annotations

from pathlib import Path

from ._deps import h5py, np, require_h5py, require_numpy


def read_frame_payload(path: str | Path) -> tuple[object, object]:
    require_numpy()
    require_h5py()
    source = Path(path)
    with h5py.File(source, "r") as handle:
        density_offset = np.asarray(handle["density_offset"], dtype=np.float32).reshape(-1)
        momentum = np.asarray(handle["momentum"], dtype=np.float32).reshape(-1)
    return density_offset, momentum


def write_frame_payload(path: str | Path, density_offset, momentum, nx: int, ny: int, nz: int) -> Path:
    require_numpy()
    require_h5py()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as handle:
        handle.create_dataset(
            "density_offset",
            data=density_offset.reshape(int(nz), int(ny), int(nx)),
            dtype="f4",
        )
        handle.create_dataset(
            "momentum",
            data=momentum.reshape(int(nz), int(ny), int(nx), 3),
            dtype="f4",
        )
    return destination


__all__ = ["read_frame_payload", "write_frame_payload"]
