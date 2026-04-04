from __future__ import annotations

from typing import Any

try:
    import h5py
except ImportError:  # pragma: no cover - depends on the local Python environment
    h5py = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on the local Python environment
    np = None


def require_numpy() -> None:
    if np is None:
        raise ImportError("numpy is required for frame construction and HDF5 IO.")


def require_h5py() -> None:
    if h5py is None:
        raise ImportError("h5py is required for frame HDF5 IO.")


def to_float32_vector(values: Any, *, name: str) -> Any:
    require_numpy()
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be convertible to a flat float32 vector.")
    return array


__all__ = ["h5py", "np", "require_h5py", "require_numpy", "to_float32_vector"]
