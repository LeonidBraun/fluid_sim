from __future__ import annotations

from pathlib import Path
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import subprocess

import matplotlib.pyplot as plt

import fluid_sim_io as fs

FILE = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VortexRingConfig:
    size = (2, 1, 1)
    resolution = 0.01
    density = 1


class VortexRing:

    def __init__(self, test_config, config=VortexRingConfig()) -> None:
        self.folder = Path(test_config["folder"])
        self.solver = Path(test_config["solver"])
        self.config = config

    def create(self):

        h = self.config.resolution
        shape = tuple([int(np.ceil(s / h)) for s in self.config.size])
        print(shape)

        origin = -0.5 * np.array(shape) * h

        X = (
            np.mgrid[: shape[0], : shape[1], : shape[2]]
            .transpose((1, 2, 3, 0))
            .astype(np.float32, order="C")
        ) * h + origin[None, None, None, :]

        u = np.exp(-np.sum(10 * (X - [[[[origin[0] / 2, 0, 0]]]]) ** 2, axis=-1))

        U = X * 0
        U[..., 0] = u

        dty_off = X[..., 0] * 0
        mom = (self.config.density + dty_off)[..., None] * U

        grid = fs.Grid(
            shape,
            h,
            cell_attributes={"momentum": mom, "density_offset": dty_off},
        )

        grid.write_hdf5(folder / "grid.h5")

        grid = fs.Grid.read_hdf5(folder / "grid.h5")

    def run(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    WORKDIR = Path("/mnt/c/Users/LeonidBraun/work_dir").resolve()
    SOLVER = Path("/home/braun/repos/fluid_sim/solver/build/fluid_sim").resolve()

    folder = WORKDIR / VortexRing.__name__
    folder.mkdir(exist_ok=True)

    test_config = {"solver": SOLVER, "folder": folder}

    case = VortexRing(test_config)
    case.create()
    case.run()
    case.evaluate()
