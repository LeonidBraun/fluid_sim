from __future__ import annotations

from pathlib import Path
import numpy as np
import subprocess

import matplotlib.pyplot as plt

import fluid_sim_io as fs

FILE = Path(__file__).resolve().parents[1]
WORKDIR = Path("/mnt/c/Users/LeonidBraun/Downloads").resolve()
SOLVER = Path("/home/braun/repos/fluid_sim/solver/build/fluid_sim").resolve()


def total_kinetic_energy(state: fs.State) -> float:
    frame_ref = state.grid.frame
    if frame_ref is None:
        raise ValueError("State does not contain frame data.")

    frame = frame_ref.data
    nx = state.grid.nx
    ny = state.grid.ny
    nz = state.grid.nz
    h = state.grid.h
    rho = state.material.reference_density + frame.density_offset_grid(nx, ny, nz)
    momentum = frame.momentum_grid(nx, ny, nz)
    momentum_squared = (momentum**2).sum(axis=-1)
    return float(0.5 * ((momentum_squared / rho).sum()) * h * h * h)


class Dambreak:
    def create(self):
        case_dir = WORKDIR / "VortexTest"

        if not SOLVER.exists():
            raise FileNotFoundError(f"Solver binary not found: {SOLVER}")

        case_dir.mkdir(parents=True, exist_ok=True)

        nx = 200
        ny = 100
        nz = 50
        h = 0.02

        x = (np.arange(nx, dtype=np.float32) + 0.5) * h
        y = (np.arange(ny, dtype=np.float32) + 0.5) * h
        z = (np.arange(nz, dtype=np.float32) + 0.5) * h
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        # Work in Python-facing field layout: (nx, ny, nz, 3), then let Frame flatten for solver IO.
        density_offset = np.zeros((nx, ny, nz), dtype=np.float32) - 0.99
        momentum = np.empty((nx, ny, nz, 3), dtype=np.float32)

        cx = 0.5 * nx * h
        cy = 0.5 * ny * h
        cz = 0.5 * nz * h

        mask = (x < 1) & (y < 1.5)
        # print(mask)
        density_offset[mask] = 0
        # print(density_offset)

        # momentum += 0.01 * (np.random.rand(*momentum.shape) - 0.5)
        initial_frame = fs.Frame.from_fields(
            density_offset=density_offset, momentum=momentum
        )

        # Filed paths are optional for in-memory case creation; write_case() will materialize defaults.
        initial_state = fs.State(
            time=0.0,
            grid=fs.StateGrid(
                nx=nx, ny=ny, nz=nz, h=h, frame=fs.Filed(data=initial_frame)
            ),
            material=fs.StateMaterialProperties(
                speed_of_sound=4.0,
                reference_density=1,
                # kinematic_viscosity=5e-5,
                kinematic_viscosity=1e-4,
                # kinematic_viscosity=0e-5,
                density_diffusivity=0.0,
            ),
        )

        # Same for the init state file reference on the run config.
        run_config = fs.RunConfig(
            solver_settings=fs.SolverSettings(cfl=1.0),
            output_settings=fs.OutputSettings(end_time=20, output_interval=20 / 300),
            init_state=fs.Filed(data=initial_state),
        )

        run_config.write_case(case_dir / "run_config.json")

    def run(self):
        print(str(SOLVER) + " " + str(WORKDIR / "VortexTest" / "run_config.json"))
        subprocess.run(
            [str(SOLVER), str(WORKDIR / "VortexTest" / "run_config.json")], check=True
        )

    def evaluate(self):
        sim_dir = WORKDIR / "VortexTest"
        config_path = sim_dir / "run_config.json"
        config = fs.read_run_config(config_path)
        time = []
        energy = []
        for output in config.outputs:
            state = fs.read_state(sim_dir / output)
            time += [state.time]
            e = total_kinetic_energy(state)
            print(output, e)
            energy += [e]

        plt.plot(time, energy)
        plt.savefig(sim_dir / "energy.png")


if __name__ == "__main__":
    case = Dambreak()
    case.create()
    case.run()
    case.evaluate()
