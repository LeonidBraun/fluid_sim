# Rectangular Fluid Sim

## Run

```bash
./solver/build/fluid_sim ../work_dir/test_sim/test_sim.json
```

The solver reads a run config JSON, an initial state JSON, and HDF5 frame data. It also writes `outputs/series.xdmf` for ParaView, but the simulation state is described by JSON files plus the referenced HDF5 frames.

## Data Model

The run config contains:

- `solver_settings`: `cfl`, `pressure_iterations`
- `output_settings`: `end_time`, `output_interval`
- `init_state`: path to the initial state JSON
- `outputs`: paths to generated output state JSON files

Example:

```json
{
  "solver_settings": {
    "cfl": 0.5,
    "pressure_iterations": 600
  },
  "output_settings": {
    "end_time": 10.0,
    "output_interval": 0.1
  },
  "init_state": "init_state.json",
  "outputs": [
    "outputs/data/state_0.json",
    "outputs/data/state_1.json"
  ]
}
```

The state JSON contains:

- `time`
- `grid`: `nx`, `ny`, `nz`, `h`, optional `frame`
- `material`: `speed_of_sound`, `reference_density`, `kinematic_viscosity`, `density_diffusivity`

Example:

```json
{
  "time": 0.0,
  "grid": {
    "nx": 16,
    "ny": 8,
    "nz": 4,
    "h": 0.01,
    "frame": "init_frame.h5"
  },
  "material": {
    "speed_of_sound": 10.0,
    "reference_density": 1.225,
    "kinematic_viscosity": 0.000015,
    "density_diffusivity": 0.00001
  }
}
```

Units are SI:

- `h` in `m`
- `time`, `end_time`, `output_interval` in `s`
- `momentum` in `kg/(m^2 s)` per cell volume convention used by the solver state
- `reference_density` and `density_offset` in `kg/m^3`
- `speed_of_sound` in `m/s`
- `kinematic_viscosity` and `density_diffusivity` in `m^2/s`

The C++ loader accepts `//` line comments and trailing commas in these JSON files.

## Python Pre/Postprocessing

The repo now includes the package under `preprocessor/fluid_sim_io/`, which mirrors the C++ IO model as frozen dataclasses:

- `Filed[T]`
- `Frame`
- `State`, `StateGrid`, `StateMaterialProperties`
- `RunConfig`, `SolverSettings`, `OutputSettings`

Each dataclass can be read from disk and written back, and `RunConfig` exposes the generated output state paths through its `outputs` array.

For local development, install it in editable mode:

```bash
source .venv/bin/activate
pip install -e ./preprocessor
```

Example:

```python
from pathlib import Path
import subprocess

import fluid_sim_io as fs
import numpy as np

config_path = Path(".../work_dir/simulation.json")
solver_path = Path(".../fluid_sim/solver/build/fluid_sim")
sim_dir = config_path.parent
nx, ny, nz = 64, 32, 1

initial_state = fs.State(
    time=0.0,
    grid=fs.StateGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        h=0.02,
        frame=fs.Filed(
            data=fs.Frame.from_fields(
                density_offset=np.zeros((nx, ny, nz), dtype=np.float32),
                momentum=np.zeros((nx, ny, nz, 3), dtype=np.float32),
            )
        ),
    ),
    material=fs.StateMaterialProperties(
        speed_of_sound=10.0,
        reference_density=1.225,
        kinematic_viscosity=1.5e-5,
        density_diffusivity=1.0e-5,
    ),
)

run_config = fs.RunConfig(
    solver_settings=fs.SolverSettings(cfl=0.05, pressure_iterations=60),
    output_settings=fs.OutputSettings(end_time=0.1, output_interval=0.05),
    init_state=fs.Filed(data=initial_state),
)

run_config.write_case(config_path)

subprocess.run([str(solver_path), str(config_path)], check=True)

config = fs.read_run_config(config_path)
last_state = fs.read_state(sim_dir / config.outputs[-1])
last_frame = last_state.grid.frame.data
momentum = last_frame.momentum_grid(
    last_state.grid.nx,
    last_state.grid.ny,
    last_state.grid.nz,
)
print(last_state.time)
print(momentum.shape)
```

The frame HDF5 payload now stores:

- `density_offset` with shape `(nx, ny, nz)`
- `momentum` with shape `(nx, ny, nz, 3)`

Reading HDF5 frames requires `numpy` and `h5py` in the Python environment you use for postprocessing.
