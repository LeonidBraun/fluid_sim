# Rectangular Fluid Sim

## Run

```bash
./build/fluid_sim /mnt/c/Users/LeonidBraun/Downloads/test_sim/test_sim.json
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
- `grid`: `nx`, `ny`, `h`, optional `frame`
- `material`: `speed_of_sound`, `reference_density`, `kinematic_viscosity`, `density_diffusivity`

Example:

```json
{
  "time": 0.0,
  "grid": {
    "nx": 400,
    "ny": 200,
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
- `State` with `State.Grid` and `State.MaterialProperties`
- `RunConfig` with `RunConfig.SolverSettings` and `RunConfig.OutputSettings`

Each dataclass can be read from disk and written back, and `RunConfig` can load the output states listed in its `outputs` array.

For local development, install it in editable mode:

```bash
source .venv/bin/activate
pip install -e ./preprocessor
```

Example:

```python
from fluid_sim_io import Filed, Frame, RunConfig, State

initial_state = State(
    time=0.0,
    grid=State.Grid(
        nx=64,
        ny=32,
        h=0.02,
        frame=Filed("default_frame.h5", Frame.zeros(64, 32)),
    ),
    material=State.MaterialProperties(
        speed_of_sound=10.0,
        reference_density=1.225,
        kinematic_viscosity=1.5e-5,
        density_diffusivity=1.0e-5,
    ),
)

config = RunConfig(
    solver_settings=RunConfig.SolverSettings(cfl=0.05, pressure_iterations=60),
    output_settings=RunConfig.OutputSettings(end_time=0.1, output_interval=0.05),
    init_state=Filed("default_state.json", initial_state),
)

config.write_case("build/default_sim/default.json")

loaded = RunConfig.read_json("build/default_sim/default.json")
outputs = loaded.load_output_states("build/default_sim/default.json")
last_state = outputs[-1].data
last_frame = last_state.grid.frame.data
momentum = last_frame.momentum_grid(last_state.grid.nx, last_state.grid.ny)
```

Reading HDF5 frames requires `numpy` and `h5py` in the Python environment you use for postprocessing.
