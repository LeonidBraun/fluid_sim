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

Each dataclass can be read from disk and written back, and `RunConfig` can load the output states listed in its `outputs` array.

For local development, install it in editable mode:

```bash
source .venv/bin/activate
pip install -e ./preprocessor
```

Example:

```python
from fluid_sim_io import Filed, Frame, OutputSettings, RunConfig, SolverSettings, State, StateGrid, StateMaterialProperties

initial_state = State(
    time=0.0,
    grid=StateGrid(
        nx=16,
        ny=8,
        nz=4,
        h=0.02,
        frame=Filed("default_frame.h5", Frame.zeros(16, 8, 4)),
    ),
    material=StateMaterialProperties(
        speed_of_sound=10.0,
        reference_density=1.225,
        kinematic_viscosity=1.5e-5,
        density_diffusivity=1.0e-5,
    ),
)

config = RunConfig(
    solver_settings=SolverSettings(cfl=0.05, pressure_iterations=60),
    output_settings=OutputSettings(end_time=0.05, output_interval=0.05),
    init_state=Filed("default_state.json", initial_state),
)

config.write_case("solver/build/default_3d/default.json")

loaded = RunConfig.read_json("solver/build/default_3d/default.json")
outputs = loaded.load_output_states("solver/build/default_3d/default.json")
last_state = outputs[-1].data
last_frame = last_state.grid.frame.data
momentum = last_frame.momentum_grid(last_state.grid.nx, last_state.grid.ny, last_state.grid.nz)
```

The HDF5 payload stores:

- `density_offset` with shape `(nz, ny, nx)`
- `momentum` with shape `(nz, ny, nx, 3)`

Reading HDF5 frames requires `numpy` and `h5py` in the Python environment you use for postprocessing.
