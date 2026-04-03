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
- `grid`: `nx`, `ny`, `h`, `initial_density`
- `material`: `reference_density`, `kinematic_viscosity`, `density_diffusivity`
- `frame`: path to the HDF5 frame

Example:

```json
{
  "time": 0.0,
  "grid": {
    "nx": 400,
    "ny": 200,
    "h": 0.01,
    "initial_density": 1.225
  },
  "material": {
    "reference_density": 1.225,
    "kinematic_viscosity": 0.000015,
    "density_diffusivity": 0.00001
  },
  "frame": "init_frame.h5"
}
```

Units are SI:

- `h` in `m`
- `time`, `end_time`, `output_interval` in `s`
- `velocity` in `m/s`
- `initial_density`, `reference_density`, and `density_offset` in `kg/m^3`
- `kinematic_viscosity` and `density_diffusivity` in `m^2/s`

The C++ loader accepts `//` line comments and trailing commas in these JSON files.

## Python Pre/Postprocessing

The repo now includes `fluid_sim_io.py`, a small Python helper for:

- creating and validating the run config plus initial state files
- reading output states from the run config `outputs` list or from `outputs/data/state_*.json`
- loading the referenced HDF5 frame data into NumPy arrays

Example:

```python
from fluid_sim_io import FluidSimIO

case = FluidSimIO("test_sim.json")
case.set_solver(cfl=0.5, pressure_iterations=600)
case.set_output(end_time=10.0, output_interval=0.1)
case.set_grid(nx=400, ny=200, h=0.01, initial_density=1.225)
case.set_material(
    reference_density=1.225,
    kinematic_viscosity=1.0e-5,
    density_diffusivity=1.0e-5,
)
case.write_case()

initial = case.read_initial_state()
latest = case.read_last_state()
print(initial.time)
print(latest.density_offset.shape)  # (ny, nx)
print(latest.velocity.shape)        # (ny, nx, 3)

x, y = case.cell_centers()
```

Reading HDF5 frames requires `numpy` and `h5py` in the Python environment you use for postprocessing.
