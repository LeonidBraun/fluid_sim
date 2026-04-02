# Rectangular Fluid Sim

## Run

```bash
./build/fluid_sim /mnt/c/Users/LeonidBraun/Downloads/test_sim/test_sim.json
```

The solver reads one JSON config file, writes `outputs/series.xdmf`, and stores HDF5 frames under `outputs/data/` next to that config.

## Config

Top-level sections:

- `grid`: `nx`, `ny`, `dx`, `dy`
- `simulation`: `end_time`, `cfl`, `reference_density`, `kinematic_viscosity`, `density_diffusivity`, `pressure_iterations`
- `output`: `output_interval`

Example:

```json
{
  "grid": {
    "nx": 256,
    "ny": 128,
    "dx": 0.02,
    "dy": 0.02
  },
  "simulation": {
    "end_time": 10.0,
    "cfl": 0.05,
    "reference_density": 1.225,
    "kinematic_viscosity": 0.000015,
    "density_diffusivity": 0.00001,
    "pressure_iterations": 60
  },
  "output": {
    "output_interval": 1.0
  }
}
```

Units are SI:

- `dx`, `dy` in `m`
- `end_time`, `output_interval` in `s`
- `velocity` in `m/s`
- `reference_density` and `density_offset` in `kg/m^3`
- `kinematic_viscosity` and `density_diffusivity` in `m^2/s`

The runner rejects the legacy keys `steps`, `dt`, `output_every`, `viscosity`, `density_diffusion`, and `density_decay`.
