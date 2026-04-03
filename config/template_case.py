from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluid_sim_io import FluidSimIO


def total_kinetic_energy(state) -> float:
    h = float(state.grid["h"])
    reference_density = float(state.material["reference_density"])
    density = reference_density + state.density_offset
    speed_squared = (state.velocity**2).sum(axis=-1)
    cell_area = h * h
    return float(0.5 * (density * speed_squared).sum() * cell_area)


def main() -> None:
    case_dir = Path(__file__).resolve().parent / "build" / "template_case"
    run_config_path = case_dir / "template_case.json"
    init_state_path = case_dir / "init_state.json"
    solver_path = ROOT / "build" / "fluid_sim"

    if not solver_path.exists():
        raise FileNotFoundError(f"Solver binary not found: {solver_path}")

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    case = FluidSimIO(run_config_path)
    case.set_solver(cfl=0.5, pressure_iterations=600)
    case.set_output(end_time=1.0, output_interval=0.1)
    case.set_grid(nx=256, ny=128, h=0.02, initial_density=1.225)
    case.set_material(
        reference_density=1.225,
        kinematic_viscosity=1.5e-5,
        density_diffusivity=1.0e-5,
    )
    case.set_init_state(time=0.0, frame="init_frame.h5")
    case.write_case(run_config_path=run_config_path, init_state_path=init_state_path)

    subprocess.run([str(solver_path), str(run_config_path)], check=True, cwd=str(ROOT))

    completed_case = FluidSimIO.from_run_config(run_config_path)
    state_refs = completed_case.list_state_files()
    if len(state_refs) < 3:
        raise RuntimeError(
            f"Expected at least 3 output states, found {len(state_refs)}."
        )

    print("Last 3 output kinetic energies:")
    for state_ref in state_refs[-3:]:
        state = completed_case.read_state(state_ref.index)
        energy = total_kinetic_energy(state)
        print(f"t = {state.time:.6f} s, KE = {energy:.12e}")


if __name__ == "__main__":
    main()
