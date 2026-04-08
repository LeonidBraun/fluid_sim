from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

try:
    from fluid_sim_io import Filed, Frame, OutputSettings, RunConfig, SolverSettings, State, StateGrid, StateMaterialProperties, load_output_states
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "fluid_sim_io is not installed. Activate the virtual environment and run "
        "'pip install -e ./preprocessor' from the repository root."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]


def total_kinetic_energy(state: State) -> float:
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
    momentum_squared = (momentum ** 2).sum(axis=-1)
    return float(0.5 * ((momentum_squared / rho).sum()) * h * h * h)


def main() -> None:
    case_dir = Path(__file__).resolve().parent / "template_case"
    config_path = case_dir / "template_case.json"
    state_path = case_dir / "template_state.json"
    frame_path = case_dir / "template_frame.h5"
    solver_path = ROOT / "solver" / "build" / "fluid_sim"

    if not solver_path.exists():
        raise FileNotFoundError(f"Solver binary not found: {solver_path}")

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    nx = 16
    ny = 8
    nz = 4
    initial_frame = Frame.zeros(nx, ny, nz)

    initial_state = State(
        time=0.0,
        grid=StateGrid(
            nx=nx,
            ny=ny,
            nz=nz,
            h=0.02,
            frame=Filed(file=frame_path.name, data=initial_frame),
        ),
        material=StateMaterialProperties(
            speed_of_sound=10.0,
            reference_density=1.225,
            kinematic_viscosity=1.5e-5,
            density_diffusivity=1.0e-5,
        ),
    )

    run_config = RunConfig(
        solver_settings=SolverSettings(cfl=0.05, pressure_iterations=60),
        output_settings=OutputSettings(end_time=0.05, output_interval=0.05),
        init_state=Filed(file=state_path.name, data=initial_state),
    )

    run_config.write_case(config_path)

    subprocess.run([str(solver_path), str(config_path)], cwd=str(ROOT), check=True)

    outputs = load_output_states(config_path)
    if len(outputs) < 2:
        raise RuntimeError(f"Expected at least 2 output states, found {len(outputs)}.")

    print("Last outputs kinetic energies:")
    for output in outputs[-2:]:
        energy = total_kinetic_energy(output.data)
        print(f"{output.file}: t = {output.data.time:.6f} s, KE = {energy:.12e}")


if __name__ == "__main__":
    main()
