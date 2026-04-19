from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CASE_DIR = Path(__file__).resolve().parent / "example_case"
SOLVER = ROOT / "solver" / "build" / "fluid_sim"
WORKDIR = Path.home() / "work_dir"

print(ROOT)
print(EXAMPLE_CASE_DIR)
print(SOLVER)
print(WORKDIR)


def stage_case(workdir: Path, case_name: str) -> Path:
    staged_case_dir = workdir / case_name
    shutil.copytree(EXAMPLE_CASE_DIR, staged_case_dir, dirs_exist_ok=True)
    return staged_case_dir


def main() -> None:
    if not SOLVER.exists():
        raise FileNotFoundError(f"Solver binary not found: {SOLVER}")

    WORKDIR.mkdir(parents=True, exist_ok=True)
    case_dir = stage_case(WORKDIR, "example_case")
    config_path = case_dir / "example_sim.json"
    print(f"Staged example case at: {case_dir}")
    print(f"Running solver: {SOLVER} {config_path}")

    subprocess.run([str(SOLVER), str(config_path)], check=True, cwd=str(ROOT))
    print(f"Finished. Outputs are in: {case_dir / 'outputs'}")


if __name__ == "__main__":
    main()
