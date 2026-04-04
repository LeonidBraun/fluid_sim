from dataclasses import replace

from .json_io import read_relaxed_json, write_json
from .paths import resolve_sibling_path
from .types import Filed, Frame, OutputSettings, RunConfig, SolverSettings, State, StateGrid, StateMaterialProperties


def read_frame(path, nx: int, ny: int) -> Frame:
    return Frame.read_hdf5(path, nx, ny)


def write_frame(path, frame: Frame, nx: int, ny: int):
    return frame.write_hdf5(path, nx, ny)


def read_state(path) -> State:
    return State.read_json(path)


def write_state(path, state: State, *, indent: int = 2):
    return state.write_json(path, indent=indent)


def read_run_config(path) -> RunConfig:
    return RunConfig.read_json(path)


def write_run_config(path, run_config: RunConfig, *, indent: int = 2):
    return run_config.write_json(path, indent=indent)


def load_output_states(config_path):
    run_config = RunConfig.read_json(config_path)
    return run_config.load_output_states(config_path)


__all__ = [
    "Filed",
    "Frame",
    "OutputSettings",
    "RunConfig",
    "SolverSettings",
    "State",
    "StateGrid",
    "StateMaterialProperties",
    "load_output_states",
    "read_frame",
    "read_relaxed_json",
    "read_run_config",
    "read_state",
    "replace",
    "resolve_sibling_path",
    "write_frame",
    "write_json",
    "write_run_config",
    "write_state",
]
