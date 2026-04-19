from .grid import Grid
from .types import (
    Filed,
    Frame,
    OutputSettings,
    RunConfig,
    SolverSettings,
    State,
    StateGrid,
    StateMaterialProperties,
)

def read_state(path) -> State:
    return State.read_json(path)

def read_run_config(path) -> RunConfig:
    return RunConfig.read_json(path)


__all__ = [
    "Filed",
    "Frame",
    "Grid",
    "OutputSettings",
    "RunConfig",
    "SolverSettings",
    "State",
    "StateGrid",
    "StateMaterialProperties",
    "read_run_config",
    "read_state",
]
