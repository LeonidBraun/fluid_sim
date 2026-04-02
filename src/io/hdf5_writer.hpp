#pragma once

#include "simulation/simulation_types.hpp"

#include <filesystem>

namespace fluid_sim {

void write_frame_hdf5(const std::filesystem::path& output_path,
                      const SimulationConfig& config,
                      double time,
                      double time_step,
                      const HostState& state);

} // namespace fluid_sim
