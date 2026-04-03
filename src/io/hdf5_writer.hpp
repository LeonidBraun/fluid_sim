#pragma once

#include "simulation_types.hpp"

#include <filesystem>

namespace io {

void write_frame_hdf5(const std::filesystem::path& output_path,
                      const SimulationConfig& config,
                      double time,
                      double time_step,
                      const HostState& state);

io::Frame read_frame_hdf5(const std::filesystem::path& input_path, int nx, int ny);

} // namespace io
