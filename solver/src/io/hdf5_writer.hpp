#pragma once

#include "io/simulation_types.hpp"

#include <filesystem>

namespace io {

void write_frame_hdf5(const std::filesystem::path& output_path, int nx, int ny, int nz, const Frame& frame);
Frame read_frame_hdf5(const std::filesystem::path& input_path, int nx, int ny, int nz);

} // namespace io
