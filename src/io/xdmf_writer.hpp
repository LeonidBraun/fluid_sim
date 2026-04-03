#pragma once

#include "io/simulation_types.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace fluid_sim {

struct SavedFrame {
  std::string file_name;
  double time = 0.0;
};

void write_xdmf_series(const std::filesystem::path& output_path,
                       const io::State& state,
                       const std::vector<SavedFrame>& frames);

} // namespace fluid_sim
