#pragma once

#include "io/xdmf_writer.hpp"
#include "simulation/simulation_types.hpp"

#include <filesystem>
#include <iosfwd>
#include <vector>

namespace fluid_sim {

class Simulation;

class IO {
 public:
  explicit IO(const std::filesystem::path& config_path);

  [[nodiscard]] const SimulationConfig& config() const {
    return config_;
  }

  [[nodiscard]] double last_saved_time() const;

  void save_frame(const Simulation& simulation);

  static void PrintUsage(std::ostream& stream);

 private:
  static SimulationConfig LoadConfig(const std::filesystem::path& config_path);
  static void ValidateConfig(const SimulationConfig& config);

  std::filesystem::path config_path_;
  std::filesystem::path output_root_;
  std::filesystem::path data_dir_;
  SimulationConfig config_{};
  std::vector<SavedFrame> saved_frames_;
};

} // namespace fluid_sim
