#pragma once

#include "simulation_types.hpp"
#include "xdmf_writer.hpp"

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

  [[nodiscard]] const HostState& initial_state() const {
    return initial_state_;
  }

  [[nodiscard]] double initial_time() const {
    return init_state_file_.time;
  }

  [[nodiscard]] double last_saved_time() const;
  [[nodiscard]] std::size_t last_output() const {
    return saved_frames_.empty() ? 0U : saved_frames_.size() - 1U;
  }

  void save_frame(const Simulation& simulation);

  static void PrintUsage(std::ostream& stream);

private:
  static io::RunConfig LoadRunConfig(const std::filesystem::path& config_path);
  static io::StateFile LoadStateFile(const std::filesystem::path& state_path);
  static io::Frame LoadFrameFile(const std::filesystem::path& frame_path);

  static void ValidateRunConfig(const RunConfig& run_config, const StateFile& init_state);
  static void WriteRunConfig(const std::filesystem::path& config_path, const RunConfig& run_config);
  static void WriteStateFile(const std::filesystem::path& state_path, const StateFile& state_file);
  static HostState LoadFrameState(const std::filesystem::path& frame_path, int nx, int ny);

  std::filesystem::path config_path_;
  std::filesystem::path root_path_;
  std::filesystem::path output_root_;
  std::filesystem::path data_dir_;
  io::RunConfig run_config_{};
  io::State init_state_{};
};

} // namespace fluid_sim
