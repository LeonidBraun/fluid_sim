#pragma once

#include "io/simulation_types.hpp"

#include <filesystem>
#include <iosfwd>
#include <vector>

namespace fluid_sim {

class Simulation;

class IO {
public:
  explicit IO(const std::filesystem::path& config_path);

  [[nodiscard]] const io::RunConfig::SolverSettings& settings() const {
    return run_config_.solver_settings;
  }

  [[nodiscard]] const io::RunConfig::OutputSettings& output_settings() const {
    return run_config_.output_settings;
  }

  [[nodiscard]] const io::State& initial_state() const {
    return init_state_;
  }

  [[nodiscard]] double initial_time() const {
    return init_state_.time;
  }

  [[nodiscard]] double last_saved_time() const;

  [[nodiscard]] std::size_t last_output() const {
    return saved_frame_count_ == 0 ? 0U : saved_frame_count_ - 1U;
  }

  void save_output(const Simulation& simulation);

  static void PrintUsage(std::ostream& stream);

private:
  static io::RunConfig LoadRunConfig(const std::filesystem::path& config_path);
  static io::State LoadStateFile(const std::filesystem::path& state_path);
  static io::Frame LoadFrameFile(const std::filesystem::path& frame_path, int nx, int ny, int nz);
  static void ValidateRunConfig(const io::RunConfig& run_config, const io::State& init_state);
  static void WriteRunConfig(const std::filesystem::path& config_path, const io::RunConfig& run_config);
  static void WriteStateFile(const std::filesystem::path& state_path, const io::State& state);

  std::filesystem::path config_path_;
  std::filesystem::path root_path_;
  std::filesystem::path output_root_;
  std::filesystem::path data_dir_;
  io::RunConfig run_config_{};
  io::State init_state_{};
  std::size_t saved_frame_count_ = 0;
};

} // namespace fluid_sim
