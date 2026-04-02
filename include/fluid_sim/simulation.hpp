#pragma once

#include "fluid_sim/cell_cloud.hpp"

#include <cstddef>
#include <vector>

namespace fluid_sim {

struct SimulationConfig {
  int nx = 256;
  int ny = 128;
  int pressure_iterations = 60;
  double end_time = 10.0;
  double cfl = 0.05;
  double output_interval = 1.0;
  double dx = 1.0;
  double dy = 1.0;
  double reference_density = 1.225;
  double kinematic_viscosity = 1.5e-5;
  double density_diffusivity = 1.0e-5;
};

struct HostState {
  std::vector<float> density_offset;
  std::vector<float> velocity_x;
  std::vector<float> velocity_y;
};

class Simulation {
 public:
  explicit Simulation(const SimulationConfig& config);
  ~Simulation() = default;

  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;
  Simulation(Simulation&&) = delete;
  Simulation& operator=(Simulation&&) = delete;

  void step(double max_dt);
  [[nodiscard]] HostState download_state() const;

  [[nodiscard]] const SimulationConfig& config() const { return config_; }
  [[nodiscard]] std::size_t cell_count() const;
  [[nodiscard]] double time() const { return time_; }
  [[nodiscard]] double last_dt() const { return last_dt_; }

 private:
  [[nodiscard]] double compute_time_step() const;

  SimulationConfig config_{};
  double time_ = 0.0;
  double last_dt_ = 0.0;
  CellCloud cloud_{};
};

}  // namespace fluid_sim
