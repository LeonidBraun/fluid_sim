#pragma once

#include "simulation/cell_cloud.hpp"
#include "simulation/simulation_types.hpp"

#include <cstddef>

namespace fluid_sim {

class Simulation {
public:
  Simulation(const SimulationConfig& config, const HostState& initial_state, const double initial_time);
  ~Simulation() = default;

  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;
  Simulation(Simulation&&) = delete;
  Simulation& operator=(Simulation&&) = delete;

  void step(double max_dt);
  [[nodiscard]] HostState download_state() const;

  [[nodiscard]] const SimulationConfig& config() const {
    return config_;
  }

  [[nodiscard]] std::size_t cell_count() const;

  [[nodiscard]] double time() const {
    return time_;
  }

  [[nodiscard]] double last_dt() const {
    return last_dt_;
  }

private:
  [[nodiscard]] double compute_time_step() const;

  SimulationConfig config_{};
  double time_ = 0.0;
  double last_dt_ = 0.0;
  CellCloud cloud_{};
};

} // namespace fluid_sim
