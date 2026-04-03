#pragma once

#include "io/simulation_types.hpp"
#include "simulation/cell_cloud.hpp"

#include <cstddef>

namespace fluid_sim {

class Simulation {
public:
  Simulation(const io::RunConfig::SolverSettings& settings, const io::State& initial_state);
  ~Simulation() = default;

  Simulation(const Simulation&) = delete;
  Simulation& operator=(const Simulation&) = delete;
  Simulation(Simulation&&) = delete;
  Simulation& operator=(Simulation&&) = delete;

  void step(double max_dt);
  [[nodiscard]] io::Frame download_frame() const;

  [[nodiscard]] const io::RunConfig::SolverSettings& settings() const {
    return settings_;
  }

  [[nodiscard]] double time() const {
    return time_;
  }

  [[nodiscard]] double last_dt() const {
    return last_dt_;
  }

private:
  [[nodiscard]] double compute_time_step() const;

  io::RunConfig::SolverSettings settings_{};
  double time_ = 0.0;
  double last_dt_ = 0.0;
  CellCloud cloud_{};
};

} // namespace fluid_sim
