#pragma once

#include <vector>

namespace fluid_sim {

struct SimulationConfig {
  int nx = 256;
  int ny = 128;
  int pressure_iterations = 60;
  double cfl = 0.05;
  double dx = 1.0;
  double dy = 1.0;
  double end_time = 10.0;
  double output_interval = 1.0;
  double reference_density = 1.225;
  double kinematic_viscosity = 1.5e-5;
  double density_diffusivity = 1.0e-5;
};

struct HostState {
  std::vector<float> density_offset;
  std::vector<float> velocity;
};

} // namespace fluid_sim
