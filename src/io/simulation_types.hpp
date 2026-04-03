#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace io {

template <typename T>
struct Filed {
  std::string file;
  T data;
};

struct Frame {
  std::vector<float> density_offset;
  std::vector<float> momentum;
};

struct State {
  double time = 0.0;

  struct Grid {
    std::optional<Filed<Frame>> frame;
    int nx = 200;
    int ny = 100;
    double h = 1.0;
  } grid;

  struct MaterialProperties {
    double speed_of_sound = 10.0;
    double reference_density = 1.225;
    double kinematic_viscosity = 1.5e-5;
    double density_diffusivity = 1.0e-5;
  } material;
};

struct RunConfig {
  struct SolverSettings {
    double cfl = 0.05;
    int pressure_iterations = 60;
  } solver_settings;

  struct OutputSettings {
    double end_time = 10.0;
    double output_interval = 1.0;
  } output_settings;

  Filed<State> init_state;
  std::vector<std::string> outputs;
};

} // namespace io
