#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace io {

template <typename T>
struct Filed {
  std::string file;
  T data;
};

template <typename T>
using Ref = std::variant<std::monostate, T, std::string>;

struct Frame {
  std::vector<float> density_offset;
  std::vector<float> momentum;
};

struct State {
  double time = 0.0;

  struct Grid {
    Filed<Frame> frame;
    std::unordered_map<std::string, std::vector<float>> cell_fields;
    std::unordered_map<std::string, std::vector<float>> point_fields;
    int nx = 200;
    int ny = 100;
    int nz = 1;
    double h = 1.0;
  } grid;

  struct MaterialProperties {
    double speed_of_sound;
    double reference_density;
    double kinematic_viscosity;
    double density_diffusivity;
  } material;
};

struct RunConfig {
  struct SolverSettings {
    double cfl;
    int pressure_iterations;
  } solver_settings;

  struct OutputSettings {
    double end_time;
    double output_interval;
  } output_settings;

  Filed<State> init_state;
  std::vector<std::string> outputs;
};

} // namespace io
