#include "fluid_sim/hdf5_writer.hpp"
#include "fluid_sim/simulation.hpp"
#include "fluid_sim/xdmf_writer.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

struct AppConfig {
  fluid_sim::SimulationConfig simulation;
};

void reject_legacy_keys(const json& object, const std::set<std::string>& keys, const char* scope) {
  for (const auto& [key, value] : object.items()) {
    (void)value;
    if (keys.find(key) != keys.end()) {
      throw std::runtime_error(std::string("Legacy config key '") + key + "' found in " + scope +
                               ". Update the JSON schema before running.");
    }
  }
}

template <typename T>
void assign_if_present(const json& object, const char* key, T& value) {
  const auto it = object.find(key);
  if (it != object.end() && !it->is_null()) {
    value = it->get<T>();
  }
}

void apply_grid_config(const json& object, fluid_sim::SimulationConfig& config) {
  assign_if_present(object, "nx", config.nx);
  assign_if_present(object, "ny", config.ny);
  assign_if_present(object, "dx", config.dx);
  assign_if_present(object, "dy", config.dy);
}

void apply_simulation_config(const json& object, fluid_sim::SimulationConfig& config) {
  assign_if_present(object, "end_time", config.end_time);
  assign_if_present(object, "cfl", config.cfl);
  assign_if_present(object, "reference_density", config.reference_density);
  assign_if_present(object, "kinematic_viscosity", config.kinematic_viscosity);
  assign_if_present(object, "density_diffusivity", config.density_diffusivity);
  assign_if_present(object, "pressure_iterations", config.pressure_iterations);
}

void apply_output_config(const json& object, AppConfig& config) {
  assign_if_present(object, "output_interval", config.simulation.output_interval);
}

AppConfig load_config(const std::filesystem::path& config_path) {
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("Unable to open config file: " + config_path.string());
  }

  json document;
  input >> document;
  if (!document.is_object()) {
    throw std::runtime_error("Config root must be a JSON object.");
  }

  reject_legacy_keys(document, {"steps", "dt", "output_every", "viscosity", "density_diffusion", "density_decay"}, "root");

  AppConfig config;
  apply_grid_config(document, config.simulation);
  apply_simulation_config(document, config.simulation);
  apply_output_config(document, config);

  if (const auto it = document.find("grid"); it != document.end()) {
    if (!it->is_object()) {
      throw std::runtime_error("Config field 'grid' must be a JSON object.");
    }
    reject_legacy_keys(*it, {"steps", "dt", "output_every", "viscosity", "density_diffusion", "density_decay"}, "grid");
    apply_grid_config(*it, config.simulation);
  }

  if (const auto it = document.find("simulation"); it != document.end()) {
    if (!it->is_object()) {
      throw std::runtime_error("Config field 'simulation' must be a JSON object.");
    }
    reject_legacy_keys(*it, {"steps", "dt", "output_every", "viscosity", "density_diffusion", "density_decay"}, "simulation");
    apply_simulation_config(*it, config.simulation);
  }

  if (const auto it = document.find("output"); it != document.end()) {
    if (!it->is_object()) {
      throw std::runtime_error("Config field 'output' must be a JSON object.");
    }
    reject_legacy_keys(*it, {"steps", "dt", "output_every", "directory"}, "output");
    apply_output_config(*it, config);
  }

  return config;
}

void validate_config(const AppConfig& config) {
  const auto& sim = config.simulation;
  if (sim.nx < 8 || sim.ny < 8) {
    throw std::runtime_error("Grid must be at least 8x8.");
  }
  if (sim.pressure_iterations < 1) {
    throw std::runtime_error("pressure_iterations must be positive.");
  }
  if (sim.end_time <= 0.0) {
    throw std::runtime_error("end_time must be positive.");
  }
  if (sim.cfl <= 0.0) {
    throw std::runtime_error("cfl must be positive.");
  }
  if (sim.output_interval <= 0.0) {
    throw std::runtime_error("output_interval must be positive.");
  }
  if (sim.dx <= 0.0 || sim.dy <= 0.0) {
    throw std::runtime_error("dx and dy must be positive.");
  }
  if (sim.reference_density <= 0.0) {
    throw std::runtime_error("reference_density must be positive.");
  }
  if (sim.kinematic_viscosity < 0.0 || sim.density_diffusivity < 0.0) {
    throw std::runtime_error("kinematic_viscosity and density_diffusivity must be non-negative.");
  }
}

void print_usage() {
  std::cout
      << "Usage: fluid_sim <config.json>\n"
      << "  config.json  Path to the simulation JSON config file\n"
      << "\n"
      << "Outputs are written next to the config file in:\n"
      << "  outputs/series.xdmf\n"
      << "  outputs/data/frame_XXXX.h5\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<fluid_sim::SavedFrame> saved_frames;

  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      print_usage();
      return 0;
    }

    if (argc != 2) {
      throw std::runtime_error("Expected exactly one argument: path to config JSON. Use --help for usage.");
    }

    const std::filesystem::path config_path = std::filesystem::absolute(argv[1]);
    const std::filesystem::path run_root = config_path.parent_path();
    const std::filesystem::path output_root = run_root / "outputs";
    const std::filesystem::path data_dir = output_root / "data";

    AppConfig app_config = load_config(config_path);
    validate_config(app_config);

    std::filesystem::create_directories(data_dir);

    fluid_sim::Simulation simulation(app_config.simulation);

    int frame_index = 0;
    double last_saved_time = -std::numeric_limits<double>::infinity();
    auto save_frame = [&]() {
      const auto state = simulation.download_state();
      std::ostringstream filename;
      filename << "frame_" << std::setw(4) << std::setfill('0') << frame_index++ << ".h5";
      const auto frame_path = data_dir / filename.str();
      fluid_sim::write_frame_hdf5(frame_path, app_config.simulation, simulation.time(),
                                  simulation.last_dt(), state);
      saved_frames.push_back({std::string("data/") + filename.str(), simulation.time()});
      fluid_sim::write_xdmf_series(output_root / "series.xdmf", app_config.simulation, saved_frames);
      last_saved_time = simulation.time();
      std::cout << "Wrote " << frame_path << '\n';
    };

    save_frame();
    double next_output_time = app_config.simulation.output_interval;
    constexpr double epsilon = 1e-12;

    while (simulation.time() + epsilon < app_config.simulation.end_time) {
      const double remaining_time = app_config.simulation.end_time - simulation.time();
      simulation.step(remaining_time);

      bool crossed_output_boundary = false;
      while (simulation.time() + epsilon >= next_output_time) {
        next_output_time += app_config.simulation.output_interval;
        crossed_output_boundary = true;
      }

      if (crossed_output_boundary) {
        save_frame();
      }
    }

    if (std::abs(simulation.time() - last_saved_time) > epsilon) {
      save_frame();
    }
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
