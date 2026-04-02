#include "io/IO.hpp"

#include "io/hdf5_writer.hpp"
#include "simulation/simulation.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fluid_sim {
namespace {

using json = nlohmann::json;

void RejectLegacyKeys(const json& object, const std::set<std::string>& keys, const char* scope) {
  for (const auto& [key, value] : object.items()) {
    (void)value;
    if (keys.find(key) != keys.end()) {
      throw std::runtime_error(std::string("Legacy config key '") + key + "' found in " + scope +
                               ". Update the JSON schema before running.");
    }
  }
}

template <typename T>
void AssignIfPresent(const json& object, const char* key, T& value) {
  const auto it = object.find(key);
  if (it != object.end() && !it->is_null()) {
    value = it->get<T>();
  }
}

void ApplyGridConfig(const json& object, SimulationConfig& config) {
  AssignIfPresent(object, "nx", config.nx);
  AssignIfPresent(object, "ny", config.ny);
  AssignIfPresent(object, "dx", config.dx);
  AssignIfPresent(object, "dy", config.dy);
}

void ApplySimulationConfig(const json& object, SimulationConfig& config) {
  AssignIfPresent(object, "end_time", config.end_time);
  AssignIfPresent(object, "cfl", config.cfl);
  AssignIfPresent(object, "reference_density", config.reference_density);
  AssignIfPresent(object, "kinematic_viscosity", config.kinematic_viscosity);
  AssignIfPresent(object, "density_diffusivity", config.density_diffusivity);
  AssignIfPresent(object, "pressure_iterations", config.pressure_iterations);
}

void ApplyOutputConfig(const json& object, SimulationConfig& config) {
  AssignIfPresent(object, "output_interval", config.output_interval);
}

} // namespace

IO::IO(const std::filesystem::path& config_path)
    : config_path_(std::filesystem::absolute(config_path)),
      output_root_(config_path_.parent_path() / "outputs"),
      data_dir_(output_root_ / "data"),
      config_(LoadConfig(config_path_)) {
  std::filesystem::create_directories(data_dir_);
}

double IO::last_saved_time() const {
  if (saved_frames_.empty()) {
    return -std::numeric_limits<double>::infinity();
  }
  return saved_frames_.back().time;
}

void IO::save_frame(const Simulation& simulation) {
  const HostState state = simulation.download_state();

  std::ostringstream filename;
  const uint32_t frame_index = static_cast<uint32_t>(saved_frames_.size());
  filename << "frame_" << std::setw(4) << std::setfill('0') << frame_index << ".h5";

  const std::filesystem::path frame_path = data_dir_ / filename.str();
  write_frame_hdf5(frame_path, simulation.config(), simulation.time(), simulation.last_dt(), state);

  saved_frames_.push_back({std::string("data/") + filename.str(), simulation.time()});
  write_xdmf_series(output_root_ / "series.xdmf", simulation.config(), saved_frames_);
}

void IO::PrintUsage(std::ostream& stream) {
  stream << "Usage: fluid_sim <config.json>\n"
         << "  config.json  Path to the simulation JSON config file\n"
         << "\n"
         << "Outputs are written next to the config file in:\n"
         << "  outputs/series.xdmf\n"
         << "  outputs/data/frame_XXXX.h5\n";
}

SimulationConfig IO::LoadConfig(const std::filesystem::path& config_path) {
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("Unable to open config file: " + config_path.string());
  }

  json document;
  input >> document;
  if (!document.is_object()) {
    throw std::runtime_error("Config root must be a JSON object.");
  }

  const std::set<std::string> legacy_keys = {"steps", "dt", "output_every",
                                             "viscosity", "density_diffusion", "density_decay"};

  SimulationConfig config;
  RejectLegacyKeys(document, legacy_keys, "root");
  ApplyGridConfig(document, config);
  ApplySimulationConfig(document, config);
  ApplyOutputConfig(document, config);

  if (const auto it = document.find("grid"); it != document.end()) {
    if (!it->is_object()) {
      throw std::runtime_error("Config field 'grid' must be a JSON object.");
    }
    RejectLegacyKeys(*it, legacy_keys, "grid");
    ApplyGridConfig(*it, config);
  }

  if (const auto it = document.find("simulation"); it != document.end()) {
    if (!it->is_object()) {
      throw std::runtime_error("Config field 'simulation' must be a JSON object.");
    }
    RejectLegacyKeys(*it, legacy_keys, "simulation");
    ApplySimulationConfig(*it, config);
  }

  if (const auto it = document.find("output"); it != document.end()) {
    if (!it->is_object()) {
      throw std::runtime_error("Config field 'output' must be a JSON object.");
    }
    RejectLegacyKeys(*it, {"steps", "dt", "output_every", "directory"}, "output");
    ApplyOutputConfig(*it, config);
  }

  ValidateConfig(config);
  return config;
}

void IO::ValidateConfig(const SimulationConfig& config) {
  if (config.nx < 8 || config.ny < 8) {
    throw std::runtime_error("Grid must be at least 8x8.");
  }
  if (config.pressure_iterations < 1) {
    throw std::runtime_error("pressure_iterations must be positive.");
  }
  if (config.end_time <= 0.0) {
    throw std::runtime_error("end_time must be positive.");
  }
  if (config.cfl <= 0.0) {
    throw std::runtime_error("cfl must be positive.");
  }
  if (config.output_interval <= 0.0) {
    throw std::runtime_error("output_interval must be positive.");
  }
  if (config.dx <= 0.0 || config.dy <= 0.0) {
    throw std::runtime_error("dx and dy must be positive.");
  }
  if (config.reference_density <= 0.0) {
    throw std::runtime_error("reference_density must be positive.");
  }
  if (config.kinematic_viscosity < 0.0 || config.density_diffusivity < 0.0) {
    throw std::runtime_error("kinematic_viscosity and density_diffusivity must be non-negative.");
  }
}

} // namespace fluid_sim
