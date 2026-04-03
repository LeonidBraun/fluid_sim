#include "io/IO.hpp"

#include "io/hdf5_writer.hpp"
#include "simulation/simulation.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace fluid_sim {
namespace {

using json = nlohmann::json;

[[nodiscard]] std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to open JSON file: " + path.string());
  }

  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

[[nodiscard]] std::string StripLineComments(const std::string& text) {
  std::string result;
  result.reserve(text.size());

  bool in_string = false;
  bool escaping = false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];

    if (in_string) {
      result.push_back(c);
      if (escaping) {
        escaping = false;
      } else if (c == '\\') {
        escaping = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      result.push_back(c);
      continue;
    }

    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      while (i < text.size() && text[i] != '\n') {
        ++i;
      }
      if (i < text.size()) {
        result.push_back(text[i]);
      }
      continue;
    }

    result.push_back(c);
  }

  return result;
}

[[nodiscard]] std::string StripTrailingCommas(const std::string& text) {
  std::string result;
  result.reserve(text.size());

  bool in_string = false;
  bool escaping = false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];

    if (in_string) {
      result.push_back(c);
      if (escaping) {
        escaping = false;
      } else if (c == '\\') {
        escaping = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      result.push_back(c);
      continue;
    }

    if (c == ',') {
      std::size_t j = i + 1;
      while (j < text.size() && std::isspace(static_cast<unsigned char>(text[j])) != 0) {
        ++j;
      }
      if (j < text.size() && (text[j] == '}' || text[j] == ']')) {
        continue;
      }
    }

    result.push_back(c);
  }

  return result;
}

[[nodiscard]] json ReadJsonFile(const std::filesystem::path& path) {
  return json::parse(StripTrailingCommas(StripLineComments(ReadTextFile(path))));
}

[[nodiscard]] const json& RequireObject(const json& object, const char* key, const char* scope) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    throw std::runtime_error(std::string("Missing required object '") + key + "' in " + scope + '.');
  }
  if (!it->is_object()) {
    throw std::runtime_error(std::string("Field '") + key + "' in " + scope + " must be a JSON object.");
  }
  return *it;
}

template <typename T>
void RequireField(const json& object, const char* key, T& value, const char* scope) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    throw std::runtime_error(std::string("Missing required field '") + key + "' in " + scope + '.');
  }
  value = it->get<T>();
}

template <typename T, typename Transform>
void RequireField(const json& object, const char* key, T& value, const char* scope, Transform&& transform) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    throw std::runtime_error(std::string("Missing required field '") + key + "' in " + scope + '.');
  }
  value = transform(*it);
}

[[nodiscard]] std::filesystem::path ResolveSiblingPath(const std::filesystem::path& base_path,
                                                       const std::filesystem::path& relative_or_absolute) {
  if (relative_or_absolute.is_absolute()) {
    return relative_or_absolute;
  }
  return (base_path.parent_path() / relative_or_absolute).lexically_normal();
}

[[nodiscard]] json ToJson(const io::RunConfig::SolverSettings& solver_settings) {
  return json{{"cfl", solver_settings.cfl}, {"pressure_iterations", solver_settings.pressure_iterations}};
}

[[nodiscard]] json ToJson(const io::RunConfig::OutputSettings& output_settings) {
  return json{{"end_time", output_settings.end_time}, {"output_interval", output_settings.output_interval}};
}

[[nodiscard]] json ToJson(const io::State::Grid& grid) {
  json result{{"nx", grid.nx}, {"ny", grid.ny}, {"h", grid.h}};
  if (grid.frame.has_value() && !grid.frame->file.empty()) {
    result["frame"] = grid.frame->file;
  }
  return result;
}

[[nodiscard]] json ToJson(const io::State::MaterialProperties& material) {
  return json{{"speed_of_sound", material.speed_of_sound},
              {"reference_density", material.reference_density},
              {"kinematic_viscosity", material.kinematic_viscosity},
              {"density_diffusivity", material.density_diffusivity}};
}

[[nodiscard]] json ToJson(const io::State& state) {
  return json{{"time", state.time},                  //
              {"grid", ToJson(state.grid)},          //
              {"material", ToJson(state.material)}}; //
}

[[nodiscard]] json ToJson(const io::RunConfig& run_config) {
  json result{{"solver_settings", ToJson(run_config.solver_settings)},
              {"output_settings", ToJson(run_config.output_settings)},
              {"init_state", run_config.init_state.file},
              {"outputs", run_config.outputs}};
  return result;
}

} // namespace

IO::IO(const std::filesystem::path& config_path)
    : config_path_(std::filesystem::absolute(config_path)) {
  root_path_ = config_path_.parent_path();
  output_root_ = root_path_ / "outputs";
  data_dir_ = output_root_ / "data";

  run_config_ = LoadRunConfig(config_path_);
  init_state_ = run_config_.init_state.data;
  ValidateRunConfig(run_config_, init_state_);

  std::filesystem::create_directories(data_dir_);

  run_config_.outputs.clear();
  saved_frames_.clear();
}

double IO::last_saved_time() const {
  if (saved_frames_.empty()) {
    return -std::numeric_limits<double>::infinity();
  }
  return saved_frames_.back().time;
}

void IO::save_output(const Simulation& simulation) {
  const io::Frame frame = simulation.download_frame();
  const std::size_t frame_index = saved_frames_.size();

  const std::string frame_name = "frame_" + std::to_string(frame_index) + ".h5";
  const std::string state_name = "state_" + std::to_string(frame_index) + ".json";

  const std::filesystem::path frame_path = data_dir_ / frame_name;
  const std::filesystem::path state_path = data_dir_ / state_name;

  write_frame_hdf5(frame_path, init_state_.grid.nx, init_state_.grid.ny, frame);

  io::State state = init_state_;
  state.time = simulation.time();
  state.grid.frame = io::Filed<io::Frame>{frame_name, frame};
  WriteStateFile(state_path, state);

  saved_frames_.push_back({std::string("data/") + frame_name, simulation.time()});
  run_config_.outputs.push_back((std::filesystem::path("outputs") / "data" / state_name).generic_string());

  WriteRunConfig(config_path_, run_config_);
  write_xdmf_series(output_root_ / "series.xdmf", init_state_, saved_frames_);
}

void IO::PrintUsage(std::ostream& stream) {
  stream << "Usage: fluid_sim <config.json>\n"
         << "  config.json  Path to the simulation run config JSON file\n"
         << "\n"
         << "The run config references an init state JSON and is updated with generated state JSON files.\n";
}

io::RunConfig IO::LoadRunConfig(const std::filesystem::path& config_path) {
  const json document = ReadJsonFile(config_path);
  if (!document.is_object()) {
    throw std::runtime_error("Run config must be a JSON object.");
  }

  io::RunConfig run_config;

  const json& solver_settings = RequireObject(document, "solver_settings", "run config");
  RequireField(solver_settings, "cfl", run_config.solver_settings.cfl, "solver_settings");
  RequireField(
      solver_settings, "pressure_iterations", run_config.solver_settings.pressure_iterations, "solver_settings");

  const json& output_settings = RequireObject(document, "output_settings", "run config");
  RequireField(output_settings, "end_time", run_config.output_settings.end_time, "output_settings");
  RequireField(output_settings, "output_interval", run_config.output_settings.output_interval, "output_settings");

  RequireField(document, "init_state", run_config.init_state, "run config", [&config_path](const json& value) {
    const std::string file = value.get<std::string>();
    return io::Filed<io::State>{file, LoadStateFile(ResolveSiblingPath(config_path, file))};
  });

  const auto outputs_it = document.find("outputs");
  if (outputs_it != document.end() && !outputs_it->is_null()) {
    if (!outputs_it->is_array()) {
      throw std::runtime_error("run config field 'outputs' must be an array of strings.");
    }
    for (const json& entry : *outputs_it) {
      if (!entry.is_string()) {
        throw std::runtime_error("run config field 'outputs' must contain only strings.");
      }
      run_config.outputs.push_back(entry.get<std::string>());
    }
  }

  return run_config;
}

io::State IO::LoadStateFile(const std::filesystem::path& state_path) {
  const json document = ReadJsonFile(state_path);
  if (!document.is_object()) {
    throw std::runtime_error("State file must be a JSON object.");
  }

  io::State state;
  RequireField(document, "time", state.time, "state");

  const json& grid = RequireObject(document, "grid", "state");
  RequireField(grid, "nx", state.grid.nx, "grid");
  RequireField(grid, "ny", state.grid.ny, "grid");
  RequireField(grid, "h", state.grid.h, "grid");
  RequireField(grid, "frame", state.grid.frame, "grid", [&state_path, &state](const json& value) {
    const std::string file = value.get<std::string>();
    return std::optional<io::Filed<io::Frame>>{
        io::Filed<io::Frame>{file, LoadFrameFile(ResolveSiblingPath(state_path, file), state.grid.nx, state.grid.ny)}};
  });

  const json& material = RequireObject(document, "material", "state");
  const auto speed_of_sound_it = material.find("speed_of_sound");
  if (speed_of_sound_it != material.end() && !speed_of_sound_it->is_null()) {
    state.material.speed_of_sound = speed_of_sound_it->get<double>();
  }
  RequireField(material, "reference_density", state.material.reference_density, "material");
  RequireField(material, "kinematic_viscosity", state.material.kinematic_viscosity, "material");
  RequireField(material, "density_diffusivity", state.material.density_diffusivity, "material");

  return state;
}

io::Frame IO::LoadFrameFile(const std::filesystem::path& frame_path, int nx, int ny) {
  return io::read_frame_hdf5(frame_path, nx, ny);
}

void IO::ValidateRunConfig(const io::RunConfig& run_config, const io::State& init_state) {
  if (run_config.init_state.file.empty()) {
    throw std::runtime_error("init_state must not be empty.");
  }
  if (run_config.solver_settings.cfl <= 0.0) {
    throw std::runtime_error("solver_settings.cfl must be positive.");
  }
  if (run_config.solver_settings.pressure_iterations < 1) {
    throw std::runtime_error("solver_settings.pressure_iterations must be positive.");
  }
  if (run_config.output_settings.output_interval <= 0.0) {
    throw std::runtime_error("output_settings.output_interval must be positive.");
  }
  if (run_config.output_settings.end_time < init_state.time) {
    throw std::runtime_error("output_settings.end_time must be >= init_state.time.");
  }
  if (init_state.grid.nx < 2 || init_state.grid.ny < 2) {
    throw std::runtime_error("State grid must be at least 2x2.");
  }
  if (init_state.grid.h <= 0.0) {
    throw std::runtime_error("grid.h must be positive.");
  }
  if (init_state.material.reference_density <= 0.0) {
    throw std::runtime_error("material.reference_density must be positive.");
  }
  if (init_state.material.kinematic_viscosity < 0.0) {
    throw std::runtime_error("material.kinematic_viscosity must be non-negative.");
  }
  if (init_state.material.density_diffusivity < 0.0) {
    throw std::runtime_error("material.density_diffusivity must be non-negative.");
  }
  if (!init_state.grid.frame.has_value()) {
    throw std::runtime_error("State file must provide grid.frame.");
  }
}

void IO::WriteRunConfig(const std::filesystem::path& config_path, const io::RunConfig& run_config) {
  std::ofstream output(config_path);
  if (!output) {
    throw std::runtime_error("Unable to write run config JSON: " + config_path.string());
  }

  output << ToJson(run_config).dump(2) << '\n';
}

void IO::WriteStateFile(const std::filesystem::path& state_path, const io::State& state) {
  std::ofstream output(state_path);
  if (!output) {
    throw std::runtime_error("Unable to write state JSON: " + state_path.string());
  }

  output << ToJson(state).dump(2) << '\n';
}

} // namespace fluid_sim
