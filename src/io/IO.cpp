#include "IO.hpp"

#include "hdf5_writer.hpp"
#include "simulation/simulation.hpp"

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

namespace fluid_sim {

using json = nlohmann::json;

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to open JSON file: " + path.string());
  }

  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string StripLineComments(const std::string& text) {
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

std::string StripTrailingCommas(const std::string& text) {
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

json ReadJsonFile(const std::filesystem::path& path) {
  const std::string sanitized = StripTrailingCommas(StripLineComments(ReadTextFile(path)));
  return json::parse(sanitized);
}

template <typename T, typename Func>
void OptionalField(const json& object, const char* key, std::optional<T>& value, const char* scope, Func&& fun) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    return;
  }
  value = fun(it);
}

template <typename T>
void OptionalField(const json& object, const char* key, std::optional<T>& value, const char* scope) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    return;
  }
  value = it->get<T>();
}

template <typename T>
void RequireField(const json& object, const char* key, T& value, const char* scope) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    throw std::runtime_error(std::string("Missing required field '") + key + "' in " + scope + '.');
  }
  value = it->get<T>();
}

const json& void RequireObject(const json& object, const char* key, const char* scope) {
  const auto it = object.find(key);
  if (it == object.end() || it->is_null()) {
    throw std::runtime_error(std::string("Missing required object '") + key + "' in " + scope + '.');
  }
  return *it;
}

std::filesystem::path ResolveSiblingPath(const std::filesystem::path& base_path,
                                         const std::filesystem::path& relative_or_absolute) {
  if (relative_or_absolute.is_absolute()) {
    return relative_or_absolute;
  }
  return (base_path.parent_path() / relative_or_absolute).lexically_normal();
}

json ToJson(const SolverSettings& solver) {
  return json{{"cfl", solver.cfl}, {"pressure_iterations", solver.pressure_iterations}};
}

json ToJson(const OutputSettings& output) {
  return json{{"end_time", output.end_time}, {"output_interval", output.output_interval}};
}

json ToJson(const GridSpec& grid) {
  return json{{"nx", grid.nx}, {"ny", grid.ny}, {"h", grid.h}, {"initial_density", grid.initial_density}};
}

json ToJson(const MaterialProperties& material) {
  return json{{"reference_density", material.reference_density},
              {"kinematic_viscosity", material.kinematic_viscosity},
              {"density_diffusivity", material.density_diffusivity}};
}

json ToJson(const StateFile& state_file) {
  return json{{"time", state_file.time},
              {"grid", ToJson(state_file.grid)},
              {"material", ToJson(state_file.material)},
              {"frame", state_file.frame.generic_string()}};
}

std::filesystem::path CurrentExecutablePath() {
  std::vector<char> buffer(4096, '\0');
  const ssize_t size = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (size < 0) {
    throw std::runtime_error("Unable to resolve /proc/self/exe.");
  }
  buffer[static_cast<std::size_t>(size)] = '\0';
  return std::filesystem::path(buffer.data());
}

std::string ShellQuote(const std::filesystem::path& path) {
  const std::string text = path.string();
  std::string quoted;
  quoted.reserve(text.size() + 2);
  quoted.push_back('\'');
  for (const char c : text) {
    if (c == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(c);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

IO::IO(const std::filesystem::path& config_path) {

  config_path_ = std::filesystem::absolute(config_path);
  root_path_ = config_path_.parent_path();
  output_root_ = root_path_ / "outputs";
  data_dir_ = output_root_ / "data";

  run_config_ = LoadRunConfig(config_path_);
  init_state_ = LoadStateFile(ResolveSiblingPath(config_path_, run_config_.init_state));

  ValidateRunConfig(run_config_, init_state_file_);
  std::filesystem::create_directories(data_dir_);
  initial_state_ = LoadFrameState(
      ResolveSiblingPath(ResolveSiblingPath(config_path_, run_config_.init_state), init_state_file_.frame),
      config_.nx,
      config_.ny);
  run_config_.outputs.clear();
}

double IO::last_saved_time() const {
  if (saved_frames_.empty()) {
    return -std::numeric_limits<double>::infinity();
  }
  return saved_frames_.back().time;
}

void IO::save_frame(const Simulation& simulation) {
  const HostState state = simulation.download_state();
  const std::size_t frame_index = saved_frames_.size();

  const std::string frame_name = "frame_" + std::to_string(frame_index) + ".h5";
  const std::string state_name = "state_" + std::to_string(frame_index) + ".json";

  const std::filesystem::path frame_path = data_dir_ / frame_name;
  const std::filesystem::path state_path = data_dir_ / state_name;

  write_frame_hdf5(frame_path, simulation.config(), simulation.time(), simulation.last_dt(), state);

  StateFile state_file;
  state_file.time = simulation.time();
  state_file.grid = init_state_file_.grid;
  state_file.material = init_state_file_.material;
  state_file.frame = frame_name;
  WriteStateFile(state_path, state_file);

  saved_frames_.push_back({std::string("data/") + frame_name, simulation.time()});
  run_config_.outputs.push_back(std::filesystem::path("outputs") / "data" / state_name);

  WriteRunConfig(config_path_, run_config_);
  write_xdmf_series(output_root_ / "series.xdmf", simulation.config(), saved_frames_);
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

  io::RunConfig rc;

  const json& solver_settings = RequireObject(document, "solver_settings", "RunConfig");
  RequireField(solver_settings, "cfl", rc.solver_settings.cfl, "solver_settings");
  RequireField(solver_settings, "pressure_iterations", rc.solver_settings.pressure_iterations, "solver_settings");

  const json& output_settings = RequireObject(document, "output_settings", "RunConfig");
  RequireField(output_settings, "end_time", rc.output_settings.end_time, "output_settings");
  RequireField(output_settings, "output_interval", rc.output_settings.output_interval, "output_settings");

  RequireField(document, "init_state", rc.init_state, "RunConfig", [](const json& j) {
    const std::string file = j.Get<std::string>();
    return Filed<io::State>{file, LoadStateFile(config_path / file)};
  });

  return rc;
}

StateFile IO::LoadStateFile(const std::filesystem::path& state_path) {
  const json document = ReadJsonFile(state_path);
  if (!document.is_object()) {
    throw std::runtime_error("State file root must be a JSON object.");
  }

  io::State state;

  RequireField(document, "time", state.time, "state");

  const json& grid = RequireObject(document, "grid", "state");
  RequireField(grid, "nx", state.grid.nx, "grid");
  RequireField(grid, "ny", state.grid.ny, "grid");
  RequireField(grid, "h", state.grid.h, "grid");
  RequireField(grid, "frame", state.grid.frame, "grid", [](const json& j) {
    const std::string file = j.Get<std::string>();
    // return Filed<io::State>{file, LoadFrameFile(state_path / file)};
    return Filed<io::State>{file, io::read_frame_hdf5(state_path / file)};
  });

  const json& material = RequireObject(document, "material", "state");
  RequireField(material, "reference_density", state.material.reference_density, "material");
  RequireField(material, "kinematic_viscosity", state.material.kinematic_viscosity, "material");
  RequireField(material, "density_diffusivity", state.material.density_diffusivity, "material");

  return state;
}

static io::Frame IO::LoadFrameFile(const std::filesystem::path& frame_path) {}

void IO::ValidateRunConfig(const RunConfig& run_config, const StateFile& init_state) {
  if (run_config.init_state.empty()) {
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
  if (init_state.grid.nx < 8 || init_state.grid.ny < 8) {
    throw std::runtime_error("State grid must be at least 8x8.");
  }
  if (init_state.grid.h <= 0.0) {
    throw std::runtime_error("grid.h must be positive.");
  }
  if (init_state.grid.initial_density <= 0.0) {
    throw std::runtime_error("grid.initial_density must be positive.");
  }
  if (init_state.material.reference_density <= 0.0) {
    throw std::runtime_error("material.reference_density must be positive.");
  }
  if (init_state.material.kinematic_viscosity < 0.0 || init_state.material.density_diffusivity < 0.0) {
    throw std::runtime_error("Material transport properties must be non-negative.");
  }
}

void IO::WriteRunConfig(const std::filesystem::path& config_path, const RunConfig& run_config) {
  json document;
  document["solver_settings"] = ToJson(run_config.solver_settings);
  document["output_settings"] = ToJson(run_config.output_settings);
  document["init_state"] = run_config.init_state.generic_string();

  json outputs = json::array();
  for (const auto& output : run_config.outputs) {
    outputs.push_back(output.generic_string());
  }
  document["outputs"] = std::move(outputs);

  std::ofstream stream(config_path);
  if (!stream) {
    throw std::runtime_error("Unable to write run config: " + config_path.string());
  }
  stream << document.dump(2) << '\n';
}

void IO::WriteStateFile(const std::filesystem::path& state_path, const StateFile& state_file) {
  std::ofstream stream(state_path);
  if (!stream) {
    throw std::runtime_error("Unable to write state file: " + state_path.string());
  }
  stream << ToJson(state_file).dump(2) << '\n';
}

HostState IO::LoadFrameState(const std::filesystem::path& frame_path, int nx, int ny) {
  const std::filesystem::path helper_path = CurrentExecutablePath().parent_path() / "fluid_frame_dump";
  if (!std::filesystem::exists(helper_path)) {
    throw std::runtime_error("Missing helper executable: " + helper_path.string());
  }

  const std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() /
      ("fluid_frame_dump_" + std::to_string(static_cast<long long>(::getpid())) + ".bin");

  const std::string command = ShellQuote(helper_path) + " " + ShellQuote(frame_path) + " " + std::to_string(nx) + " " +
                              std::to_string(ny) + " " + ShellQuote(temp_path) + " >/dev/null 2>/dev/null";
  const int exit_code = std::system(command.c_str());
  if (exit_code != 0) {
    std::error_code cleanup_error;
    std::filesystem::remove(temp_path, cleanup_error);
    throw std::runtime_error("Unable to load state frame: " + frame_path.string());
  }

  std::ifstream input(temp_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Unable to open dumped frame data: " + temp_path.string());
  }

  std::uint64_t density_size = 0;
  std::uint64_t velocity_size = 0;
  input.read(reinterpret_cast<char*>(&density_size), sizeof(density_size));
  HostState state;
  state.density_offset.resize(static_cast<std::size_t>(density_size));
  input.read(reinterpret_cast<char*>(state.density_offset.data()),
             static_cast<std::streamsize>(state.density_offset.size() * sizeof(float)));
  input.read(reinterpret_cast<char*>(&velocity_size), sizeof(velocity_size));
  state.velocity.resize(static_cast<std::size_t>(velocity_size));
  input.read(reinterpret_cast<char*>(state.velocity.data()),
             static_cast<std::streamsize>(state.velocity.size() * sizeof(float)));

  std::error_code cleanup_error;
  std::filesystem::remove(temp_path, cleanup_error);

  const std::size_t expected_density_size = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  if (state.density_offset.size() != expected_density_size || state.velocity.size() != expected_density_size * 3U) {
    throw std::runtime_error("Loaded frame data size does not match requested grid dimensions.");
  }

  return state;
}

} // namespace fluid_sim
