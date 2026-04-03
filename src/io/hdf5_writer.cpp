#include "hdf5_writer.hpp"
#include "simulation_types.hpp"

#include <hdf5.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace io {

void check_hdf5(herr_t status, const char* message) {
  if (status < 0) {
    throw std::runtime_error(message);
  }
}

class H5Handle {
public:
  using CloseFunction = herr_t (*)(hid_t);

  H5Handle() = default;
  H5Handle(hid_t id, CloseFunction close_fn)
      : id_(id),
        close_fn_(close_fn) {}

  ~H5Handle() {
    if (id_ >= 0 && close_fn_ != nullptr) {
      close_fn_(id_);
    }
  }

  H5Handle(const H5Handle&) = delete;
  H5Handle& operator=(const H5Handle&) = delete;

  H5Handle(H5Handle&& other) noexcept
      : id_(other.id_),
        close_fn_(other.close_fn_) {
    other.id_ = -1;
    other.close_fn_ = nullptr;
  }

  H5Handle& operator=(H5Handle&& other) noexcept {
    if (this != &other) {
      if (id_ >= 0 && close_fn_ != nullptr) {
        close_fn_(id_);
      }
      id_ = other.id_;
      close_fn_ = other.close_fn_;
      other.id_ = -1;
      other.close_fn_ = nullptr;
    }
    return *this;
  }

  [[nodiscard]] hid_t get() const {
    return id_;
  }

private:
  hid_t id_ = -1;
  CloseFunction close_fn_ = nullptr;
};

bool IsMountedWindowsPath(const std::filesystem::path& path) {
  const std::string text = path.generic_string();
  return text.rfind("/mnt/", 0) == 0;
}

std::filesystem::path MakeTempCopyIfNeeded(const std::filesystem::path& input_path) {
  if (!IsMountedWindowsPath(input_path)) {
    return input_path;
  }

  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() / ("fluid_sim_init_" + std::to_string(static_cast<long long>(::getpid())) +
                                                "_" + std::to_string(static_cast<long long>(stamp)) + ".h5");
  std::filesystem::copy_file(input_path, temp_path, std::filesystem::copy_options::overwrite_existing);
  return temp_path;
}

void write_scalar_dataset(hid_t file, const char* name, int nx, int ny, const std::vector<float>& values) {
  const std::array<hsize_t, 3> dims = {1, static_cast<hsize_t>(ny), static_cast<hsize_t>(nx)};
  H5Handle dataspace(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 dataspace.");
  }

  H5Handle dataset(H5Dcreate2(file, name, H5T_NATIVE_FLOAT, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                   H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to create dataset: ") + name);
  }

  check_hdf5(H5Dwrite(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
             "Unable to write dataset.");
}

void write_vector_dataset(hid_t file, const char* name, int nx, int ny, const std::vector<float>& values) {
  const std::array<hsize_t, 4> dims = {1, static_cast<hsize_t>(ny), static_cast<hsize_t>(nx), 3};
  H5Handle dataspace(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 vector dataspace.");
  }

  H5Handle dataset(H5Dcreate2(file, name, H5T_NATIVE_FLOAT, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                   H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to create dataset: ") + name);
  }

  check_hdf5(H5Dwrite(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
             "Unable to write vector dataset.");
}

std::vector<float> read_dataset(hid_t file, const char* name, std::size_t expected_size) {
  H5Handle dataset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to open dataset: ") + name);
  }

  std::vector<float> values(expected_size);
  check_hdf5(H5Dread(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
             "Unable to read dataset.");
  return values;
}

void write_frame_hdf5(const std::filesystem::path& output_path, const io::Frame& state) {
  const std::size_t expected_size = static_cast<std::size_t>(config.nx) * static_cast<std::size_t>(config.ny);
  if (state.density_offset.size() != expected_size || state.velocity.size() != expected_size * 3U) {
    throw std::runtime_error("HostState size does not match simulation dimensions.");
  }

  H5Handle file(H5Fcreate(output_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
  if (file.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 output file.");
  }

  write_scalar_dataset(file.get(), "density_offset", config.nx, config.ny, state.density_offset);
  write_vector_dataset(file.get(), "velocity", config.nx, config.ny, state.velocity);
}

io::Frame read_frame_hdf5(const std::filesystem::path& input_path, int nx, int ny) {
  const std::filesystem::path local_input_path = MakeTempCopyIfNeeded(input_path);

  try {
    H5Handle file(H5Fopen(local_input_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
    if (file.get() < 0) {
      throw std::runtime_error("Unable to open HDF5 input file: " + local_input_path.string());
    }

    const std::size_t cell_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    io::Frame frame;
    frame.density_offset = read_dataset(file.get(), "density_offset", cell_count);
    frame.velocity = read_dataset(file.get(), "velocity", cell_count * 3U);

    if (local_input_path != input_path) {
      std::error_code cleanup_error;
      std::filesystem::remove(local_input_path, cleanup_error);
    }

    return frame;
  } catch (...) {
    if (local_input_path != input_path) {
      std::error_code cleanup_error;
      std::filesystem::remove(local_input_path, cleanup_error);
    }
    throw;
  }
}

} // namespace io
