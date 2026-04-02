#include "fluid_sim/hdf5_writer.hpp"

#include <hdf5.h>

#include <array>
#include <stdexcept>
#include <string>

namespace fluid_sim {
namespace {

void check_hdf5(herr_t status, const char* message) {
  if (status < 0) {
    throw std::runtime_error(message);
  }
}

class H5Handle {
 public:
  using CloseFunction = herr_t (*)(hid_t);

  H5Handle() = default;
  H5Handle(hid_t id, CloseFunction close_fn) : id_(id), close_fn_(close_fn) {}
  ~H5Handle() {
    if (id_ >= 0 && close_fn_ != nullptr) {
      close_fn_(id_);
    }
  }

  H5Handle(const H5Handle&) = delete;
  H5Handle& operator=(const H5Handle&) = delete;

  H5Handle(H5Handle&& other) noexcept : id_(other.id_), close_fn_(other.close_fn_) {
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

  [[nodiscard]] hid_t get() const { return id_; }

 private:
  hid_t id_ = -1;
  CloseFunction close_fn_ = nullptr;
};

void write_scalar_attribute(hid_t object, const char* name, double value) {
  H5Handle dataspace(H5Screate(H5S_SCALAR), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 scalar dataspace.");
  }

  H5Handle attribute(H5Acreate2(object, name, H5T_NATIVE_DOUBLE, dataspace.get(), H5P_DEFAULT,
                                H5P_DEFAULT),
                     H5Aclose);
  if (attribute.get() < 0) {
    throw std::runtime_error(std::string("Unable to create HDF5 attribute: ") + name);
  }

  check_hdf5(H5Awrite(attribute.get(), H5T_NATIVE_DOUBLE, &value), "Unable to write double attribute.");
}

void write_scalar_attribute(hid_t object, const char* name, int value) {
  H5Handle dataspace(H5Screate(H5S_SCALAR), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 scalar dataspace.");
  }

  H5Handle attribute(H5Acreate2(object, name, H5T_NATIVE_INT, dataspace.get(), H5P_DEFAULT,
                                H5P_DEFAULT),
                     H5Aclose);
  if (attribute.get() < 0) {
    throw std::runtime_error(std::string("Unable to create HDF5 attribute: ") + name);
  }

  check_hdf5(H5Awrite(attribute.get(), H5T_NATIVE_INT, &value), "Unable to write int attribute.");
}

void write_scalar_dataset(hid_t file, const char* name, int nx, int ny, const std::vector<float>& values) {
  const std::array<hsize_t, 3> dims = {1, static_cast<hsize_t>(ny), static_cast<hsize_t>(nx)};
  H5Handle dataspace(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 dataspace.");
  }

  H5Handle dataset(
      H5Dcreate2(file, name, H5T_NATIVE_FLOAT, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to create dataset: ") + name);
  }

  check_hdf5(H5Dwrite(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
             "Unable to write dataset.");
}

void write_vector_dataset(hid_t file, const char* name, int nx, int ny,
                          const std::vector<float>& velocity_x,
                          const std::vector<float>& velocity_y) {
  const std::array<hsize_t, 4> dims = {1, static_cast<hsize_t>(ny), static_cast<hsize_t>(nx), 3};
  std::vector<float> values(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * 3U, 0.0f);

  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const std::size_t scalar_idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(nx) +
                                     static_cast<std::size_t>(x);
      const std::size_t vector_idx = scalar_idx * 3U;
      values[vector_idx] = velocity_x[scalar_idx];
      values[vector_idx + 1U] = velocity_y[scalar_idx];
    }
  }

  H5Handle dataspace(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 vector dataspace.");
  }

  H5Handle dataset(
      H5Dcreate2(file, name, H5T_NATIVE_FLOAT, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to create dataset: ") + name);
  }

  check_hdf5(H5Dwrite(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
             "Unable to write vector dataset.");
}

}  // namespace

void write_frame_hdf5(const std::filesystem::path& output_path,
                      const SimulationConfig& config,
                      double time,
                      double time_step,
                      const HostState& state) {
  const std::size_t expected_size = static_cast<std::size_t>(config.nx) * static_cast<std::size_t>(config.ny);
  if (state.density_offset.size() != expected_size || state.velocity_x.size() != expected_size ||
      state.velocity_y.size() != expected_size) {
    throw std::runtime_error("HostState size does not match simulation dimensions.");
  }

  H5Handle file(H5Fcreate(output_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
  if (file.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 output file.");
  }

  write_scalar_attribute(file.get(), "time", time);
  write_scalar_attribute(file.get(), "time_step", time_step);
  write_scalar_attribute(file.get(), "end_time", config.end_time);
  write_scalar_attribute(file.get(), "cfl", config.cfl);
  write_scalar_attribute(file.get(), "output_interval", config.output_interval);
  write_scalar_attribute(file.get(), "reference_density", config.reference_density);
  write_scalar_attribute(file.get(), "kinematic_viscosity", config.kinematic_viscosity);
  write_scalar_attribute(file.get(), "density_diffusivity", config.density_diffusivity);
  write_scalar_attribute(file.get(), "nx", config.nx);
  write_scalar_attribute(file.get(), "ny", config.ny);
  write_scalar_attribute(file.get(), "dx", config.dx);
  write_scalar_attribute(file.get(), "dy", config.dy);

  write_scalar_dataset(file.get(), "density_offset", config.nx, config.ny, state.density_offset);
  write_vector_dataset(file.get(), "velocity", config.nx, config.ny, state.velocity_x, state.velocity_y);
}

}  // namespace fluid_sim
