#include "io/hdf5_common.hpp"
#include "io/hdf5_writer.hpp"

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace io {
namespace {

void write_float_dataset(hid_t file, const char* name, const std::vector<hsize_t>& dims, const float* values) {
  hdf5::Handle dataspace(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 dataspace.");
  }

  hdf5::Handle dataset(
      H5Dcreate2(file, name, H5T_NATIVE_FLOAT, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to create dataset: ") + name);
  }

  hdf5::check_status(H5Dwrite(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values),
                     "Unable to write float dataset.");
}

[[nodiscard]] std::vector<float> read_float_dataset(hid_t file, const char* name, std::size_t expected_size) {
  hdf5::Handle dataset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to open dataset: ") + name);
  }

  std::vector<float> values(expected_size);
  hdf5::check_status(H5Dread(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
                     "Unable to read float dataset.");
  return values;
}

} // namespace

void write_frame_hdf5(const std::filesystem::path& output_path, int nx, int ny, int nz, const Frame& frame) {
  const std::size_t cell_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
  if (frame.density_offset.size() != cell_count || frame.momentum.size() != cell_count * 3U) {
    throw std::runtime_error("Frame payload size does not match the target grid.");
  }

  hdf5::Handle file(H5Fcreate(output_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
  if (file.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 output file.");
  }

  write_float_dataset(
      file.get(),
      "density_offset",
      {static_cast<hsize_t>(nz), static_cast<hsize_t>(ny), static_cast<hsize_t>(nx)},
      frame.density_offset.data());
  write_float_dataset(
      file.get(),
      "momentum",
      {static_cast<hsize_t>(nz), static_cast<hsize_t>(ny), static_cast<hsize_t>(nx), 3},
      frame.momentum.data());
}

Frame read_frame_hdf5(const std::filesystem::path& input_path, int nx, int ny, int nz) {
  hdf5::Handle file(H5Fopen(input_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (file.get() < 0) {
    throw std::runtime_error("Unable to open HDF5 input file: " + input_path.string());
  }

  const std::size_t cell_count =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
  Frame frame;
  frame.density_offset = read_float_dataset(file.get(), "density_offset", cell_count);
  frame.momentum = read_float_dataset(file.get(), "momentum", cell_count * 3U);
  return frame;
}

} // namespace io
