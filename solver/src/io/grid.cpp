#include "io/hdf5_common.hpp"
#include "io/grid.hpp"

#include <hdf5.h>

#include <array>
#include <filesystem>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace io {
namespace {

[[nodiscard]] std::size_t Product(const std::vector<std::size_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>{});
}

[[nodiscard]] std::vector<std::size_t> ToSizeVector(const std::array<int, 3>& shape) {
  return {
      static_cast<std::size_t>(shape[0]),
      static_cast<std::size_t>(shape[1]),
      static_cast<std::size_t>(shape[2]),
  };
}

[[nodiscard]] std::vector<std::size_t> ToPointShape(const std::array<int, 3>& shape) {
  return {
      static_cast<std::size_t>(shape[0] + 1),
      static_cast<std::size_t>(shape[1] + 1),
      static_cast<std::size_t>(shape[2] + 1),
  };
}

void write_scalar_dataset(hid_t file, const char* name, float value) {
  hdf5::Handle dataspace(H5Screate(H5S_SCALAR), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create scalar HDF5 dataspace.");
  }

  hdf5::Handle dataset(
      H5Dcreate2(file, name, H5T_NATIVE_FLOAT, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to create dataset: ") + name);
  }

  hdf5::check_status(H5Dwrite(dataset.get(), H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value),
                     "Unable to write scalar dataset.");
}

template <typename T, std::size_t N>
void write_fixed_vector_dataset(hid_t file, const char* name, const std::array<T, N>& values, hid_t dtype) {
  const std::array<hsize_t, 1> dims = {N};
  hdf5::Handle dataspace(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to create fixed vector HDF5 dataspace.");
  }

  hdf5::Handle dataset(
      H5Dcreate2(file, name, dtype, dataspace.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to create dataset: ") + name);
  }

  hdf5::check_status(H5Dwrite(dataset.get(), dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
                     "Unable to write fixed vector dataset.");
}

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

[[nodiscard]] std::vector<int> read_int_dataset(hid_t file, const char* name, std::size_t expected_size) {
  hdf5::Handle dataset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to open dataset: ") + name);
  }

  std::vector<int> values(expected_size);
  hdf5::check_status(H5Dread(dataset.get(), H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()),
                     "Unable to read int dataset.");
  return values;
}

[[nodiscard]] std::vector<hsize_t> read_dataset_shape(hid_t file, const char* name) {
  hdf5::Handle dataset(H5Dopen2(file, name, H5P_DEFAULT), H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error(std::string("Unable to open dataset: ") + name);
  }

  hdf5::Handle dataspace(H5Dget_space(dataset.get()), H5Sclose);
  if (dataspace.get() < 0) {
    throw std::runtime_error("Unable to access dataset dataspace.");
  }

  const int rank = H5Sget_simple_extent_ndims(dataspace.get());
  if (rank < 0) {
    throw std::runtime_error("Unable to query dataset rank.");
  }

  std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
  hdf5::check_status(H5Sget_simple_extent_dims(dataspace.get(), dims.data(), nullptr), "Unable to read dataset shape.");
  return dims;
}

void write_attribute_group(hid_t file, const char* group_name,
                           const std::unordered_map<std::string, GridAttribute>& attributes) {
  hdf5::Handle group(H5Gcreate2(file, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
  if (group.get() < 0) {
    throw std::runtime_error(std::string("Unable to create group: ") + group_name);
  }

  for (const auto& [name, attribute] : attributes) {
    std::vector<hsize_t> dims(attribute.shape.begin(), attribute.shape.end());
    write_float_dataset(group.get(), name.c_str(), dims, attribute.values.data());
  }
}

[[nodiscard]] GridAttribute read_attribute_dataset(hid_t group, const std::string& name) {
  hdf5::Handle dataset(H5Dopen2(group, name.c_str(), H5P_DEFAULT), H5Dclose);
  if (dataset.get() < 0) {
    throw std::runtime_error("Unable to open attribute dataset: " + name);
  }

  const std::vector<hsize_t> dims = read_dataset_shape(group, name.c_str());
  const int rank = static_cast<int>(dims.size());
  if (rank < 3) {
    throw std::runtime_error("Grid attribute '" + name + "' must have rank at least 3.");
  }

  GridAttribute attribute;
  attribute.shape.assign(dims.begin(), dims.end());
  attribute.values.resize(Product(attribute.shape));
  attribute.values = read_float_dataset(group, name.c_str(), attribute.values.size());
  return attribute;
}

[[nodiscard]] std::unordered_map<std::string, GridAttribute> read_attribute_group(hid_t file, const char* group_name) {
  std::unordered_map<std::string, GridAttribute> attributes;

  if (H5Lexists(file, group_name, H5P_DEFAULT) <= 0) {
    return attributes;
  }

  hdf5::Handle group(H5Gopen2(file, group_name, H5P_DEFAULT), H5Gclose);
  if (group.get() < 0) {
    throw std::runtime_error(std::string("Unable to open group: ") + group_name);
  }

  H5G_info_t group_info{};
  hdf5::check_status(H5Gget_info(group.get(), &group_info), "Unable to inspect attribute group.");

  for (hsize_t i = 0; i < group_info.nlinks; ++i) {
    const ssize_t name_size = H5Lget_name_by_idx(
        group.get(), ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
    if (name_size < 0) {
      throw std::runtime_error("Unable to inspect attribute dataset name.");
    }

    std::string name(static_cast<std::size_t>(name_size) + 1, '\0');
    hdf5::check_status(H5Lget_name_by_idx(
                           group.get(), ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), name.size(), H5P_DEFAULT),
                       "Unable to read attribute dataset name.");
    name.resize(static_cast<std::size_t>(name_size));
    attributes.emplace(name, read_attribute_dataset(group.get(), name));
  }
  return attributes;
}

} // namespace

std::size_t GridAttribute::element_count() const {
  return Product(shape);
}

void Grid::validate() const {
  for (const int axis : shape) {
    if (axis <= 0) {
      throw std::runtime_error("Grid shape entries must be positive.");
    }
  }

  const auto expected_cell_shape = ToSizeVector(shape);
  const auto expected_point_shape = ToPointShape(shape);

  for (const auto& [name, attribute] : cell_attributes) {
    if (attribute.shape.size() < 3) {
      throw std::runtime_error("Cell attribute '" + name + "' must have rank at least 3.");
    }
    if (!std::equal(expected_cell_shape.begin(), expected_cell_shape.end(), attribute.shape.begin())) {
      throw std::runtime_error("Cell attribute '" + name + "' shape does not match the grid cell shape.");
    }
    if (attribute.values.size() != attribute.element_count()) {
      throw std::runtime_error("Cell attribute '" + name + "' payload size does not match its shape.");
    }
  }

  for (const auto& [name, attribute] : point_attributes) {
    if (attribute.shape.size() < 3) {
      throw std::runtime_error("Point attribute '" + name + "' must have rank at least 3.");
    }
    if (!std::equal(expected_point_shape.begin(), expected_point_shape.end(), attribute.shape.begin())) {
      throw std::runtime_error("Point attribute '" + name + "' shape does not match the grid point shape.");
    }
    if (attribute.values.size() != attribute.element_count()) {
      throw std::runtime_error("Point attribute '" + name + "' payload size does not match its shape.");
    }
  }
}

void Grid::write_hdf5(const std::filesystem::path& output_path) const {
  validate();
  std::filesystem::create_directories(output_path.parent_path());

  hdf5::Handle file(H5Fcreate(output_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), H5Fclose);
  if (file.get() < 0) {
    throw std::runtime_error("Unable to create HDF5 output file.");
  }

  write_fixed_vector_dataset(file.get(), "shape", shape, H5T_NATIVE_INT);
  write_scalar_dataset(file.get(), "h", h);
  write_fixed_vector_dataset(file.get(), "origin", origin, H5T_NATIVE_FLOAT);
  write_attribute_group(file.get(), "cell_attributes", cell_attributes);
  write_attribute_group(file.get(), "point_attributes", point_attributes);
}

Grid Grid::read_hdf5(const std::filesystem::path& input_path) {
  hdf5::Handle file(H5Fopen(input_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (file.get() < 0) {
    throw std::runtime_error("Unable to open HDF5 input file: " + input_path.string());
  }

  const std::vector<int> shape_values = read_int_dataset(file.get(), "shape", 3);
  const std::vector<float> h_values = read_float_dataset(file.get(), "h", 1);
  const std::vector<float> origin_values = read_float_dataset(file.get(), "origin", 3);

  Grid grid;
  grid.shape = {shape_values[0], shape_values[1], shape_values[2]};
  grid.h = h_values[0];
  grid.origin = {origin_values[0], origin_values[1], origin_values[2]};
  grid.cell_attributes = read_attribute_group(file.get(), "cell_attributes");
  grid.point_attributes = read_attribute_group(file.get(), "point_attributes");
  grid.validate();
  return grid;
}

} // namespace io
