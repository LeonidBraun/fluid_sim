#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace io {

struct GridAttribute {
  std::vector<std::size_t> shape;
  std::vector<float> values;

  [[nodiscard]] std::size_t element_count() const;
};

struct Grid {
  std::array<int, 3> shape{200, 100, 50};
  float h = 1.0f;
  std::array<float, 3> origin{0.0f, 0.0f, 0.0f};
  std::unordered_map<std::string, GridAttribute> cell_attributes;
  std::unordered_map<std::string, GridAttribute> point_attributes;

  void validate() const;
  void write_hdf5(const std::filesystem::path& output_path) const;
  [[nodiscard]] static Grid read_hdf5(const std::filesystem::path& input_path);
};

} // namespace io
