#pragma once

#include "fluid_sim/gpu_vector.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fluid_sim {

struct CellState {
  float density_offset = 0.0f;
  float3 velocity = make_float3(0.0f, 0.0f, 0.0f);
};

struct CellCloudView {
  uint32_t size_x = 0;
  uint32_t size_y = 0;
  CellState* cell_state = nullptr;
  CellState* cell_state_tmp = nullptr;
  float* pressure = nullptr;
  float* pressure_tmp = nullptr;
  float* divergence = nullptr;
};

struct CellCloud {
  uint32_t size_x = 0;
  uint32_t size_y = 0;
  GPUVector<CellState> cell_state;
  GPUVector<CellState> cell_state_tmp;
  GPUVector<float> pressure;
  GPUVector<float> pressure_tmp;
  GPUVector<float> divergence;

  void resize(uint32_t new_size_x, uint32_t new_size_y) {
    size_x = new_size_x;
    size_y = new_size_y;
    const std::size_t count = cell_count();
    cell_state.resize(count);
    cell_state_tmp.resize(count);
    pressure.resize(count);
    pressure_tmp.resize(count);
    divergence.resize(count);

    const CellState zero_state{};
    cell_state.fill(zero_state);
    cell_state_tmp.fill(zero_state);
    pressure.fill(0.0f);
    pressure_tmp.fill(0.0f);
    divergence.fill(0.0f);
  }

  [[nodiscard]] std::size_t cell_count() const {
    return static_cast<std::size_t>(size_x) * static_cast<std::size_t>(size_y);
  }

  [[nodiscard]] CellCloudView view() {
    return CellCloudView{size_x, size_y, cell_state.data(), cell_state_tmp.data(), pressure.data(),
                         pressure_tmp.data(), divergence.data()};
  }
};

}  // namespace fluid_sim
