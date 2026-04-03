#pragma once

#include "simulation/gpu_vector.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fluid_sim {

struct CellState {
  float density_offset = 0.0f;
  float3 velocity = make_float3(0.0f, 0.0f, 0.0f);
};

struct CellCloudView {
  uint32_t s_x = 0;
  uint32_t s_y = 0;
  float kin_visc = 0.001f;
  float dty_visc = 0.001f;
  float ref_dty = 1.225;
  CellState* cell_state = nullptr;
  CellState* cell_state_tmp = nullptr;
  [[nodiscard]] uint32_t size() const {
    return s_x * s_y;
  }
};

struct CellCloud {
  uint32_t s_x = 0;
  uint32_t s_y = 0;
  float kin_visc = 0.001f;
  float dty_visc = 0.001f;
  float ref_dty = 1.225;
  GPUVector<CellState> cell_state;
  GPUVector<CellState> cell_state_tmp;

  void resize(const uint32_t new_size_x, const uint32_t new_size_y) {
    s_x = new_size_x;
    s_y = new_size_y;

    cell_state.resize(size());
    cell_state_tmp.resize(size());
  }

  [[nodiscard]] size_t size() const {
    return static_cast<size_t>(s_x) * static_cast<size_t>(s_y);
  }

  [[nodiscard]] CellCloudView view() {
    return CellCloudView{s_x, s_y, kin_visc, dty_visc, ref_dty, cell_state.data(), cell_state_tmp.data()};
  }
};

} // namespace fluid_sim
