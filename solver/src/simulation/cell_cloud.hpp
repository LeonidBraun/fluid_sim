#pragma once

#include "math/vector.hpp"
#include "simulation/gpu_vector.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fluid_sim {

struct CellState {
  float density_offset = 0.0f;
  V3 momentum = V3(0.0f, 0.0f, 0.0f);
};

struct CellCloudView {
  uint32_t size_x = 0;
  uint32_t size_y = 0;
  uint32_t size_z = 0;
  CellState* cell_state = nullptr;
  CellState* cell_state_tmp = nullptr;
  float h = 1.0;
  float kin_visc = 0.0;
  float dty_visc = 0.0;
  float ref_dty = 0.0;
  float sos = 0.0;
};

struct CellCloud {
  uint32_t size_x = 0;
  uint32_t size_y = 0;
  uint32_t size_z = 0;
  GPUVector<CellState> cell_state;
  GPUVector<CellState> cell_state_tmp;

  float h;
  float kin_visc;
  float dty_visc;
  float ref_dty;
  float sos;

  void resize(const uint32_t new_size_x, const uint32_t new_size_y, const uint32_t new_size_z) {
    size_x = new_size_x;
    size_y = new_size_y;
    size_z = new_size_z;

    cell_state.resize(size());
    cell_state_tmp.resize(size());
  }

  [[nodiscard]] uint32_t size() const {
    return size_x * size_y * size_z;
  }

  [[nodiscard]] CellCloudView view() {
    return CellCloudView{
        size_x, size_y, size_z, cell_state.data(), cell_state_tmp.data(), h, kin_visc, dty_visc, ref_dty, sos};
  }
};

} // namespace fluid_sim
