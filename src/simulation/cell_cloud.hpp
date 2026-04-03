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
  uint32_t size_x = 0;
  uint32_t size_y = 0;
  CellState* cell_state = nullptr;
  CellState* cell_state_tmp = nullptr;
  CellState* cell_state_flux = nullptr;

  float h = 1.0;
  float kin_visc = 0.0;
  float dty_visc = 0.0;
  float ref_dty = 0.0;
};

struct CellCloud {
  uint32_t size_x = 0;
  uint32_t size_y = 0;
  GPUVector<CellState> cell_state;
  GPUVector<CellState> cell_state_tmp;
  GPUVector<CellState> cell_state_flux;

  float h;
  float kin_visc;
  float dty_visc;
  float ref_dty;

  void resize(const uint32_t new_size_x, const uint32_t new_size_y) {
    size_x = new_size_x;
    size_y = new_size_y;

    cell_state.resize(size());
    cell_state_tmp.resize(size());
    cell_state_flux.resize(size());
  }

  [[nodiscard]] std::size_t size() const {
    return static_cast<std::size_t>(size_x) * static_cast<std::size_t>(size_y);
  }

  [[nodiscard]] CellCloudView view() {
    return CellCloudView{size_x,
                         size_y,
                         cell_state.data(),
                         cell_state_tmp.data(),
                         cell_state_flux.data(),
                         h,
                         kin_visc,
                         dty_visc,
                         ref_dty};
  }
};

} // namespace fluid_sim
