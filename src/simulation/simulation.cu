#include "math/vector.hpp"
#include "simulation/simulation.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fluid_sim {
namespace {

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    const cudaError_t error_code = (call);                                                                             \
    if (error_code != cudaSuccess) {                                                                                   \
      throw std::runtime_error(std::string("CUDA failure: ") + cudaGetErrorString(error_code));                        \
    }                                                                                                                  \
  } while (false)

struct MaxAbsVelocity {
  __host__ __device__ float operator()(const CellState& cell) const {
    const float max = fmaxf(fmaxf(fabsf(cell.momentum[0]), fabsf(cell.momentum[1])), fabsf(cell.momentum[2]));
    return max / (ref_dty + cell.density_offset);
  }
  const float ref_dty;
};

template <typename T>
__device__ T index_2d(T x, T y, T nx) {
  return y * nx + x;
}
template __device__ int index_2d<int>(int x, int y, int nx);
template __device__ uint32_t index_2d<uint32_t>(uint32_t x, uint32_t y, uint32_t nx);

__device__ float prs(const float dty_offset, const float sos) {
  return sos * sos * dty_offset;
}

__device__ CellState
FakeRiemann(const CellState& ls, const CellState& rs, const V3& normal, const float ref_dty, const float sos) {
  const float rho_l = fmaxf(ref_dty + ls.density_offset, 1.0e-6f);
  const float rho_r = fmaxf(ref_dty + rs.density_offset, 1.0e-6f);
  const float un_l = dot(ls.momentum, normal) / rho_l;
  const float un_r = dot(rs.momentum, normal) / rho_r;
  const float p_l = prs(ls.density_offset, sos);
  const float p_r = prs(rs.density_offset, sos);

  const CellState flux_l{dot(ls.momentum, normal), ls.momentum * un_l + p_l * normal};
  const CellState flux_r{dot(rs.momentum, normal), rs.momentum * un_r + p_r * normal};

  const float a = fmaxf(fabsf(un_l) + sos, fabsf(un_r) + sos);

  CellState F;
  F.density_offset =
      0.5f * (flux_l.density_offset + flux_r.density_offset) - 0.5f * a * (rs.density_offset - ls.density_offset);
  F.momentum = 0.5f * (flux_l.momentum + flux_r.momentum) - 0.5f * a * (rs.momentum - ls.momentum);
  return F;
}

__global__ void
BuildStageState(const CellState* base, const CellState* rhs, CellState* out, const float scale, const uint32_t count) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    out[i].density_offset = base[i].density_offset + scale * rhs[i].density_offset;
    out[i].momentum = base[i].momentum + scale * rhs[i].momentum;
  }
}

__global__ void FinishRK4(const CellState* base,
                          const CellState* k1,
                          const CellState* k2,
                          const CellState* k3,
                          const CellState* k4,
                          CellState* out,
                          const float dt,
                          const uint32_t count) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    out[i].density_offset =
        base[i].density_offset +
        dt * (k1[i].density_offset + 2.0f * k2[i].density_offset + 2.0f * k3[i].density_offset + k4[i].density_offset) /
            6.0f;
    out[i].momentum = base[i].momentum +
                      dt * (k1[i].momentum + 2.0f * k2[i].momentum + 2.0f * k3[i].momentum + k4[i].momentum) / 6.0f;
  }
}

__global__ void RHS(CellCloudView cloud, [[maybe_unused]] float t) {
  const float inv_h = 1.0f / fmaxf(cloud.h, 1.0e-6f);
  const V3 nx_pos(1.0f, 0.0f, 0.0f);
  const V3 nx_neg(-1.0f, 0.0f, 0.0f);
  const V3 ny_pos(0.0f, 1.0f, 0.0f);
  const V3 ny_neg(0.0f, -1.0f, 0.0f);

  for (uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; y < cloud.size_y; y += blockDim.y * gridDim.y) {
    for (uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; x < cloud.size_x; x += blockDim.x * gridDim.x) {
      const uint32_t c = index_2d(x, y, cloud.size_x);
      const CellState state = cloud.cell_state[c];
      CellState tmp{0, V3(0)};

      {
        CellState rs = state;
        if (x == cloud.size_x - 1) {
          rs.momentum = state.momentum - 2.0f * dot(state.momentum, nx_pos) * nx_pos;
        } else {
          const uint32_t r = index_2d(x + 1, y, cloud.size_x);
          rs = cloud.cell_state[r];
        }
        const CellState flux = FakeRiemann(state, rs, nx_pos, cloud.ref_dty, cloud.sos);
        tmp.density_offset -= inv_h * flux.density_offset;
        tmp.momentum -= inv_h * flux.momentum;
      }

      {
        CellState rs = state;
        if (x == 0) {
          rs.momentum = state.momentum - 2.0f * dot(state.momentum, nx_neg) * nx_neg;
        } else {
          const uint32_t l = index_2d(x - 1, y, cloud.size_x);
          rs = cloud.cell_state[l];
        }
        const CellState flux = FakeRiemann(state, rs, nx_neg, cloud.ref_dty, cloud.sos);
        tmp.density_offset -= inv_h * flux.density_offset;
        tmp.momentum -= inv_h * flux.momentum;
      }

      {
        CellState rs = state;
        if (y == cloud.size_y - 1) {
          rs.momentum = state.momentum - 2.0f * dot(state.momentum, ny_pos) * ny_pos;
        } else {
          const uint32_t u = index_2d(x, y + 1, cloud.size_x);
          rs = cloud.cell_state[u];
        }
        const CellState flux = FakeRiemann(state, rs, ny_pos, cloud.ref_dty, cloud.sos);
        tmp.density_offset -= inv_h * flux.density_offset;
        tmp.momentum -= inv_h * flux.momentum;
      }

      {
        CellState rs = state;
        if (y == 0) {
          rs.momentum = state.momentum - 2.0f * dot(state.momentum, ny_neg) * ny_neg;
        } else {
          const uint32_t d = index_2d(x, y - 1, cloud.size_x);
          rs = cloud.cell_state[d];
        }
        const CellState flux = FakeRiemann(state, rs, ny_neg, cloud.ref_dty, cloud.sos);
        tmp.density_offset -= inv_h * flux.density_offset;
        tmp.momentum -= inv_h * flux.momentum;
      }

      cloud.cell_state_tmp[c] = tmp;

      if ((x * x + y * y) * cloud.h * cloud.h < 0.5)
        cloud.cell_state_tmp[c].momentum[0] += cloud.sos * (cloud.cell_state[c].density_offset + cloud.ref_dty);
    }
  }
}

} // namespace

Simulation::Simulation(const io::RunConfig::SolverSettings& settings, const io::State& initial_state)
    : settings_(settings),
      time_(initial_state.time) {
  if (initial_state.grid.nx < 2 || initial_state.grid.ny < 2) {
    throw std::runtime_error("Simulation dimensions must be at least 2x2.");
  }

  cloud_.resize(static_cast<uint32_t>(initial_state.grid.nx), static_cast<uint32_t>(initial_state.grid.ny));

  base_state.resize(cloud_.size());
  stage_state.resize(cloud_.size());
  k1.resize(cloud_.size());
  k2.resize(cloud_.size());
  k3.resize(cloud_.size());
  k4.resize(cloud_.size());

  cloud_.h = initial_state.grid.h;
  cloud_.kin_visc = initial_state.material.kinematic_viscosity;
  cloud_.dty_visc = initial_state.material.density_diffusivity;
  cloud_.ref_dty = initial_state.material.reference_density;
  cloud_.sos = initial_state.material.speed_of_sound;

  if (initial_state.grid.frame.has_value()) {
    const io::Frame& frame = initial_state.grid.frame->data;
    if (frame.density_offset.size() != cloud_.size() || frame.momentum.size() != cloud_.size() * 3U) {
      throw std::runtime_error("Initial state size does not match simulation dimensions.");
    }
    std::vector<CellState> device_state(cloud_.size());
    for (std::size_t i = 0; i < cloud_.size(); ++i) {
      device_state[i].density_offset = frame.density_offset[i];
      device_state[i].momentum = V3(frame.momentum[3 * i], frame.momentum[3 * i + 1], frame.momentum[3 * i + 2]);
    }
    cloud_.cell_state.assign(device_state);
  } else {
    cloud_.cell_state.fill(CellState{});
  }

  cloud_.cell_state_tmp.fill(CellState{});

  {
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));

    const dim3 block(16, 16, 1);
    const int target_blocks = std::max(32, device_properties.multiProcessorCount * 4);
    const int grid_x = 8;
    const int grid_y = std::max(4, (target_blocks + grid_x - 1) / grid_x);
    launch = {block, dim3(grid_x, grid_y, 1)};
  }
}

double Simulation::compute_time_step() const {
  thrust::device_ptr<const CellState> begin(cloud_.cell_state.data());
  thrust::device_ptr<const CellState> end = begin + cloud_.cell_state.size();
  const float max_vel = thrust::transform_reduce(
      thrust::device, begin, end, MaxAbsVelocity{cloud_.ref_dty}, 0.0f, thrust::maximum<float>{});
  const float max_speed = max_vel + cloud_.sos;
  return settings_.cfl * cloud_.h / max_speed;
}

void Simulation::step(double max_dt) {
  const double current_dt = std::min(compute_time_step(), max_dt);
  const float dt = static_cast<float>(current_dt);
  const float current_time = static_cast<float>(time_);
  const uint32_t cell_count = cloud_.size();
  const dim3 linear_block(256, 1, 1);
  const dim3 linear_grid(std::max<uint32_t>(1U, (cell_count + linear_block.x - 1) / linear_block.x), 1, 1);

  CUDA_CHECK(cudaMemcpy(
      base_state.data(), cloud_.cell_state.data(), cell_count * sizeof(CellState), cudaMemcpyDeviceToDevice));

  const auto make_view = [&](CellState* state, CellState* rhs) {
    return CellCloudView{cloud_.size_x,
                         cloud_.size_y,
                         state,
                         rhs,
                         cloud_.h,
                         cloud_.kin_visc,
                         cloud_.dty_visc,
                         cloud_.ref_dty,
                         cloud_.sos};
  };

  RHS<<<launch.grid, launch.block>>>(make_view(base_state.data(), k1.data()), current_time);
  CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(
      base_state.data(), k1.data(), stage_state.data(), 0.5f * dt, cell_count);
  CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k2.data()), current_time + 0.5f * dt);
  CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(
      base_state.data(), k2.data(), stage_state.data(), 0.5f * dt, cell_count);
  CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k3.data()), current_time + 0.5f * dt);
  CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(base_state.data(), k3.data(), stage_state.data(), dt, cell_count);
  CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k4.data()), current_time + dt);
  CUDA_CHECK(cudaGetLastError());

  FinishRK4<<<linear_grid, linear_block>>>(
      base_state.data(), k1.data(), k2.data(), k3.data(), k4.data(), cloud_.cell_state.data(), dt, cell_count);
  CUDA_CHECK(cudaGetLastError());

  last_dt_ = current_dt;
  time_ += current_dt;
}

io::Frame Simulation::download_frame() const {
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  io::Frame frame;
  const auto device_state = ToStdVector(cloud_.cell_state);
  frame.density_offset.resize(device_state.size());
  frame.momentum.resize(device_state.size() * 3U);

  for (size_t i = 0; i < device_state.size(); ++i) {
    frame.density_offset[i] = device_state[i].density_offset;
    for (uint32_t k = 0; k < 3; k++)
      frame.momentum[i * 3 + k] = device_state[i].momentum[k];
  }

  return frame;
}

} // namespace fluid_sim
