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

  const float rho_l = ref_dty + ls.density_offset;
  const float rho_r = ref_dty + rs.density_offset;

  const float rho = sqrt(rho_l * rho_r);
  const V3 u = (rs.momentum + ls.momentum) / (rho_r + rho_l);
  const float u_n = dot(u, normal);

  CellState F;
  F.density_offset = rho * u_n - 0.0001f * sos * (rs.density_offset - ls.density_offset);
  F.momentum = rho * u_n * u + normal * prs(rho - ref_dty, sos);
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
  const float inv_h = 1.0f / cloud.h;
  const float inv_h2 = inv_h * inv_h;
  constexpr V3 nx_pos(1.0f, 0.0f, 0.0f);
  constexpr V3 nx_neg(-1.0f, 0.0f, 0.0f);
  constexpr V3 ny_pos(0.0f, 1.0f, 0.0f);
  constexpr V3 ny_neg(0.0f, -1.0f, 0.0f);

  for (uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; y < cloud.size_y; y += blockDim.y * gridDim.y) {
    for (uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; x < cloud.size_x; x += blockDim.x * gridDim.x) {
      const uint32_t c = index_2d(x, y, cloud.size_x);
      const CellState state = cloud.cell_state[c];
      const float rho_c = cloud.ref_dty + state.density_offset;
      const V3 vel_c = state.momentum / rho_c;
      CellState tmp{0, V3(0)};

      {
        CellState rs = state;
        if (x == cloud.size_x - 1) {
          rs.density_offset = state.density_offset;
          rs.momentum = -state.momentum;
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
          rs.density_offset = state.density_offset;
          rs.momentum = -state.momentum;
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
          rs.density_offset = state.density_offset;
          rs.momentum = -state.momentum;
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
          rs.density_offset = rs.density_offset;
          rs.momentum = -state.momentum;
        } else {
          const uint32_t d = index_2d(x, y - 1, cloud.size_x);
          rs = cloud.cell_state[d];
        }
        const CellState flux = FakeRiemann(state, rs, ny_neg, cloud.ref_dty, cloud.sos);
        tmp.density_offset -= inv_h * flux.density_offset;
        tmp.momentum -= inv_h * flux.momentum;
      }

      if (cloud.kin_visc > 0.0f) {
        V3 vel_xp = -vel_c;
        V3 vel_xm = -vel_c;
        V3 vel_yp = -vel_c;
        V3 vel_ym = -vel_c;

        if (x + 1 < cloud.size_x) {
          const CellState nbr = cloud.cell_state[index_2d(x + 1, y, cloud.size_x)];
          vel_xp = nbr.momentum / rho_c;
        }
        if (x > 0) {
          const CellState nbr = cloud.cell_state[index_2d(x - 1, y, cloud.size_x)];
          vel_xm = nbr.momentum / rho_c;
        }
        if (y + 1 < cloud.size_y) {
          const CellState nbr = cloud.cell_state[index_2d(x, y + 1, cloud.size_x)];
          vel_yp = nbr.momentum / rho_c;
        }
        if (y > 0) {
          const CellState nbr = cloud.cell_state[index_2d(x, y - 1, cloud.size_x)];
          vel_ym = nbr.momentum / rho_c;
        }

        const V3 lap_vel = (vel_xp + vel_xm + vel_yp + vel_ym - 4.0f * vel_c) * inv_h2;
        tmp.momentum += cloud.kin_visc * cloud.ref_dty * lap_vel;
      }
      const float X = x * cloud.h;
      const float Y = y * cloud.h;
      if (((X - 2) * (X - 2) + Y * Y) < 0.25)
        tmp.momentum[0] += 0.5f * (1.0f - cloud.cell_state[c].momentum[0]);
      // tmp.momentum[0] += (1.0 - tmp.momentum[0]);
      cloud.cell_state_tmp[c] = tmp;
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
  const double advective_dt = settings_.cfl * cloud_.h / std::max<double>(max_speed, 1.0e-6);
  if (cloud_.kin_visc <= 0.0f) {
    return advective_dt;
  }
  const double diffusive_dt = 0.25 * cloud_.h * cloud_.h / cloud_.kin_visc;
  return std::min(advective_dt, diffusive_dt);
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

  // std::cout << cloud_.kin_visc << "\n";
  RHS<<<launch.grid, launch.block>>>(make_view(base_state.data(), k1.data()), current_time);
  // CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(
      base_state.data(), k1.data(), stage_state.data(), 0.5f * dt, cell_count);
  // CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k2.data()), current_time + 0.5f * dt);
  // CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(
      base_state.data(), k2.data(), stage_state.data(), 0.5f * dt, cell_count);
  // CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k3.data()), current_time + 0.5f * dt);
  // CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(base_state.data(), k3.data(), stage_state.data(), dt, cell_count);
  // CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k4.data()), current_time + dt);
  // CUDA_CHECK(cudaGetLastError());

  FinishRK4<<<linear_grid, linear_block>>>(
      base_state.data(), k1.data(), k2.data(), k3.data(), k4.data(), cloud_.cell_state.data(), dt, cell_count);
  // CUDA_CHECK(cudaGetLastError());

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
