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

struct LaunchConfig {
  dim3 block;
  dim3 grid;
};

struct MaxAbsVelocity {
  __host__ __device__ float operator()(const CellState& cell) const {
    return fmaxf(fabsf(cell.velocity.x), fabsf(cell.velocity.y));
  }
};

__device__ int index_2d(int x, int y, int nx) {
  return y * nx + x;
}

__device__ float clampf(float value, float lower, float upper) {
  return fminf(fmaxf(value, lower), upper);
}

__device__ float clamp_density_offset(float value, float reference_density) {
  return fmaxf(value, -0.99f * reference_density);
}

__device__ float sample_density_offset(const CellState* state, float x, float y, int nx, int ny) {
  const float gx = clampf(x, 0.0f, static_cast<float>(nx - 1));
  const float gy = clampf(y, 0.0f, static_cast<float>(ny - 1));

  const int x0 = static_cast<int>(floorf(gx));
  const int y0 = static_cast<int>(floorf(gy));
  const int x1 = min(x0 + 1, nx - 1);
  const int y1 = min(y0 + 1, ny - 1);

  const float tx = gx - static_cast<float>(x0);
  const float ty = gy - static_cast<float>(y0);

  const float s00 = state[index_2d(x0, y0, nx)].density_offset;
  const float s10 = state[index_2d(x1, y0, nx)].density_offset;
  const float s01 = state[index_2d(x0, y1, nx)].density_offset;
  const float s11 = state[index_2d(x1, y1, nx)].density_offset;

  const float sx0 = s00 + tx * (s10 - s00);
  const float sx1 = s01 + tx * (s11 - s01);
  return sx0 + ty * (sx1 - sx0);
}

__device__ float3 sample_velocity(const CellState* state, float x, float y, int nx, int ny) {
  const float gx = clampf(x, 0.0f, static_cast<float>(nx - 1));
  const float gy = clampf(y, 0.0f, static_cast<float>(ny - 1));

  const int x0 = static_cast<int>(floorf(gx));
  const int y0 = static_cast<int>(floorf(gy));
  const int x1 = min(x0 + 1, nx - 1);
  const int y1 = min(y0 + 1, ny - 1);

  const float tx = gx - static_cast<float>(x0);
  const float ty = gy - static_cast<float>(y0);

  const float3 s00 = state[index_2d(x0, y0, nx)].velocity;
  const float3 s10 = state[index_2d(x1, y0, nx)].velocity;
  const float3 s01 = state[index_2d(x0, y1, nx)].velocity;
  const float3 s11 = state[index_2d(x1, y1, nx)].velocity;

  const float3 sx0 = make_float3(s00.x + tx * (s10.x - s00.x), s00.y + tx * (s10.y - s00.y), 0.0f);
  const float3 sx1 = make_float3(s01.x + tx * (s11.x - s01.x), s01.y + tx * (s11.y - s01.y), 0.0f);
  return make_float3(sx0.x + ty * (sx1.x - sx0.x), sx0.y + ty * (sx1.y - sx0.y), 0.0f);
}

LaunchConfig make_launch_config() {
  cudaDeviceProp device_properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));

  const dim3 block(16, 16, 1);
  const int target_blocks = std::max(32, device_properties.multiProcessorCount * 4);
  const int grid_x = 8;
  const int grid_y = std::max(4, (target_blocks + grid_x - 1) / grid_x);
  return {block, dim3(grid_x, grid_y, 1)};
}

__global__ void clear_scalar_kernel(CellCloudView cloud, float* field) {
  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < static_cast<int>(cloud.size_y); y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < static_cast<int>(cloud.size_x);
         x += blockDim.x * gridDim.x) {
      field[index_2d(x, y, static_cast<int>(cloud.size_x))] = 0.0f;
    }
  }
}

__global__ void add_source_kernel(CellCloudView cloud, float dt, float time, float h, float reference_density) {
  const float domain_width = static_cast<float>(cloud.size_x) * h;
  const float domain_height = static_cast<float>(cloud.size_y) * h;
  const float source_center_x = 0.5f * domain_width;
  const float source_half_width = 0.12f * domain_width;
  const float source_height = 0.14f * domain_height;
  const float density_offset_rate = 0.25f;
  const float vertical_acceleration = 8.0f;
  const float horizontal_acceleration = 1.5f;

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < static_cast<int>(cloud.size_y); y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < static_cast<int>(cloud.size_x);
         x += blockDim.x * gridDim.x) {
      const float cell_x = (static_cast<float>(x) + 0.5f) * h;
      const float cell_y = (static_cast<float>(y) + 0.5f) * h;
      const bool in_source = fabsf(cell_x - source_center_x) <= source_half_width && cell_y <= source_height;
      if (!in_source) {
        continue;
      }

      const int idx = index_2d(x, y, static_cast<int>(cloud.size_x));
      const float pulse = 0.85f + 0.15f * sinf(time * 1.7f);
      const float swirl = sinf(4.0f * cell_x + time * 1.3f);
      cloud.cell_state[idx].density_offset = clamp_density_offset(
          cloud.cell_state[idx].density_offset + density_offset_rate * dt * pulse, reference_density);
      cloud.cell_state[idx].velocity.y += vertical_acceleration * dt * pulse;
      cloud.cell_state[idx].velocity.x += horizontal_acceleration * dt * swirl;
      cloud.cell_state[idx].velocity.z = 0.0f;
    }
  }
}

__global__ void apply_boundary_kernel(CellCloudView cloud) {
  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < static_cast<int>(cloud.size_y); y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < static_cast<int>(cloud.size_x);
         x += blockDim.x * gridDim.x) {
      if (x == 0 || y == 0 || x == static_cast<int>(cloud.size_x) - 1 || y == static_cast<int>(cloud.size_y) - 1) {
        const int idx = index_2d(x, y, static_cast<int>(cloud.size_x));
        cloud.cell_state[idx].density_offset = 0.0f;
        cloud.cell_state[idx].velocity = make_float3(0.0f, 0.0f, 0.0f);
      }
    }
  }
}

__global__ void diffuse_state_kernel(CellCloudView cloud,
                                     float dt,
                                     float h,
                                     float kinematic_viscosity,
                                     float density_diffusivity,
                                     float reference_density) {
  const float inv_h2 = 1.0f / (h * h);

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < static_cast<int>(cloud.size_y); y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < static_cast<int>(cloud.size_x);
         x += blockDim.x * gridDim.x) {
      const int idx = index_2d(x, y, static_cast<int>(cloud.size_x));
      if (x == 0 || y == 0 || x == static_cast<int>(cloud.size_x) - 1 || y == static_cast<int>(cloud.size_y) - 1) {
        cloud.cell_state_tmp[idx].density_offset = 0.0f;
        cloud.cell_state_tmp[idx].velocity = make_float3(0.0f, 0.0f, 0.0f);
        continue;
      }

      const CellState center = cloud.cell_state[idx];
      const CellState left = cloud.cell_state[index_2d(x - 1, y, static_cast<int>(cloud.size_x))];
      const CellState right = cloud.cell_state[index_2d(x + 1, y, static_cast<int>(cloud.size_x))];
      const CellState down = cloud.cell_state[index_2d(x, y - 1, static_cast<int>(cloud.size_x))];
      const CellState up = cloud.cell_state[index_2d(x, y + 1, static_cast<int>(cloud.size_x))];

      const float lap_density = (left.density_offset - 2.0f * center.density_offset + right.density_offset) * inv_h2 +
                                (down.density_offset - 2.0f * center.density_offset + up.density_offset) * inv_h2;
      const float lap_vel_x = (left.velocity.x - 2.0f * center.velocity.x + right.velocity.x) * inv_h2 +
                              (down.velocity.x - 2.0f * center.velocity.x + up.velocity.x) * inv_h2;
      const float lap_vel_y = (left.velocity.y - 2.0f * center.velocity.y + right.velocity.y) * inv_h2 +
                              (down.velocity.y - 2.0f * center.velocity.y + up.velocity.y) * inv_h2;

      cloud.cell_state_tmp[idx].density_offset =
          clamp_density_offset(center.density_offset + density_diffusivity * dt * lap_density, reference_density);
      cloud.cell_state_tmp[idx].velocity = make_float3(center.velocity.x + kinematic_viscosity * dt * lap_vel_x,
                                                       center.velocity.y + kinematic_viscosity * dt * lap_vel_y,
                                                       0.0f);
    }
  }
}

__global__ void advect_state_kernel(CellCloudView cloud, float dt, float h, float reference_density) {
  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < static_cast<int>(cloud.size_y); y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < static_cast<int>(cloud.size_x);
         x += blockDim.x * gridDim.x) {
      const int idx = index_2d(x, y, static_cast<int>(cloud.size_x));
      const float backtrace_x = static_cast<float>(x) - dt * cloud.cell_state[idx].velocity.x / h;
      const float backtrace_y = static_cast<float>(y) - dt * cloud.cell_state[idx].velocity.y / h;
      cloud.cell_state_tmp[idx].density_offset =
          clamp_density_offset(sample_density_offset(cloud.cell_state,
                                                     backtrace_x,
                                                     backtrace_y,
                                                     static_cast<int>(cloud.size_x),
                                                     static_cast<int>(cloud.size_y)),
                               reference_density);
      cloud.cell_state_tmp[idx].velocity = sample_velocity(
          cloud.cell_state, backtrace_x, backtrace_y, static_cast<int>(cloud.size_x), static_cast<int>(cloud.size_y));
      cloud.cell_state_tmp[idx].velocity.z = 0.0f;
    }
  }
}

__global__ void compute_divergence_kernel(CellCloudView cloud, float h) {
  const float inv_2h = 0.5f / h;

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < static_cast<int>(cloud.size_y); y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < static_cast<int>(cloud.size_x);
         x += blockDim.x * gridDim.x) {
      const int idx = index_2d(x, y, static_cast<int>(cloud.size_x));
      if (x == 0 || y == 0 || x == static_cast<int>(cloud.size_x) - 1 || y == static_cast<int>(cloud.size_y) - 1) {
        cloud.divergence[idx] = 0.0f;
        continue;
      }

      const float dudx = (cloud.cell_state[index_2d(x + 1, y, static_cast<int>(cloud.size_x))].velocity.x -
                          cloud.cell_state[index_2d(x - 1, y, static_cast<int>(cloud.size_x))].velocity.x) *
                         inv_2h;
      const float dvdy = (cloud.cell_state[index_2d(x, y + 1, static_cast<int>(cloud.size_x))].velocity.y -
                          cloud.cell_state[index_2d(x, y - 1, static_cast<int>(cloud.size_x))].velocity.y) *
                         inv_2h;
      cloud.divergence[idx] = dudx + dvdy;
    }
  }
}

__global__ void pressure_jacobi_kernel(CellCloudView cloud, float h) {
  const float h2 = h * h;
  const float denominator = 4.0f * h2;

  for (uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; y < cloud.size_y; y += blockDim.y * gridDim.y) {
    for (uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; x < cloud.size_x; x += blockDim.x * gridDim.x) {
      const int idx = index_2d(x, y, static_cast<int>(cloud.size_x));
      if (x == 0 || y == 0 || x == static_cast<int>(cloud.size_x) - 1 || y == static_cast<int>(cloud.size_y) - 1) {
        cloud.pressure_tmp[idx] = 0.0f;
        continue;
      }

      const float px = cloud.pressure[index_2d(x - 1, y, static_cast<int>(cloud.size_x))] +
                       cloud.pressure[index_2d(x + 1, y, static_cast<int>(cloud.size_x))];
      const float py = cloud.pressure[index_2d(x, y - 1, static_cast<int>(cloud.size_x))] +
                       cloud.pressure[index_2d(x, y + 1, static_cast<int>(cloud.size_x))];
      cloud.pressure_tmp[idx] = (px + py - cloud.divergence[idx] * h2) / denominator;
    }
  }
}

__global__ void project_velocity_kernel(CellCloudView cloud, float h) {
  const float inv_2h = 0.5f / h;

  for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < static_cast<int>(cloud.size_y); y += blockDim.y * gridDim.y) {
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < static_cast<int>(cloud.size_x);
         x += blockDim.x * gridDim.x) {
      const int idx = index_2d(x, y, static_cast<int>(cloud.size_x));
      if (x == 0 || y == 0 || x == static_cast<int>(cloud.size_x) - 1 || y == static_cast<int>(cloud.size_y) - 1) {
        cloud.cell_state[idx].velocity = make_float3(0.0f, 0.0f, 0.0f);
        continue;
      }

      const float grad_x = (cloud.pressure[index_2d(x + 1, y, static_cast<int>(cloud.size_x))] -
                            cloud.pressure[index_2d(x - 1, y, static_cast<int>(cloud.size_x))]) *
                           inv_2h;
      const float grad_y = (cloud.pressure[index_2d(x, y + 1, static_cast<int>(cloud.size_x))] -
                            cloud.pressure[index_2d(x, y - 1, static_cast<int>(cloud.size_x))]) *
                           inv_2h;
      cloud.cell_state[idx].velocity.x -= grad_x;
      cloud.cell_state[idx].velocity.y -= grad_y;
      cloud.cell_state[idx].velocity.z = 0.0f;
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

  cloud_.h = initial_state.grid.h;
  cloud_.kin_visc = initial_state.material.kinematic_viscosity;
  cloud_.dty_visc = initial_state.material.density_diffusivity;
  cloud_.ref_dty = initial_state.material.reference_density;

  if (initial_state.grid.frame.has_value()) {
    const io::Frame& frame = initial_state.grid.frame->data;
    if (frame.density_offset.size() != cloud_.size() || frame.velocity.size() != cloud_.size() * 3U) {
      throw std::runtime_error("Initial state size does not match simulation dimensions.");
    }
    std::vector<CellState> device_state(cloud_.size());
    for (std::size_t i = 0; i < cloud_.size(); ++i) {
      device_state[i].density_offset = frame.density_offset[i];
      device_state[i].velocity =
          make_float3(frame.velocity[i * 3U], frame.velocity[i * 3U + 1U], frame.velocity[i * 3U + 2U]);
    }
    cloud_.cell_state.assign(device_state);
  }

  cloud_.cell_state_tmp.fill(CellState{});
}

double Simulation::compute_time_step() const {
  thrust::device_ptr<const CellState> begin(cloud_.cell_state.data());
  thrust::device_ptr<const CellState> end = begin + cloud_.cell_state.size();
  const float max_speed =
      thrust::transform_reduce(thrust::device, begin, end, MaxAbsVelocity{}, 0.0f, thrust::maximum<float>{});

  const double characteristic_speed = std::max(static_cast<double>(max_speed), 1.0);
  return settings_.cfl * cloud_.h / characteristic_speed;
}

void Simulation::step(double max_dt) {
  const double current_dt = std::min(compute_time_step(), max_dt);
  const float dt = static_cast<float>(current_dt);
  const float current_time = static_cast<float>(time_);

  const LaunchConfig launch = make_launch_config();
  CellCloudView cloud = cloud_.view();

  add_source_kernel<<<launch.grid, launch.block>>>(cloud, dt, current_time, cloud_.h, cloud_.ref_dty);

  diffuse_state_kernel<<<launch.grid, launch.block>>>(
      cloud, dt, cloud.h, cloud.kin_visc, cloud.dty_visc, cloud.ref_dty);
  std::swap(cloud_.cell_state, cloud_.cell_state_tmp);
  cloud = cloud_.view();

  advect_state_kernel<<<launch.grid, launch.block>>>(cloud, dt, cloud.h, cloud.ref_dty);
  std::swap(cloud_.cell_state, cloud_.cell_state_tmp);
  cloud = cloud_.view();

  apply_boundary_kernel<<<launch.grid, launch.block>>>(cloud);

  compute_divergence_kernel<<<launch.grid, launch.block>>>(cloud, cloud.h);

  clear_scalar_kernel<<<launch.grid, launch.block>>>(cloud, cloud.pressure);
  clear_scalar_kernel<<<launch.grid, launch.block>>>(cloud, cloud.pressure_tmp);

  for (int iter = 0; iter < settings_.pressure_iterations; ++iter) {
    pressure_jacobi_kernel<<<launch.grid, launch.block>>>(cloud, cloud.h);
    std::swap(cloud_.pressure, cloud_.pressure_tmp);
    cloud = cloud_.view();
  }
  project_velocity_kernel<<<launch.grid, launch.block>>>(cloud, cloud.h);
  apply_boundary_kernel<<<launch.grid, launch.block>>>(cloud);

  last_dt_ = current_dt;
  time_ += current_dt;
}

io::Frame Simulation::download_frame() const {
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  io::Frame frame;
  const auto device_state = ToStdVector(cloud_.cell_state);
  frame.density_offset.resize(device_state.size());
  frame.velocity.resize(device_state.size() * 3U);

  for (size_t i = 0; i < device_state.size(); ++i) {
    frame.density_offset[i] = device_state[i].density_offset;
    frame.velocity[i * 3U] = device_state[i].velocity.x;
    frame.velocity[i * 3U + 1U] = device_state[i].velocity.y;
    frame.velocity[i * 3U + 2U] = device_state[i].velocity.z;
  }

  return frame;
}

} // namespace fluid_sim
