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
    return fmaxf(fabsf(cell.momentum.x), fabsf(cell.momentum.y));
  }
};

template <typename T>
__device__ T index_2d(T x, T y, T nx) {
  return y * nx + x;
}
template __device__ int index_2d<int>(int x, int y, int nx);
template __device__ uint32_t index_2d<uint32_t>(uint32_t x, uint32_t y, uint32_t nx);

__global__ void RHS(CellCloudView cloud, float t) {
  for (uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; y < cloud.size_y; y += blockDim.y * gridDim.y) {
    for (uint32_t x = blockIdx.x * blockDim.x + threadIdx.x; x < cloud.size_x; x += blockDim.x * gridDim.x) {
      const uint32_t c = index_2d(x, y, cloud.size_x);
      cloud.cell_state[c].density_offset = cosf(t);

      if ((x * x + y * y) * cloud.h * cloud.h < 0.5)
        cloud.cell_state[c].momentum.x = 1.0 * (cloud.cell_state[c].density_offset + cloud.ref_dty);
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
    if (frame.density_offset.size() != cloud_.size() || frame.momentum.size() != cloud_.size() * 3U) {
      throw std::runtime_error("Initial state size does not match simulation dimensions.");
    }
    std::vector<CellState> device_state(cloud_.size());
    for (std::size_t i = 0; i < cloud_.size(); ++i) {
      device_state[i].density_offset = frame.density_offset[i];
      device_state[i].momentum =
          make_float3(frame.momentum[i * 3U], frame.momentum[i * 3U + 1U], frame.momentum[i * 3U + 2U]);
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
  const float max_speed =
      thrust::transform_reduce(thrust::device, begin, end, MaxAbsVelocity{}, 0.0f, thrust::maximum<float>{});

  return settings_.cfl * cloud_.h / max_speed;
}

void Simulation::step(double max_dt) {
  const double current_dt = std::min(compute_time_step(), max_dt);
  const float dt = static_cast<float>(current_dt);
  const float current_time = static_cast<float>(time_);

  CellCloudView cloud = cloud_.view();

  RHS<<<launch.grid, launch.block>>>(cloud, static_cast<float>(time_));

  // CUDA_CHECK(cudaGetLastError());
  // CUDA_CHECK(cudaDeviceSynchronize());

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
    frame.momentum[i * 3U] = device_state[i].momentum.x;
    frame.momentum[i * 3U + 1U] = device_state[i].momentum.y;
    frame.momentum[i * 3U + 2U] = device_state[i].momentum.z;
  }

  return frame;
}

} // namespace fluid_sim
