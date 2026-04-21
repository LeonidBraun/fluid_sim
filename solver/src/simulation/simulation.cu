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
    const float rho_floor = 1.0e-4f * ref_dty;
    const float rho_total = max(rho_floor, ref_dty + cell.density_offset);
    const float momentum_l1 = fabsf(cell.momentum[0]) + fabsf(cell.momentum[1]) + fabsf(cell.momentum[2]);
    return momentum_l1 / rho_total;
  }
  const float ref_dty;
};

template <typename T>
__device__ T index_3d(T x, T y, T z, T ny, T nz) {
  return (x * ny + y) * nz + z;
}
template __device__ int index_3d<int>(int x, int y, int z, int ny, int nz);
template __device__ uint32_t index_3d<uint32_t>(uint32_t x, uint32_t y, uint32_t z, uint32_t ny, uint32_t nz);

__device__ float scale(const float dty_offset) {
  const float scale = dty_offset < 0 ? 0.0f : 1.0f;
  return scale;
}
__device__ float prs(const float dty_offset, const float sos) {
  // return sos * sos * dty_offset;
  // return sos * sos * max(0.f, dty_offset);
  return sos * sos * scale(dty_offset) * dty_offset;
}
__device__ float comp_energy(const float dty_offset, const float sos, const float ref_dty) {
  return sos * sos * scale(dty_offset) *
         (log1p(dty_offset / ref_dty) - dty_offset / max(1e-5f * ref_dty, dty_offset + ref_dty));
}
__device__ float dq(const float dty_offset, const float sos, const float ref_dty) {
  // return sos * sos * log1p(rel_dty_offset);
  // return sos * sos * log1p(max(0.01f * rel_dty_offset, rel_dty_offset));
  return comp_energy(dty_offset, sos, ref_dty) + prs(dty_offset, sos) / max(1e-5f * ref_dty, dty_offset + ref_dty);
}

__device__ float clamp01(const float value) {
  return fminf(1.0f, fmaxf(0.0f, value));
}

__device__ float safe_density_total(const float density_offset, const float ref_dty, const float rho_floor) {
  return max(rho_floor, ref_dty + density_offset);
}

__device__ float stage_dt_floor(const CellCloudView& cloud, const float dt) {
  return max(dt, 1.0e-6f * cloud.h / max(cloud.sos, 1.0e-6f));
}

struct FluxBudget {
  float min_mass_flux;
  float max_mass_flux;
  float target_mass_flux;
};

__device__ FluxBudget mass_flux_budget(const float centered_mass_flux,
                                       const float rho_tot_l,
                                       const float rho_tot_r,
                                       const float rho_floor,
                                       const CellCloudView& cloud,
                                       const float dt) {
  constexpr float positivity_safety = 0.9f;
  constexpr float faces_per_cell = 6.0f;
  const float stage_dt = stage_dt_floor(cloud, dt);
  const float left_budget =
      positivity_safety * max(0.0f, rho_tot_l - rho_floor) * cloud.h / (faces_per_cell * stage_dt);
  const float right_budget =
      positivity_safety * max(0.0f, rho_tot_r - rho_floor) * cloud.h / (faces_per_cell * stage_dt);
  const float min_mass_flux = -right_budget;
  const float max_mass_flux = left_budget;
  const float target_mass_flux =
      min_mass_flux <= max_mass_flux ? fminf(max_mass_flux, fmaxf(min_mass_flux, centered_mass_flux)) : 0.0f;
  return FluxBudget{min_mass_flux, max_mass_flux, target_mass_flux};
}

struct EntropyData {
  float residual;
  float momentum_damping;
};

__device__ EntropyData entropy_data(const CellState& flux,
                                    const float rho_l,
                                    const float rho_r,
                                    const V3& u_l,
                                    const V3& u_r,
                                    const float u_n_l,
                                    const float u_n_r,
                                    const V3& d_m_mom,
                                    const CellCloudView& cloud) {
  const float deta_1 = dq(rho_r, cloud.sos, cloud.ref_dty) - dq(rho_l, cloud.sos, cloud.ref_dty) -
                       0.5f * (dot(u_r, u_r) - dot(u_l, u_l));
  const V3 deta_2 = u_r - u_l;
  const float deta_F = prs(rho_r, cloud.sos) * u_n_r - prs(rho_l, cloud.sos) * u_n_l;
  return EntropyData{
      flux.density_offset * deta_1 + dot(flux.momentum, deta_2) - deta_F,
      dot(d_m_mom, deta_2),
  };
}

__device__ float
near_vacuum_blend(const float rho_tot_l, const float rho_tot_r, const float rho_floor, const float ref_dty) {
  const float rho_min = min(rho_tot_l, rho_tot_r);
  return clamp01((5.0e-2f * ref_dty - rho_min) / (5.0e-2f * ref_dty - rho_floor));
}

__device__ void sanitize_budgeted_flux(CellState& flux,
                                       const float min_mass_flux,
                                       const float max_mass_flux,
                                       const float near_vacuum,
                                       const float wave_speed,
                                       const float rho_tot_l,
                                       const float rho_tot_r) {
  flux.density_offset = fminf(max_mass_flux, fmaxf(min_mass_flux, flux.density_offset));

  const float momentum_cap = 2.0f * wave_speed * max(rho_tot_l, rho_tot_r);
  for (int k = 0; k < 3; ++k) {
    if (!isfinite(flux.momentum[k])) {
      flux.momentum[k] = 0.0f;
    } else {
      flux.momentum[k] = fminf(momentum_cap, fmaxf(-momentum_cap, flux.momentum[k]));
    }
  }
  if (!isfinite(flux.density_offset)) {
    flux.density_offset = 0.0f;
  }

  if (near_vacuum > 0.0f) {
    const float vacuum_keep = 1.0f - near_vacuum;
    flux.density_offset *= vacuum_keep;
    flux.momentum *= vacuum_keep;
  }
}

__device__ CellState FakeRiemann(const CellState& ls,        //
                                 const CellState& rs,        //
                                 const V3& normal,           //
                                 const CellCloudView& cloud, //
                                 const float dt) {

  const float rho_l = ls.density_offset;
  const float rho_r = rs.density_offset;
  const float dty_l = max(1e-5f * cloud.ref_dty, rho_l + cloud.ref_dty);
  const float dty_r = max(1e-5f * cloud.ref_dty, rho_r + cloud.ref_dty);
  const V3 u_l = ls.momentum / dty_l;
  const V3 u_r = rs.momentum / dty_r;

  // const float rho = 0.5f * (rho_l + rho_r);
  // const float rho = 0.5f * (rho_l + rho_r);
  // const float dty = rho + cloud.ref_dty;
  const float dty = sqrt(dty_l * dty_r);
  const V3 u = 0.5f * (u_l + u_r);
  const float u_n = dot(u, normal);

  CellState F;
  // F.density_offset = dty * u_n;
  // F.momentum = dty * u_n * u + normal * prs(dty - cloud.ref_dty, cloud.sos);
  // F.density_offset = max(0.0f, dot(ls.momentum, normal));
  // F.density_offset += min(0.0f, dot(rs.momentum, normal));
  // F.density_offset += p;
  // {
  //   const float rho_up = F.density_offset < 0.0f ? rho_l : rho_r;
  //   const float max_rho_flux = min(fabsf(F.density_offset), 0.3f * (rho_up + cloud.ref_dty) * cloud.h / dt);
  //   F.density_offset = F.density_offset > 0.0f ? max_rho_flux : -max_rho_flux;
  // }
  // F.momentum = ls.momentum * max(0.0f, dot(ls.momentum, normal)) / dty_l;
  // F.momentum += rs.momentum * min(0.0f, dot(rs.momentum, normal)) / dty_r;
  // const float p = prs(sqrt(dty_l * dty_r) - ls.density_offset, cloud.sos);
  // F.momentum += normal * p;

  {
    const float deta_1 = dq(rho_r, cloud.sos, cloud.ref_dty) - dq(rho_l, cloud.sos, cloud.ref_dty) -
                         0.5f * (dot(u_r, u_r) - dot(u_l, u_l)); // - cloud.h * dot(cloud.gravity, normal);
    const V3 deta_2 = u_r - u_l;
    const float deta_F = prs(rho_r, cloud.sos) * dot(u_r, normal) - prs(rho_l, cloud.sos) * dot(u_l, normal);

    const float phi = max(.0f, F.density_offset * deta_1 + dot(F.momentum, deta_2) - deta_F);
    // const float phi = (F.density_offset * deta_1 + dot(F.momentum, deta_2) - deta_F);
    if (phi != 0.f) {
      // const float deta_sq = deta_1 + dot(deta_2, u);
      const float deta_sq = deta_1 * deta_1 + dot(deta_2, deta_2);
      if (deta_sq != 0.f) {
        constexpr float overdamp = 2.f;
        // F.density_offset -= overdamp * phi / deta_sq;
        // F.momentum -= u * (overdamp * dty * phi / deta_sq);
        F.density_offset -= overdamp * deta_1 * phi / deta_sq;
        F.momentum -= deta_2 * (overdamp * phi / deta_sq);
      }
    }
  }
  // const float rho_min = min(rho_l, rho_r);
  // if (rho_min) {
  // const float a = max(fabsf(dot(u_l, normal)) + cloud.sos, fabsf(dot(u_r, normal)) + cloud.sos);
  // F.density_offset += 0.25f * a * (rho_l - rho_r);
  // F.momentum += 0.25f * a * (ls.momentum - rs.momentum);
  // }
  // F.momentum += normal * (0.5f * cloud.dty_visc * cloud.sos * dot(normal, (ls.momentum - rs.momentum)));
  // F.momentum += normal * (0.5f * 0.1f * cloud.sos * dot(normal, (ls.momentum - rs.momentum)));
  // F.density_offset += cloud.dty_visc / cloud.sos * (prs(rho_l, cloud.sos) - prs(rho_r, cloud.sos));
  F.density_offset += cloud.dty_visc * cloud.sos * (rho_l - rho_r);

  F.momentum += (cloud.kin_visc * min(dty_l, dty_r) / cloud.h) * (u_l - u_r);
  return F;
}

__device__ CellState ComputeFlux(const CellState& ls,        //
                                 const CellState& rs,        //
                                 const V3& normal,           //
                                 const CellCloudView& cloud, //
                                 const float dt) {
  const float rho_l = ls.density_offset;
  const float rho_r = rs.density_offset;
  const float rho_tot_l = max(1e-5f * cloud.ref_dty, cloud.ref_dty + rho_l);
  const float rho_tot_r = max(1e-5f * cloud.ref_dty, cloud.ref_dty + rho_r);
  const V3 u_l = ls.momentum / rho_tot_l;
  const V3 u_r = rs.momentum / rho_tot_r;
  const float u_n_l = dot(u_l, normal);
  const float u_n_r = dot(u_r, normal);
  const float wave_speed = max(fabsf(u_n_l) + cloud.sos, fabsf(u_n_r) + cloud.sos);

  CellState low;
  low.density_offset = 0.5f * dot(ls.momentum + rs.momentum, normal) - 0.5f * wave_speed * (rho_r - rho_l);
  low.momentum = 0.5f * ls.momentum * u_n_l + 0.5f * rs.momentum * u_n_r;
  low.momentum += 0.5f * normal * (prs(rho_l, cloud.sos) + prs(rho_r, cloud.sos));
  low.momentum -= 0.5f * wave_speed * (rs.momentum - ls.momentum);

  CellState high = FakeRiemann(ls, rs, normal, cloud, dt);

  const float rho_floor = 1.0e-4f * cloud.ref_dty;
  const float rho_blend = 5.0e-2f * cloud.ref_dty;
  const float rho_min = min(rho_tot_l, rho_tot_r);
  const float high_weight = clamp01((rho_min - rho_floor) / max(rho_blend - rho_floor, 1.0e-6f * cloud.ref_dty));

  CellState F;
  F.density_offset = low.density_offset + high_weight * (high.density_offset - low.density_offset);
  F.momentum = low.momentum + high_weight * (high.momentum - low.momentum);

  const float deta_1 = dq(rho_r, cloud.sos, cloud.ref_dty) - dq(rho_l, cloud.sos, cloud.ref_dty) -
                       0.5f * (dot(u_r, u_r) - dot(u_l, u_l));
  const V3 deta_2 = u_r - u_l;
  const float deta_F = prs(rho_r, cloud.sos) * u_n_r - prs(rho_l, cloud.sos) * u_n_l;
  const float phi = max(0.0f, F.density_offset * deta_1 + dot(F.momentum, deta_2) - deta_F);
  if (phi > 0.0f) {
    const float deta_sq = deta_1 * deta_1 + dot(deta_2, deta_2);
    if (deta_sq > 0.0f) {
      constexpr float overdamp = 1.0f;
      F.density_offset -= overdamp * deta_1 * phi / deta_sq;
      F.momentum -= deta_2 * (overdamp * phi / deta_sq);
    }
  }

  const float stage_dt = max(dt, 1.0e-6f * cloud.h / max(cloud.sos, 1.0e-6f));
  constexpr float faces_per_cell = 6.0f;
  constexpr float outgoing_safety = 0.9f;
  const float left_budget = outgoing_safety * rho_tot_l * cloud.h / (faces_per_cell * stage_dt);
  const float right_budget = outgoing_safety * rho_tot_r * cloud.h / (faces_per_cell * stage_dt);
  const float unclamped_mass_flux = F.density_offset;
  if (unclamped_mass_flux > 0.0f) {
    F.density_offset = fminf(unclamped_mass_flux, left_budget);
  } else {
    F.density_offset = fmaxf(unclamped_mass_flux, -right_budget);
  }

  if (fabsf(unclamped_mass_flux) > 0.0f) {
    const float flux_scale = fabsf(F.density_offset) / fabsf(unclamped_mass_flux);
    F.momentum = low.momentum + flux_scale * (F.momentum - low.momentum);
  }

  return F;
}

__device__ CellState BudgetedRoeFlux(const CellState& ls,        //
                                     const CellState& rs,        //
                                     const V3& normal,           //
                                     const CellCloudView& cloud, //
                                     const float dt) {
  const float rho_floor = 1.0e-4f * cloud.ref_dty;
  const float rho_total_l = safe_density_total(ls.density_offset, cloud.ref_dty, rho_floor);
  const float rho_total_r = safe_density_total(rs.density_offset, cloud.ref_dty, rho_floor);
  const float rho_offset_l = rho_total_l - cloud.ref_dty;
  const float rho_offset_r = rho_total_r - cloud.ref_dty;
  const V3 vel_l = ls.momentum / rho_total_l;
  const V3 vel_r = rs.momentum / rho_total_r;
  const float vel_n_l = dot(vel_l, normal);
  const float vel_n_r = dot(vel_r, normal);

  const float sqrt_rho_l = sqrtf(rho_total_l);
  const float sqrt_rho_r = sqrtf(rho_total_r);
  const float inv_sqrt_sum = 1.0f / (sqrt_rho_l + sqrt_rho_r);
  const V3 vel_roe = (sqrt_rho_l * vel_l + sqrt_rho_r * vel_r) * inv_sqrt_sum;
  const float vel_n_roe = dot(vel_roe, normal);
  const float wave_speed = max(fabsf(vel_n_l), fabsf(vel_n_r)) + cloud.sos;

  const V3 momentum_advective = 0.5f * ls.momentum * vel_n_l + 0.5f * rs.momentum * vel_n_r;
  const V3 momentum_pressure = 0.5f * normal * (prs(rho_offset_l, cloud.sos) + prs(rho_offset_r, cloud.sos));

  CellState F;
  F.density_offset = 0.5f * dot(ls.momentum + rs.momentum, normal);
  F.momentum = momentum_advective + momentum_pressure;

  const float drho = rho_offset_r - rho_offset_l;
  const V3 dm = rs.momentum - ls.momentum;
  const float d_rho_mass = 0.5f * wave_speed * drho;
  const V3 d_rho_mom = V3(0.0f);
  const float d_m_mass = 0.0f;
  const V3 d_m_mom = 0.5f * wave_speed * dm;

  const float centered_mass_flux = F.density_offset;
  const FluxBudget budget = mass_flux_budget(centered_mass_flux, rho_total_l, rho_total_r, rho_floor, cloud, dt);

  float theta_rho = 0.0f;
  if (fabsf(d_rho_mass) > 1.0e-12f) {
    theta_rho = clamp01((centered_mass_flux - budget.target_mass_flux) / d_rho_mass);
  }

  const float mass_flux_scale =
      fabsf(centered_mass_flux) > 1.0e-12f ? fabsf(budget.target_mass_flux) / fabsf(centered_mass_flux) : 0.0f;
  F.density_offset -= theta_rho * d_rho_mass;
  F.momentum -= theta_rho * d_rho_mom;
  F.density_offset = budget.target_mass_flux;
  F.momentum = clamp01(mass_flux_scale) * momentum_advective + momentum_pressure;

  const EntropyData entropy =
      entropy_data(F, rho_offset_l, rho_offset_r, vel_l, vel_r, vel_n_l, vel_n_r, d_m_mom, cloud);

  float theta_m = 0.0f;
  if (entropy.residual > 0.0f && entropy.momentum_damping > 0.0f) {
    theta_m = clamp01(entropy.residual / entropy.momentum_damping);
  }

  F.density_offset -= theta_m * d_m_mass;
  F.momentum -= theta_m * d_m_mom;

  if (entropy.residual > 0.0f && entropy.momentum_damping <= 0.0f) {
    F.density_offset -= 0.5f * wave_speed * drho;
    F.momentum -= 0.5f * wave_speed * dm;
  }

  const float near_vacuum = near_vacuum_blend(rho_total_l, rho_total_r, rho_floor, cloud.ref_dty);
  const float mass_flux_ratio =
      fabsf(centered_mass_flux) > 1.0e-12f ? fabsf(F.density_offset) / fabsf(centered_mass_flux) : 0.0f;
  const float momentum_keep = fmaxf(mass_flux_ratio, 1.0f - near_vacuum);
  F.momentum *= clamp01(momentum_keep);

  const float roe_visc = 0.25f * wave_speed * fabsf(vel_n_roe - 0.5f * (vel_n_l + vel_n_r));
  F.momentum -= roe_visc * dm;
  F.momentum += (cloud.kin_visc * cloud.ref_dty / cloud.h) * (vel_l - vel_r);
  sanitize_budgeted_flux(
      F, budget.min_mass_flux, budget.max_mass_flux, near_vacuum, wave_speed, rho_total_l, rho_total_r);
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
    // if (out[i].density_offset < 1e-5f - 1.0f)
    //   out[i].momentum *= 0.0f;
  }
}

__global__ void RHS(CellCloudView cloud, [[maybe_unused]] const float t, const float dt) {
  const float inv_h = 1.0f / cloud.h;

  using V3i = Vector<int, 3>;
  constexpr Vector<V3i, 6> dirs{V3i{1, 0, 0}, V3i{-1, 0, 0}, V3i{0, 1, 0}, V3i{0, -1, 0}, V3i{0, 0, 1}, V3i{0, 0, -1}};

  for (uint32_t x = blockIdx.z * blockDim.z + threadIdx.z; x < cloud.size_x; x += blockDim.z * gridDim.z) {
    for (uint32_t y = blockIdx.y * blockDim.y + threadIdx.y; y < cloud.size_y; y += blockDim.y * gridDim.y) {
      for (uint32_t z = blockIdx.x * blockDim.x + threadIdx.x; z < cloud.size_z; z += blockDim.x * gridDim.x) {
        const uint32_t c = index_3d(x, y, z, cloud.size_y, cloud.size_z);
        const CellState cs = cloud.cell_state[c];
        const float rho_c = cloud.ref_dty + cs.density_offset;
        const V3 vel_c = cs.momentum / rho_c;
        CellState tmp{0.f, V3(0.f)};

        for (V3i dir : dirs) {
          CellState ns;
          const uint32_t nx = x + static_cast<uint32_t>(dir[0]);
          const uint32_t ny = y + static_cast<uint32_t>(dir[1]);
          const uint32_t nz = z + static_cast<uint32_t>(dir[2]);
          if (nx < cloud.size_x && ny < cloud.size_y && nz < cloud.size_z) {
            const uint32_t n = index_3d(nx, ny, nz, cloud.size_y, cloud.size_z);
            ns = cloud.cell_state[n];
          } else {
            ns.density_offset = cs.density_offset;
            ns.momentum = -cs.momentum;
          }

          const CellState flux = ComputeFlux(cs, ns, V3(dir), cloud, dt);
          // tmp.momentum +=
          //     inv_h * cloud.kin_visc * cloud.ref_dty * (ns.momentum / (cloud.ref_dty + ns.density_offset) - vel_c);
          tmp.density_offset -= flux.density_offset;
          tmp.momentum -= flux.momentum;
        }

        // const float X = x * cloud.h;
        // const float Y = y * cloud.h;
        // if (((X - 2) * (X - 2) + Y * Y) < 0.25)
        //   tmp.momentum += 0.0001f * cloud.sos / cloud.h * (V3{1.f, 0.f, 0.f} - vel_c);

        tmp.density_offset = inv_h * tmp.density_offset;
        tmp.momentum = inv_h * tmp.momentum;
        tmp.momentum += (cs.density_offset + cloud.ref_dty) * cloud.gravity;
        cloud.cell_state_tmp[c] = tmp;
      }
    }
  }
}

} // namespace

Simulation::Simulation(const io::RunConfig::SolverSettings& settings, const io::State& initial_state)
    : settings_(settings),
      time_(initial_state.time) {
  if (initial_state.grid.nx < 2 || initial_state.grid.ny < 2 || initial_state.grid.nz < 1) {
    throw std::runtime_error("Simulation dimensions must be at least 2x2x1.");
  }

  cloud_.resize(static_cast<uint32_t>(initial_state.grid.nx),
                static_cast<uint32_t>(initial_state.grid.ny),
                static_cast<uint32_t>(initial_state.grid.nz));

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

  const io::Frame& frame = initial_state.grid.frame.data;
  if (frame.density_offset.size() != cloud_.size() || frame.momentum.size() != cloud_.size() * 3U) {
    throw std::runtime_error("Initial state size does not match simulation dimensions.");
  }
  std::vector<CellState> device_state(cloud_.size());
  for (std::size_t i = 0; i < cloud_.size(); ++i) {
    device_state[i].density_offset = frame.density_offset[i];
    device_state[i].momentum = V3(frame.momentum[3 * i], frame.momentum[3 * i + 1], frame.momentum[3 * i + 2]);
  }
  cloud_.cell_state.assign(device_state);

  cloud_.cell_state_tmp.fill(CellState{});

  {
    cudaDeviceProp device_properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));

    const dim3 block(8, 8, 4);
    const int target_blocks = std::max(32, device_properties.multiProcessorCount * 4);
    const int grid_z = 4;
    const int grid_y = 4;
    const int grid_x = std::max(2, (target_blocks + grid_z * grid_y - 1) / (grid_z * grid_y));
    launch = {block, dim3(grid_x, grid_y, grid_z)};
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
                         cloud_.size_z,
                         state,
                         rhs,
                         cloud_.h,
                         cloud_.kin_visc,
                         cloud_.dty_visc,
                         cloud_.ref_dty,
                         cloud_.sos,
                         cloud_.gravity};
  };

  // std::cout << cloud_.kin_visc << "\n";
  RHS<<<launch.grid, launch.block>>>(make_view(base_state.data(), k1.data()), current_time, dt);
  // CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(
      base_state.data(), k1.data(), stage_state.data(), 0.5f * dt, cell_count);
  // CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k2.data()), current_time + 0.5f * dt, dt);
  // CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(
      base_state.data(), k2.data(), stage_state.data(), 0.5f * dt, cell_count);
  // CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k3.data()), current_time + 0.5f * dt, dt);
  // CUDA_CHECK(cudaGetLastError());

  BuildStageState<<<linear_grid, linear_block>>>(base_state.data(), k3.data(), stage_state.data(), dt, cell_count);
  // CUDA_CHECK(cudaGetLastError());
  RHS<<<launch.grid, launch.block>>>(make_view(stage_state.data(), k4.data()), current_time + dt, dt);
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
