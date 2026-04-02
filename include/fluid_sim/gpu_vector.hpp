#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fluid_sim {

inline void ThrowIfCudaError(cudaError_t error, const char* context) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(error));
  }
}

template <typename T>
class GPUVector {
 public:
  GPUVector() = default;
  explicit GPUVector(std::size_t size) {
    resize(size);
  }

  GPUVector(const GPUVector&) = delete;
  GPUVector& operator=(const GPUVector&) = delete;

  GPUVector(GPUVector&& other) noexcept {
    swap(other);
  }

  GPUVector& operator=(GPUVector&& other) noexcept {
    if (this != &other) {
      clear();
      swap(other);
    }
    return *this;
  }

  ~GPUVector() {
    clear();
  }

  void resize(std::size_t new_size) {
    if (new_size == size_) {
      return;
    }

    clear();
    if (new_size == 0) {
      return;
    }

    ThrowIfCudaError(cudaMalloc(&data_, new_size * sizeof(T)), "cudaMalloc failed in GPUVector::resize");
    size_ = new_size;
  }

  void fill(const T& value) {
    if (size_ == 0) {
      return;
    }

    if (IsAllZero(value)) {
      ThrowIfCudaError(cudaMemset(data_, 0, size_ * sizeof(T)), "cudaMemset failed in GPUVector::fill");
      return;
    }

    std::vector<T> host(size_, value);
    assign(host);
  }

  void assign(const std::vector<T>& host) {
    if (host.size() != size_) {
      resize(host.size());
    }
    if (size_ == 0) {
      return;
    }

    ThrowIfCudaError(cudaMemcpy(data_, host.data(), size_ * sizeof(T), cudaMemcpyHostToDevice),
                     "cudaMemcpy host-to-device failed in GPUVector::assign");
  }

  std::vector<T> to_std_vector() const {
    std::vector<T> host(size_);
    if (size_ == 0) {
      return host;
    }

    ThrowIfCudaError(cudaMemcpy(host.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost),
                     "cudaMemcpy device-to-host failed in GPUVector::to_std_vector");
    return host;
  }

  void clear() {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
    }
    size_ = 0;
  }

  [[nodiscard]] T* data() { return data_; }
  [[nodiscard]] const T* data() const { return data_; }
  [[nodiscard]] std::size_t size() const { return size_; }

 private:
  static bool IsAllZero(const T& value) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      if (bytes[i] != 0U) {
        return false;
      }
    }
    return true;
  }

  void swap(GPUVector& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

template <typename T>
GPUVector<T> ToGPUVector(const std::vector<T>& host) {
  GPUVector<T> device(host.size());
  device.assign(host);
  return device;
}

template <typename T>
std::vector<T> ToStdVector(const GPUVector<T>& device) {
  return device.to_std_vector();
}

}  // namespace fluid_sim
