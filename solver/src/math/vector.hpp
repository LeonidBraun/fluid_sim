#pragma once

#include <cmath>
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <utility>

#if defined(__CUDACC__) || defined(__HIPCC__)
#ifndef INLINE
#define INLINE __host__ __device__ inline
#endif
#else
#ifndef INLINE
#define INLINE inline
#endif
#endif

namespace fluid_sim {
namespace detail {

INLINE float vector_sqrt(const float value) {
  return sqrtf(value);
}

INLINE double vector_sqrt(const double value) {
  return sqrt(value);
}

template <typename T>
INLINE auto vector_sqrt(const T value) {
  return vector_sqrt(static_cast<double>(value));
}

} // namespace detail

template <typename T, std::size_t N>
class Vector {
public:
  static_assert(N > 0, "Vector dimension must be positive.");

  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  INLINE constexpr Vector() {}

  INLINE constexpr explicit Vector(const T& value) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] = value;
    }
  }

  template <typename... Args,
            typename = std::enable_if_t<sizeof...(Args) == N && (std::is_convertible_v<Args, T> && ...)>>
  INLINE constexpr explicit Vector(Args... args)
      : values_{static_cast<T>(args)...} {}

  [[nodiscard]] INLINE static constexpr Vector Zero() {
    return Vector{};
  }

  [[nodiscard]] INLINE static constexpr Vector Filled(const T& value) {
    return Vector(value);
  }

  [[nodiscard]] INLINE static constexpr std::size_t size() {
    return N;
  }

  [[nodiscard]] INLINE T* data() {
    return values_;
  }

  [[nodiscard]] INLINE const T* data() const {
    return values_;
  }

  [[nodiscard]] INLINE T& operator[](const std::size_t index) {
    return values_[index];
  }

  [[nodiscard]] INLINE const T& operator[](const std::size_t index) const {
    return values_[index];
  }

  [[nodiscard]] INLINE T& at(const std::size_t index) {
    return values_[index];
  }

  [[nodiscard]] INLINE const T& at(const std::size_t index) const {
    return values_[index];
  }

  [[nodiscard]] INLINE iterator begin() {
    return values_;
  }

  [[nodiscard]] INLINE const_iterator begin() const {
    return values_;
  }

  [[nodiscard]] INLINE const_iterator cbegin() const {
    return values_;
  }

  [[nodiscard]] INLINE iterator end() {
    return values_ + N;
  }

  [[nodiscard]] INLINE const_iterator end() const {
    return values_ + N;
  }

  [[nodiscard]] INLINE const_iterator cend() const {
    return values_ + N;
  }

  INLINE Vector& operator+=(const Vector& other) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] += other[i];
    }
    return *this;
  }

  INLINE Vector& operator-=(const Vector& other) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] -= other[i];
    }
    return *this;
  }

  INLINE Vector& operator*=(const T& scalar) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] *= scalar;
    }
    return *this;
  }

  INLINE Vector& operator/=(const T& scalar) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] /= scalar;
    }
    return *this;
  }

  INLINE Vector& operator+=(const T& scalar) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] += scalar;
    }
    return *this;
  }

  INLINE Vector& operator-=(const T& scalar) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] -= scalar;
    }
    return *this;
  }

  [[nodiscard]] INLINE constexpr Vector operator+() const {
    return *this;
  }

  [[nodiscard]] INLINE Vector operator-() const {
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = -values_[i];
    }
    return result;
  }

  [[nodiscard]] INLINE bool operator==(const Vector& other) const {
    for (std::size_t i = 0; i < N; ++i) {
      if (values_[i] != other[i]) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] INLINE bool operator!=(const Vector& other) const {
    return !(*this == other);
  }

private:
  T values_[N] = {};
};

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator+(Vector<T, N> left, const Vector<T, N>& right) {
  left += right;
  return left;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator-(Vector<T, N> left, const Vector<T, N>& right) {
  left -= right;
  return left;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator+(Vector<T, N> vector, const T& scalar) {
  vector += scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator+(const T& scalar, Vector<T, N> vector) {
  vector += scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator-(Vector<T, N> vector, const T& scalar) {
  vector -= scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator-(const T& scalar, const Vector<T, N>& vector) {
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar - vector[i];
  }
  return result;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator*(Vector<T, N> vector, const T& scalar) {
  vector *= scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator*(const T& scalar, Vector<T, N> vector) {
  vector *= scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator/(Vector<T, N> vector, const T& scalar) {
  vector /= scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> operator/(const T& scalar, const Vector<T, N>& vector) {
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar / vector[i];
  }
  return result;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr T dot(const Vector<T, N>& left, const Vector<T, N>& right) {
  T result{};
  for (std::size_t i = 0; i < N; ++i) {
    result += left[i] * right[i];
  }
  return result;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr T inner_product(const Vector<T, N>& left, const Vector<T, N>& right) {
  return dot(left, right);
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr Vector<T, N> hadamard_product(const Vector<T, N>& left, const Vector<T, N>& right) {
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = left[i] * right[i];
  }
  return result;
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr T norm_squared(const Vector<T, N>& vector) {
  return dot(vector, vector);
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE auto norm(const Vector<T, N>& vector) {
  return detail::vector_sqrt(norm_squared(vector));
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE auto length(const Vector<T, N>& vector) {
  return norm(vector);
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE Vector<T, N> normalized(const Vector<T, N>& vector) {
  const auto magnitude = norm(vector);
  if (magnitude == 0) {
    return Vector<T, N>::Zero();
  }
  return vector / static_cast<T>(magnitude);
}

template <typename T>
[[nodiscard]] INLINE constexpr T cross(const Vector<T, 2>& left, const Vector<T, 2>& right) {
  return left[0] * right[1] - left[1] * right[0];
}

template <typename T>
[[nodiscard]] INLINE constexpr Vector<T, 3> cross(const Vector<T, 3>& left, const Vector<T, 3>& right) {
  return Vector<T, 3>(left[1] * right[2] - left[2] * right[1],
                      left[2] * right[0] - left[0] * right[2],
                      left[0] * right[1] - left[1] * right[0]);
}

template <typename T, std::size_t N>
[[nodiscard]] INLINE constexpr T sum(const Vector<T, N>& vector) {
  T result{};
  for (std::size_t i = 0; i < N; ++i) {
    result += vector[i];
  }
  return result;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& stream, const Vector<T, N>& vector) {
  stream << '(';
  for (std::size_t i = 0; i < N; ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << vector[i];
  }
  stream << ')';
  return stream;
}

template <typename T>
using Vec2 = Vector<T, 2>;

template <typename T>
using Vec3 = Vector<T, 3>;

template <typename T>
using Vec4 = Vector<T, 4>;

using V2 = Vector<float, 2>;
using V3 = Vector<float, 3>;
using V4 = Vector<float, 4>;

} // namespace fluid_sim
