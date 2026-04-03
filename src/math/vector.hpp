#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace math {

template <typename T, std::size_t N>
class Vector {
public:
  static_assert(N > 0, "Vector dimension must be positive.");

  using value_type = T;
  using storage_type = std::array<T, N>;
  using iterator = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;

  constexpr Vector() = default;

  constexpr explicit Vector(const T& value) {
    values_.fill(value);
  }

  template <typename... Args,
            typename = std::enable_if_t<sizeof...(Args) == N && (std::is_convertible_v<Args, T> && ...)>>
  constexpr explicit Vector(Args... args)
      : values_{static_cast<T>(args)...} {}

  constexpr Vector(std::initializer_list<T> values) {
    if (values.size() != N) {
      throw std::runtime_error("Vector initializer size does not match dimension.");
    }
    std::copy(values.begin(), values.end(), values_.begin());
  }

  [[nodiscard]] static constexpr Vector Zero() {
    return Vector{};
  }

  [[nodiscard]] static constexpr Vector Filled(const T& value) {
    return Vector(value);
  }

  [[nodiscard]] constexpr std::size_t size() const {
    return N;
  }

  [[nodiscard]] constexpr T* data() {
    return values_.data();
  }

  [[nodiscard]] constexpr const T* data() const {
    return values_.data();
  }

  [[nodiscard]] constexpr T& operator[](std::size_t index) {
    return values_[index];
  }

  [[nodiscard]] constexpr const T& operator[](std::size_t index) const {
    return values_[index];
  }

  [[nodiscard]] constexpr T& at(std::size_t index) {
    return values_.at(index);
  }

  [[nodiscard]] constexpr const T& at(std::size_t index) const {
    return values_.at(index);
  }

  [[nodiscard]] constexpr iterator begin() {
    return values_.begin();
  }

  [[nodiscard]] constexpr const_iterator begin() const {
    return values_.begin();
  }

  [[nodiscard]] constexpr const_iterator cbegin() const {
    return values_.cbegin();
  }

  [[nodiscard]] constexpr iterator end() {
    return values_.end();
  }

  [[nodiscard]] constexpr const_iterator end() const {
    return values_.end();
  }

  [[nodiscard]] constexpr const_iterator cend() const {
    return values_.cend();
  }

  constexpr Vector& operator+=(const Vector& other) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] += other[i];
    }
    return *this;
  }

  constexpr Vector& operator-=(const Vector& other) {
    for (std::size_t i = 0; i < N; ++i) {
      values_[i] -= other[i];
    }
    return *this;
  }

  constexpr Vector& operator*=(const T& scalar) {
    for (auto& value : values_) {
      value *= scalar;
    }
    return *this;
  }

  constexpr Vector& operator/=(const T& scalar) {
    for (auto& value : values_) {
      value /= scalar;
    }
    return *this;
  }

  constexpr Vector& operator+=(const T& scalar) {
    for (auto& value : values_) {
      value += scalar;
    }
    return *this;
  }

  constexpr Vector& operator-=(const T& scalar) {
    for (auto& value : values_) {
      value -= scalar;
    }
    return *this;
  }

  [[nodiscard]] constexpr Vector operator+() const {
    return *this;
  }

  [[nodiscard]] constexpr Vector operator-() const {
    Vector result;
    for (std::size_t i = 0; i < N; ++i) {
      result[i] = -values_[i];
    }
    return result;
  }

  [[nodiscard]] constexpr bool operator==(const Vector& other) const {
    return values_ == other.values_;
  }

  [[nodiscard]] constexpr bool operator!=(const Vector& other) const {
    return !(*this == other);
  }

private:
  storage_type values_{};
};

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator+(Vector<T, N> left, const Vector<T, N>& right) {
  left += right;
  return left;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator-(Vector<T, N> left, const Vector<T, N>& right) {
  left -= right;
  return left;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator+(Vector<T, N> vector, const T& scalar) {
  vector += scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator+(const T& scalar, Vector<T, N> vector) {
  vector += scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator-(Vector<T, N> vector, const T& scalar) {
  vector -= scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator-(const T& scalar, const Vector<T, N>& vector) {
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = scalar - vector[i];
  }
  return result;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator*(Vector<T, N> vector, const T& scalar) {
  vector *= scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator*(const T& scalar, Vector<T, N> vector) {
  vector *= scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> operator/(Vector<T, N> vector, const T& scalar) {
  vector /= scalar;
  return vector;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr T dot(const Vector<T, N>& left, const Vector<T, N>& right) {
  T result{};
  for (std::size_t i = 0; i < N; ++i) {
    result += left[i] * right[i];
  }
  return result;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr T inner_product(const Vector<T, N>& left, const Vector<T, N>& right) {
  return dot(left, right);
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr Vector<T, N> hadamard_product(const Vector<T, N>& left, const Vector<T, N>& right) {
  Vector<T, N> result;
  for (std::size_t i = 0; i < N; ++i) {
    result[i] = left[i] * right[i];
  }
  return result;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr T norm_squared(const Vector<T, N>& vector) {
  return dot(vector, vector);
}

template <typename T, std::size_t N>
[[nodiscard]] auto norm(const Vector<T, N>& vector) {
  using result_type = decltype(std::sqrt(norm_squared(vector)));
  return static_cast<result_type>(std::sqrt(static_cast<result_type>(norm_squared(vector))));
}

template <typename T, std::size_t N>
[[nodiscard]] auto length(const Vector<T, N>& vector) {
  return norm(vector);
}

template <typename T, std::size_t N>
[[nodiscard]] Vector<T, N> normalized(const Vector<T, N>& vector) {
  const auto magnitude = norm(vector);
  if (magnitude == 0) {
    throw std::runtime_error("Cannot normalize a zero-length vector.");
  }
  return vector / static_cast<T>(magnitude);
}

template <typename T>
[[nodiscard]] constexpr T cross(const Vector<T, 2>& left, const Vector<T, 2>& right) {
  return left[0] * right[1] - left[1] * right[0];
}

template <typename T>
[[nodiscard]] constexpr Vector<T, 3> cross(const Vector<T, 3>& left, const Vector<T, 3>& right) {
  return Vector<T, 3>(left[1] * right[2] - left[2] * right[1],
                      left[2] * right[0] - left[0] * right[2],
                      left[0] * right[1] - left[1] * right[0]);
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr T sum(const Vector<T, N>& vector) {
  T result{};
  for (const T& value : vector) {
    result += value;
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

} // namespace math
