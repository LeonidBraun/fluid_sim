#include "io/hdf5_common.hpp"

#include <stdexcept>

namespace io::hdf5 {

void check_status(herr_t status, const char* message) {
  if (status < 0) {
    throw std::runtime_error(message);
  }
}

Handle::Handle(hid_t id, CloseFunction close_function)
    : id_(id),
      close_function_(close_function) {}

Handle::~Handle() {
  if (id_ >= 0 && close_function_ != nullptr) {
    close_function_(id_);
  }
}

Handle::Handle(Handle&& other) noexcept
    : id_(other.id_),
      close_function_(other.close_function_) {
  other.id_ = -1;
  other.close_function_ = nullptr;
}

Handle& Handle::operator=(Handle&& other) noexcept {
  if (this != &other) {
    if (id_ >= 0 && close_function_ != nullptr) {
      close_function_(id_);
    }
    id_ = other.id_;
    close_function_ = other.close_function_;
    other.id_ = -1;
    other.close_function_ = nullptr;
  }
  return *this;
}

hid_t Handle::get() const {
  return id_;
}

} // namespace io::hdf5
