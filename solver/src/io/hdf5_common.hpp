#pragma once

#include <hdf5.h>

#include <filesystem>

namespace io::hdf5 {

void check_status(herr_t status, const char* message);

class Handle {
public:
  using CloseFunction = herr_t (*)(hid_t);

  Handle() = default;
  Handle(hid_t id, CloseFunction close_function);
  ~Handle();

  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept;
  Handle& operator=(Handle&& other) noexcept;

  [[nodiscard]] hid_t get() const;

private:
  hid_t id_ = -1;
  CloseFunction close_function_ = nullptr;
};

} // namespace io::hdf5
