#pragma once

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

namespace fluid_sim {

class ConsoleLog {
public:
  using Clock = std::chrono::steady_clock;

  explicit ConsoleLog(Clock::time_point last_log_time, std::ostream& stream);
  ~ConsoleLog();

  void print_progress(const double simulation_time,
                      const double end_time,
                      const size_t last_output,
                      const bool force = false);
  void finish_line();

private:
  [[nodiscard]] std::string
  format_progress(const double simulation_time, const double end_time, const size_t last_output) const;

  Clock::time_point last_log_time_;
  std::ostream* stream_;
  std::size_t last_line_width_ = 0;
  bool has_active_line_ = false;
};

inline ConsoleLog::ConsoleLog(Clock::time_point last_log_time, std::ostream& stream)
    : last_log_time_(last_log_time),
      stream_(&stream) {
  (*stream_) << "\n";
  (*stream_) << "        Simulation Status:\n";
  (*stream_) << "\n";
}

inline ConsoleLog::~ConsoleLog() {
  finish_line();
}

inline void ConsoleLog::print_progress(const double simulation_time,
                                       const double end_time,
                                       const size_t last_output,
                                       const bool force) {
  const Clock::time_point now = Clock::now();
  if (now - last_log_time_ < std::chrono::seconds(1) && !force) {
    return;
  }

  last_log_time_ = now;
  const std::string line = format_progress(simulation_time, end_time, last_output);

  (*stream_) << '\r' << line;
  if (line.size() < last_line_width_) {
    (*stream_) << std::string(last_line_width_ - line.size(), ' ');
  }
  stream_->flush();

  last_line_width_ = line.size();
  has_active_line_ = true;
}

inline void ConsoleLog::finish_line() {
  if (!has_active_line_) {
    return;
  }

  (*stream_) << '\n';
  stream_->flush();
  has_active_line_ = false;
  last_line_width_ = 0;
}

inline std::string ConsoleLog::format_progress(double simulation_time, double end_time, std::size_t last_output) const {
  const double progress = end_time > 0.0 ? (100.0 * simulation_time / end_time) : 0.0;

  std::ostringstream line;
  line << std::fixed << std::setprecision(3) << "time " << std::setw(8) << simulation_time << " / " << std::setw(8)
       << end_time << " s"
       << " | " << std::setprecision(1) << std::setw(5) << progress << '%' << " | last output #" << last_output;
  return line.str();
}

} // namespace fluid_sim
