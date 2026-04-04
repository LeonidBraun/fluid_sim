#include "io/IO.hpp"
#include "io/console_log.hpp"
#include "simulation/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      fluid_sim::IO::PrintUsage(std::cout);
      return 0;
    }

    if (argc != 2) {
      throw std::runtime_error("Expected exactly one argument: path to config JSON. Use --help for usage.");
    }

    fluid_sim::IO io(std::filesystem::absolute(argv[1]));
    fluid_sim::Simulation simulation(io.settings(), io.initial_state());

    io.save_output(simulation);

    const io::RunConfig::OutputSettings& output_settings = io.output_settings();
    const double epsilon = 1e-12 * std::max(1.0, output_settings.output_interval);
    const auto next_output_boundary = [interval = output_settings.output_interval, epsilon](const double time) {
      const double interval_index = std::floor((time + epsilon) / interval);
      const double boundary = (interval_index + 1.0) * interval;
      if (boundary <= time + epsilon) {
        return boundary + interval;
      }
      return boundary;
    };

    double next_output_time = next_output_boundary(simulation.time());
    fluid_sim::ConsoleLog console_log(std::chrono::steady_clock::now(), std::cout);

    while (simulation.time() + epsilon < output_settings.end_time) {
      const double remaining_to_end = output_settings.end_time - simulation.time();
      const double remaining_to_output = next_output_time - simulation.time();
      const double max_step = std::min(remaining_to_end, remaining_to_output);

      console_log.print_progress(simulation.time(), output_settings.end_time, io.last_output());

      if (max_step <= epsilon) {
        io.save_output(simulation);
        next_output_time = next_output_boundary(simulation.time() + output_settings.output_interval);
        continue;
      }

      simulation.step(max_step);

      if (simulation.time() + epsilon >= next_output_time) {
        io.save_output(simulation);
        next_output_time = next_output_boundary(simulation.time());
      }
    }

    if (std::abs(simulation.time() - io.last_saved_time()) > epsilon) {
      io.save_output(simulation);
    }
    console_log.print_progress(simulation.time(), output_settings.end_time, io.last_output(), true);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
