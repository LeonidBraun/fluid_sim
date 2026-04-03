#include "io/IO.hpp"
#include "io/console_log.hpp"
#include "simulation/simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
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
    double next_output_time = simulation.time() + output_settings.output_interval;
    const double epsilon = 1e-12 * std::max(1.0, output_settings.output_interval);
    fluid_sim::ConsoleLog console_log(std::chrono::steady_clock::now(), std::cout);

    while (simulation.time() + epsilon < output_settings.end_time) {
      const double remaining_time = output_settings.end_time - simulation.time();
      console_log.print_progress(simulation.time(), output_settings.end_time, io.last_output());

      simulation.step(remaining_time);

      bool crossed_output_boundary = false;
      while (simulation.time() + epsilon >= next_output_time) {
        next_output_time += output_settings.output_interval;
        crossed_output_boundary = true;
      }

      if (crossed_output_boundary) {
        io.save_output(simulation);
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
