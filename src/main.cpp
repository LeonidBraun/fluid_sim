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
    fluid_sim::Simulation simulation(io.config(), io.initial_state(), io.initial_time());

    io.save_frame(simulation);

    const fluid_sim::SimulationConfig& config = io.config();
    double next_output_time = simulation.time() + config.output_interval;
    const double epsilon = 1e-12 * std::max(1.0, config.output_interval);
    fluid_sim::ConsoleLog console_log(std::chrono::steady_clock::now(), std::cout);

    while (simulation.time() + epsilon < config.end_time) {
      const double remaining_time = config.end_time - simulation.time();
      console_log.print_progress(simulation.time(), config.end_time, io.last_output());

      simulation.step(remaining_time);

      bool crossed_output_boundary = false;
      while (simulation.time() + epsilon >= next_output_time) {
        next_output_time += config.output_interval;
        crossed_output_boundary = true;
      }

      if (crossed_output_boundary) {
        io.save_frame(simulation);
      }
    }

    if (std::abs(simulation.time() - io.last_saved_time()) > epsilon) {
      io.save_frame(simulation);
    }
    console_log.print_progress(simulation.time(), config.end_time, io.last_output(), true);
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
