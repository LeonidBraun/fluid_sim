#include "io/hdf5_writer.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: fluid_frame_dump <input.h5> <nx> <ny> <output.bin>\n";
    return 2;
  }

  try {
    const int nx = std::stoi(argv[2]);
    const int ny = std::stoi(argv[3]);
    const fluid_sim::HostState state = fluid_sim::read_frame_hdf5(argv[1], nx, ny);

    std::ofstream output(argv[4], std::ios::binary);
    if (!output) {
      throw std::runtime_error(std::string("Unable to open output file: ") + argv[4]);
    }

    const std::uint64_t density_size = static_cast<std::uint64_t>(state.density_offset.size());
    const std::uint64_t velocity_size = static_cast<std::uint64_t>(state.velocity.size());
    output.write(reinterpret_cast<const char*>(&density_size), sizeof(density_size));
    output.write(reinterpret_cast<const char*>(state.density_offset.data()),
                 static_cast<std::streamsize>(state.density_offset.size() * sizeof(float)));
    output.write(reinterpret_cast<const char*>(&velocity_size), sizeof(velocity_size));
    output.write(reinterpret_cast<const char*>(state.velocity.data()),
                 static_cast<std::streamsize>(state.velocity.size() * sizeof(float)));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
