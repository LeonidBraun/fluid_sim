#include "fluid_sim/xdmf_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fluid_sim {

void write_xdmf_series(const std::filesystem::path& output_path,
                       const SimulationConfig& config,
                       const std::vector<SavedFrame>& frames) {
  std::ofstream stream(output_path);
  if (!stream) {
    throw std::runtime_error("Unable to create XDMF output file.");
  }

  stream << std::setprecision(17);
  stream << "<?xml version=\"1.0\" ?>\n";
  stream << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
  stream << "<Xdmf Version=\"3.0\">\n";
  stream << "  <Domain>\n";
  stream << "    <Grid Name=\"fluid_series\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

  for (std::size_t i = 0; i < frames.size(); ++i) {
    const auto& frame = frames[i];
    stream << "      <Grid Name=\"frame_" << std::setw(4) << std::setfill('0') << i
           << "\" GridType=\"Uniform\">\n";
    stream << "        <Time Value=\"" << frame.time << "\"/>\n";
    stream << "        <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"2 "
           << (config.ny + 1) << ' ' << (config.nx + 1) << "\"/>\n";
    stream << "        <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
    stream << "          <DataItem Dimensions=\"3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">0 0 0</DataItem>\n";
    stream << "          <DataItem Dimensions=\"3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">1 "
           << config.dy << ' ' << config.dx << "</DataItem>\n";
    stream << "        </Geometry>\n";
    stream << "        <Attribute Name=\"density_offset\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
    stream << "          <DataItem Dimensions=\"1 " << config.ny << ' ' << config.nx
           << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
           << frame.file_name << ":/density_offset</DataItem>\n";
    stream << "        </Attribute>\n";
    stream << "        <Attribute Name=\"velocity\" AttributeType=\"Vector\" Center=\"Cell\">\n";
    stream << "          <DataItem Dimensions=\"1 " << config.ny << ' ' << config.nx
           << " 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
           << frame.file_name << ":/velocity</DataItem>\n";
    stream << "        </Attribute>\n";
    stream << "      </Grid>\n";
  }

  stream << "    </Grid>\n";
  stream << "  </Domain>\n";
  stream << "</Xdmf>\n";
}

}  // namespace fluid_sim
