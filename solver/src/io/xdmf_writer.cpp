#include "io/xdmf_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fluid_sim {

void write_xdmf_series(const std::filesystem::path& output_path,
                       const io::State& state,
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
    stream << "      <Grid Name=\"frame_" << i << "\" GridType=\"Uniform\">\n";
    stream << "        <Time Value=\"" << frame.time << "\"/>\n";
    stream << "        <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\"2 " << (state.grid.ny + 1) << ' '
           << (state.grid.nx + 1) << "\"/>\n";
    stream << "        <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
    stream << "          <DataItem Dimensions=\"3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">0 0 "
              "0</DataItem>\n";
    stream << "          <DataItem Dimensions=\"3\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">1 "
           << state.grid.h << ' ' << state.grid.h << "</DataItem>\n";
    stream << "        </Geometry>\n";
    stream << "        <Attribute Name=\"density_offset\" AttributeType=\"Scalar\" Center=\"Cell\">\n";
    stream << "          <DataItem Dimensions=\"1 " << state.grid.ny << ' ' << state.grid.nx
           << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">" << frame.file_name
           << ":/density_offset</DataItem>\n";
    stream << "        </Attribute>\n";
    stream << "        <Attribute Name=\"momentum\" AttributeType=\"Vector\" Center=\"Cell\">\n";
    stream << "          <DataItem Dimensions=\"1 " << state.grid.ny << ' ' << state.grid.nx
           << " 3\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">" << frame.file_name
           << ":/momentum</DataItem>\n";
    stream << "        </Attribute>\n";
    stream << "      </Grid>\n";
  }

  stream << "    </Grid>\n";
  stream << "  </Domain>\n";
  stream << "</Xdmf>\n";
}

} // namespace fluid_sim
