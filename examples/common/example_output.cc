#include "example_output.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace examples {

void SnapshotWriter::Submit(std::function<void()> task) {
  pending_.push_back(std::async(std::launch::async, std::move(task)));
}

void SnapshotWriter::Wait() {
  for (auto& f : pending_) f.get();
  pending_.clear();
}

std::string NumberedFileName(const std::string& prefix, int step) {
  std::ostringstream name;
  name << prefix << "_" << std::setfill('0') << std::setw(5) << step << ".dat";
  return name.str();
}

void WriteTraces(const std::string& path, const std::vector<std::string>& names, float dt,
                 const std::vector<std::vector<float>>& traces) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write " + path);
  out << "# time";
  for (const auto& name : names) out << " " << name;
  out << "\n" << std::scientific << std::setprecision(8);
  std::size_t const num_samples = traces.empty() ? 0 : traces[0].size();
  for (std::size_t n = 0; n < num_samples; ++n) {
    out << (n + 1) * dt;
    for (const auto& trace : traces) out << " " << trace[n];
    out << "\n";
  }
}

void WriteNodalSliceZ(const std::string& path, const vectorReal::host_mirror_type& field, int nx, int ny, int iz) {
  std::ofstream out(path);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      if (ix > 0) out << " ";
      out << field(ix + iy * nx + iz * nx * ny);
    }
    out << "\n";
  }
}

void WriteElementalSliceZ(const std::string& path, const arrayReal::host_mirror_type& field, int ex, int ey, int n1d,
                          int element_layer, int node_layer) {
  std::ofstream out(path);
  for (int ey_idx = 0; ey_idx < ey; ++ey_idx) {
    for (int jy = 0; jy < n1d; ++jy) {
      for (int ex_idx = 0; ex_idx < ex; ++ex_idx) {
        int const element = ex_idx + ey_idx * ex + element_layer * ex * ey;
        for (int ix = 0; ix < n1d; ++ix) {
          if (ex_idx > 0 || ix > 0) out << " ";
          out << field(element, ix + jy * n1d + node_layer * n1d * n1d);
        }
      }
      out << "\n";
    }
  }
}

void PrintRunSummary(const std::string& method, const std::string& physics, const CommonOptions& opt,
                     const Mesh& mesh, float dt, int num_samples) {
  std::cout << "\n==========================================================\n"
            << "  " << method << " - " << physics << "\n"
            << "==========================================================\n"
            << "  Mesh:            " << opt.mesh << ", " << mesh.getNumberOfElements() << " elements, "
            << mesh.getNumberOfNodes() << " nodes, order " << opt.order << "\n"
            << "  Model:           on " << (opt.is_model_on_nodes ? "nodes" : "elements") << "\n"
            << "  Time step:       " << dt << " s, " << num_samples << " steps (" << opt.timemax << " s)\n"
            << "  Snapshots:       ";
  if (opt.snapshots) {
    std::cout << "every " << opt.snap_interval << " steps\n";
  } else {
    std::cout << "off\n";
  }
  std::cout << "==========================================================\n\n";
}

void PrintTimings(const Timings& timings) {
  double const total = timings.init + timings.compute + timings.io;
  auto line = [total](const char* label, double seconds) {
    double const percent = (total > 0) ? 100.0 * seconds / total : 0.0;
    std::cout << "  " << label << std::fixed << std::setprecision(4) << std::setw(10) << seconds << " s ("
              << std::setprecision(1) << std::setw(5) << percent << " %)\n";
  };
  std::cout << "\n==========================================================\n";
  line("Initialization: ", timings.init);
  line("Solver:         ", timings.compute);
  line("Outputs:        ", timings.io);
  std::cout << "  Total:          " << std::fixed << std::setprecision(4) << std::setw(10) << total << " s\n"
            << "==========================================================\n\n";
}

}  // namespace examples
