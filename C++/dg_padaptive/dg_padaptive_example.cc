/**
 * @file dg_padaptive_example.cc
 * @brief Acoustic wave propagation with the p-adaptive DG solver of FUnTiDES.
 *
 * The domain is split by a horizontal plane z = --dg-padaptive-boundary-z: elements below it
 * use the low order --order-min (pMin), elements above it the high order --order (pMax).
 * Each part keeps its own pressure per element. The run is single-process.
 *
 * Outputs: receiver_trace.txt (pressure at the receiver, read in the part containing it)
 * and, with --snapshots, slice_dgpadaptive_pmin_<step>.dat and slice_dgpadaptive_pmax_<step>.dat
 * (x-y planes through the middle element layer of each part).
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "data_type.h"
#include "dg_padaptive_solver_data.h"
#include "example_acquisition.h"
#include "example_mesh.h"
#include "example_options.h"
#include "example_output.h"
#include "example_runtime.h"
#include "sem_enums.h"
#include "solver_factory.h"
#include "source_time_function.h"

namespace {

using utils::enums::methodType;
using utils::enums::physicType;

struct DgPAdaptiveOptions : examples::CommonOptions {
  int order_min = 1;       ///< Polynomial order of the pMin part, 0 < order_min < order.
  float boundary_z = 0.f;  ///< Height of the pMin/pMax interface (m); 0 means lz / 2.

  static void BindCli(cxxopts::Options& cli, DgPAdaptiveOptions& o) {
    examples::CommonOptions::BindCli(cli, o);
    cli.add_options()("order-min", "Polynomial order of the pMin part (default 1)",
                      cxxopts::value<int>(o.order_min))(
        "dg-padaptive-boundary-z", "Height of the pMin/pMax interface (m, default lz/2)",
        cxxopts::value<float>(o.boundary_z));
  }

  void Validate() const {
    examples::CommonOptions::Validate();
    if (order_min < 1 || order_min >= order) throw std::runtime_error("order-min must satisfy 1 <= order-min < order");
    if (boundary_z < 0 || boundary_z > lz) throw std::runtime_error("dg-padaptive-boundary-z must lie in [0, lz]");
  }

  float InterfaceZ() const { return (boundary_z > 0.f) ? boundary_z : 0.5f * lz; }
};

void Run(const DgPAdaptiveOptions& opt) {
  examples::Timings timings;
  auto const start_init = std::chrono::steady_clock::now();

  // Mesh and model of the whole domain, built at the high order.
  examples::MeshParams const params = examples::MakeLocalParams(opt, /*is_elastic=*/false, /*rank=*/0, /*size=*/1);
  auto const mesh_type = examples::ParseMeshType(opt.mesh);
  auto mesh = examples::BuildMesh(params, mesh_type, opt.free_surface);
  float const dt = examples::ComputeTimeStep(opt, *mesh);
  int const num_samples = static_cast<int>(opt.timemax / dt);
  float const interface_z = opt.InterfaceZ();

  // Solver; the interface must be set before computeFEInit().
  auto solver = solver::solver_factory::createSolver(methodType::kDgPAdaptive, examples::ParseImplemType(opt.implem),
                                                     mesh_type, examples::ModelLocation(opt), physicType::kAcoustic,
                                                     opt.order, opt.order_min);
  solver->setZBoundary(interface_z);

  // Source and receiver, with weights at the order of the part containing them.
  SourceTimeFunction wavelet;
  std::vector<float> const signal = wavelet.computeSourceTerm(num_samples, dt, opt.f0, opt.ricker_order, opt.tpeak);
  bool const source_in_pmin = (opt.srcz <= interface_z);
  bool const receiver_in_pmax = (opt.rcvz >= interface_z);
  std::array<float, 3> const source_coord = {opt.srcx, opt.srcy, opt.srcz};
  std::array<float, 3> const receiver_coord = {opt.rcvx, opt.rcvy, opt.rcvz};
  auto const source_pmin = examples::LocatePoint(*mesh, params, source_coord, opt.order_min);
  auto const source_pmax = examples::LocatePoint(*mesh, params, source_coord, opt.order);
  auto const receiver = examples::LocatePoint(*mesh, params, receiver_coord, receiver_in_pmax ? opt.order : opt.order_min);
  int const n1d_min = opt.order_min + 1;
  int const n1d_max = opt.order + 1;
  int const num_dofs_min = n1d_min * n1d_min * n1d_min;
  int const num_dofs_max = n1d_max * n1d_max * n1d_max;
  // Only the part containing the source receives non-zero weights.
  arrayReal const weights_pmin = source_in_pmin ? source_pmin.weights : arrayReal("weightsPMin", 1, num_dofs_min);
  arrayReal const weights_pmax = source_in_pmin ? arrayReal("weightsPMax", 1, num_dofs_max) : source_pmax.weights;
  bool const source_on_rank = source_pmax.on_rank;

  // Pressure per element of each part at time steps n-1 and n.
  int const num_elements = mesh->getNumberOfElements();
  arrayReal p_pmin_prev = allocateArray2D<arrayReal>(num_elements, num_dofs_min, "pnPMinDGPrev");
  arrayReal p_pmin_curr = allocateArray2D<arrayReal>(num_elements, num_dofs_min, "pnPMinDGCurr");
  arrayReal p_pmax_prev = allocateArray2D<arrayReal>(num_elements, num_dofs_max, "pnPMaxDGPrev");
  arrayReal p_pmax_curr = allocateArray2D<arrayReal>(num_elements, num_dofs_max, "pnPMaxDGCurr");
  solver::DGPAdaptiveWavefieldAcoustic wavefield(p_pmin_prev, p_pmin_curr, p_pmax_prev, p_pmax_curr);
  solver::DGPAdaptiveRhsAcoustic rhs(
      examples::MakeSourceTerm(signal, source_on_rank && source_in_pmin, "rhsTermPMin"),
      examples::MakeSourceTerm(signal, source_on_rank && !source_in_pmin, "rhsTermPMax"),
      examples::MakeElementList(source_pmax.element), weights_pmin, weights_pmax);
  solver::DGPAdaptiveSolverData data(wavefield, rhs);

  float const b = opt.boundaries_size;
  solver->computeFEInit(*mesh, {b, b, b}, opt.surface_sponge, opt.taper_delta);
  timings.init = examples::SecondsSince(start_init);
  examples::PrintRunSummary("DG p-adaptive", "acoustic", opt, *mesh, dt, num_samples);
  int const ez_pmin = static_cast<int>(std::round(params.ez * interface_z / params.lz));
  std::cout << "Interface z = " << interface_z << " m (" << ez_pmin << " pMin element layers), orders "
            << opt.order_min << "/" << opt.order << ", receiver in the " << (receiver_in_pmax ? "pMax" : "pMin")
            << " part\n";

  std::vector<float> trace(num_samples);
  examples::SnapshotWriter snapshot_writer;
  auto h_snapshot_pmin = Kokkos::create_mirror(p_pmin_prev);
  auto h_snapshot_pmax = Kokkos::create_mirror(p_pmax_prev);

  for (int step = 0; step < num_samples; ++step) {
    auto start = std::chrono::steady_clock::now();
    solver->computeOneStep(dt, step, data);
    Kokkos::fence();
    timings.compute += examples::SecondsSince(start);

    start = std::chrono::steady_clock::now();
    // The new pressures are written into the previous-step buffers; swapWavefields() makes them current.
    arrayReal const p_pmin_new = data.m_wavefield.getPMinPreviousField(0);
    arrayReal const p_pmax_new = data.m_wavefield.getPMaxPreviousField(0);
    trace[step] = examples::InterpolateElemental(receiver_in_pmax ? p_pmax_new : p_pmin_new, receiver);
    if (step % 50 == 0) std::cout << "step " << step << "/" << num_samples << "  p(receiver)=" << trace[step] << "\n";

    if (opt.snapshots && step % opt.snap_interval == 0) {
      snapshot_writer.Wait();
      Kokkos::deep_copy(h_snapshot_pmin, p_pmin_new);
      Kokkos::deep_copy(h_snapshot_pmax, p_pmax_new);
      snapshot_writer.Submit([&params, &opt, h_snapshot_pmin, h_snapshot_pmax, ez_pmin, step] {
        examples::WriteElementalSliceZ(examples::NumberedFileName("slice_dgpadaptive_pmin", step), h_snapshot_pmin,
                                       params.ex, params.ey, opt.order_min + 1, ez_pmin / 2, opt.order_min / 2);
        examples::WriteElementalSliceZ(examples::NumberedFileName("slice_dgpadaptive_pmax", step), h_snapshot_pmax,
                                       params.ex, params.ey, opt.order + 1, ez_pmin + (params.ez - ez_pmin) / 2,
                                       opt.order / 2);
      });
    }
    data.swapWavefields();
    timings.io += examples::SecondsSince(start);
  }

  snapshot_writer.Wait();
  examples::WriteTraces("receiver_trace.txt", {"pressure"}, dt, {trace});
  examples::PrintTimings(timings);
}

}  // namespace

int main(int argc, char** argv) {
  examples::Runtime runtime(argc, argv);

  DgPAdaptiveOptions opt;
  if (auto exit_code = examples::ParseOptions(argc, argv, "funtides-dg-padaptive",
                                              "Acoustic propagation, p-adaptive DG solver", opt)) {
    return *exit_code;
  }

  try {
    Run(opt);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
