/**
 * @file dg_example.cc
 * @brief Acoustic wave propagation with the discontinuous Galerkin (DG) solver of FUnTiDES.
 *
 * The pressure is stored per element, shape (elements, (order+1)^3): the elements do not
 * share nodes. The run is single-process.
 *
 * Outputs: receiver_trace.txt (pressure at the receiver) and, with --snapshots,
 * slice_dg_<step>.dat (x-y plane through the middle element layer).
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "data_type.h"
#include "dg_solver_data.h"
#include "dg_wavefield_acoustic.h"
#include "example_acquisition.h"
#include "example_mesh.h"
#include "example_options.h"
#include "example_output.h"
#include "example_runtime.h"
#include "rhs_acoustic.h"
#include "sem_enums.h"
#include "solver_factory.h"
#include "source_time_function.h"

namespace {

using utils::enums::methodType;
using utils::enums::physicType;

void Run(const examples::CommonOptions& opt) {
  examples::Timings timings;
  auto const start_init = std::chrono::steady_clock::now();

  // Mesh and model of the whole domain.
  examples::MeshParams const params = examples::MakeLocalParams(opt, /*is_elastic=*/false, /*rank=*/0, /*size=*/1);
  auto const mesh_type = examples::ParseMeshType(opt.mesh);
  auto mesh = examples::BuildMesh(params, mesh_type, opt.free_surface);
  float const dt = examples::ComputeTimeStep(opt, *mesh);
  int const num_samples = static_cast<int>(opt.timemax / dt);

  // Solver.
  auto solver = solver::solver_factory::createSolver(methodType::kDg, examples::ParseImplemType(opt.implem),
                                                     mesh_type, examples::ModelLocation(opt), physicType::kAcoustic,
                                                     opt.order);

  // Source (Ricker wavelet) and receiver.
  SourceTimeFunction wavelet;
  std::vector<float> const signal = wavelet.computeSourceTerm(num_samples, dt, opt.f0, opt.ricker_order, opt.tpeak);
  auto const source = examples::LocatePoint(*mesh, params, {opt.srcx, opt.srcy, opt.srcz}, opt.order);
  auto const receiver = examples::LocatePoint(*mesh, params, {opt.rcvx, opt.rcvy, opt.rcvz}, opt.order);

  // Pressure at time steps n-1 and n, and the data handed to the solver.
  int const num_elements = mesh->getNumberOfElements();
  int const num_dofs = mesh->getNumberOfPointsPerElement();
  arrayReal p_prev = allocateArray2D<arrayReal>(num_elements, num_dofs, "pnDGPrev");
  arrayReal p_curr = allocateArray2D<arrayReal>(num_elements, num_dofs, "pnDGCurr");
  solver::DGWavefieldAcoustic wavefield(p_prev, p_curr);
  solver::RhsAcoustic rhs(examples::MakeSourceTerm(signal, source.on_rank, "rhsTerm"),
                          examples::MakeElementList(source.element), source.weights);
  solver::DGsolverDataAcoustic data(wavefield, rhs);

  // Mass and stiffness operators, sponge layers.
  float const b = opt.boundaries_size;
  solver->computeFEInit(*mesh, {b, b, b}, opt.surface_sponge, opt.taper_delta);
  timings.init = examples::SecondsSince(start_init);
  examples::PrintRunSummary("DG", "acoustic", opt, *mesh, dt, num_samples);

  std::vector<float> trace(num_samples);
  examples::SnapshotWriter snapshot_writer;
  auto h_snapshot = Kokkos::create_mirror(p_prev);

  for (int step = 0; step < num_samples; ++step) {
    auto start = std::chrono::steady_clock::now();
    solver->computeOneStep(dt, step, data);
    Kokkos::fence();
    timings.compute += examples::SecondsSince(start);

    start = std::chrono::steady_clock::now();
    // The new pressure is written into the previous-step buffer; swapWavefields() makes it the current one.
    arrayReal const p_new = data.getPreviousField(0);
    trace[step] = examples::InterpolateElemental(p_new, receiver);
    if (step % 50 == 0) std::cout << "step " << step << "/" << num_samples << "  p(receiver)=" << trace[step] << "\n";

    if (opt.snapshots && step % opt.snap_interval == 0) {
      snapshot_writer.Wait();
      Kokkos::deep_copy(h_snapshot, p_new);
      snapshot_writer.Submit([&params, &opt, h_snapshot, step] {
        examples::WriteElementalSliceZ(examples::NumberedFileName("slice_dg", step), h_snapshot, params.ex, params.ey,
                                       opt.order + 1, params.ez / 2, opt.order / 2);
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

  examples::CommonOptions opt;
  if (auto exit_code = examples::ParseOptions(argc, argv, "funtides-dg", "Acoustic propagation, DG solver", opt)) {
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
