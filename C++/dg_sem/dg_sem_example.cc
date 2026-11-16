/**
 * @file dg_sem_example.cc
 * @brief Acoustic wave propagation with the coupled DG-SEM solver of FUnTiDES.
 *
 * The domain is split by a horizontal plane z = --dg-sem-boundary-z: DG below it, SEM above.
 * Each part keeps its own pressure (per element for DG, per global node for SEM) and the
 * solver couples them across the interface. The run is single-process.
 *
 * Outputs: receiver_trace.txt (pressure at the receiver, read in the part containing it)
 * and, with --snapshots, slice_dgsem_xz_<step>.dat (x-z plane at mid y, DG rows then SEM rows).
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "data_type.h"
#include "dgsem_solver_data.h"
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

struct DgSemOptions : examples::CommonOptions {
  float boundary_z = 0.f;  ///< Height of the DG/SEM interface (m); 0 means lz / 2.

  static void BindCli(cxxopts::Options& cli, DgSemOptions& o) {
    examples::CommonOptions::BindCli(cli, o);
    cli.add_options()("dg-sem-boundary-z", "Height of the DG/SEM interface (m, default lz/2)",
                      cxxopts::value<float>(o.boundary_z));
  }

  void Validate() const {
    examples::CommonOptions::Validate();
    if (boundary_z < 0 || boundary_z > lz) throw std::runtime_error("dg-sem-boundary-z must lie in [0, lz]");
  }

  float InterfaceZ() const { return (boundary_z > 0.f) ? boundary_z : 0.5f * lz; }
};

// x-z plane at mid y: the DG element layers below the interface, then the SEM node rows above it.
void WriteXzSlice(const std::string& path, const arrayReal::host_mirror_type& dg, const vectorReal::host_mirror_type& sem,
                  const examples::MeshParams& params, int order, int ez_dg) {
  int const ex = params.ex;
  int const ey = params.ey;
  int const nx = ex * order + 1;
  int const ny = ey * order + 1;
  int const nz = params.ez * order + 1;
  int const n1d = order + 1;
  int const ey_mid = ey / 2;
  int const jy_mid = order / 2;

  std::ofstream out(path);
  for (int ez_idx = 0; ez_idx < ez_dg; ++ez_idx) {
    for (int kz = 0; kz < n1d; ++kz) {
      for (int ix = 0; ix < nx; ++ix) {
        // The last node column belongs to the last element.
        int const element_x = (ix == nx - 1) ? ex - 1 : ix / order;
        int const node_x = (ix == nx - 1) ? order : ix % order;
        int const element = element_x + ey_mid * ex + ez_idx * ex * ey;
        if (ix > 0) out << " ";
        out << dg(element, node_x + jy_mid * n1d + kz * n1d * n1d);
      }
      out << "\n";
    }
  }
  for (int iz = ez_dg * order; iz < nz; ++iz) {
    for (int ix = 0; ix < nx; ++ix) {
      if (ix > 0) out << " ";
      out << sem(ix + (ny / 2) * nx + iz * nx * ny);
    }
    out << "\n";
  }
}

void Run(const DgSemOptions& opt) {
  examples::Timings timings;
  auto const start_init = std::chrono::steady_clock::now();

  // Mesh and model of the whole domain.
  examples::MeshParams const params = examples::MakeLocalParams(opt, /*is_elastic=*/false, /*rank=*/0, /*size=*/1);
  auto const mesh_type = examples::ParseMeshType(opt.mesh);
  auto mesh = examples::BuildMesh(params, mesh_type, opt.free_surface);
  float const dt = examples::ComputeTimeStep(opt, *mesh);
  int const num_samples = static_cast<int>(opt.timemax / dt);
  float const interface_z = opt.InterfaceZ();

  // Solver; the interface must be set before computeFEInit().
  auto solver = solver::solver_factory::createSolver(methodType::kDgSem, examples::ParseImplemType(opt.implem),
                                                     mesh_type, examples::ModelLocation(opt), physicType::kAcoustic,
                                                     opt.order);
  solver->setZBoundary(interface_z);

  // Source and receiver. The source term goes to the part containing the source.
  SourceTimeFunction wavelet;
  std::vector<float> const signal = wavelet.computeSourceTerm(num_samples, dt, opt.f0, opt.ricker_order, opt.tpeak);
  auto const source = examples::LocatePoint(*mesh, params, {opt.srcx, opt.srcy, opt.srcz}, opt.order);
  auto const receiver = examples::LocatePoint(*mesh, params, {opt.rcvx, opt.rcvy, opt.rcvz}, opt.order);
  bool const source_in_dg = (opt.srcz < interface_z);
  bool const receiver_in_sem = (opt.rcvz >= interface_z);

  // DG pressure per element, SEM pressure per global node, each at time steps n-1 and n.
  int const num_elements = mesh->getNumberOfElements();
  int const num_dofs = mesh->getNumberOfPointsPerElement();
  int const num_nodes = mesh->getNumberOfNodes();
  arrayReal p_dg_prev = allocateArray2D<arrayReal>(num_elements, num_dofs, "pnDGPrev");
  arrayReal p_dg_curr = allocateArray2D<arrayReal>(num_elements, num_dofs, "pnDGCurr");
  vectorReal p_sem_prev = allocateVector<vectorReal>(num_nodes, "pnSEMPrev");
  vectorReal p_sem_curr = allocateVector<vectorReal>(num_nodes, "pnSEMCurr");
  solver::DGSEMWavefieldAcoustic wavefield(p_dg_prev, p_dg_curr, p_sem_prev, p_sem_curr);
  solver::DGSEMRhsAcoustic rhs(examples::MakeSourceTerm(signal, source.on_rank && source_in_dg, "rhsTermDG"),
                               examples::MakeSourceTerm(signal, source.on_rank && !source_in_dg, "rhsTermSEM"),
                               examples::MakeElementList(source.element), source.weights);
  solver::DGSEMsolverData data(wavefield, rhs);

  float const b = opt.boundaries_size;
  solver->computeFEInit(*mesh, {b, b, b}, opt.surface_sponge, opt.taper_delta);
  timings.init = examples::SecondsSince(start_init);
  examples::PrintRunSummary("DG-SEM", "acoustic", opt, *mesh, dt, num_samples);
  std::cout << "Interface z = " << interface_z << " m, receiver in the " << (receiver_in_sem ? "SEM" : "DG")
            << " part\n";

  std::vector<float> trace(num_samples);
  auto h_sem = Kokkos::create_mirror(p_sem_prev);
  examples::SnapshotWriter snapshot_writer;
  auto h_snapshot_dg = Kokkos::create_mirror(p_dg_prev);
  auto h_snapshot_sem = Kokkos::create_mirror(p_sem_prev);
  int const ez_dg = static_cast<int>(std::round(params.ez * interface_z / params.lz));

  for (int step = 0; step < num_samples; ++step) {
    auto start = std::chrono::steady_clock::now();
    solver->computeOneStep(dt, step, data);
    Kokkos::fence();
    timings.compute += examples::SecondsSince(start);

    start = std::chrono::steady_clock::now();
    // The new pressures are written into the previous-step buffers; swapWavefields() makes them current.
    arrayReal const p_dg_new = data.m_wavefield.getDGPreviousField(0);
    vectorReal const p_sem_new = data.m_wavefield.getSEMPreviousField(0);
    if (receiver_in_sem) {
      Kokkos::deep_copy(h_sem, p_sem_new);
      trace[step] = examples::InterpolateNodal(*mesh, h_sem, receiver);
    } else {
      trace[step] = examples::InterpolateElemental(p_dg_new, receiver);
    }
    if (step % 50 == 0) std::cout << "step " << step << "/" << num_samples << "  p(receiver)=" << trace[step] << "\n";

    if (opt.snapshots && step % opt.snap_interval == 0) {
      snapshot_writer.Wait();
      Kokkos::deep_copy(h_snapshot_dg, p_dg_new);
      Kokkos::deep_copy(h_snapshot_sem, p_sem_new);
      snapshot_writer.Submit([&params, &opt, h_snapshot_dg, h_snapshot_sem, ez_dg, step] {
        WriteXzSlice(examples::NumberedFileName("slice_dgsem_xz", step), h_snapshot_dg, h_snapshot_sem, params,
                     opt.order, ez_dg);
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

  DgSemOptions opt;
  if (auto exit_code =
          examples::ParseOptions(argc, argv, "funtides-dg-sem", "Acoustic propagation, coupled DG-SEM solver", opt)) {
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
