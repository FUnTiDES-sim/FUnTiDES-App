/**
 * @file sem2d_example.cc
 * @brief Acoustic wave propagation in 2-D with the quadrilateral SEM solver of FUnTiDES.
 *
 * The domain is the rectangle [0, lx] x [0, lz] of the (x, z) plane (z up, free surface on
 * z = lz with --free-surface), meshed with ex x ez quadrilaterals. The y options of the common
 * command line (--ey, --ly, --srcy, --rcvy) are ignored. Point source and receiver are
 * interpolated with the Lagrange (GLL) shape functions of their element. Single process.
 * Requires a FUnTiDES built with COMPILE_SEM_2D.
 *
 * Output: receiver_trace.txt (pressure at the receiver).
 */

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cartesian_unstruct_2d_builder.h"
#include "data_type.h"
#include "example_acquisition.h"
#include "example_mesh.h"
#include "example_options.h"
#include "example_output.h"
#include "example_runtime.h"
#include "gllpoints.h"
#include "rhs_acoustic.h"
#include "sem_enums.h"
#include "sem_solver_data.h"
#include "solver_factory.h"
#include "source_time_function.h"
#include "wavefield_acoustic.h"

namespace {

using utils::enums::implemType;
using utils::enums::meshType;
using utils::enums::methodType;
using utils::enums::physicType;

/// A point of the 2-D mesh: its element, and its weights on the (order+1)^2 element nodes.
struct Point2D {
  int element = 0;
  std::vector<float> weights;
};

/// Value at xi of the 1-D Lagrange polynomial of GLL node k, order @p order.
double Lagrange1D(int order, int k, double xi) {
  double l = 1.0;
  double const xk = GLLPoints::get(order, k);
  for (int j = 0; j <= order; ++j)
    if (j != k) l *= (xi - GLLPoints::get(order, j)) / (xk - GLLPoints::get(order, j));
  return l;
}

/// Element and interpolation weights of (x, z) in the ex x ez Cartesian grid of [0, lx] x [0, lz].
Point2D Locate(const examples::CommonOptions& opt, float x, float z) {
  if (x < 0 || x > opt.lx || z < 0 || z > opt.lz) throw std::invalid_argument("point outside the 2-D domain");
  float const hx = opt.lx / opt.ex, hz = opt.lz / opt.ez;
  int const ix = std::min(static_cast<int>(x / hx), opt.ex - 1);
  int const iz = std::min(static_cast<int>(z / hz), opt.ez - 1);
  double const xi = 2.0 * (x - ix * hx) / hx - 1.0;
  double const eta = 2.0 * (z - iz * hz) / hz - 1.0;
  Point2D p;
  p.element = ix + iz * opt.ex;
  int const n = opt.order + 1;
  p.weights.resize(n * n);
  for (int c = 0; c < n; ++c)
    for (int a = 0; a < n; ++a)
      p.weights[a + c * n] = static_cast<float>(Lagrange1D(opt.order, a, xi) * Lagrange1D(opt.order, c, eta));
  return p;
}

void Run(const examples::CommonOptions& opt) {
  examples::Timings timings;
  auto const start_init = std::chrono::steady_clock::now();

  // 2-D mesh and homogeneous model (vp = 1500 m/s, rho = 1).
  examples::MeshParams params(opt.order, opt.ex, 1, opt.ez, opt.lx, 1.f, opt.lz, opt.is_model_on_nodes, false);
  auto mesh = model::CartesianUnstruct2DBuilder<float, int>(params).getModel(opt.free_surface);
  float const dt = examples::ComputeTimeStep(opt, *mesh);
  int const num_samples = static_cast<int>(opt.timemax / dt);

  auto solver = solver::solver_factory::createSolver(methodType::kSem, implemType::kMakutu, meshType::kUnstruct,
                                                     examples::ModelLocation(opt), physicType::kAcoustic, opt.order,
                                                     0, /*dim=*/2);

  // Source (Ricker wavelet) and receiver.
  SourceTimeFunction wavelet;
  std::vector<float> const signal = wavelet.computeSourceTerm(num_samples, dt, opt.f0, opt.ricker_order, opt.tpeak);
  Point2D const source = Locate(opt, opt.srcx, opt.srcz);
  Point2D const receiver = Locate(opt, opt.rcvx, opt.rcvz);
  int const npe = mesh->getNumberOfPointsPerElement();
  arrayReal weights = allocateArray2D<arrayReal>(1, npe, "rhsWeights");
  {
    auto h = Kokkos::create_mirror_view(weights);
    for (int q = 0; q < npe; ++q) h(0, q) = source.weights[q];
    Kokkos::deep_copy(weights, h);
  }
  std::vector<int> receiver_nodes(npe);
  for (int q = 0; q < npe; ++q)
    receiver_nodes[q] = mesh->globalNodeIndex(receiver.element, q % (opt.order + 1), 0, q / (opt.order + 1));

  // Pressure at time steps n-1 and n.
  int const num_nodes = mesh->getNumberOfNodes();
  vectorReal p_prev = allocateVector<vectorReal>(num_nodes, "pnPrev");
  vectorReal p_curr = allocateVector<vectorReal>(num_nodes, "pnCurr");
  solver::WavefieldAcoustic wavefield(p_prev, p_curr);
  solver::RhsAcoustic rhs(examples::MakeSourceTerm(signal, true, "rhsTerm"), examples::MakeElementList(source.element),
                          weights);
  solver::SEMsolverDataAcoustic data(wavefield, rhs);

  float const b = opt.boundaries_size;
  solver->computeFEInit(*mesh, {b, 0.f, b}, opt.surface_sponge, opt.taper_delta);
  timings.init = examples::SecondsSince(start_init);
  examples::PrintRunSummary("SEM 2-D", "acoustic", opt, *mesh, dt, num_samples);

  std::vector<float> trace(num_samples);
  // A separate host copy: create_mirror_view() would alias the UVM wavefield buffers
  auto h_p = Kokkos::create_mirror(p_prev);
  for (int step = 0; step < num_samples; ++step) {
    auto start = std::chrono::steady_clock::now();
    solver->computeOneStep(dt, step, data);
    Kokkos::fence();
    timings.compute += examples::SecondsSince(start);

    start = std::chrono::steady_clock::now();
    // The new pressure is written into the previous-step buffer; swapWavefields() makes it the current one.
    Kokkos::deep_copy(h_p, data.getPreviousField(0));
    float value = 0.f;
    for (int q = 0; q < npe; ++q) value += receiver.weights[q] * h_p(receiver_nodes[q]);
    trace[step] = value;
    if (step % 100 == 0) std::cout << "step " << step << "/" << num_samples << "  p(receiver)=" << value << "\n";
    data.swapWavefields();
    timings.io += examples::SecondsSince(start);
  }

  examples::WriteTraces("receiver_trace.txt", {"pressure"}, dt, {trace});
  examples::PrintTimings(timings);
}

}  // namespace

int main(int argc, char** argv) {
  examples::Runtime runtime(argc, argv);

  examples::CommonOptions opt;
  if (auto exit_code =
          examples::ParseOptions(argc, argv, "funtides-sem2d", "Acoustic propagation in 2-D (x, z), SEM solver", opt)) {
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
