/**
 * @file sem_example.cc
 * @brief Wave propagation with the spectral element (SEM) solver of FUnTiDES.
 *
 * Physics: acoustic (default), elastic (--is-elastic) or coupled acousto-elastic
 * (--is-acousto-elastic, fluid above --acousto-elastic-boundary-z). Fields are stored per
 * global node. Built with USE_MPI, the domain is split along x across the MPI ranks and the
 * ranks exchange their boundary nodes; acousto-elastic runs are single-process.
 *
 * Outputs, written by the rank holding the receiver:
 * - receiver_trace.txt: pressure (acoustic, or acousto-elastic receiver in the fluid) or
 *   ux uy uz (elastic, or acousto-elastic receiver in the solid);
 * - das_trace.txt with --das-type (elastic only);
 * - with --snapshots, slice_<field>_<step>.dat: x-y plane at mid z of p, ux or uz
 *   (slice_<field>_r<rank>_<step>.dat in a distributed run).
 */

#include <Kokkos_Core.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "boundary_synchronizer.h"
#include "data_type.h"
#include "example_acquisition.h"
#include "example_mesh.h"
#include "example_options.h"
#include "example_output.h"
#include "example_runtime.h"
#include "parallel_topology.h"
#include "rhs_acoustic.h"
#include "rhs_acoustoelastic.h"
#include "rhs_elastic.h"
#include "sem_enums.h"
#include "sem_solver_acoustoelastic.h"
#include "sem_solver_data.h"
#include "solver_factory.h"
#include "source_and_receiver_utils.h"
#include "source_time_function.h"
#include "topology_factory.h"
#include "wavefield_acoustic.h"
#include "wavefield_acoustoelastic.h"
#include "wavefield_elastic.h"
#ifdef USE_MPI
#include "mpi_backend.h"
#endif

namespace {

using SourceAndReceiverUtils::DASType;
using utils::enums::methodType;
using utils::enums::physicType;

struct SemOptions : examples::CommonOptions {
  bool is_elastic = false;                  ///< Elastic propagation.
  bool is_acousto_elastic = false;          ///< Coupled acoustic (top) / elastic (bottom) propagation.
  float acousto_elastic_boundary_z = 0.f;   ///< Height of the fluid/solid interface (m).
  std::string anisotropy = "iso";           ///< Elastic anisotropy: iso | vti | tti.
  float qp = -1.f;                          ///< P-wave quality factor; negative for no attenuation.
  float qs = -1.f;                          ///< S-wave quality factor; negative for no attenuation.
  std::vector<float> sls_frequencies;       ///< SLS reference angular frequencies (rad/s).
  std::vector<float> sls_coefficients;      ///< SLS anelasticity coefficients, one per frequency.
  std::string das_type = "none";            ///< DAS receiver: none | dipole | strain.
  float das_dip = 0.f;                      ///< Fiber dip from the horizontal (degrees).
  float das_azimuth = 0.f;                  ///< Fiber azimuth in the x-y plane from x (degrees).
  float das_gauge_length = 1.f;             ///< Gauge length (m).
  int das_samples = 5;                      ///< Integration points along the fiber (strain).

  static void BindCli(cxxopts::Options& cli, SemOptions& o) {
    examples::CommonOptions::BindCli(cli, o);
    cli.add_options()("is-elastic", "Elastic propagation", cxxopts::value<bool>(o.is_elastic))(
        "is-acousto-elastic", "Coupled acoustic/elastic propagation", cxxopts::value<bool>(o.is_acousto_elastic))(
        "acousto-elastic-boundary-z", "Height of the fluid/solid interface (m)",
        cxxopts::value<float>(o.acousto_elastic_boundary_z))(
        "anisotropy", "Elastic anisotropy: iso|vti|tti (default iso)", cxxopts::value<std::string>(o.anisotropy))(
        "qp", "P-wave quality factor (default: no attenuation)", cxxopts::value<float>(o.qp))(
        "qs", "S-wave quality factor (default: no attenuation)", cxxopts::value<float>(o.qs))(
        "sls-reference-angular-frequencies", "Comma-separated SLS reference angular frequencies (rad/s)",
        cxxopts::value<std::vector<float>>(o.sls_frequencies))(
        "sls-anelasticity-coefficients", "Comma-separated SLS anelasticity coefficients, one per frequency",
        cxxopts::value<std::vector<float>>(o.sls_coefficients))(
        "das-type", "DAS receiver: none|dipole|strain (default none, elastic only)",
        cxxopts::value<std::string>(o.das_type))("das-dip", "DAS fiber dip (degrees)",
                                                 cxxopts::value<float>(o.das_dip))(
        "das-azimuth", "DAS fiber azimuth (degrees)", cxxopts::value<float>(o.das_azimuth))(
        "das-gauge-length", "DAS gauge length (m, default 1)", cxxopts::value<float>(o.das_gauge_length))(
        "das-samples", "DAS integration points along the fiber (default 5)", cxxopts::value<int>(o.das_samples));
  }

  void Validate() const {
    examples::CommonOptions::Validate();
    if (anisotropy != "iso" && anisotropy != "vti" && anisotropy != "tti") {
      throw std::runtime_error("anisotropy must be iso, vti or tti");
    }
    if (!sls_coefficients.empty() && sls_coefficients.size() != sls_frequencies.size()) {
      throw std::runtime_error("sls-anelasticity-coefficients must have one value per reference frequency");
    }
    if (das_type != "none" && das_type != "dipole" && das_type != "strain") {
      throw std::runtime_error("das-type must be none, dipole or strain");
    }
    if (das_type != "none") {
      if (!is_elastic || is_acousto_elastic) throw std::runtime_error("DAS receivers require an elastic run");
      if (das_gauge_length <= 0) throw std::runtime_error("das-gauge-length must be > 0");
      if (das_type == "strain" && das_samples < 1) throw std::runtime_error("das-samples must be >= 1");
    }
  }

  physicType Physics() const {
    if (is_acousto_elastic) return physicType::kAcoustoElastic;
    return is_elastic ? physicType::kElastic : physicType::kAcoustic;
  }

  model::AnisotropyType Anisotropy() const {
    if (anisotropy == "vti") return model::AnisotropyType::kVTI;
    if (anisotropy == "tti") return model::AnisotropyType::kTTI;
    return model::AnisotropyType::kIso;
  }
};

/// Everything a time loop needs, set up once by Run().
struct SemRun {
  const SemOptions& opt;
  int rank = 0;
  examples::MeshParams params;
  std::shared_ptr<examples::Mesh> mesh;
  std::unique_ptr<solver::Solver> solver;
  parallel::ParallelTopology topology;
  std::unique_ptr<solver::BoundarySynchronizer> syncer;
  float dt = 0.f;
  int num_samples = 0;
  std::vector<float> signal;
  examples::PointInElement source;
  examples::PointInElement receiver;
  examples::Timings timings;
};

/// Distributed Acoustic Sensing receiver: weighted sum of the displacement projected on the fiber.
struct DasProbe {
  DASType type = DASType::kNone;
  std::array<float, 3> projection = {1, 0, 0};  ///< Fiber direction, divided by the gauge length for a dipole.
  std::vector<int> nodes;                       ///< Global nodes of the integration points; -1 when unused.
  std::vector<float> weights;                   ///< Weight of each node.

  float Read(const vectorReal::host_mirror_type& ux, const vectorReal::host_mirror_type& uy,
             const vectorReal::host_mirror_type& uz) const {
    float value = 0.f;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      int const n = nodes[i];
      if (n >= 0) value += (ux(n) * projection[0] + uy(n) * projection[1] + uz(n) * projection[2]) * weights[i];
    }
    return value;
  }
};

// ComputeDASWeightsForSample() takes the order as a template parameter.
template <int ORDER>
void DasWeights(real_t const (&corners)[8][3], const std::array<float, 3>& coord, const std::array<float, 3>& direction,
                float integration_constant, DASType type, float* weights) {
  SourceAndReceiverUtils::ComputeDASWeightsForSample<ORDER>(corners, coord, direction, integration_constant, type,
                                                            weights);
}

DasProbe BuildDasProbe(const SemRun& run) {
  const SemOptions& opt = run.opt;
  DasProbe das;
  if (opt.das_type == "none") return das;

  das.type = (opt.das_type == "dipole") ? DASType::kDipole : DASType::kStrainIntegration;
  int const num_points = (das.type == DASType::kDipole) ? 2 : opt.das_samples;
  std::array<float, 3> const direction = SourceAndReceiverUtils::ComputeDASVector(opt.das_dip, opt.das_azimuth);
  das.projection = direction;
  if (das.type == DASType::kDipole) {
    for (float& p : das.projection) p /= opt.das_gauge_length;
  }

  std::array<float, 3> const receiver_coord = {opt.rcvx, opt.rcvy, opt.rcvz};
  int const order = opt.order;
  int const n1d = order + 1;
  int const num_dofs = n1d * n1d * n1d;
  das.nodes.assign(num_points * num_dofs, -1);
  das.weights.assign(num_points * num_dofs, 0.f);

  for (int p = 0; p < num_points; ++p) {
    // Points spread over the gauge length, centered on the receiver; a dipole takes the difference of its two ends.
    float const offset = (num_points == 1) ? 0.f : -0.5f + static_cast<float>(p) / (num_points - 1);
    float const integration_constant =
        (das.type == DASType::kDipole) ? (p == 0 ? -1.f : 1.f) : 1.f / static_cast<float>(num_points);
    std::array<float, 3> coord;
    for (int d = 0; d < 3; ++d) coord[d] = receiver_coord[d] + direction[d] * opt.das_gauge_length * offset;
    int element = examples::FindElement(run.params, coord);
    if (element < 0) element = 0;

    real_t corners[8][3];
    examples::ElementCorners(*run.mesh, element, corners);
    int const base = p * num_dofs;
    for (int k = 0; k < n1d; ++k) {
      for (int j = 0; j < n1d; ++j) {
        for (int i = 0; i < n1d; ++i) {
          das.nodes[base + i + j * n1d + k * n1d * n1d] = run.mesh->globalNodeIndex(element, i, j, k);
        }
      }
    }
    float* weights = &das.weights[base];
    switch (order) {
      case 1:
        DasWeights<1>(corners, coord, direction, integration_constant, das.type, weights);
        break;
      case 2:
        DasWeights<2>(corners, coord, direction, integration_constant, das.type, weights);
        break;
      case 3:
        DasWeights<3>(corners, coord, direction, integration_constant, das.type, weights);
        break;
      case 4:
        DasWeights<4>(corners, coord, direction, integration_constant, das.type, weights);
        break;
      case 5:
        DasWeights<5>(corners, coord, direction, integration_constant, das.type, weights);
        break;
      case 6:
        DasWeights<6>(corners, coord, direction, integration_constant, das.type, weights);
        break;
      default:
        throw std::runtime_error("DAS order " + std::to_string(order) + " not supported (1-6)");
    }
  }
  return das;
}

vectorReal ToDevice(const std::vector<float>& values, const std::string& label) {
  vectorReal view(label, values.size());
  auto h_view = Kokkos::create_mirror_view(view);
  for (std::size_t i = 0; i < values.size(); ++i) h_view(i) = values[i];
  Kokkos::deep_copy(view, h_view);
  return view;
}

// Quality factors on the model and standard linear solid (SLS) mechanisms on the solver.
void SetupAttenuation(const SemOptions& opt, examples::Mesh& mesh, solver::Solver& solver) {
  bool const has_q = (opt.qp > 0 || opt.qs > 0);
  if (has_q) mesh.setQualityFactors(opt.qp > 0 ? opt.qp : 1e9f, opt.qs > 0 ? opt.qs : 1e9f);

  if (!opt.sls_frequencies.empty()) {
    solver.setSLSAttenuation(ToDevice(opt.sls_frequencies, "slsFreqs"), ToDevice(opt.sls_coefficients, "slsCoeffs"));
  } else if (has_q) {
    // One mechanism at the dominant angular frequency of the source.
    solver.setSLSAttenuation(ToDevice({2.f * static_cast<float>(M_PI) * opt.f0}, "slsFreqAuto"));
  }
}

std::unique_ptr<solver::BoundarySynchronizer::Backend> MakeSyncBackend(int size) {
#ifdef USE_MPI
  if (size > 1) return std::make_unique<solver::MPIBackend>();
#endif
  return std::make_unique<solver::SerialBackend>();
}

/// One time step: forces, exchange of the force vectors between ranks, then update of the fields.
void AdvanceOneStep(SemRun& run, solver::Solver::DataStruct& data, int step) {
  auto const start = std::chrono::steady_clock::now();
  if (run.opt.Physics() == physicType::kAcoustoElastic) {
    run.solver->computeOneStep(run.dt, step, data);
  } else {
    run.solver->computeForces(run.dt, step, data);
    if (run.topology.isDistributed()) {
      for (int c = 0; c < run.solver->getNumComponents(); ++c) {
        run.syncer->synchronize(run.solver->getForceVector(c), run.topology);
      }
    }
    run.solver->updateSolutionForward(run.dt, data);
  }
  Kokkos::fence();
  run.timings.compute += examples::SecondsSince(start);
}

/// Copies @p field to a host buffer and writes its mid-z x-y plane in the background.
class SliceWriter {
 public:
  SliceWriter(const SemRun& run, const vectorReal& like, const std::string& field_name)
      : buffer_(Kokkos::create_mirror(like)), nx_(run.params.ex * run.opt.order + 1),
        ny_(run.params.ey * run.opt.order + 1), nz_(run.params.ez * run.opt.order + 1),
        prefix_("slice_" + field_name + (run.topology.isDistributed() ? "_r" + std::to_string(run.rank) : "")) {}

  void Write(examples::SnapshotWriter& writer, const vectorReal& field, int step) {
    Kokkos::deep_copy(buffer_, field);
    writer.Submit([buffer = buffer_, nx = nx_, ny = ny_, nz = nz_, name = examples::NumberedFileName(prefix_, step)] {
      examples::WriteNodalSliceZ(name, buffer, nx, ny, nz / 2);
    });
  }

 private:
  vectorReal::host_mirror_type buffer_;
  int nx_, ny_, nz_;
  std::string prefix_;
};

void RunAcoustic(SemRun& run) {
  const SemOptions& opt = run.opt;
  int const num_nodes = run.mesh->getNumberOfNodes();
  vectorReal p_prev = allocateVector<vectorReal>(num_nodes, "pnGlobalPrev");
  vectorReal p_curr = allocateVector<vectorReal>(num_nodes, "pnGlobalCurr");
  solver::WavefieldAcoustic wavefield(p_prev, p_curr);
  solver::RhsAcoustic rhs(examples::MakeSourceTerm(run.signal, run.source.on_rank, "rhsTerm"),
                          examples::MakeElementList(run.source.element), run.source.weights);
  solver::SEMsolverDataAcoustic data(wavefield, rhs);

  std::vector<float> trace(run.num_samples);
  auto h_p = Kokkos::create_mirror(p_prev);
  examples::SnapshotWriter snapshot_writer;
  SliceWriter slice_p(run, p_prev, "p");

  for (int step = 0; step < run.num_samples; ++step) {
    AdvanceOneStep(run, data, step);

    auto const start = std::chrono::steady_clock::now();
    // The new field is written into the previous-step buffer; swapWavefields() makes it the current one.
    vectorReal const p_new = data.getPreviousField(0);
    if (run.receiver.on_rank) {
      Kokkos::deep_copy(h_p, p_new);
      trace[step] = examples::InterpolateNodal(*run.mesh, h_p, run.receiver);
      if (step % 50 == 0) std::cout << "step " << step << "/" << run.num_samples << "  p(receiver)=" << trace[step] << "\n";
    }
    if (opt.snapshots && step % opt.snap_interval == 0) {
      snapshot_writer.Wait();
      slice_p.Write(snapshot_writer, p_new, step);
    }
    data.swapWavefields();
    run.timings.io += examples::SecondsSince(start);
  }

  snapshot_writer.Wait();
  if (run.receiver.on_rank) examples::WriteTraces("receiver_trace.txt", {"pressure"}, run.dt, {trace});
}

void RunElastic(SemRun& run) {
  const SemOptions& opt = run.opt;
  int const num_nodes = run.mesh->getNumberOfNodes();
  std::array<vectorReal, 3> u_prev, u_curr;
  for (int c = 0; c < 3; ++c) {
    u_prev[c] = allocateVector<vectorReal>(num_nodes, "unGlobalPrev");
    u_curr[c] = allocateVector<vectorReal>(num_nodes, "unGlobalCurr");
  }
  solver::WavefieldElastic wavefield(u_prev[0], u_curr[0], u_prev[1], u_curr[1], u_prev[2], u_curr[2]);
  // The same signal drives the three components of the force.
  bool const active = run.source.on_rank;
  solver::RhsElastic rhs(examples::MakeSourceTerm(run.signal, active, "rhsTermX"),
                         examples::MakeSourceTerm(run.signal, active, "rhsTermY"),
                         examples::MakeSourceTerm(run.signal, active, "rhsTermZ"),
                         examples::MakeElementList(run.source.element), run.source.weights);
  solver::SEMsolverDataElastic data(wavefield, rhs);

  DasProbe const das = BuildDasProbe(run);
  std::vector<std::vector<float>> traces(3, std::vector<float>(run.num_samples));
  std::vector<float> das_trace(run.num_samples);
  std::array<vectorReal::host_mirror_type, 3> h_u;
  for (int c = 0; c < 3; ++c) h_u[c] = Kokkos::create_mirror(u_prev[c]);
  examples::SnapshotWriter snapshot_writer;
  SliceWriter slice_ux(run, u_prev[0], "ux");

  for (int step = 0; step < run.num_samples; ++step) {
    AdvanceOneStep(run, data, step);

    auto const start = std::chrono::steady_clock::now();
    if (run.receiver.on_rank) {
      for (int c = 0; c < 3; ++c) {
        Kokkos::deep_copy(h_u[c], data.getPreviousField(c));
        traces[c][step] = examples::InterpolateNodal(*run.mesh, h_u[c], run.receiver);
      }
      if (das.type != DASType::kNone) das_trace[step] = das.Read(h_u[0], h_u[1], h_u[2]);
      if (step % 50 == 0) std::cout << "step " << step << "/" << run.num_samples << "  ux(receiver)=" << traces[0][step] << "\n";
    }
    if (opt.snapshots && step % opt.snap_interval == 0) {
      snapshot_writer.Wait();
      slice_ux.Write(snapshot_writer, data.getPreviousField(0), step);
    }
    data.swapWavefields();
    run.timings.io += examples::SecondsSince(start);
  }

  snapshot_writer.Wait();
  if (run.receiver.on_rank) {
    examples::WriteTraces("receiver_trace.txt", {"ux", "uy", "uz"}, run.dt, traces);
    if (das.type != DASType::kNone) examples::WriteTraces("das_trace.txt", {"das"}, run.dt, {das_trace});
  }
}

void RunAcoustoElastic(SemRun& run) {
  const SemOptions& opt = run.opt;
  int const num_nodes = run.mesh->getNumberOfNodes();
  // Field 0 is the pressure, fields 1-3 the displacement.
  std::array<vectorReal, 4> f_prev, f_curr;
  for (int f = 0; f < 4; ++f) {
    f_prev[f] = allocateVector<vectorReal>(num_nodes, "fieldGlobalPrev");
    f_curr[f] = allocateVector<vectorReal>(num_nodes, "fieldGlobalCurr");
  }
  solver::WavefieldAcoustoElastic wavefield(f_prev[0], f_curr[0], f_prev[1], f_curr[1], f_prev[2], f_curr[2],
                                            f_prev[3], f_curr[3]);
  // A source in the fluid is a pressure source; in the solid, a force with the same signal on each component.
  bool const source_in_fluid = (opt.srcz >= opt.acousto_elastic_boundary_z);
  bool const in_fluid = run.source.on_rank && source_in_fluid;
  bool const in_solid = run.source.on_rank && !source_in_fluid;
  solver::RhsAcoustoElastic rhs(examples::MakeSourceTerm(run.signal, in_fluid, "rhsTerm"),
                                examples::MakeElementList(run.source.element), run.source.weights,
                                examples::MakeSourceTerm(run.signal, in_solid, "rhsTermX"),
                                examples::MakeSourceTerm(run.signal, in_solid, "rhsTermY"),
                                examples::MakeSourceTerm(run.signal, in_solid, "rhsTermZ"));
  solver::SEMsolverDataAcoustoElastic data(wavefield, rhs);

  // A receiver in the fluid records the pressure, in the solid the displacement.
  bool const receiver_in_fluid = (opt.rcvz >= opt.acousto_elastic_boundary_z);
  int const first_field = receiver_in_fluid ? 0 : 1;
  int const num_traces = receiver_in_fluid ? 1 : 3;
  std::vector<std::vector<float>> traces(num_traces, std::vector<float>(run.num_samples));
  auto h_field = Kokkos::create_mirror(f_prev[0]);
  examples::SnapshotWriter snapshot_writer;
  SliceWriter slice_p(run, f_prev[0], "p");
  SliceWriter slice_uz(run, f_prev[3], "uz");

  for (int step = 0; step < run.num_samples; ++step) {
    AdvanceOneStep(run, data, step);

    auto const start = std::chrono::steady_clock::now();
    if (run.receiver.on_rank) {
      for (int t = 0; t < num_traces; ++t) {
        Kokkos::deep_copy(h_field, data.getPreviousField(first_field + t));
        traces[t][step] = examples::InterpolateNodal(*run.mesh, h_field, run.receiver);
      }
      if (step % 50 == 0) std::cout << "step " << step << "/" << run.num_samples << "  receiver=" << traces[0][step] << "\n";
    }
    if (opt.snapshots && step % opt.snap_interval == 0) {
      snapshot_writer.Wait();
      slice_p.Write(snapshot_writer, data.getPreviousField(0), step);
      slice_uz.Write(snapshot_writer, data.getPreviousField(3), step);
    }
    data.swapWavefields();
    run.timings.io += examples::SecondsSince(start);
  }

  snapshot_writer.Wait();
  if (run.receiver.on_rank) {
    if (receiver_in_fluid) {
      examples::WriteTraces("receiver_trace.txt", {"pressure"}, run.dt, traces);
    } else {
      examples::WriteTraces("receiver_trace.txt", {"ux", "uy", "uz"}, run.dt, traces);
    }
  }
}

void Run(const SemOptions& opt, const examples::Runtime& runtime) {
  auto const start_init = std::chrono::steady_clock::now();
  physicType const physics = opt.Physics();
  if (physics == physicType::kAcoustoElastic && runtime.size() > 1) {
    throw std::runtime_error("acousto-elastic runs are single-process");
  }

  SemRun run{opt};
  run.rank = runtime.rank();

  // Mesh and model of the subdomain of this rank.
  run.params = examples::MakeLocalParams(opt, opt.is_elastic, runtime.rank(), runtime.size());
  run.params.isAcoustoElastic = opt.is_acousto_elastic;
  run.params.acoustoElasticBoundaryZ = opt.acousto_elastic_boundary_z;
  auto const mesh_type = examples::ParseMeshType(opt.mesh);
  run.mesh = examples::BuildMesh(run.params, mesh_type, opt.free_surface);
  run.dt = examples::ComputeTimeStep(opt, *run.mesh);
  run.num_samples = static_cast<int>(opt.timemax / run.dt);

  // Solver, anisotropy and attenuation.
  run.solver = solver::solver_factory::createSolver(methodType::kSem, examples::ParseImplemType(opt.implem), mesh_type,
                                                    examples::ModelLocation(opt), physics, opt.order);
  if (physics != physicType::kAcoustic) {
    run.solver->setAnisotropyType(opt.Anisotropy());
    // On elements, the rotated TTI tensors are precomputed on the model.
    if (opt.Anisotropy() == model::AnisotropyType::kTTI && !opt.is_model_on_nodes) {
      run.mesh->initElasticityTensors(model::AnisotropyType::kTTI);
    }
  }
  SetupAttenuation(opt, *run.mesh, *run.solver);

  // Neighbor ranks and exchange of the nodes they share.
  run.topology = parallel::TopologyFactory::createFromMesh(*run.mesh, runtime.rank(), runtime.size(),
                                                           run.params.origin_x, run.params.lx);
  run.syncer = std::make_unique<solver::BoundarySynchronizer>(MakeSyncBackend(runtime.size()));

  // Source (Ricker wavelet) and receiver.
  SourceTimeFunction wavelet;
  run.signal = wavelet.computeSourceTerm(run.num_samples, run.dt, opt.f0, opt.ricker_order, opt.tpeak);
  run.source = examples::LocatePoint(*run.mesh, run.params, {opt.srcx, opt.srcy, opt.srcz}, opt.order);
  run.receiver = examples::LocatePoint(*run.mesh, run.params, {opt.rcvx, opt.rcvy, opt.rcvz}, opt.order);

  // Mass and stiffness operators, sponge layers. The diagonal mass and damping matrices are
  // summed over the ranks sharing a node.
  float const b = opt.boundaries_size;
  run.solver->computeFEInit(*run.mesh, {b, b, b}, opt.surface_sponge, opt.taper_delta);
  if (run.topology.isDistributed()) {
    if (physics != physicType::kElastic) run.syncer->synchronize(run.solver->getMassMatrixAcoustic(), run.topology);
    if (physics != physicType::kAcoustic) run.syncer->synchronize(run.solver->getMassMatrixElastic(), run.topology);
    for (int c = 0; c < run.solver->getNumComponents(); ++c) {
      run.syncer->synchronize(run.solver->getDampingMatrix(c), run.topology);
    }
  }
  run.timings.init = examples::SecondsSince(start_init);

  if (runtime.rank() == 0) {
    examples::PrintRunSummary("SEM", utils::enums::to_string(physics), opt, *run.mesh, run.dt, run.num_samples);
  }
  switch (physics) {
    case physicType::kAcoustic:
      RunAcoustic(run);
      break;
    case physicType::kElastic:
      RunElastic(run);
      break;
    case physicType::kAcoustoElastic:
      RunAcoustoElastic(run);
      break;
  }
  if (runtime.rank() == 0) examples::PrintTimings(run.timings);
}

}  // namespace

int main(int argc, char** argv) {
  examples::Runtime runtime(argc, argv);

  SemOptions opt;
  if (auto exit_code = examples::ParseOptions(argc, argv, "funtides-sem",
                                              "Acoustic, elastic or acousto-elastic propagation, SEM solver", opt)) {
    return *exit_code;
  }

  try {
    Run(opt, runtime);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
