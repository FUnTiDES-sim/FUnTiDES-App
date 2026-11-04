/**
 * @file example_options.h
 * @brief Command-line options shared by every FUnTiDES example, and their parsing.
 */

#ifndef FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_OPTIONS_H_
#define FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_OPTIONS_H_

#include <cstdlib>
#include <cxxopts.hpp>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "sem_enums.h"

namespace examples {

/**
 * @brief Options common to every method: mesh, time stepping, source, receiver and sponge layers.
 *
 * Lengths are in meters and times in seconds. An example that needs more flags derives
 * its own option set from this one and redefines BindCli() and Validate().
 */
struct CommonOptions {
  int order = 2;                                      ///< Polynomial order of the elements.
  int ex = 50, ey = 50, ez = 50;                      ///< Number of elements along x, y, z.
  float lx = 2000.f, ly = 2000.f, lz = 2000.f;        ///< Domain size along x, y, z.
  float srcx = 1010.f, srcy = 1010.f, srcz = 1010.f;  ///< Source position.
  float rcvx = 1310.f, rcvy = 1310.f, rcvz = 1310.f;  ///< Receiver position.
  float f0 = 5.f;                                     ///< Dominant frequency of the Ricker source (Hz).
  int ricker_order = 2;                               ///< Derivative order of the Ricker wavelet.
  float tpeak = 0.2f;                                 ///< Peak time of the Ricker wavelet.

  std::string implem = "makutu";   ///< Integral back-end: makutu.
  std::string mesh = "cartesian";  ///< Mesh kind: cartesian | ucartesian.
  float dt = 0.006f;               ///< Time step, ignored when auto_dt is set.
  float timemax = 0.7f;            ///< Simulated duration.
  bool auto_dt = false;            ///< Derive the time step from a CFL condition.
  bool snapshots = false;          ///< Write field slices during the run.
  int snap_interval = 20;          ///< Number of time steps between two slices.
  float boundaries_size = 0.f;     ///< Thickness of the sponge layers.
  bool surface_sponge = false;     ///< Forwarded to Solver::computeFEInit().
  float taper_delta = 0.015f;      ///< Taper coefficient of the sponge layers.
  bool is_model_on_nodes = false;  ///< Model given on nodes (true) or on elements (false).
  bool free_surface = false;       ///< Free surface on the top boundary (z+).
  std::string model_file;          ///< Heterogeneous .ftmd model file; empty for a homogeneous model.

  /**
   * @brief Registers the common flags.
   * @param[in,out] cli Parser receiving the flags.
   * @param[in,out] o Instance written by the parser; must outlive the parsing.
   */
  static void BindCli(cxxopts::Options& cli, CommonOptions& o) {
    cli.add_options()
        ("o,order", "Polynomial order of the elements", cxxopts::value<int>(o.order))
        ("ex", "Number of elements along x", cxxopts::value<int>(o.ex))
        ("ey", "Number of elements along y", cxxopts::value<int>(o.ey))
        ("ez", "Number of elements along z", cxxopts::value<int>(o.ez))
        ("lx", "Domain size along x (m)", cxxopts::value<float>(o.lx))
        ("ly", "Domain size along y (m)", cxxopts::value<float>(o.ly))
        ("lz", "Domain size along z (m)", cxxopts::value<float>(o.lz))
        ("srcx", "Source position x (m)", cxxopts::value<float>(o.srcx))
        ("srcy", "Source position y (m)", cxxopts::value<float>(o.srcy))
        ("srcz", "Source position z (m)", cxxopts::value<float>(o.srcz))
        ("rcvx", "Receiver position x (m)", cxxopts::value<float>(o.rcvx))
        ("rcvy", "Receiver position y (m)", cxxopts::value<float>(o.rcvy))
        ("rcvz", "Receiver position z (m)", cxxopts::value<float>(o.rcvz))
        ("f0", "Dominant frequency of the source (Hz)", cxxopts::value<float>(o.f0))
        ("ricker-order", "Derivative order of the Ricker wavelet", cxxopts::value<int>(o.ricker_order))
        ("tpeak", "Peak time of the Ricker wavelet (s)", cxxopts::value<float>(o.tpeak))
        ("implem", "Integral back-end: makutu", cxxopts::value<std::string>(o.implem))
        ("mesh", "Mesh: cartesian|ucartesian", cxxopts::value<std::string>(o.mesh))
        ("dt", "Time step (s, default 0.006)", cxxopts::value<float>(o.dt))
        ("timemax", "Simulated duration (s, default 0.7)", cxxopts::value<float>(o.timemax))
        ("auto-dt", "Derive the time step from a CFL condition", cxxopts::value<bool>(o.auto_dt))
        ("s,snapshots", "Write field slices during the run", cxxopts::value<bool>(o.snapshots))
        ("snap-interval", "Time steps between two slices (default 20)", cxxopts::value<int>(o.snap_interval))
        ("boundaries-size", "Thickness of the sponge layers (m)", cxxopts::value<float>(o.boundaries_size))
        ("sponge-surface", "Treat the top surface nodes as non-sponge nodes",
         cxxopts::value<bool>(o.surface_sponge))
        ("taper-delta", "Taper coefficient of the sponge layers", cxxopts::value<float>(o.taper_delta))
        ("is-model-on-nodes", "Model given on nodes (true) or on elements (false)",
         cxxopts::value<bool>(o.is_model_on_nodes))
        ("free-surface", "Free surface on the top boundary (z+), default false",
         cxxopts::value<bool>(o.free_surface))
        ("model-file", "Heterogeneous .ftmd model file (optional)", cxxopts::value<std::string>(o.model_file));
  }

  /**
   * @brief Checks the sizes, the time stepping and the snapshot interval.
   * @throws std::runtime_error on the first invalid value.
   */
  void Validate() const {
    if (order < 1) throw std::runtime_error("order must be >= 1");
    if (ex <= 0 || ey <= 0 || ez <= 0) throw std::runtime_error("ex/ey/ez must be > 0");
    if (lx <= 0 || ly <= 0 || lz <= 0) throw std::runtime_error("lx/ly/lz must be > 0");
    if (!auto_dt && dt <= 0) throw std::runtime_error("dt must be > 0");
    if (timemax <= 0) throw std::runtime_error("timemax must be > 0");
    if (snapshots && snap_interval <= 0) throw std::runtime_error("snap-interval must be > 0");
    auto inside = [this](float x, float y, float z) {
      return x >= 0 && x < lx && y >= 0 && y < ly && z >= 0 && z < lz;
    };
    if (!inside(srcx, srcy, srcz)) throw std::runtime_error("the source must lie inside the domain");
    if (!inside(rcvx, rcvy, rcvz)) throw std::runtime_error("the receiver must lie inside the domain");
  }
};

/**
 * @brief Parses the command line into @p opt.
 *
 * Unknown flags are accepted so that Kokkos flags pass through. @p Options provides a
 * static BindCli(cxxopts::Options&, Options&) and a Validate() const member.
 *
 * @return An exit code when the program must stop (help printed or invalid options), nothing otherwise.
 */
template <typename Options>
std::optional<int> ParseOptions(int argc, char** argv, const std::string& program, const std::string& description,
                                Options& opt) {
  cxxopts::Options cli(program, description);
  cli.allow_unrecognised_options();
  cli.add_options()("h,help", "Print this help");
  Options::BindCli(cli, opt);

  try {
    auto result = cli.parse(argc, argv);
    if (result.count("help")) {
      std::cout << cli.help() << std::endl;
      return EXIT_SUCCESS;
    }
    opt.Validate();
  } catch (const std::exception& e) {
    std::cerr << "Invalid options: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return std::nullopt;
}

/// Mesh kind of a --mesh value.
inline utils::enums::meshType ParseMeshType(const std::string& name) {
  if (name == "cartesian") return utils::enums::meshType::kStruct;
  if (name == "ucartesian") return utils::enums::meshType::kUnstruct;
  throw std::invalid_argument("Unknown mesh '" + name + "': expected cartesian or ucartesian");
}

/// Integral back-end of an --implem value.
inline utils::enums::implemType ParseImplemType(const std::string& name) {
  if (name == "makutu") return utils::enums::implemType::kMakutu;
  throw std::invalid_argument("Unknown implementation '" + name + "': only makutu is supported");
}

/// Where the model parameters are stored.
inline utils::enums::modelLocationType ModelLocation(const CommonOptions& opt) {
  return opt.is_model_on_nodes ? utils::enums::modelLocationType::kOnNodes
                               : utils::enums::modelLocationType::kOnElements;
}

}  // namespace examples

#endif  // FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_OPTIONS_H_
