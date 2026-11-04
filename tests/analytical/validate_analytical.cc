/**
 * @file validate_analytical.cc
 * @brief Runs one acoustic example and compares its receiver trace with the exact solution.
 *
 * The medium is the default homogeneous model (Vp = 1500 m/s, rho = 1) of a 2000 m cube.
 * For a point source f(t) the solved equation (1/(rho c^2)) p'' - div((1/rho) grad p) = f(t) delta(x - xs)
 * has the closed-form solution p(x, t) = rho f(t - r/c) / (4 pi r), r = |x - xs|, compared
 * without normalization. The cases are those of the FUnTiDES Python validation
 * (tests/validation/cases.py): same positions, meshes, time steps and tolerances.
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "source_time_function.h"

#ifndef FUNTIDES_BIN_DIR
#define FUNTIDES_BIN_DIR "./bin"
#endif

namespace {

constexpr double kDomainSize = 2000.0;
constexpr double kVp = 1500.0;
constexpr double kRho = 1.0;
constexpr float kF0 = 5.f;
constexpr int kRickerOrder = 2;
constexpr float kTpeak = 0.2f;
constexpr double kTimeMax = 0.7;
constexpr double kPi = 3.14159265358979323846;

struct Case {
  std::string method = "sem";  // sem | dg | dg-sem | dg-padaptive
  std::string mesh = "cartesian";
  int order = 1;
  int order_min = 0;  // pMin of dg-padaptive
  double tolerance = 0.0;

  bool IsCoupled() const { return method == "dg-sem" || method == "dg-padaptive"; }

  // Coupled solvers split the domain at z = 1000 m (their default): the source lies 150 m below the
  // interface and the receiver 165 m above, off the element faces where DG fields are discontinuous.
  std::vector<double> Source() const {
    return IsCoupled() ? std::vector<double>{1010, 1010, 850} : std::vector<double>{1010, 1010, 1010};
  }
  std::vector<double> Receiver() const {
    return IsCoupled() ? std::vector<double>{1310, 1310, 1165} : std::vector<double>{1310, 1310, 1310};
  }

  // Same number of nodes per direction for every order; DG fields have no shared nodes, hence fewer.
  int ElementsPerDirection() const { return (method == "sem" ? 400 : 300) / (order + 1); }

  // SEM: 1 ms. DG: stable only for dt ~ h / (vp * order^2), scaled by a measured factor and floored to 1e-5 s.
  double TimeStep() const {
    if (method == "sem") return 0.001;
    double const cfl = (order == 1) ? 0.15 : 0.2;
    double const h = kDomainSize / ElementsPerDirection();
    return std::floor(cfl * h / (kVp * order * order) / 1e-5) * 1e-5;
  }

  std::string Command() const {
    int const n = ElementsPerDirection();
    std::ostringstream cmd;
    cmd << std::setprecision(9) << FUNTIDES_BIN_DIR << "/funtides-" << method << " --order " << order
        << " --mesh=" << mesh << " --ex " << n << " --ey " << n << " --ez " << n << " --dt " << TimeStep()
        << " --timemax " << kTimeMax << " --f0 " << kF0 << " --ricker-order " << kRickerOrder << " --tpeak "
        << kTpeak;
    auto const src = Source();
    auto const rcv = Receiver();
    cmd << " --srcx " << src[0] << " --srcy " << src[1] << " --srcz " << src[2] << " --rcvx " << rcv[0] << " --rcvy "
        << rcv[1] << " --rcvz " << rcv[2];
    if (method == "dg-padaptive") cmd << " --order-min " << order_min;
    return cmd.str();
  }
};

// Times (column 0) and pressures (column 1) of receiver_trace.txt.
void ReadTrace(const std::string& path, std::vector<double>& time, std::vector<double>& pressure) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open " + path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    double t = 0.0, p = 0.0;
    if (row >> t >> p) {
      time.push_back(t);
      pressure.push_back(p);
    }
  }
  if (time.empty()) throw std::runtime_error(path + " holds no sample");
}

double AnalyticalPressure(double t, double r) {
  SourceTimeFunction wavelet;
  return kRho * wavelet.evaluateRicker(static_cast<float>(t - r / kVp), kF0, kRickerOrder, kTpeak) / (4.0 * kPi * r);
}

Case ParseCase(int argc, char** argv) {
  Case c;
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    bool const has_value = (i + 1 < argc);
    if (arg == "--method" && has_value) {
      c.method = argv[++i];
    } else if (arg == "--mesh" && has_value) {
      c.mesh = argv[++i];
    } else if (arg == "--order" && has_value) {
      c.order = std::stoi(argv[++i]);
    } else if (arg == "--order-min" && has_value) {
      c.order_min = std::stoi(argv[++i]);
    } else if (arg == "--tolerance" && has_value) {
      c.tolerance = std::stod(argv[++i]);
    } else {
      throw std::invalid_argument("Unknown argument " + arg +
                                  "; usage: --method M --mesh M --order N [--order-min N] --tolerance T");
    }
  }
  if (c.tolerance <= 0) throw std::invalid_argument("--tolerance must be > 0");
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Case const c = ParseCase(argc, argv);
    std::string const command = c.Command();
    std::cout << "Running: " << command << std::endl;
    if (std::system(command.c_str()) != 0) {
      std::cerr << "Solver run failed" << std::endl;
      return EXIT_FAILURE;
    }

    std::vector<double> time, numerical;
    ReadTrace("receiver_trace.txt", time, numerical);

    auto const src = c.Source();
    auto const rcv = c.Receiver();
    double const r = std::sqrt((rcv[0] - src[0]) * (rcv[0] - src[0]) + (rcv[1] - src[1]) * (rcv[1] - src[1]) +
                               (rcv[2] - src[2]) * (rcv[2] - src[2]));
    double error = 0.0, norm = 0.0;
    for (std::size_t n = 0; n < time.size(); ++n) {
      double const reference = AnalyticalPressure(time[n], r);
      error += (numerical[n] - reference) * (numerical[n] - reference);
      norm += reference * reference;
    }
    double const relative_l2 = std::sqrt(error / norm);

    bool const passed = relative_l2 < c.tolerance;
    std::cout << std::scientific << std::setprecision(3) << "\n" << c.method << " order " << c.order;
    if (c.order_min > 0) std::cout << " (order-min " << c.order_min << ")";
    std::cout << " " << c.mesh << ", " << time.size() << " samples, dt " << c.TimeStep() << " s\n"
              << "Relative L2 error: " << relative_l2 << "  (tolerance " << c.tolerance << ")  "
              << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
