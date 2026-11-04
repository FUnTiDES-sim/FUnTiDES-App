/**
 * @file example_output.h
 * @brief Text outputs of the examples (receiver traces, field slices), background writing and timing report.
 */

#ifndef FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_OUTPUT_H_
#define FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_OUTPUT_H_

#include <chrono>
#include <functional>
#include <future>
#include <string>
#include <vector>

#include "data_type.h"
#include "example_mesh.h"
#include "example_options.h"

namespace examples {

/**
 * @brief Runs snapshot writes on background threads so that the time loop does not wait for the disk.
 *
 * Tasks must only read host buffers that the time loop does not touch until Wait() returns.
 */
class SnapshotWriter {
 public:
  SnapshotWriter() = default;
  SnapshotWriter(const SnapshotWriter&) = delete;
  SnapshotWriter& operator=(const SnapshotWriter&) = delete;
  ~SnapshotWriter() { Wait(); }

  /// Starts @p task on a background thread.
  void Submit(std::function<void()> task);

  /// Blocks until every submitted task is done.
  void Wait();

 private:
  std::vector<std::future<void>> pending_;
};

/// "<prefix>_<step on 5 digits>.dat".
std::string NumberedFileName(const std::string& prefix, int step);

/**
 * @brief Writes receiver traces, one line per time step: time, then one value per trace.
 * @param[in] path Output file.
 * @param[in] names Column names of the traces, written in the header line.
 * @param[in] dt Time step; traces[c][n] is the value at time (n + 1) * dt.
 * @param[in] traces One vector per trace, all of the same size.
 */
void WriteTraces(const std::string& path, const std::vector<std::string>& names, float dt,
                 const std::vector<std::vector<float>>& traces);

/**
 * @brief Writes the x-y plane of index @p iz of a nodal field on a structured node grid, one line per y.
 */
void WriteNodalSliceZ(const std::string& path, const vectorReal::host_mirror_type& field, int nx, int ny, int iz);

/**
 * @brief Writes an x-y plane of an element-wise field, one line per y node row.
 * @param[in] field Host copy of the field, shape (elements, n1d^3), elements x fastest then y then z.
 * @param[in] ex,ey Number of elements along x and y.
 * @param[in] n1d Number of nodes per direction in an element.
 * @param[in] element_layer Index along z of the element layer.
 * @param[in] node_layer Index along z of the node plane inside the element layer.
 */
void WriteElementalSliceZ(const std::string& path, const arrayReal::host_mirror_type& field, int ex, int ey, int n1d,
                          int element_layer, int node_layer);

/// Wall-clock time spent in each phase of a run, in seconds.
struct Timings {
  double init = 0.0;     ///< Mesh, solver and finite element initialization.
  double compute = 0.0;  ///< Solver time steps.
  double io = 0.0;       ///< Receiver readout and outputs.
};

/// Seconds elapsed since @p start.
inline double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// Prints the configuration of a run before the time loop.
void PrintRunSummary(const std::string& method, const std::string& physics, const CommonOptions& opt,
                     const Mesh& mesh, float dt, int num_samples);

/// Prints the time spent in each phase.
void PrintTimings(const Timings& timings);

}  // namespace examples

#endif  // FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_OUTPUT_H_
