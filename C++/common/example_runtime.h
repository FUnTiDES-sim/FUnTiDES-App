/**
 * @file example_runtime.h
 * @brief Initialization and finalization of MPI and Kokkos for the examples.
 */

#ifndef FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_RUNTIME_H_
#define FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_RUNTIME_H_

#include <Kokkos_Core.hpp>
#include <cstdlib>
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace examples {

/**
 * @brief Initializes MPI (when built with USE_MPI) then Kokkos, and finalizes them in reverse order.
 *
 * Create it first in main(); every Kokkos view must be released before it is destroyed.
 */
class Runtime {
 public:
  Runtime(int& argc, char**& argv) {
    // Thread binding of the OpenMP back-end; values already set by the user are kept.
    setenv("OMP_PROC_BIND", "spread", 0);
    setenv("OMP_PLACES", "threads", 0);
#ifdef USE_MPI
    int provided = 0;
    // Snapshots are written from background threads.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      std::cerr << "WARNING: MPI_THREAD_MULTIPLE not supported, asynchronous snapshots may be unstable.\n";
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
#endif
    Kokkos::initialize(argc, argv);
  }

  ~Runtime() {
    Kokkos::finalize();
#ifdef USE_MPI
    MPI_Finalize();
#endif
  }

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  int rank() const { return rank_; }  ///< MPI rank, 0 without MPI.
  int size() const { return size_; }  ///< Number of MPI ranks, 1 without MPI.

 private:
  int rank_ = 0;
  int size_ = 1;
};

}  // namespace examples

#endif  // FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_RUNTIME_H_
