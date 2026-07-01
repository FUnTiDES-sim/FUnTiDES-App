#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# env.sh  —  Setup environment for makutu + FUnTiDES
#
# Usage:
#   source env.sh
# ---------------------------------------------------------------------------

# Resolve the directory containing this script (works with `source`)
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. FUnTiDES paths (pyfuntides, kokkos, LD_LIBRARY_PATH)
# ---------------------------------------------------------------------------
source "${_ENV_SH_DIR}/../proxy-FUn/FUn-Libs/envpython.sh"

# ---------------------------------------------------------------------------
# 2. CUDA: pin to a single GPU so Kokkos UVM allocations work correctly.
#    On multi-GPU systems, UVM synchronisation across all devices causes
#    cudaErrorIllegalAddress inside CudaUVMSpace::impl_allocate.
#    Override by setting CUDA_VISIBLE_DEVICES before sourcing this file.
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---------------------------------------------------------------------------
# 3. makutu (top-level root so `import utilities`, `import workflows` work)
# ---------------------------------------------------------------------------
export PYTHONPATH="${_ENV_SH_DIR}:${PYTHONPATH}"
