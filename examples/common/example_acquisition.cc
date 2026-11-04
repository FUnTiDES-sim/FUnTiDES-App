#include "example_acquisition.h"

#include <Kokkos_Core.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

#include "source_and_receiver_utils.h"

namespace examples {

namespace {

// ComputeRHSWeights() takes the order as a template parameter.
void ComputeWeights(int order, real_t const (&corners)[8][3], const std::array<float, 3>& coord,
                    arrayReal::host_mirror_type& weights) {
  switch (order) {
    case 1:
      SourceAndReceiverUtils::ComputeRHSWeights<1>(corners, coord, weights);
      break;
    case 2:
      SourceAndReceiverUtils::ComputeRHSWeights<2>(corners, coord, weights);
      break;
    case 3:
      SourceAndReceiverUtils::ComputeRHSWeights<3>(corners, coord, weights);
      break;
    case 4:
      SourceAndReceiverUtils::ComputeRHSWeights<4>(corners, coord, weights);
      break;
    case 5:
      SourceAndReceiverUtils::ComputeRHSWeights<5>(corners, coord, weights);
      break;
    case 6:
      SourceAndReceiverUtils::ComputeRHSWeights<6>(corners, coord, weights);
      break;
    default:
      throw std::runtime_error("Interpolation order " + std::to_string(order) + " not supported (1-6)");
  }
}

}  // namespace

int FindElement(const MeshParams& params, const std::array<float, 3>& coord) {
  float const rel[3] = {coord[0] - params.origin_x, coord[1] - params.origin_y, coord[2] - params.origin_z};
  int const num[3] = {params.ex, params.ey, params.ez};
  float const size[3] = {params.lx, params.ly, params.lz};
  int idx[3];
  for (int d = 0; d < 3; ++d) {
    idx[d] = static_cast<int>(std::floor(rel[d] * num[d] / size[d]));
    if (rel[d] < 0 || idx[d] >= num[d]) return -1;
  }
  return idx[0] + idx[1] * num[0] + idx[2] * num[0] * num[1];
}

void ElementCorners(const Mesh& mesh, int element, real_t (&corners)[8][3]) {
  int const ends[2] = {0, mesh.getOrder()};
  int c = 0;
  for (int k : ends) {
    for (int j : ends) {
      for (int i : ends) {
        int const node = mesh.globalNodeIndex(element, i, j, k);
        for (int d = 0; d < 3; ++d) corners[c][d] = mesh.nodeCoord(node, d);
        ++c;
      }
    }
  }
}

PointInElement LocatePoint(const Mesh& mesh, const MeshParams& params, const std::array<float, 3>& coord, int order) {
  PointInElement point;
  int const element = FindElement(params, coord);
  point.on_rank = (element >= 0);
  point.element = point.on_rank ? element : 0;

  int const n1d = order + 1;
  point.weights = arrayReal("pointWeights", 1, n1d * n1d * n1d);
  point.h_weights = Kokkos::create_mirror_view(point.weights);
  Kokkos::deep_copy(point.h_weights, 0.f);

  if (point.on_rank) {
    real_t corners[8][3];
    ElementCorners(mesh, point.element, corners);
    ComputeWeights(order, corners, coord, point.h_weights);
  }
  Kokkos::deep_copy(point.weights, point.h_weights);
  return point;
}

vectorInt MakeElementList(int element) {
  vectorInt list("elementList", 1);
  auto h_list = Kokkos::create_mirror_view(list);
  h_list(0) = element;
  Kokkos::deep_copy(list, h_list);
  return list;
}

arrayReal MakeSourceTerm(const std::vector<float>& signal, bool active, const std::string& label) {
  int const num_samples = static_cast<int>(signal.size());
  arrayReal term(label, 1, num_samples);
  if (active) {
    auto h_term = Kokkos::create_mirror_view(term);
    for (int t = 0; t < num_samples; ++t) h_term(0, t) = signal[t];
    Kokkos::deep_copy(term, h_term);
  }
  return term;
}

float InterpolateNodal(const Mesh& mesh, const vectorReal::host_mirror_type& field, const PointInElement& point) {
  int const n1d = mesh.getOrder() + 1;
  float value = 0.f;
  for (int k = 0; k < n1d; ++k) {
    for (int j = 0; j < n1d; ++j) {
      for (int i = 0; i < n1d; ++i) {
        int const node = mesh.globalNodeIndex(point.element, i, j, k);
        value += field(node) * point.h_weights(0, i + j * n1d + k * n1d * n1d);
      }
    }
  }
  return value;
}

float InterpolateElemental(const arrayReal& field, const PointInElement& point) {
  auto dofs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                  Kokkos::subview(field, point.element, Kokkos::ALL()));
  float value = 0.f;
  for (int d = 0; d < static_cast<int>(dofs.extent(0)); ++d) value += dofs(d) * point.h_weights(0, d);
  return value;
}

}  // namespace examples
