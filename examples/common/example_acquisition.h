/**
 * @file example_acquisition.h
 * @brief Point sources and receivers of the examples: location, interpolation weights, source terms and readout.
 */

#ifndef FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_ACQUISITION_H_
#define FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_ACQUISITION_H_

#include <array>
#include <string>
#include <vector>

#include "data_type.h"
#include "example_mesh.h"

namespace examples {

/// A point of the domain, as seen by the solver: its element and its interpolation weights.
struct PointInElement {
  int element = 0;                        ///< Element containing the point; 0 when the point is not on this rank.
  bool on_rank = false;                   ///< True if the point lies in the local subdomain.
  arrayReal weights;                      ///< Weights of the point on the element nodes, shape (1, (order+1)^3).
  arrayReal::host_mirror_type h_weights;  ///< Host copy of weights.
};

/// Index of the element containing @p coord in the local subdomain, or -1 when the point is outside it.
int FindElement(const MeshParams& params, const std::array<float, 3>& coord);

/// Coordinates of the 8 corners of @p element, x fastest then y then z.
void ElementCorners(const Mesh& mesh, int element, real_t (&corners)[8][3]);

/**
 * @brief Locates @p coord in the local subdomain and computes its Lagrange interpolation weights.
 *
 * The weights are zero when the point is not on this rank.
 * @param[in] mesh Mesh of the subdomain.
 * @param[in] params Subdomain parameters, used to find the element.
 * @param[in] coord Global coordinates of the point.
 * @param[in] order Order of the weights; differs from the mesh order on the pMin side of the p-adaptive method.
 */
PointInElement LocatePoint(const Mesh& mesh, const MeshParams& params, const std::array<float, 3>& coord, int order);

/// Device list holding a single element index, as expected by the solver right-hand sides.
vectorInt MakeElementList(int element);

/// Source term of one source, shape (1, signal size): @p signal when @p active, zeros otherwise.
arrayReal MakeSourceTerm(const std::vector<float>& signal, bool active, const std::string& label);

/**
 * @brief Value at a point of a nodal (SEM) field already copied to the host.
 * @param[in] mesh Mesh, used to map element nodes to global nodes.
 * @param[in] field Host copy of the field.
 * @param[in] point Point and weights from LocatePoint().
 */
float InterpolateNodal(const Mesh& mesh, const vectorReal::host_mirror_type& field, const PointInElement& point);

/**
 * @brief Value at a point of an element-wise (DG) field of shape (elements, local dofs).
 *
 * Only the dofs of the point's element are copied to the host.
 */
float InterpolateElemental(const arrayReal& field, const PointInElement& point);

}  // namespace examples

#endif  // FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_ACQUISITION_H_
