/**
 * @file example_mesh.h
 * @brief Cartesian mesh construction and time step selection for the examples.
 */

#ifndef FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_MESH_H_
#define FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_MESH_H_

#include <memory>
#include <string>

#include "cartesian_params.h"
#include "example_options.h"
#include "model.h"
#include "sem_enums.h"

namespace examples {

using Mesh = model::ModelApi<float, int>;
using MeshParams = model::CartesianParams<float, int>;

/**
 * @brief Parameters of the subdomain owned by @p rank, the domain being split along x.
 *
 * Physics-specific fields (isAcoustoElastic, acoustoElasticBoundaryZ) keep their defaults;
 * set them on the result before BuildMesh().
 */
MeshParams MakeLocalParams(const CommonOptions& opt, bool is_elastic, int rank, int size);

/**
 * @brief Builds the mesh and model of a subdomain.
 * @param[in] params Subdomain parameters, see MakeLocalParams().
 * @param[in] mesh_type Structured or unstructured storage of the cartesian mesh.
 * @param[in] free_surface Free surface on the top boundary.
 * @throws std::runtime_error for a structured mesh of order outside 1-6.
 */
std::shared_ptr<Mesh> BuildMesh(const MeshParams& params, utils::enums::meshType mesh_type, bool free_surface);

/**
 * @brief Time step: --dt, or a CFL estimate from the smallest node spacing and the largest speed with --auto-dt.
 */
float ComputeTimeStep(const CommonOptions& opt, const Mesh& mesh);

}  // namespace examples

#endif  // FUNTIDES_APP_EXAMPLES_COMMON_EXAMPLE_MESH_H_
