#include "example_mesh.h"

#include <stdexcept>
#include <string>

#include "cartesian_partitioner.h"
#include "cartesian_struct_builder.h"
#include "cartesian_unstruct_builder.h"

namespace examples {

namespace {

// The structured builder takes the order as a template parameter.
template <int ORDER>
std::shared_ptr<Mesh> BuildStructMesh(const MeshParams& p, bool free_surface) {
  // Global sizes of -1 mean "same as the local subdomain".
  model::CartesianStructBuilder<float, int, ORDER> builder(
      p.ex, p.lx, p.ey, p.ly, p.ez, p.lz, p.isModelOnNodes, p.isElastic, p.origin_x, p.origin_y, p.origin_z, -1.f,
      -1.f, -1.f, 0.f, 0.f, 0.f, p.isAcoustoElastic, p.acoustoElasticBoundaryZ, p.DgSemBoundaryZ, p.model_file);
  return builder.getModel(free_surface);
}

}  // namespace

MeshParams MakeLocalParams(const CommonOptions& opt, bool is_elastic, int rank, int size) {
  MeshParams global(opt.order, opt.ex, opt.ey, opt.ez, opt.lx, opt.ly, opt.lz, opt.is_model_on_nodes, is_elastic);
  model::CartesianXPartitioner<float, int> partitioner;
  MeshParams local = partitioner.partition(global, rank, size);
  local.model_file = opt.model_file;
  return local;
}

std::shared_ptr<Mesh> BuildMesh(const MeshParams& params, utils::enums::meshType mesh_type, bool free_surface) {
  if (mesh_type == utils::enums::meshType::kUnstruct) {
    model::CartesianUnstructBuilder<float, int> builder(params);
    return builder.getModel(free_surface);
  }
  switch (params.order) {
    case 1:
      return BuildStructMesh<1>(params, free_surface);
    case 2:
      return BuildStructMesh<2>(params, free_surface);
    case 3:
      return BuildStructMesh<3>(params, free_surface);
    case 4:
      return BuildStructMesh<4>(params, free_surface);
    case 5:
      return BuildStructMesh<5>(params, free_surface);
    case 6:
      return BuildStructMesh<6>(params, free_surface);
    default:
      throw std::runtime_error("Structured mesh order " + std::to_string(params.order) + " not supported (1-6)");
  }
}

float ComputeTimeStep(const CommonOptions& opt, const Mesh& mesh) {
  if (!opt.auto_dt) return opt.dt;
  float const cfl_factor = (opt.order == 2) ? 0.5f : 0.7f;
  float const sqrt_dim = 1.73f;
  return cfl_factor * mesh.getMinSpacing() / (sqrt_dim * mesh.getMaxSpeed());
}

}  // namespace examples
