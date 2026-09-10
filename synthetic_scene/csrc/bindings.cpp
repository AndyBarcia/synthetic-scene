#include <torch/extension.h>

#include <c10/cuda/CUDAFunctions.h>
#include <vector>

namespace py = pybind11;
namespace {

constexpr int kRandomSceneMaxSpheres = 64;
constexpr int kRandomSceneMaxBoxes = 64;
constexpr int kRandomSceneMaxPrisms = 64;
constexpr int kRandomSceneMaxCylinders = 64;
}  // namespace

void render_scene_cuda(
    torch::Tensor image,
    torch::Tensor instance_map,
    torch::Tensor semantic_map,
    torch::Tensor visible_primitive_masks,
    torch::Tensor sphere_centers,
    torch::Tensor sphere_radii,
    torch::Tensor sphere_counts,
    torch::Tensor sphere_class_ids,
    torch::Tensor sphere_instance_ids,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor plane_counts,
    torch::Tensor terrain_base_heights,
    torch::Tensor terrain_depth_limits,
    torch::Tensor terrain_phase_xs,
    torch::Tensor terrain_phase_zs,
    torch::Tensor terrain_dz,
    torch::Tensor terrain_dz_growth,
    torch::Tensor terrain_counts,
    torch::Tensor terrain_class_ids,
    torch::Tensor terrain_instance_ids,
    torch::Tensor box_centers,
    torch::Tensor box_half_sizes,
    torch::Tensor box_axes,
    torch::Tensor box_counts,
    torch::Tensor box_class_ids,
    torch::Tensor box_instance_ids,
    torch::Tensor prism_centers,
    torch::Tensor prism_half_sizes,
    torch::Tensor prism_axes,
    torch::Tensor prism_counts,
    torch::Tensor prism_class_ids,
    torch::Tensor prism_instance_ids,
    torch::Tensor cylinder_centers,
    torch::Tensor cylinder_radii,
    torch::Tensor cylinder_half_heights,
    torch::Tensor cylinder_axes,
    torch::Tensor cylinder_counts,
    torch::Tensor cylinder_class_ids,
    torch::Tensor cylinder_instance_ids,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors,
    torch::Tensor plane_colors,
    torch::Tensor terrain_colors,
    torch::Tensor box_colors,
    torch::Tensor prism_colors,
    torch::Tensor cylinder_colors,
    double ambient,
    bool shadows,
    double shadow_strength);

py::dict require_dict(py::dict values, const char* key) {
  const py::str py_key(key);
  TORCH_CHECK(values.contains(py_key), "missing dictionary key: ", key);
  return values[py_key].cast<py::dict>();
}

torch::Tensor require_tensor(py::dict values, const char* key) {
  const py::str py_key(key);
  TORCH_CHECK(values.contains(py_key), "missing dictionary key: ", key);
  return values[py_key].cast<torch::Tensor>();
}

double require_double(py::dict values, const char* key) {
  const py::str py_key(key);
  TORCH_CHECK(values.contains(py_key), "missing dictionary key: ", key);
  return values[py_key].cast<double>();
}

bool require_bool(py::dict values, const char* key) {
  const py::str py_key(key);
  TORCH_CHECK(values.contains(py_key), "missing dictionary key: ", key);
  return values[py_key].cast<bool>();
}

void random_scene_cuda(
    const std::vector<torch::Tensor>& outputs,
    int64_t seed,
    float scatter_radius,
    float ground_y,
    float depth_limit,
    float dz,
    float dz_growth,
    float fov_degrees,
    float aspect_ratio,
    int house_count,
    int tree_count,
    int cloud_count,
    int car_count,
    int person_count);

struct PackedViews {
  std::vector<torch::Tensor> scene;
  torch::Tensor plane_counts;
  torch::Tensor terrain_class_ids;
  torch::Tensor terrain_instance_ids;
};

PackedViews unpack_scene(
    torch::Tensor float_data,
    torch::Tensor integer_data,
    int64_t batch_size,
    int64_t sphere_count,
    int64_t box_count,
    int64_t prism_count,
    int64_t cylinder_count) {
  int64_t float_offset = 0;
  int64_t integer_offset = 0;
  auto take_float = [&](std::vector<int64_t> shape) {
    const int64_t size = c10::multiply_integers(shape);
    auto view = float_data.narrow(0, float_offset, size).view(shape);
    float_offset += size;
    return view;
  };
  auto take_integer = [&](std::vector<int64_t> shape) {
    const int64_t size = c10::multiply_integers(shape);
    auto view = integer_data.narrow(0, integer_offset, size).view(shape);
    integer_offset += size;
    return view;
  };

  std::vector<torch::Tensor> scene = {
      take_float({batch_size, sphere_count, 3}),
      take_float({batch_size, sphere_count}),
      take_float({batch_size, sphere_count, 3}),
      take_integer({batch_size}),
      take_integer({batch_size, sphere_count}),
      take_integer({batch_size, sphere_count}),
      take_float({batch_size, 1}),
      take_float({batch_size, 1}),
      take_float({batch_size, 1}),
      take_float({batch_size, 1}),
      take_float({batch_size, 1}),
      take_float({batch_size, 1}),
      take_float({batch_size, 1, 3}),
      take_integer({batch_size}),
      take_float({batch_size, box_count, 3}),
      take_float({batch_size, box_count, 3}),
      take_float({batch_size, box_count, 3, 3}),
      take_float({batch_size, box_count, 3}),
      take_integer({batch_size}),
      take_integer({batch_size, box_count}),
      take_integer({batch_size, box_count}),
      take_float({batch_size, prism_count, 3}),
      take_float({batch_size, prism_count, 3}),
      take_float({batch_size, prism_count, 3, 3}),
      take_float({batch_size, prism_count, 3}),
      take_integer({batch_size}),
      take_integer({batch_size, prism_count}),
      take_integer({batch_size, prism_count}),
      take_float({batch_size, cylinder_count, 3}),
      take_float({batch_size, cylinder_count}),
      take_float({batch_size, cylinder_count}),
      take_float({batch_size, cylinder_count, 3, 3}),
      take_float({batch_size, cylinder_count, 3}),
      take_integer({batch_size}),
      take_integer({batch_size, cylinder_count}),
      take_integer({batch_size, cylinder_count}),
  };
  auto plane_counts = take_integer({batch_size});
  auto terrain_class_ids = take_integer({batch_size, 1});
  auto terrain_instance_ids = take_integer({batch_size, 1});
  TORCH_CHECK(float_offset == float_data.numel() && integer_offset == integer_data.numel(),
              "packed scene size does not match its capacities");
  return {std::move(scene), plane_counts, terrain_class_ids, terrain_instance_ids};
}

py::tuple generate_random_scene_packed(
    int64_t seed,
    int batch_size,
    float scatter_radius,
    float ground_y,
    float depth_limit,
    float dz,
    float dz_growth,
    float fov_degrees,
    float aspect_ratio,
    int house_count,
    int tree_count,
    int cloud_count,
    int car_count,
    int person_count) {
  TORCH_CHECK(c10::cuda::device_count() > 0, "CUDA is required");
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(house_count >= 0 && tree_count >= 0 && cloud_count >= 0 &&
              car_count >= 0 && person_count >= 0, "object counts must be non-negative");
  TORCH_CHECK(house_count + tree_count + cloud_count + car_count + person_count > 0,
              "at least one composite object is required");

  const int64_t sphere_count = tree_count + cloud_count * 3 + person_count;
  const int64_t box_count = house_count + car_count + person_count * 5;
  const int64_t prism_count = house_count;
  const int64_t cylinder_count = tree_count + car_count * 4;
  TORCH_CHECK(sphere_count <= kRandomSceneMaxSpheres && box_count <= kRandomSceneMaxBoxes &&
              prism_count <= kRandomSceneMaxPrisms && cylinder_count <= kRandomSceneMaxCylinders,
              "generated scene exceeds renderer capacities");

  const int64_t float_size = batch_size *
      (7 * sphere_count + 9 + 18 * box_count + 18 * prism_count + 17 * cylinder_count);
  const int64_t integer_size = batch_size *
      (8 + 2 * (sphere_count + box_count + prism_count + cylinder_count));
  auto float_data = torch::empty(
      {float_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto integer_data = torch::empty(
      {integer_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
  auto views = unpack_scene(
      float_data, integer_data, batch_size, sphere_count, box_count, prism_count, cylinder_count);
  views.plane_counts.zero_();
  views.terrain_class_ids.fill_(2);
  views.terrain_instance_ids.fill_(sphere_count + 1);

  random_scene_cuda(
      views.scene, seed, scatter_radius, ground_y, depth_limit, dz, dz_growth,
      fov_degrees, aspect_ratio, house_count, tree_count, cloud_count, car_count, person_count);
  return py::make_tuple(float_data, integer_data);
}

void render_packed_scene(
    torch::Tensor image,
    torch::Tensor instance_map,
    torch::Tensor semantic_map,
    torch::Tensor visible_masks,
    torch::Tensor float_data,
    torch::Tensor integer_data,
    int sphere_count,
    int box_count,
    int prism_count,
    int cylinder_count,
    torch::Tensor light_direction,
    double fov_degrees,
    torch::Tensor background,
    double ambient,
    bool shadows,
    double shadow_strength) {
  auto packed = unpack_scene(
      float_data, integer_data, image.size(0), sphere_count, box_count, prism_count, cylinder_count);
  const auto& scene = packed.scene;
  const auto empty_planes = float_data.narrow(0, 0, 0).view({image.size(0), 0, 3});
  render_scene_cuda(
      image, instance_map, semantic_map, visible_masks,
      scene[0], scene[1], scene[3], scene[4], scene[5],
      empty_planes, empty_planes, packed.plane_counts,
      scene[6], scene[7], scene[8], scene[9], scene[10], scene[11], scene[13],
      packed.terrain_class_ids, packed.terrain_instance_ids,
      scene[14], scene[15], scene[16], scene[18], scene[19], scene[20],
      scene[21], scene[22], scene[23], scene[25], scene[26], scene[27],
      scene[28], scene[29], scene[30], scene[31], scene[33], scene[34], scene[35],
      light_direction, fov_degrees, background,
      scene[2], empty_planes, scene[12], scene[17], scene[24], scene[32],
      ambient, shadows, shadow_strength);
}

void render_scene(
    torch::Tensor image,
    torch::Tensor instance_map,
    torch::Tensor semantic_map,
    torch::Tensor visible_primitive_masks,
    py::dict scene,
    py::dict options) {
  const py::dict spheres = require_dict(scene, "spheres");
  const py::dict planes = require_dict(scene, "planes");
  const py::dict terrain = require_dict(scene, "terrain");
  const py::dict boxes = require_dict(scene, "boxes");
  const py::dict prisms = require_dict(scene, "prisms");
  const py::dict cylinders = require_dict(scene, "cylinders");

  const torch::Tensor light_dir = require_tensor(options, "light_dir");
  const torch::Tensor background = require_tensor(options, "background");
  const double fov_degrees = require_double(options, "fov_degrees");
  const double ambient = require_double(options, "ambient");
  const bool shadows = require_bool(options, "shadows");
  const double shadow_strength = require_double(options, "shadow_strength");

  const torch::Tensor sphere_centers = require_tensor(spheres, "centers");
  const torch::Tensor sphere_radii = require_tensor(spheres, "radii");
  const torch::Tensor sphere_colors = require_tensor(spheres, "colors");
  const torch::Tensor sphere_counts = require_tensor(spheres, "counts");
  const torch::Tensor sphere_class_ids = require_tensor(spheres, "class_ids");
  const torch::Tensor sphere_instance_ids = require_tensor(spheres, "instance_ids");

  const torch::Tensor plane_points = require_tensor(planes, "points");
  const torch::Tensor plane_normals = require_tensor(planes, "normals");
  const torch::Tensor plane_colors = require_tensor(planes, "colors");
  const torch::Tensor plane_counts = require_tensor(planes, "counts");

  const torch::Tensor terrain_base_heights = require_tensor(terrain, "base_heights");
  const torch::Tensor terrain_depth_limits = require_tensor(terrain, "depth_limits");
  const torch::Tensor terrain_phase_xs = require_tensor(terrain, "phase_xs");
  const torch::Tensor terrain_phase_zs = require_tensor(terrain, "phase_zs");
  const torch::Tensor terrain_dz = require_tensor(terrain, "dz");
  const torch::Tensor terrain_dz_growth = require_tensor(terrain, "dz_growth");
  const torch::Tensor terrain_colors = require_tensor(terrain, "colors");
  const torch::Tensor terrain_counts = require_tensor(terrain, "counts");
  const torch::Tensor terrain_class_ids = require_tensor(terrain, "class_ids");
  const torch::Tensor terrain_instance_ids = require_tensor(terrain, "instance_ids");

  const torch::Tensor box_centers = require_tensor(boxes, "centers");
  const torch::Tensor box_half_sizes = require_tensor(boxes, "half_sizes");
  const torch::Tensor box_axes = require_tensor(boxes, "axes");
  const torch::Tensor box_colors = require_tensor(boxes, "colors");
  const torch::Tensor box_counts = require_tensor(boxes, "counts");
  const torch::Tensor box_class_ids = require_tensor(boxes, "class_ids");
  const torch::Tensor box_instance_ids = require_tensor(boxes, "instance_ids");
  const torch::Tensor prism_centers = require_tensor(prisms, "centers");
  const torch::Tensor prism_half_sizes = require_tensor(prisms, "half_sizes");
  const torch::Tensor prism_axes = require_tensor(prisms, "axes");
  const torch::Tensor prism_colors = require_tensor(prisms, "colors");
  const torch::Tensor prism_counts = require_tensor(prisms, "counts");
  const torch::Tensor prism_class_ids = require_tensor(prisms, "class_ids");
  const torch::Tensor prism_instance_ids = require_tensor(prisms, "instance_ids");
  const torch::Tensor cylinder_centers = require_tensor(cylinders, "centers");
  const torch::Tensor cylinder_radii = require_tensor(cylinders, "radii");
  const torch::Tensor cylinder_half_heights = require_tensor(cylinders, "half_heights");
  const torch::Tensor cylinder_axes = require_tensor(cylinders, "axes");
  const torch::Tensor cylinder_colors = require_tensor(cylinders, "colors");
  const torch::Tensor cylinder_counts = require_tensor(cylinders, "counts");
  const torch::Tensor cylinder_class_ids = require_tensor(cylinders, "class_ids");
  const torch::Tensor cylinder_instance_ids = require_tensor(cylinders, "instance_ids");

  TORCH_CHECK(image.is_cuda(), "image must be a CUDA tensor");
  TORCH_CHECK(instance_map.is_cuda() && semantic_map.is_cuda() && visible_primitive_masks.is_cuda(), "segmentation tensors must be CUDA tensors");
  TORCH_CHECK(image.dtype() == torch::kFloat32, "image must be float32");
  TORCH_CHECK(instance_map.dtype() == torch::kInt32 && semantic_map.dtype() == torch::kInt32 && visible_primitive_masks.dtype() == torch::kInt32, "segmentation tensors must be int32");
  TORCH_CHECK(
      visible_primitive_masks.numel() == 0 ||
          (visible_primitive_masks.dim() == 2 && visible_primitive_masks.size(0) == image.size(0) &&
           visible_primitive_masks.size(1) == 9),
      "visible primitive masks must be empty or B x 9");
  TORCH_CHECK(image.dim() == 4 && image.size(1) == 3, "image must be B x 3 x H x W");
  TORCH_CHECK(
      instance_map.numel() == 0 ||
          (instance_map.dim() == 3 && instance_map.size(0) == image.size(0) && instance_map.size(1) == image.size(2) &&
           instance_map.size(2) == image.size(3)),
      "instance_map must be empty or B x H x W");
  TORCH_CHECK(
      semantic_map.numel() == 0 ||
          (semantic_map.dim() == 3 && semantic_map.size(0) == image.size(0) && semantic_map.size(1) == image.size(2) &&
           semantic_map.size(2) == image.size(3)),
      "semantic_map must be empty or B x H x W");
  TORCH_CHECK(sphere_centers.is_cuda() && sphere_radii.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(plane_points.is_cuda() && plane_normals.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(
      terrain_base_heights.is_cuda() && terrain_depth_limits.is_cuda() && terrain_phase_xs.is_cuda() &&
          terrain_phase_zs.is_cuda() && terrain_dz.is_cuda() && terrain_dz_growth.is_cuda(),
      "scene tensors must be CUDA tensors");
  TORCH_CHECK(box_centers.is_cuda() && box_half_sizes.is_cuda() && box_axes.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(prism_centers.is_cuda() && prism_half_sizes.is_cuda() && prism_axes.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(cylinder_centers.is_cuda() && cylinder_radii.is_cuda() && cylinder_half_heights.is_cuda() && cylinder_axes.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(
      light_dir.is_cuda() && background.is_cuda() && sphere_colors.is_cuda() && plane_colors.is_cuda() &&
          terrain_colors.is_cuda() && box_colors.is_cuda() && prism_colors.is_cuda() && cylinder_colors.is_cuda(),
      "scene tensors must be CUDA tensors");
  TORCH_CHECK(
      sphere_counts.is_cuda() && plane_counts.is_cuda() && terrain_counts.is_cuda() && box_counts.is_cuda() &&
          prism_counts.is_cuda() && cylinder_counts.is_cuda(),
      "primitive count tensors must be CUDA tensors");
  TORCH_CHECK(
      sphere_class_ids.is_cuda() && sphere_instance_ids.is_cuda() && terrain_class_ids.is_cuda() &&
          terrain_instance_ids.is_cuda() && box_class_ids.is_cuda() && box_instance_ids.is_cuda() &&
          prism_class_ids.is_cuda() && prism_instance_ids.is_cuda() && cylinder_class_ids.is_cuda() &&
          cylinder_instance_ids.is_cuda(),
      "primitive metadata tensors must be CUDA tensors");
  TORCH_CHECK(sphere_centers.dim() == 3 && sphere_centers.size(2) == 3, "sphere_centers must be B x N x 3");
  TORCH_CHECK(sphere_radii.dim() == 2, "sphere_radii must be B x N");
  TORCH_CHECK(plane_points.dim() == 3 && plane_points.size(2) == 3, "plane_points must be B x N x 3");
  TORCH_CHECK(plane_normals.dim() == 3 && plane_normals.size(2) == 3, "plane_normals must be B x N x 3");
  TORCH_CHECK(terrain_base_heights.dim() == 2, "terrain_base_heights must be B x N");
  TORCH_CHECK(terrain_depth_limits.dim() == 2, "terrain_depth_limits must be B x N");
  TORCH_CHECK(terrain_phase_xs.dim() == 2, "terrain_phase_xs must be B x N");
  TORCH_CHECK(terrain_phase_zs.dim() == 2, "terrain_phase_zs must be B x N");
  TORCH_CHECK(terrain_dz.dim() == 2, "terrain_dz must be B x N");
  TORCH_CHECK(terrain_dz_growth.dim() == 2, "terrain_dz_growth must be B x N");
  TORCH_CHECK(box_centers.dim() == 3 && box_centers.size(2) == 3, "box_centers must be B x N x 3");
  TORCH_CHECK(box_half_sizes.dim() == 3 && box_half_sizes.size(2) == 3, "box_half_sizes must be B x N x 3");
  TORCH_CHECK(box_axes.dim() == 4 && box_axes.size(2) == 3 && box_axes.size(3) == 3, "box_axes must be B x N x 3 x 3");
  TORCH_CHECK(prism_centers.dim() == 3 && prism_centers.size(2) == 3, "prism_centers must be B x N x 3");
  TORCH_CHECK(prism_half_sizes.dim() == 3 && prism_half_sizes.size(2) == 3, "prism_half_sizes must be B x N x 3");
  TORCH_CHECK(prism_axes.dim() == 4 && prism_axes.size(2) == 3 && prism_axes.size(3) == 3, "prism_axes must be B x N x 3 x 3");
  TORCH_CHECK(cylinder_centers.dim() == 3 && cylinder_centers.size(2) == 3, "cylinder_centers must be B x N x 3");
  TORCH_CHECK(cylinder_radii.dim() == 2, "cylinder_radii must be B x N");
  TORCH_CHECK(cylinder_half_heights.dim() == 2, "cylinder_half_heights must be B x N");
  TORCH_CHECK(cylinder_axes.dim() == 4 && cylinder_axes.size(2) == 3 && cylinder_axes.size(3) == 3, "cylinder_axes must be B x N x 3 x 3");
  TORCH_CHECK(sphere_class_ids.dim() == 2 && sphere_instance_ids.dim() == 2, "sphere metadata must be B x N");
  TORCH_CHECK(terrain_class_ids.dim() == 2 && terrain_instance_ids.dim() == 2, "terrain metadata must be B x N");
  TORCH_CHECK(box_class_ids.dim() == 2 && box_instance_ids.dim() == 2, "box metadata must be B x N");
  TORCH_CHECK(prism_class_ids.dim() == 2 && prism_instance_ids.dim() == 2, "prism metadata must be B x N");
  TORCH_CHECK(cylinder_class_ids.dim() == 2 && cylinder_instance_ids.dim() == 2, "cylinder metadata must be B x N");
  TORCH_CHECK(sphere_colors.dim() == 3 && sphere_colors.size(2) == 3, "sphere_colors must be B x N x 3");
  TORCH_CHECK(plane_colors.dim() == 3 && plane_colors.size(2) == 3, "plane_colors must be B x N x 3");
  TORCH_CHECK(terrain_colors.dim() == 3 && terrain_colors.size(2) == 3, "terrain_colors must be B x N x 3");
  TORCH_CHECK(box_colors.dim() == 3 && box_colors.size(2) == 3, "box_colors must be B x N x 3");
  TORCH_CHECK(prism_colors.dim() == 3 && prism_colors.size(2) == 3, "prism_colors must be B x N x 3");
  TORCH_CHECK(cylinder_colors.dim() == 3 && cylinder_colors.size(2) == 3, "cylinder_colors must be B x N x 3");
  TORCH_CHECK(
      sphere_counts.dim() == 1 && plane_counts.dim() == 1 && terrain_counts.dim() == 1 && box_counts.dim() == 1 &&
          prism_counts.dim() == 1 && cylinder_counts.dim() == 1,
      "primitive counts must be B");
  TORCH_CHECK(sphere_centers.size(0) == image.size(0), "scene batch size must match image batch size");
  TORCH_CHECK(
      plane_points.size(0) == image.size(0) && terrain_depth_limits.size(0) == image.size(0) &&
          box_centers.size(0) == image.size(0) && prism_centers.size(0) == image.size(0) &&
          cylinder_centers.size(0) == image.size(0),
      "scene batch size must match image batch size");
  TORCH_CHECK(
      sphere_counts.size(0) == image.size(0) && plane_counts.size(0) == image.size(0) &&
          terrain_counts.size(0) == image.size(0) && box_counts.size(0) == image.size(0) &&
          prism_counts.size(0) == image.size(0) && cylinder_counts.size(0) == image.size(0),
      "primitive count batch size must match image batch size");
  TORCH_CHECK(sphere_centers.size(1) == sphere_radii.size(1), "sphere_centers and sphere_radii must have matching lengths");
  TORCH_CHECK(sphere_centers.size(1) == sphere_colors.size(1), "sphere_centers and sphere_colors must have matching lengths");
  TORCH_CHECK(plane_points.size(1) == plane_normals.size(1), "plane_points and plane_normals must have matching lengths");
  TORCH_CHECK(plane_points.size(1) == plane_colors.size(1), "plane_points and plane_colors must have matching lengths");
  TORCH_CHECK(terrain_base_heights.sizes() == terrain_depth_limits.sizes(), "terrain fields must have matching lengths");
  TORCH_CHECK(terrain_phase_xs.sizes() == terrain_depth_limits.sizes(), "terrain fields must have matching lengths");
  TORCH_CHECK(terrain_phase_zs.sizes() == terrain_depth_limits.sizes(), "terrain fields must have matching lengths");
  TORCH_CHECK(terrain_dz.sizes() == terrain_depth_limits.sizes(), "terrain fields must have matching lengths");
  TORCH_CHECK(terrain_dz_growth.sizes() == terrain_depth_limits.sizes(), "terrain fields must have matching lengths");
  TORCH_CHECK(box_centers.size(1) == box_half_sizes.size(1), "box_centers and box_half_sizes must have matching lengths");
  TORCH_CHECK(box_centers.size(1) == box_axes.size(1), "box_centers and box_axes must have matching lengths");
  TORCH_CHECK(box_centers.size(1) == box_colors.size(1), "box_centers and box_colors must have matching lengths");
  TORCH_CHECK(prism_centers.size(1) == prism_half_sizes.size(1), "prism_centers and prism_half_sizes must have matching lengths");
  TORCH_CHECK(prism_centers.size(1) == prism_axes.size(1), "prism_centers and prism_axes must have matching lengths");
  TORCH_CHECK(prism_centers.size(1) == prism_colors.size(1), "prism_centers and prism_colors must have matching lengths");
  TORCH_CHECK(cylinder_centers.size(1) == cylinder_radii.size(1), "cylinder_centers and cylinder_radii must have matching lengths");
  TORCH_CHECK(cylinder_centers.size(1) == cylinder_half_heights.size(1), "cylinder_centers and cylinder_half_heights must have matching lengths");
  TORCH_CHECK(cylinder_centers.size(1) == cylinder_axes.size(1), "cylinder_centers and cylinder_axes must have matching lengths");
  TORCH_CHECK(cylinder_centers.size(1) == cylinder_colors.size(1), "cylinder_centers and cylinder_colors must have matching lengths");
  TORCH_CHECK(sphere_class_ids.sizes() == sphere_radii.sizes(), "sphere metadata must match sphere slots");
  TORCH_CHECK(sphere_instance_ids.sizes() == sphere_radii.sizes(), "sphere metadata must match sphere slots");
  TORCH_CHECK(terrain_class_ids.sizes() == terrain_depth_limits.sizes(), "terrain metadata must match terrain slots");
  TORCH_CHECK(terrain_instance_ids.sizes() == terrain_depth_limits.sizes(), "terrain metadata must match terrain slots");
  TORCH_CHECK(terrain_colors.size(0) == image.size(0) && terrain_colors.size(1) == terrain_depth_limits.size(1), "terrain_colors must match terrain slots");
  TORCH_CHECK(box_class_ids.size(0) == image.size(0) && box_class_ids.size(1) == box_centers.size(1), "box metadata must match box slots");
  TORCH_CHECK(box_instance_ids.size(0) == image.size(0) && box_instance_ids.size(1) == box_centers.size(1), "box metadata must match box slots");
  TORCH_CHECK(prism_class_ids.size(0) == image.size(0) && prism_class_ids.size(1) == prism_centers.size(1), "prism metadata must match prism slots");
  TORCH_CHECK(prism_instance_ids.size(0) == image.size(0) && prism_instance_ids.size(1) == prism_centers.size(1), "prism metadata must match prism slots");
  TORCH_CHECK(cylinder_class_ids.size(0) == image.size(0) && cylinder_class_ids.size(1) == cylinder_centers.size(1), "cylinder metadata must match cylinder slots");
  TORCH_CHECK(cylinder_instance_ids.size(0) == image.size(0) && cylinder_instance_ids.size(1) == cylinder_centers.size(1), "cylinder metadata must match cylinder slots");
  TORCH_CHECK(
      sphere_centers.size(1) > 0 || plane_points.size(1) > 0 || terrain_depth_limits.size(1) > 0 ||
          box_centers.size(1) > 0 || prism_centers.size(1) > 0 || cylinder_centers.size(1) > 0,
      "at least one object slot is required");
  TORCH_CHECK(light_dir.numel() == 3 && background.numel() == 3, "light/background vectors must be vec3");
  TORCH_CHECK(ambient >= 0.0 && ambient <= 1.0, "ambient must be in the range [0, 1]");
  TORCH_CHECK(shadow_strength >= 0.0 && shadow_strength <= 1.0, "shadow_strength must be in the range [0, 1]");
  TORCH_CHECK(sphere_centers.dtype() == torch::kFloat32, "sphere_centers must be float32");
  TORCH_CHECK(sphere_radii.dtype() == torch::kFloat32, "sphere_radii must be float32");
  TORCH_CHECK(plane_points.dtype() == torch::kFloat32, "plane_points must be float32");
  TORCH_CHECK(plane_normals.dtype() == torch::kFloat32, "plane_normals must be float32");
  TORCH_CHECK(terrain_base_heights.dtype() == torch::kFloat32, "terrain_base_heights must be float32");
  TORCH_CHECK(terrain_depth_limits.dtype() == torch::kFloat32, "terrain_depth_limits must be float32");
  TORCH_CHECK(terrain_phase_xs.dtype() == torch::kFloat32, "terrain_phase_xs must be float32");
  TORCH_CHECK(terrain_phase_zs.dtype() == torch::kFloat32, "terrain_phase_zs must be float32");
  TORCH_CHECK(terrain_dz.dtype() == torch::kFloat32, "terrain_dz must be float32");
  TORCH_CHECK(terrain_dz_growth.dtype() == torch::kFloat32, "terrain_dz_growth must be float32");
  TORCH_CHECK(box_centers.dtype() == torch::kFloat32, "box_centers must be float32");
  TORCH_CHECK(box_half_sizes.dtype() == torch::kFloat32, "box_half_sizes must be float32");
  TORCH_CHECK(box_axes.dtype() == torch::kFloat32, "box_axes must be float32");
  TORCH_CHECK(prism_centers.dtype() == torch::kFloat32, "prism_centers must be float32");
  TORCH_CHECK(prism_half_sizes.dtype() == torch::kFloat32, "prism_half_sizes must be float32");
  TORCH_CHECK(prism_axes.dtype() == torch::kFloat32, "prism_axes must be float32");
  TORCH_CHECK(cylinder_centers.dtype() == torch::kFloat32, "cylinder_centers must be float32");
  TORCH_CHECK(cylinder_radii.dtype() == torch::kFloat32, "cylinder_radii must be float32");
  TORCH_CHECK(cylinder_half_heights.dtype() == torch::kFloat32, "cylinder_half_heights must be float32");
  TORCH_CHECK(cylinder_axes.dtype() == torch::kFloat32, "cylinder_axes must be float32");
  TORCH_CHECK(light_dir.dtype() == torch::kFloat32, "light_dir must be float32");
  TORCH_CHECK(background.dtype() == torch::kFloat32, "background must be float32");
  TORCH_CHECK(sphere_colors.dtype() == torch::kFloat32, "sphere_colors must be float32");
  TORCH_CHECK(plane_colors.dtype() == torch::kFloat32, "plane_colors must be float32");
  TORCH_CHECK(terrain_colors.dtype() == torch::kFloat32, "terrain_colors must be float32");
  TORCH_CHECK(box_colors.dtype() == torch::kFloat32, "box_colors must be float32");
  TORCH_CHECK(prism_colors.dtype() == torch::kFloat32, "prism_colors must be float32");
  TORCH_CHECK(cylinder_colors.dtype() == torch::kFloat32, "cylinder_colors must be float32");
  TORCH_CHECK(
      sphere_counts.dtype() == torch::kInt32 && plane_counts.dtype() == torch::kInt32 &&
          terrain_counts.dtype() == torch::kInt32 && box_counts.dtype() == torch::kInt32 &&
          prism_counts.dtype() == torch::kInt32 && cylinder_counts.dtype() == torch::kInt32,
      "primitive counts must be int32");
  TORCH_CHECK(
      sphere_class_ids.dtype() == torch::kInt32 && sphere_instance_ids.dtype() == torch::kInt32 &&
          terrain_class_ids.dtype() == torch::kInt32 && terrain_instance_ids.dtype() == torch::kInt32 &&
          box_class_ids.dtype() == torch::kInt32 && box_instance_ids.dtype() == torch::kInt32 &&
          prism_class_ids.dtype() == torch::kInt32 && prism_instance_ids.dtype() == torch::kInt32 &&
          cylinder_class_ids.dtype() == torch::kInt32 && cylinder_instance_ids.dtype() == torch::kInt32,
      "primitive metadata must be int32");
  TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
  TORCH_CHECK(instance_map.is_contiguous() && semantic_map.is_contiguous() && visible_primitive_masks.is_contiguous(), "segmentation tensors must be contiguous");
  TORCH_CHECK(sphere_centers.is_contiguous() && sphere_radii.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(plane_points.is_contiguous() && plane_normals.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(
      terrain_base_heights.is_contiguous() && terrain_depth_limits.is_contiguous() &&
          terrain_phase_xs.is_contiguous() && terrain_phase_zs.is_contiguous() &&
          terrain_dz.is_contiguous() && terrain_dz_growth.is_contiguous(),
      "scene tensors must be contiguous");
  TORCH_CHECK(box_centers.is_contiguous() && box_half_sizes.is_contiguous() && box_axes.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(prism_centers.is_contiguous() && prism_half_sizes.is_contiguous() && prism_axes.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(cylinder_centers.is_contiguous() && cylinder_radii.is_contiguous() && cylinder_half_heights.is_contiguous() && cylinder_axes.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(
      light_dir.is_contiguous() && background.is_contiguous() && sphere_colors.is_contiguous() && plane_colors.is_contiguous() &&
          terrain_colors.is_contiguous() && box_colors.is_contiguous() && prism_colors.is_contiguous() && cylinder_colors.is_contiguous(),
      "scene tensors must be contiguous");
  TORCH_CHECK(
      sphere_counts.is_contiguous() && plane_counts.is_contiguous() && terrain_counts.is_contiguous() &&
          box_counts.is_contiguous() && prism_counts.is_contiguous() && cylinder_counts.is_contiguous(),
      "primitive count tensors must be contiguous");
  TORCH_CHECK(
      sphere_class_ids.is_contiguous() && sphere_instance_ids.is_contiguous() &&
          terrain_class_ids.is_contiguous() && terrain_instance_ids.is_contiguous() &&
          box_class_ids.is_contiguous() && box_instance_ids.is_contiguous() &&
          prism_class_ids.is_contiguous() && prism_instance_ids.is_contiguous() &&
          cylinder_class_ids.is_contiguous() && cylinder_instance_ids.is_contiguous(),
      "primitive metadata tensors must be contiguous");

  render_scene_cuda(
      image,
      instance_map,
      semantic_map,
      visible_primitive_masks,
      sphere_centers,
      sphere_radii,
      sphere_counts,
      sphere_class_ids,
      sphere_instance_ids,
      plane_points,
      plane_normals,
      plane_counts,
      terrain_base_heights,
      terrain_depth_limits,
      terrain_phase_xs,
      terrain_phase_zs,
      terrain_dz,
      terrain_dz_growth,
      terrain_counts,
      terrain_class_ids,
      terrain_instance_ids,
      box_centers,
      box_half_sizes,
      box_axes,
      box_counts,
      box_class_ids,
      box_instance_ids,
      prism_centers,
      prism_half_sizes,
      prism_axes,
      prism_counts,
      prism_class_ids,
      prism_instance_ids,
      cylinder_centers,
      cylinder_radii,
      cylinder_half_heights,
      cylinder_axes,
      cylinder_counts,
      cylinder_class_ids,
      cylinder_instance_ids,
      light_dir,
      fov_degrees,
      background,
      sphere_colors,
      plane_colors,
      terrain_colors,
      box_colors,
      prism_colors,
      cylinder_colors,
      ambient,
      shadows,
      shadow_strength);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "generate_random_scene",
      &generate_random_scene_packed,
      py::arg("seed"),
      py::arg("batch_size"),
      py::arg("scatter_radius"),
      py::arg("ground_y"),
      py::arg("depth_limit"),
      py::arg("dz"),
      py::arg("dz_growth"),
      py::arg("fov_degrees"),
      py::arg("aspect_ratio"),
      py::arg("house_count"),
      py::arg("tree_count"),
      py::arg("cloud_count"),
      py::arg("car_count"),
      py::arg("person_count"),
      "Generate a random scene in two packed CUDA buffers");
  m.def("render_packed_scene", &render_packed_scene, "Render directly from packed CUDA buffers");
  m.def("render_scene", &render_scene, "Render Lambert-shaded geometric objects (CUDA)");
}
