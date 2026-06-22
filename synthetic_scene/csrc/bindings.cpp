#include <torch/extension.h>

#include <ATen/CPUGeneratorImpl.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cmath>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kRandomSceneMaxSpheres = 64;
constexpr int kRandomSceneMaxBoxes = 64;
constexpr int kRandomTerrainWidth = 33;
constexpr int kRandomTerrainHeight = 33;
constexpr float kTau = 6.28318530717958647692f;

struct Vec3 {
  float x;
  float y;
  float z;
};

struct Mat3 {
  Vec3 rows[3];
};

float rand_float(at::Generator& generator, float low, float high) {
  if (low == high) {
    return low;
  }
  return torch::empty({}, torch::TensorOptions().dtype(torch::kFloat32))
      .uniform_(low, high, generator)
      .item<float>();
}

int rand_int(at::Generator& generator, int low, int high) {
  if (low == high) {
    return low;
  }
  return torch::randint(
             low,
             high + 1,
             {},
             c10::optional<at::Generator>(generator),
             torch::TensorOptions().dtype(torch::kInt64))
      .item<int64_t>();
}

Vec3 normalize3(Vec3 vector) {
  const float length = std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
  TORCH_CHECK(length > 1.0e-8f, "expected a non-zero vector");
  return Vec3{vector.x / length, vector.y / length, vector.z / length};
}

Vec3 cross3(Vec3 a, Vec3 b) {
  return Vec3{
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

Vec3 random_color(at::Generator& generator) {
  const float hue = rand_float(generator, 0.0f, 1.0f);
  const float saturation = rand_float(generator, 0.55f, 0.95f);
  const float value = rand_float(generator, 0.55f, 1.0f);
  int sector = static_cast<int>(hue * 6.0f);
  const float frac = hue * 6.0f - static_cast<float>(sector);
  const float p = value * (1.0f - saturation);
  const float q = value * (1.0f - frac * saturation);
  const float t = value * (1.0f - (1.0f - frac) * saturation);
  sector %= 6;
  if (sector == 0) {
    return Vec3{value, t, p};
  }
  if (sector == 1) {
    return Vec3{q, value, p};
  }
  if (sector == 2) {
    return Vec3{p, value, t};
  }
  if (sector == 3) {
    return Vec3{p, q, value};
  }
  if (sector == 4) {
    return Vec3{t, p, value};
  }
  return Vec3{value, p, q};
}

Vec3 random_frustum_point(
    at::Generator& generator,
    float fov_degrees,
    float aspect_ratio,
    float min_distance,
    float max_distance,
    float min_ndc_y,
    float max_ndc_y) {
  const float fov_radians = fov_degrees * 0.017453292519943295f;
  const float image_plane_scale = std::tan(0.5f * fov_radians);
  const float ndc_x = rand_float(generator, -1.0f, 1.0f);
  const float ndc_y = rand_float(generator, min_ndc_y, max_ndc_y);
  const float px = ndc_x * aspect_ratio * image_plane_scale;
  const float py = ndc_y * image_plane_scale;
  const Vec3 ray_dir = normalize3(Vec3{px, py, -1.0f});
  const float distance = rand_float(generator, min_distance, max_distance);
  return Vec3{ray_dir.x * distance, ray_dir.y * distance, ray_dir.z * distance};
}

Mat3 yaw_axes(float yaw) {
  const float cos_yaw = std::cos(yaw);
  const float sin_yaw = std::sin(yaw);
  return Mat3{{Vec3{cos_yaw, 0.0f, -sin_yaw}, Vec3{0.0f, 1.0f, 0.0f}, Vec3{sin_yaw, 0.0f, cos_yaw}}};
}

void append_vec3(std::vector<float>& values, Vec3 vector) {
  values.push_back(vector.x);
  values.push_back(vector.y);
  values.push_back(vector.z);
}

void append_mat3(std::vector<float>& values, Mat3 matrix) {
  for (const Vec3& row : matrix.rows) {
    append_vec3(values, row);
  }
}

float smooth_height(float x, float z, float phase_x, float phase_z) {
  return 0.28f * std::sin(1.35f * x + phase_x) + 0.18f * std::cos(1.85f * z + phase_z) +
      0.08f * std::sin(1.1f * (x + z) + phase_x * 0.7f);
}

float sample_heightmap(
    const std::vector<float>& heights,
    int batch_idx,
    float x,
    float z,
    Vec3 origin,
    float cell_size) {
  const float gx = (x - origin.x) / cell_size;
  const float gz = (z - origin.z) / cell_size;
  const float clamped_x = std::max(0.0f, std::min(gx, static_cast<float>(kRandomTerrainWidth - 1)));
  const float clamped_z = std::max(0.0f, std::min(gz, static_cast<float>(kRandomTerrainHeight - 1)));
  const int x0 = std::min(static_cast<int>(std::floor(clamped_x)), kRandomTerrainWidth - 2);
  const int z0 = std::min(static_cast<int>(std::floor(clamped_z)), kRandomTerrainHeight - 2);
  const float tx = clamped_x - static_cast<float>(x0);
  const float tz = clamped_z - static_cast<float>(z0);
  const int base = batch_idx * kRandomTerrainHeight * kRandomTerrainWidth;
  const float h00 = heights[base + z0 * kRandomTerrainWidth + x0];
  const float h10 = heights[base + z0 * kRandomTerrainWidth + x0 + 1];
  const float h01 = heights[base + (z0 + 1) * kRandomTerrainWidth + x0];
  const float h11 = heights[base + (z0 + 1) * kRandomTerrainWidth + x0 + 1];
  const float hx0 = h00 * (1.0f - tx) + h10 * tx;
  const float hx1 = h01 * (1.0f - tx) + h11 * tx;
  return hx0 * (1.0f - tz) + hx1 * tz;
}

torch::Tensor make_cuda_tensor(std::vector<float>&& values, std::vector<int64_t> shape) {
  int64_t numel = 1;
  for (const int64_t dim : shape) {
    numel *= dim;
  }
  if (numel == 0) {
    return torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  }
  auto cpu_tensor = torch::from_blob(values.data(), shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
  return cpu_tensor.to(torch::kCUDA, /*non_blocking=*/true).contiguous();
}

torch::Tensor make_cuda_int_tensor(std::vector<int32_t>&& values, std::vector<int64_t> shape) {
  int64_t numel = 1;
  for (const int64_t dim : shape) {
    numel *= dim;
  }
  if (numel == 0) {
    return torch::empty(shape, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
  }
  auto cpu_tensor = torch::from_blob(values.data(), shape, torch::TensorOptions().dtype(torch::kInt32)).clone();
  return cpu_tensor.to(torch::kCUDA, /*non_blocking=*/true).contiguous();
}

}  // namespace

void render_scene_cuda(
    torch::Tensor image,
    torch::Tensor instance_map,
    torch::Tensor semantic_map,
    torch::Tensor sphere_centers,
    torch::Tensor sphere_radii,
    torch::Tensor sphere_counts,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor plane_counts,
    torch::Tensor terrain_heightmaps,
    torch::Tensor terrain_origins,
    torch::Tensor terrain_cell_sizes,
    torch::Tensor terrain_counts,
    torch::Tensor box_centers,
    torch::Tensor box_half_sizes,
    torch::Tensor box_axes,
    torch::Tensor box_counts,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors,
    torch::Tensor plane_colors,
    torch::Tensor terrain_colors,
    torch::Tensor box_colors,
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

py::dict random_scene(
    int64_t seed,
    int ground_objects,
    int floating_objects,
    int batch_size,
    float scatter_radius,
    float ground_y,
    float fov_degrees,
    float aspect_ratio) {
  TORCH_CHECK(c10::cuda::device_count() > 0, "CUDA is required to generate a native random scene");
  TORCH_CHECK(ground_objects >= 0 && floating_objects >= 0, "object counts must be non-negative");
  TORCH_CHECK(ground_objects + floating_objects > 0, "at least one non-plane object is required");
  TORCH_CHECK(batch_size > 0, "batch_size must be positive");
  TORCH_CHECK(scatter_radius > 0.0f, "scatter_radius must be positive");
  TORCH_CHECK(fov_degrees > 0.0f && fov_degrees < 180.0f, "fov_degrees must be in the open interval (0, 180)");
  TORCH_CHECK(aspect_ratio > 0.0f, "aspect_ratio must be positive");

  at::Generator generator = at::detail::createCPUGenerator(static_cast<uint64_t>(seed));
  std::vector<float> sphere_centers;
  std::vector<float> sphere_radii;
  std::vector<float> sphere_colors;
  std::vector<int32_t> sphere_counts;
  std::vector<float> box_centers;
  std::vector<float> box_half_sizes;
  std::vector<float> box_axes;
  std::vector<float> box_colors;
  std::vector<int32_t> box_counts;
  std::vector<float> terrain_heightmaps;
  std::vector<float> terrain_origins;
  std::vector<float> terrain_cell_sizes;
  std::vector<float> terrain_colors;
  std::vector<int32_t> terrain_counts;
  const int max_objects = ground_objects + floating_objects;
  const int64_t sphere_count = static_cast<int64_t>(max_objects);
  const int64_t box_count = static_cast<int64_t>(max_objects);
  TORCH_CHECK(
      sphere_count <= kRandomSceneMaxSpheres && box_count <= kRandomSceneMaxBoxes,
      "random_scene generated more primitives than the renderer supports");

  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const float terrain_cell_size = std::max(0.25f, scatter_radius * 2.8f / static_cast<float>(kRandomTerrainWidth - 1));
    const float terrain_span = static_cast<float>(kRandomTerrainWidth - 1) * terrain_cell_size;
    const Vec3 terrain_origin{-0.5f * terrain_span, ground_y, -2.0f - scatter_radius - 0.5f * terrain_span};
    const float phase_x = rand_float(generator, 0.0f, kTau);
    const float phase_z = rand_float(generator, 0.0f, kTau);
    append_vec3(terrain_origins, terrain_origin);
    terrain_cell_sizes.push_back(terrain_cell_size);
    append_vec3(terrain_colors, Vec3{0.34f, 0.46f, 0.28f});
    terrain_counts.push_back(1);
    for (int row = 0; row < kRandomTerrainHeight; ++row) {
      for (int col = 0; col < kRandomTerrainWidth; ++col) {
        const float x = terrain_origin.x + static_cast<float>(col) * terrain_cell_size;
        const float z = terrain_origin.z + static_cast<float>(row) * terrain_cell_size;
        terrain_heightmaps.push_back(terrain_origin.y + smooth_height(x, z, phase_x, phase_z));
      }
    }

    int scene_spheres = 0;
    int scene_boxes = 0;
    auto add_sphere = [&](Vec3 center, float radius) {
      append_vec3(sphere_centers, center);
      sphere_radii.push_back(radius);
      append_vec3(sphere_colors, random_color(generator));
      ++scene_spheres;
    };
    auto add_box = [&](Vec3 center, Vec3 half_size, Mat3 axes) {
      append_vec3(box_centers, center);
      append_vec3(box_half_sizes, half_size);
      append_mat3(box_axes, axes);
      append_vec3(box_colors, random_color(generator));
      ++scene_boxes;
    };

    for (int i = 0; i < ground_objects; ++i) {
      const Vec3 frustum_point = random_frustum_point(
          generator,
          fov_degrees,
          aspect_ratio,
          2.0f,
          2.0f + scatter_radius,
          -0.82f,
          -0.18f);
      const float x = frustum_point.x;
      const float z = frustum_point.z;
      const float terrain_y = sample_heightmap(terrain_heightmaps, batch_idx, x, z, terrain_origin, terrain_cell_size);
      if (rand_float(generator, 0.0f, 1.0f) < 0.55f) {
        const float radius = rand_float(generator, 0.18f, 0.65f);
        add_sphere(Vec3{x, terrain_y + radius, z}, radius);
      } else {
        const Vec3 half_size{
            rand_float(generator, 0.18f, 0.6f),
            rand_float(generator, 0.18f, 0.75f),
            rand_float(generator, 0.18f, 0.6f),
        };
        add_box(Vec3{x, terrain_y + half_size.y, z}, half_size, yaw_axes(rand_float(generator, 0.0f, kTau)));
      }
    }

    for (int i = 0; i < floating_objects; ++i) {
      const Vec3 frustum_point = random_frustum_point(
          generator,
          fov_degrees,
          aspect_ratio,
          1.8f,
          2.0f + scatter_radius * 0.9f,
          0.15f,
          0.85f);
      const float x = frustum_point.x;
      const float y = frustum_point.y;
      const float z = frustum_point.z;
      if (rand_float(generator, 0.0f, 1.0f) < 0.65f) {
        const float radius = rand_float(generator, 0.15f, 0.5f);
        add_sphere(Vec3{x, y, z}, radius);
      } else {
        const Vec3 half_size{
            rand_float(generator, 0.14f, 0.45f),
            rand_float(generator, 0.14f, 0.45f),
            rand_float(generator, 0.14f, 0.45f),
        };
        add_box(Vec3{x, y, z}, half_size, yaw_axes(rand_float(generator, 0.0f, kTau)));
      }
    }

    const int real_spheres = scene_spheres;
    const int real_boxes = scene_boxes;
    while (scene_spheres < max_objects) {
      append_vec3(sphere_centers, Vec3{0.0f, 0.0f, -1.0f});
      sphere_radii.push_back(1.0f);
      append_vec3(sphere_colors, Vec3{0.0f, 0.0f, 0.0f});
      ++scene_spheres;
    }
    while (scene_boxes < max_objects) {
      append_vec3(box_centers, Vec3{0.0f, 0.0f, -1.0f});
      append_vec3(box_half_sizes, Vec3{1.0f, 1.0f, 1.0f});
      append_mat3(box_axes, yaw_axes(0.0f));
      append_vec3(box_colors, Vec3{0.0f, 0.0f, 0.0f});
      ++scene_boxes;
    }
    sphere_counts.push_back(static_cast<int32_t>(real_spheres));
    box_counts.push_back(static_cast<int32_t>(real_boxes));
  }

  std::vector<float> plane_points;
  std::vector<float> plane_normals;
  std::vector<float> plane_colors;
  py::dict result;
  result["sphere_centers"] = make_cuda_tensor(std::move(sphere_centers), {batch_size, sphere_count, 3});
  result["sphere_radii"] = make_cuda_tensor(std::move(sphere_radii), {batch_size, sphere_count});
  result["sphere_colors"] = make_cuda_tensor(std::move(sphere_colors), {batch_size, sphere_count, 3});
  result["sphere_counts"] = make_cuda_int_tensor(std::move(sphere_counts), {batch_size});
  result["plane_points"] = make_cuda_tensor(std::move(plane_points), {batch_size, 0, 3});
  result["plane_normals"] = make_cuda_tensor(std::move(plane_normals), {batch_size, 0, 3});
  result["plane_colors"] = make_cuda_tensor(std::move(plane_colors), {batch_size, 0, 3});
  result["plane_counts"] = make_cuda_int_tensor(std::vector<int32_t>(batch_size, 0), {batch_size});
  result["terrain_heightmaps"] = make_cuda_tensor(std::move(terrain_heightmaps), {batch_size, kRandomTerrainHeight, kRandomTerrainWidth});
  result["terrain_origins"] = make_cuda_tensor(std::move(terrain_origins), {batch_size, 1, 3});
  result["terrain_cell_sizes"] = make_cuda_tensor(std::move(terrain_cell_sizes), {batch_size, 1});
  result["terrain_colors"] = make_cuda_tensor(std::move(terrain_colors), {batch_size, 1, 3});
  result["terrain_counts"] = make_cuda_int_tensor(std::move(terrain_counts), {batch_size});
  result["box_centers"] = make_cuda_tensor(std::move(box_centers), {batch_size, box_count, 3});
  result["box_half_sizes"] = make_cuda_tensor(std::move(box_half_sizes), {batch_size, box_count, 3});
  result["box_axes"] = make_cuda_tensor(std::move(box_axes), {batch_size, box_count, 3, 3});
  result["box_colors"] = make_cuda_tensor(std::move(box_colors), {batch_size, box_count, 3});
  result["box_counts"] = make_cuda_int_tensor(std::move(box_counts), {batch_size});
  return result;
}

void render_scene(
    torch::Tensor image,
    torch::Tensor instance_map,
    torch::Tensor semantic_map,
    py::dict scene,
    py::dict options) {
  const py::dict spheres = require_dict(scene, "spheres");
  const py::dict planes = require_dict(scene, "planes");
  const py::dict terrain = require_dict(scene, "terrain");
  const py::dict boxes = require_dict(scene, "boxes");

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

  const torch::Tensor plane_points = require_tensor(planes, "points");
  const torch::Tensor plane_normals = require_tensor(planes, "normals");
  const torch::Tensor plane_colors = require_tensor(planes, "colors");
  const torch::Tensor plane_counts = require_tensor(planes, "counts");

  const torch::Tensor terrain_heightmaps = require_tensor(terrain, "heightmaps");
  const torch::Tensor terrain_origins = require_tensor(terrain, "origins");
  const torch::Tensor terrain_cell_sizes = require_tensor(terrain, "cell_sizes");
  const torch::Tensor terrain_colors = require_tensor(terrain, "colors");
  const torch::Tensor terrain_counts = require_tensor(terrain, "counts");

  const torch::Tensor box_centers = require_tensor(boxes, "centers");
  const torch::Tensor box_half_sizes = require_tensor(boxes, "half_sizes");
  const torch::Tensor box_axes = require_tensor(boxes, "axes");
  const torch::Tensor box_colors = require_tensor(boxes, "colors");
  const torch::Tensor box_counts = require_tensor(boxes, "counts");

  TORCH_CHECK(image.is_cuda(), "image must be a CUDA tensor");
  TORCH_CHECK(instance_map.is_cuda() && semantic_map.is_cuda(), "segmentation maps must be CUDA tensors");
  TORCH_CHECK(image.dtype() == torch::kFloat32, "image must be float32");
  TORCH_CHECK(instance_map.dtype() == torch::kInt32 && semantic_map.dtype() == torch::kInt32, "segmentation maps must be int32");
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
  TORCH_CHECK(terrain_heightmaps.is_cuda() && terrain_origins.is_cuda() && terrain_cell_sizes.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(box_centers.is_cuda() && box_half_sizes.is_cuda() && box_axes.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(
      light_dir.is_cuda() && background.is_cuda() && sphere_colors.is_cuda() && plane_colors.is_cuda() &&
          terrain_colors.is_cuda() && box_colors.is_cuda(),
      "scene tensors must be CUDA tensors");
  TORCH_CHECK(sphere_counts.is_cuda() && plane_counts.is_cuda() && terrain_counts.is_cuda() && box_counts.is_cuda(), "primitive count tensors must be CUDA tensors");
  TORCH_CHECK(sphere_centers.dim() == 3 && sphere_centers.size(2) == 3, "sphere_centers must be B x N x 3");
  TORCH_CHECK(sphere_radii.dim() == 2, "sphere_radii must be B x N");
  TORCH_CHECK(plane_points.dim() == 3 && plane_points.size(2) == 3, "plane_points must be B x N x 3");
  TORCH_CHECK(plane_normals.dim() == 3 && plane_normals.size(2) == 3, "plane_normals must be B x N x 3");
  TORCH_CHECK(terrain_heightmaps.dim() == 3, "terrain_heightmaps must be B x H x W");
  TORCH_CHECK(terrain_origins.dim() == 3 && terrain_origins.size(1) == 1 && terrain_origins.size(2) == 3, "terrain_origins must be B x 1 x 3");
  TORCH_CHECK(terrain_cell_sizes.dim() == 2 && terrain_cell_sizes.size(1) == 1, "terrain_cell_sizes must be B x 1");
  TORCH_CHECK(box_centers.dim() == 3 && box_centers.size(2) == 3, "box_centers must be B x N x 3");
  TORCH_CHECK(box_half_sizes.dim() == 3 && box_half_sizes.size(2) == 3, "box_half_sizes must be B x N x 3");
  TORCH_CHECK(box_axes.dim() == 4 && box_axes.size(2) == 3 && box_axes.size(3) == 3, "box_axes must be B x N x 3 x 3");
  TORCH_CHECK(sphere_colors.dim() == 3 && sphere_colors.size(2) == 3, "sphere_colors must be B x N x 3");
  TORCH_CHECK(plane_colors.dim() == 3 && plane_colors.size(2) == 3, "plane_colors must be B x N x 3");
  TORCH_CHECK(terrain_colors.dim() == 3 && terrain_colors.size(1) == 1 && terrain_colors.size(2) == 3, "terrain_colors must be B x 1 x 3");
  TORCH_CHECK(box_colors.dim() == 3 && box_colors.size(2) == 3, "box_colors must be B x N x 3");
  TORCH_CHECK(sphere_counts.dim() == 1 && plane_counts.dim() == 1 && terrain_counts.dim() == 1 && box_counts.dim() == 1, "primitive counts must be B");
  TORCH_CHECK(sphere_centers.size(0) == image.size(0), "scene batch size must match image batch size");
  TORCH_CHECK(
      plane_points.size(0) == image.size(0) && terrain_heightmaps.size(0) == image.size(0) && box_centers.size(0) == image.size(0),
      "scene batch size must match image batch size");
  TORCH_CHECK(
      sphere_counts.size(0) == image.size(0) && plane_counts.size(0) == image.size(0) &&
          terrain_counts.size(0) == image.size(0) && box_counts.size(0) == image.size(0),
      "primitive count batch size must match image batch size");
  TORCH_CHECK(sphere_centers.size(1) == sphere_radii.size(1), "sphere_centers and sphere_radii must have matching lengths");
  TORCH_CHECK(sphere_centers.size(1) == sphere_colors.size(1), "sphere_centers and sphere_colors must have matching lengths");
  TORCH_CHECK(plane_points.size(1) == plane_normals.size(1), "plane_points and plane_normals must have matching lengths");
  TORCH_CHECK(plane_points.size(1) == plane_colors.size(1), "plane_points and plane_colors must have matching lengths");
  TORCH_CHECK(box_centers.size(1) == box_half_sizes.size(1), "box_centers and box_half_sizes must have matching lengths");
  TORCH_CHECK(box_centers.size(1) == box_axes.size(1), "box_centers and box_axes must have matching lengths");
  TORCH_CHECK(box_centers.size(1) == box_colors.size(1), "box_centers and box_colors must have matching lengths");
  TORCH_CHECK(
      sphere_centers.size(1) > 0 || plane_points.size(1) > 0 || terrain_heightmaps.size(1) > 0 || box_centers.size(1) > 0,
      "at least one object slot is required");
  TORCH_CHECK(light_dir.numel() == 3 && background.numel() == 3, "light/background vectors must be vec3");
  TORCH_CHECK(ambient >= 0.0 && ambient <= 1.0, "ambient must be in the range [0, 1]");
  TORCH_CHECK(shadow_strength >= 0.0 && shadow_strength <= 1.0, "shadow_strength must be in the range [0, 1]");
  TORCH_CHECK(sphere_centers.dtype() == torch::kFloat32, "sphere_centers must be float32");
  TORCH_CHECK(sphere_radii.dtype() == torch::kFloat32, "sphere_radii must be float32");
  TORCH_CHECK(plane_points.dtype() == torch::kFloat32, "plane_points must be float32");
  TORCH_CHECK(plane_normals.dtype() == torch::kFloat32, "plane_normals must be float32");
  TORCH_CHECK(terrain_heightmaps.dtype() == torch::kFloat32, "terrain_heightmaps must be float32");
  TORCH_CHECK(terrain_origins.dtype() == torch::kFloat32, "terrain_origins must be float32");
  TORCH_CHECK(terrain_cell_sizes.dtype() == torch::kFloat32, "terrain_cell_sizes must be float32");
  TORCH_CHECK(box_centers.dtype() == torch::kFloat32, "box_centers must be float32");
  TORCH_CHECK(box_half_sizes.dtype() == torch::kFloat32, "box_half_sizes must be float32");
  TORCH_CHECK(box_axes.dtype() == torch::kFloat32, "box_axes must be float32");
  TORCH_CHECK(light_dir.dtype() == torch::kFloat32, "light_dir must be float32");
  TORCH_CHECK(background.dtype() == torch::kFloat32, "background must be float32");
  TORCH_CHECK(sphere_colors.dtype() == torch::kFloat32, "sphere_colors must be float32");
  TORCH_CHECK(plane_colors.dtype() == torch::kFloat32, "plane_colors must be float32");
  TORCH_CHECK(terrain_colors.dtype() == torch::kFloat32, "terrain_colors must be float32");
  TORCH_CHECK(box_colors.dtype() == torch::kFloat32, "box_colors must be float32");
  TORCH_CHECK(
      sphere_counts.dtype() == torch::kInt32 && plane_counts.dtype() == torch::kInt32 &&
          terrain_counts.dtype() == torch::kInt32 && box_counts.dtype() == torch::kInt32,
      "primitive counts must be int32");
  TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
  TORCH_CHECK(instance_map.is_contiguous() && semantic_map.is_contiguous(), "segmentation maps must be contiguous");
  TORCH_CHECK(sphere_centers.is_contiguous() && sphere_radii.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(plane_points.is_contiguous() && plane_normals.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(terrain_heightmaps.is_contiguous() && terrain_origins.is_contiguous() && terrain_cell_sizes.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(box_centers.is_contiguous() && box_half_sizes.is_contiguous() && box_axes.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(
      light_dir.is_contiguous() && background.is_contiguous() && sphere_colors.is_contiguous() && plane_colors.is_contiguous() &&
          terrain_colors.is_contiguous() && box_colors.is_contiguous(),
      "scene tensors must be contiguous");
  TORCH_CHECK(sphere_counts.is_contiguous() && plane_counts.is_contiguous() && terrain_counts.is_contiguous() && box_counts.is_contiguous(), "primitive count tensors must be contiguous");

  render_scene_cuda(
      image,
      instance_map,
      semantic_map,
      sphere_centers,
      sphere_radii,
      sphere_counts,
      plane_points,
      plane_normals,
      plane_counts,
      terrain_heightmaps,
      terrain_origins,
      terrain_cell_sizes,
      terrain_counts,
      box_centers,
      box_half_sizes,
      box_axes,
      box_counts,
      light_dir,
      fov_degrees,
      background,
      sphere_colors,
      plane_colors,
      terrain_colors,
      box_colors,
      ambient,
      shadows,
      shadow_strength);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "random_scene",
      &random_scene,
      py::arg("seed"),
      py::arg("ground_objects"),
      py::arg("floating_objects"),
      py::arg("batch_size"),
      py::arg("scatter_radius"),
      py::arg("ground_y"),
      py::arg("fov_degrees"),
      py::arg("aspect_ratio"),
      "Generate random camera-space scene tensors directly from the native extension");
  m.def("render_scene", &render_scene, "Render Lambert-shaded geometric objects (CUDA)");
}
