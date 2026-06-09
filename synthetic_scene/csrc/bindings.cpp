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

Mat3 look_at_orientation(Vec3 origin, Vec3 target) {
  const Vec3 forward = normalize3(Vec3{target.x - origin.x, target.y - origin.y, target.z - origin.z});
  const Vec3 back{-forward.x, -forward.y, -forward.z};
  const Vec3 world_up{0.0f, 1.0f, 0.0f};
  Vec3 right = cross3(world_up, back);
  if (right.x * right.x + right.y * right.y + right.z * right.z <= 1.0e-8f) {
    right = Vec3{1.0f, 0.0f, 0.0f};
  } else {
    right = normalize3(right);
  }
  const Vec3 up = normalize3(cross3(back, right));
  return Mat3{{right, up, back}};
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

std::tuple<float, float> random_ground_xz(at::Generator& generator, float scatter_radius) {
  const float radius = scatter_radius * std::sqrt(rand_float(generator, 0.0f, 1.0f));
  const float angle = rand_float(generator, 0.0f, kTau);
  return {radius * std::cos(angle), radius * std::sin(angle)};
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

std::pair<float, float> require_range(const char* name, py::tuple value) {
  TORCH_CHECK(value.size() == 2, name, " must contain exactly two values");
  const float low = value[0].cast<float>();
  const float high = value[1].cast<float>();
  TORCH_CHECK(low <= high, name, " lower bound must be <= upper bound");
  return {low, high};
}

}  // namespace

void render_scene_cuda(
    torch::Tensor image,
    torch::Tensor instance_map,
    torch::Tensor semantic_map,
    torch::Tensor camera_origin,
    torch::Tensor camera_orientation,
    torch::Tensor sphere_centers,
    torch::Tensor sphere_radii,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor box_centers,
    torch::Tensor box_half_sizes,
    torch::Tensor box_axes,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors,
    torch::Tensor plane_colors,
    torch::Tensor box_colors);

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

py::dict random_scene(
    int64_t seed,
    int ground_objects,
    int floating_objects,
    int cameras,
    float scatter_radius,
    float ground_y,
    py::tuple camera_distance,
    py::tuple camera_height,
    float fov_degrees) {
  TORCH_CHECK(c10::cuda::device_count() > 0, "CUDA is required to generate a native random scene");
  TORCH_CHECK(ground_objects >= 0 && floating_objects >= 0, "object counts must be non-negative");
  TORCH_CHECK(ground_objects + floating_objects > 0, "at least one non-plane object is required");
  TORCH_CHECK(cameras > 0, "cameras must be positive");
  TORCH_CHECK(scatter_radius > 0.0f, "scatter_radius must be positive");
  TORCH_CHECK(fov_degrees > 0.0f && fov_degrees < 180.0f, "fov_degrees must be in the open interval (0, 180)");
  const auto camera_distance_range = require_range("camera_distance", camera_distance);
  const float camera_distance_min = camera_distance_range.first;
  const float camera_distance_max = camera_distance_range.second;
  const auto camera_height_range = require_range("camera_height", camera_height);
  const float camera_height_min = camera_height_range.first;
  const float camera_height_max = camera_height_range.second;
  TORCH_CHECK(camera_distance_min > 0.0f, "camera_distance values must be positive");

  at::Generator generator = at::detail::createCPUGenerator(static_cast<uint64_t>(seed));
  std::vector<float> sphere_centers;
  std::vector<float> sphere_radii;
  std::vector<float> sphere_colors;
  std::vector<float> box_centers;
  std::vector<float> box_half_sizes;
  std::vector<float> box_axes;
  std::vector<float> box_colors;
  std::vector<Vec3> targets;

  auto add_sphere = [&](Vec3 center, float radius) {
    append_vec3(sphere_centers, center);
    sphere_radii.push_back(radius);
    append_vec3(sphere_colors, random_color(generator));
    targets.push_back(center);
  };
  auto add_box = [&](Vec3 center, Vec3 half_size, Mat3 axes) {
    append_vec3(box_centers, center);
    append_vec3(box_half_sizes, half_size);
    append_mat3(box_axes, axes);
    append_vec3(box_colors, random_color(generator));
    targets.push_back(center);
  };

  for (int i = 0; i < ground_objects; ++i) {
    const auto ground_xz = random_ground_xz(generator, scatter_radius);
    const float x = std::get<0>(ground_xz);
    const float z = std::get<1>(ground_xz);
    if (rand_float(generator, 0.0f, 1.0f) < 0.55f) {
      const float radius = rand_float(generator, 0.18f, 0.65f);
      add_sphere(Vec3{x, ground_y + radius, z}, radius);
    } else {
      const Vec3 half_size{
          rand_float(generator, 0.18f, 0.6f),
          rand_float(generator, 0.18f, 0.75f),
          rand_float(generator, 0.18f, 0.6f),
      };
      add_box(Vec3{x, ground_y + half_size.y, z}, half_size, yaw_axes(rand_float(generator, 0.0f, kTau)));
    }
  }

  for (int i = 0; i < floating_objects; ++i) {
    const auto ground_xz = random_ground_xz(generator, scatter_radius * 0.9f);
    const float x = std::get<0>(ground_xz);
    const float z = std::get<1>(ground_xz);
    if (rand_float(generator, 0.0f, 1.0f) < 0.65f) {
      const float radius = rand_float(generator, 0.15f, 0.5f);
      add_sphere(Vec3{x, rand_float(generator, ground_y + 1.0f, ground_y + 3.2f), z}, radius);
    } else {
      const Vec3 half_size{
          rand_float(generator, 0.14f, 0.45f),
          rand_float(generator, 0.14f, 0.45f),
          rand_float(generator, 0.14f, 0.45f),
      };
      add_box(Vec3{x, rand_float(generator, ground_y + 1.1f, ground_y + 3.4f), z}, half_size, yaw_axes(rand_float(generator, 0.0f, kTau)));
    }
  }

  const int64_t sphere_count = static_cast<int64_t>(sphere_radii.size());
  const int64_t box_count = static_cast<int64_t>(box_centers.size() / 3);
  TORCH_CHECK(
      sphere_count <= kRandomSceneMaxSpheres && box_count <= kRandomSceneMaxBoxes,
      "random_scene generated more primitives than the renderer supports");

  std::vector<float> camera_origins;
  std::vector<float> camera_orientations;
  for (int i = 0; i < cameras; ++i) {
    const Vec3 target = targets[static_cast<size_t>(rand_int(generator, 0, static_cast<int>(targets.size()) - 1))];
    const float distance = rand_float(generator, camera_distance_min, camera_distance_max);
    const float angle = rand_float(generator, 0.0f, kTau);
    const float height = rand_float(generator, camera_height_min, camera_height_max);
    const Vec3 origin{
        target.x + distance * std::cos(angle),
        target.y + height,
        target.z + distance * std::sin(angle),
    };
    append_vec3(camera_origins, origin);
    append_mat3(camera_orientations, look_at_orientation(origin, target));
  }

  py::dict result;
  result["camera_origins"] = make_cuda_tensor(std::move(camera_origins), {cameras, 3});
  result["camera_orientations"] = make_cuda_tensor(std::move(camera_orientations), {cameras, 3, 3});
  result["sphere_centers"] = make_cuda_tensor(std::move(sphere_centers), {sphere_count, 3});
  result["sphere_radii"] = make_cuda_tensor(std::move(sphere_radii), {sphere_count});
  result["sphere_colors"] = make_cuda_tensor(std::move(sphere_colors), {sphere_count, 3});
  result["plane_points"] = make_cuda_tensor(std::vector<float>{0.0f, ground_y, 0.0f}, {1, 3});
  result["plane_normals"] = make_cuda_tensor(std::vector<float>{0.0f, 1.0f, 0.0f}, {1, 3});
  result["plane_colors"] = make_cuda_tensor(std::vector<float>{0.45f, 0.48f, 0.43f}, {1, 3});
  result["box_centers"] = make_cuda_tensor(std::move(box_centers), {box_count, 3});
  result["box_half_sizes"] = make_cuda_tensor(std::move(box_half_sizes), {box_count, 3});
  result["box_axes"] = make_cuda_tensor(std::move(box_axes), {box_count, 3, 3});
  result["box_colors"] = make_cuda_tensor(std::move(box_colors), {box_count, 3});
  return result;
}

void render_scene(
    torch::Tensor image,
    torch::Tensor instance_map,
    torch::Tensor semantic_map,
    py::dict camera,
    py::dict scene,
    py::dict options) {
  const py::dict spheres = require_dict(scene, "spheres");
  const py::dict planes = require_dict(scene, "planes");
  const py::dict boxes = require_dict(scene, "boxes");

  const torch::Tensor camera_origin = require_tensor(camera, "origin");
  const torch::Tensor camera_orientation = require_tensor(camera, "orientation");
  const double fov_degrees = require_double(camera, "fov_degrees");
  const torch::Tensor light_dir = require_tensor(options, "light_dir");
  const torch::Tensor background = require_tensor(options, "background");

  const torch::Tensor sphere_centers = require_tensor(spheres, "centers");
  const torch::Tensor sphere_radii = require_tensor(spheres, "radii");
  const torch::Tensor sphere_colors = require_tensor(spheres, "colors");

  const torch::Tensor plane_points = require_tensor(planes, "points");
  const torch::Tensor plane_normals = require_tensor(planes, "normals");
  const torch::Tensor plane_colors = require_tensor(planes, "colors");

  const torch::Tensor box_centers = require_tensor(boxes, "centers");
  const torch::Tensor box_half_sizes = require_tensor(boxes, "half_sizes");
  const torch::Tensor box_axes = require_tensor(boxes, "axes");
  const torch::Tensor box_colors = require_tensor(boxes, "colors");

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
  TORCH_CHECK(camera_origin.is_cuda() && camera_orientation.is_cuda(), "camera tensors must be CUDA tensors");
  TORCH_CHECK(sphere_centers.is_cuda() && sphere_radii.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(plane_points.is_cuda() && plane_normals.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(box_centers.is_cuda() && box_half_sizes.is_cuda() && box_axes.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(light_dir.is_cuda() && background.is_cuda() && sphere_colors.is_cuda() && plane_colors.is_cuda() && box_colors.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(camera_origin.dim() == 2 && camera_origin.size(1) == 3, "camera_origin must be B x 3");
  TORCH_CHECK(
      camera_orientation.dim() == 3 && camera_orientation.size(1) == 3 && camera_orientation.size(2) == 3,
      "camera_orientation must be B x 3 x 3");
  TORCH_CHECK(camera_origin.size(0) == image.size(0), "camera_origin batch size must match image batch size");
  TORCH_CHECK(camera_orientation.size(0) == image.size(0), "camera_orientation batch size must match image batch size");
  TORCH_CHECK(sphere_centers.dim() == 2 && sphere_centers.size(1) == 3, "sphere_centers must be N x 3");
  TORCH_CHECK(sphere_radii.dim() == 1, "sphere_radii must be N");
  TORCH_CHECK(plane_points.dim() == 2 && plane_points.size(1) == 3, "plane_points must be N x 3");
  TORCH_CHECK(plane_normals.dim() == 2 && plane_normals.size(1) == 3, "plane_normals must be N x 3");
  TORCH_CHECK(box_centers.dim() == 2 && box_centers.size(1) == 3, "box_centers must be N x 3");
  TORCH_CHECK(box_half_sizes.dim() == 2 && box_half_sizes.size(1) == 3, "box_half_sizes must be N x 3");
  TORCH_CHECK(box_axes.dim() == 3 && box_axes.size(1) == 3 && box_axes.size(2) == 3, "box_axes must be N x 3 x 3");
  TORCH_CHECK(sphere_colors.dim() == 2 && sphere_colors.size(1) == 3, "sphere_colors must be N x 3");
  TORCH_CHECK(plane_colors.dim() == 2 && plane_colors.size(1) == 3, "plane_colors must be N x 3");
  TORCH_CHECK(box_colors.dim() == 2 && box_colors.size(1) == 3, "box_colors must be N x 3");
  TORCH_CHECK(sphere_centers.size(0) == sphere_radii.size(0), "sphere_centers and sphere_radii must have matching lengths");
  TORCH_CHECK(sphere_centers.size(0) == sphere_colors.size(0), "sphere_centers and sphere_colors must have matching lengths");
  TORCH_CHECK(plane_points.size(0) == plane_normals.size(0), "plane_points and plane_normals must have matching lengths");
  TORCH_CHECK(plane_points.size(0) == plane_colors.size(0), "plane_points and plane_colors must have matching lengths");
  TORCH_CHECK(box_centers.size(0) == box_half_sizes.size(0), "box_centers and box_half_sizes must have matching lengths");
  TORCH_CHECK(box_centers.size(0) == box_axes.size(0), "box_centers and box_axes must have matching lengths");
  TORCH_CHECK(box_centers.size(0) == box_colors.size(0), "box_centers and box_colors must have matching lengths");
  TORCH_CHECK(sphere_centers.size(0) > 0 || plane_points.size(0) > 0 || box_centers.size(0) > 0, "at least one object is required");
  TORCH_CHECK(light_dir.numel() == 3 && background.numel() == 3, "light/background vectors must be vec3");
  TORCH_CHECK(camera_origin.dtype() == torch::kFloat32, "camera_origin must be float32");
  TORCH_CHECK(camera_orientation.dtype() == torch::kFloat32, "camera_orientation must be float32");
  TORCH_CHECK(sphere_centers.dtype() == torch::kFloat32, "sphere_centers must be float32");
  TORCH_CHECK(sphere_radii.dtype() == torch::kFloat32, "sphere_radii must be float32");
  TORCH_CHECK(plane_points.dtype() == torch::kFloat32, "plane_points must be float32");
  TORCH_CHECK(plane_normals.dtype() == torch::kFloat32, "plane_normals must be float32");
  TORCH_CHECK(box_centers.dtype() == torch::kFloat32, "box_centers must be float32");
  TORCH_CHECK(box_half_sizes.dtype() == torch::kFloat32, "box_half_sizes must be float32");
  TORCH_CHECK(box_axes.dtype() == torch::kFloat32, "box_axes must be float32");
  TORCH_CHECK(light_dir.dtype() == torch::kFloat32, "light_dir must be float32");
  TORCH_CHECK(background.dtype() == torch::kFloat32, "background must be float32");
  TORCH_CHECK(sphere_colors.dtype() == torch::kFloat32, "sphere_colors must be float32");
  TORCH_CHECK(plane_colors.dtype() == torch::kFloat32, "plane_colors must be float32");
  TORCH_CHECK(box_colors.dtype() == torch::kFloat32, "box_colors must be float32");
  TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
  TORCH_CHECK(instance_map.is_contiguous() && semantic_map.is_contiguous(), "segmentation maps must be contiguous");
  TORCH_CHECK(camera_origin.is_contiguous() && camera_orientation.is_contiguous(), "camera tensors must be contiguous");
  TORCH_CHECK(sphere_centers.is_contiguous() && sphere_radii.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(plane_points.is_contiguous() && plane_normals.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(box_centers.is_contiguous() && box_half_sizes.is_contiguous() && box_axes.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(light_dir.is_contiguous() && background.is_contiguous() && sphere_colors.is_contiguous() && plane_colors.is_contiguous() && box_colors.is_contiguous(), "scene tensors must be contiguous");

  render_scene_cuda(
      image,
      instance_map,
      semantic_map,
      camera_origin,
      camera_orientation,
      sphere_centers,
      sphere_radii,
      plane_points,
      plane_normals,
      box_centers,
      box_half_sizes,
      box_axes,
      light_dir,
      fov_degrees,
      background,
      sphere_colors,
      plane_colors,
      box_colors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "random_scene",
      &random_scene,
      py::arg("seed"),
      py::arg("ground_objects"),
      py::arg("floating_objects"),
      py::arg("cameras"),
      py::arg("scatter_radius"),
      py::arg("ground_y"),
      py::arg("camera_distance"),
      py::arg("camera_height"),
      py::arg("fov_degrees"),
      "Generate random scene tensors directly from the native extension");
  m.def("render_scene", &render_scene, "Render Lambert-shaded geometric objects (CUDA)");
}
