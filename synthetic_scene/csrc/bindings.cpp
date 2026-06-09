#include <torch/extension.h>

namespace py = pybind11;

void render_scene_cuda(
    torch::Tensor image,
    torch::Tensor instance_map,
    torch::Tensor semantic_map,
    torch::Tensor camera_origin,
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
  TORCH_CHECK(camera_origin.is_cuda() && sphere_centers.is_cuda() && sphere_radii.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(plane_points.is_cuda() && plane_normals.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(box_centers.is_cuda() && box_half_sizes.is_cuda() && box_axes.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(light_dir.is_cuda() && background.is_cuda() && sphere_colors.is_cuda() && plane_colors.is_cuda() && box_colors.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(camera_origin.dim() == 2 && camera_origin.size(1) == 3, "camera_origin must be B x 3");
  TORCH_CHECK(camera_origin.size(0) == image.size(0), "camera_origin batch size must match image batch size");
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
  TORCH_CHECK(camera_origin.is_contiguous() && sphere_centers.is_contiguous() && sphere_radii.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(plane_points.is_contiguous() && plane_normals.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(box_centers.is_contiguous() && box_half_sizes.is_contiguous() && box_axes.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(light_dir.is_contiguous() && background.is_contiguous() && sphere_colors.is_contiguous() && plane_colors.is_contiguous() && box_colors.is_contiguous(), "scene tensors must be contiguous");

  render_scene_cuda(
      image,
      instance_map,
      semantic_map,
      camera_origin,
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
  m.def("render_scene", &render_scene, "Render Lambert-shaded geometric objects (CUDA)");
}
