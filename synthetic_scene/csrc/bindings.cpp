#include <torch/extension.h>

void render_spheres_cuda(
    torch::Tensor image,
    torch::Tensor camera_origin,
    torch::Tensor sphere_centers,
    torch::Tensor sphere_radii,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors);

void render_spheres(
    torch::Tensor image,
    torch::Tensor camera_origin,
    torch::Tensor sphere_centers,
    torch::Tensor sphere_radii,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors) {
  TORCH_CHECK(image.is_cuda(), "image must be a CUDA tensor");
  TORCH_CHECK(image.dtype() == torch::kFloat32, "image must be float32");
  TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "image must be H x W x 3");
  TORCH_CHECK(camera_origin.is_cuda() && sphere_centers.is_cuda() && sphere_radii.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(light_dir.is_cuda() && background.is_cuda() && sphere_colors.is_cuda(), "scene tensors must be CUDA tensors");
  TORCH_CHECK(camera_origin.numel() == 3, "camera_origin must be vec3");
  TORCH_CHECK(sphere_centers.dim() == 2 && sphere_centers.size(1) == 3, "sphere_centers must be N x 3");
  TORCH_CHECK(sphere_radii.dim() == 1, "sphere_radii must be N");
  TORCH_CHECK(sphere_colors.dim() == 2 && sphere_colors.size(1) == 3, "sphere_colors must be N x 3");
  TORCH_CHECK(sphere_centers.size(0) == sphere_radii.size(0), "sphere_centers and sphere_radii must have matching lengths");
  TORCH_CHECK(sphere_centers.size(0) == sphere_colors.size(0), "sphere_centers and sphere_colors must have matching lengths");
  TORCH_CHECK(sphere_centers.size(0) > 0, "at least one sphere is required");
  TORCH_CHECK(light_dir.numel() == 3 && background.numel() == 3, "light/background vectors must be vec3");
  TORCH_CHECK(camera_origin.dtype() == torch::kFloat32, "camera_origin must be float32");
  TORCH_CHECK(sphere_centers.dtype() == torch::kFloat32, "sphere_centers must be float32");
  TORCH_CHECK(sphere_radii.dtype() == torch::kFloat32, "sphere_radii must be float32");
  TORCH_CHECK(light_dir.dtype() == torch::kFloat32, "light_dir must be float32");
  TORCH_CHECK(background.dtype() == torch::kFloat32, "background must be float32");
  TORCH_CHECK(sphere_colors.dtype() == torch::kFloat32, "sphere_colors must be float32");
  TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
  TORCH_CHECK(camera_origin.is_contiguous() && sphere_centers.is_contiguous() && sphere_radii.is_contiguous(), "scene tensors must be contiguous");
  TORCH_CHECK(light_dir.is_contiguous() && background.is_contiguous() && sphere_colors.is_contiguous(), "scene tensors must be contiguous");

  render_spheres_cuda(
      image,
      camera_origin,
      sphere_centers,
      sphere_radii,
      light_dir,
      fov_degrees,
      background,
      sphere_colors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_spheres", &render_spheres, "Render Lambert-shaded spheres (CUDA)");
}
