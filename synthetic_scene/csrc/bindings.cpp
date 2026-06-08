#include <torch/extension.h>

void render_sphere_cuda(
    torch::Tensor image,
    torch::Tensor camera_origin,
    torch::Tensor sphere_center,
    double sphere_radius,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_color);

void render_sphere(
    torch::Tensor image,
    torch::Tensor camera_origin,
    torch::Tensor sphere_center,
    double sphere_radius,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_color) {
  TORCH_CHECK(image.is_cuda(), "image must be a CUDA tensor");
  TORCH_CHECK(image.dtype() == torch::kFloat32, "image must be float32");
  TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "image must be H x W x 3");
  TORCH_CHECK(camera_origin.is_cuda() && sphere_center.is_cuda(), "scene vectors must be CUDA tensors");
  TORCH_CHECK(light_dir.is_cuda() && background.is_cuda() && sphere_color.is_cuda(), "scene vectors must be CUDA tensors");
  TORCH_CHECK(camera_origin.numel() == 3 && sphere_center.numel() == 3, "camera and sphere center must be vec3");
  TORCH_CHECK(light_dir.numel() == 3 && background.numel() == 3 && sphere_color.numel() == 3, "light/color vectors must be vec3");
  TORCH_CHECK(camera_origin.dtype() == torch::kFloat32, "camera_origin must be float32");
  TORCH_CHECK(sphere_center.dtype() == torch::kFloat32, "sphere_center must be float32");
  TORCH_CHECK(light_dir.dtype() == torch::kFloat32, "light_dir must be float32");
  TORCH_CHECK(background.dtype() == torch::kFloat32, "background must be float32");
  TORCH_CHECK(sphere_color.dtype() == torch::kFloat32, "sphere_color must be float32");
  TORCH_CHECK(image.is_contiguous(), "image must be contiguous");
  TORCH_CHECK(camera_origin.is_contiguous() && sphere_center.is_contiguous(), "vectors must be contiguous");
  TORCH_CHECK(light_dir.is_contiguous() && background.is_contiguous() && sphere_color.is_contiguous(), "vectors must be contiguous");

  render_sphere_cuda(
      image,
      camera_origin,
      sphere_center,
      sphere_radius,
      light_dir,
      fov_degrees,
      background,
      sphere_color);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_sphere", &render_sphere, "Render a Lambert-shaded sphere (CUDA)");
}
