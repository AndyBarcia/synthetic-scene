#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

namespace {

struct Vec3 {
  float x;
  float y;
  float z;
};

__device__ __forceinline__ Vec3 make_vec3(float x, float y, float z) {
  return Vec3{x, y, z};
}

__device__ __forceinline__ Vec3 load_vec3(const float* ptr) {
  return make_vec3(ptr[0], ptr[1], ptr[2]);
}

__device__ __forceinline__ Vec3 add(Vec3 a, Vec3 b) {
  return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Vec3 sub(Vec3 a, Vec3 b) {
  return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3 mul(Vec3 a, float s) {
  return make_vec3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 normalize(Vec3 v) {
  const float len2 = fmaxf(dot(v, v), 1.0e-20f);
  return mul(v, rsqrtf(len2));
}

__global__ void render_sphere_kernel(
    float* image,
    int width,
    int height,
    const float* camera_origin_ptr,
    const float* sphere_center_ptr,
    float sphere_radius,
    const float* light_dir_ptr,
    float fov_degrees,
    const float* background_ptr,
    const float* sphere_color_ptr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Vec3 camera_origin = load_vec3(camera_origin_ptr);
  const Vec3 sphere_center = load_vec3(sphere_center_ptr);
  const Vec3 light_dir = normalize(load_vec3(light_dir_ptr));
  const Vec3 background = load_vec3(background_ptr);
  const Vec3 sphere_color = load_vec3(sphere_color_ptr);

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float fov_radians = fov_degrees * 0.017453292519943295f;
  const float image_plane_scale = tanf(0.5f * fov_radians);
  const float px = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f) *
      aspect * image_plane_scale;
  const float py = (1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2.0f) *
      image_plane_scale;

  const Vec3 ray_origin = camera_origin;
  const Vec3 ray_dir = normalize(make_vec3(px, py, -1.0f));

  const Vec3 oc = sub(ray_origin, sphere_center);
  const float a = dot(ray_dir, ray_dir);
  const float b = 2.0f * dot(oc, ray_dir);
  const float c = dot(oc, oc) - sphere_radius * sphere_radius;
  const float discriminant = b * b - 4.0f * a * c;

  Vec3 color = background;
  if (discriminant >= 0.0f) {
    const float sqrt_disc = sqrtf(discriminant);
    const float inv_2a = 0.5f / a;
    float t = (-b - sqrt_disc) * inv_2a;
    if (t <= 1.0e-4f) {
      t = (-b + sqrt_disc) * inv_2a;
    }
    if (t > 1.0e-4f) {
      const Vec3 hit = add(ray_origin, mul(ray_dir, t));
      const Vec3 normal = normalize(sub(hit, sphere_center));
      const float shade = fmaxf(dot(normal, light_dir), 0.0f);
      const float ambient = 0.08f;
      color = mul(sphere_color, ambient + (1.0f - ambient) * shade);
    }
  }

  const int offset = (y * width + x) * 3;
  image[offset + 0] = color.x;
  image[offset + 1] = color.y;
  image[offset + 2] = color.z;
}

}  // namespace

void render_sphere_cuda(
    torch::Tensor image,
    torch::Tensor camera_origin,
    torch::Tensor sphere_center,
    double sphere_radius,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_color) {
  const int height = static_cast<int>(image.size(0));
  const int width = static_cast<int>(image.size(1));
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  render_sphere_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      image.data_ptr<float>(),
      width,
      height,
      camera_origin.data_ptr<float>(),
      sphere_center.data_ptr<float>(),
      static_cast<float>(sphere_radius),
      light_dir.data_ptr<float>(),
      static_cast<float>(fov_degrees),
      background.data_ptr<float>(),
      sphere_color.data_ptr<float>());

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
