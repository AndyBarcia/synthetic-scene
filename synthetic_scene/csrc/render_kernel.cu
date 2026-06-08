#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

namespace {

constexpr int kMaxSpheres = 64;

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

__global__ void render_spheres_kernel(
    float* image,
    int width,
    int height,
    const float* camera_origin_ptr,
    const float* sphere_centers_ptr,
    const float* sphere_radii_ptr,
    int sphere_count,
    const float* light_dir_ptr,
    float fov_degrees,
    const float* background_ptr,
    const float* sphere_colors_ptr) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Vec3 camera_origin = load_vec3(camera_origin_ptr);
  const Vec3 light_dir = normalize(load_vec3(light_dir_ptr));
  const Vec3 background = load_vec3(background_ptr);

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float fov_radians = fov_degrees * 0.017453292519943295f;
  const float image_plane_scale = tanf(0.5f * fov_radians);
  const float px = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f) *
      aspect * image_plane_scale;
  const float py = (1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2.0f) *
      image_plane_scale;

  const Vec3 ray_origin = camera_origin;
  const Vec3 ray_dir = normalize(make_vec3(px, py, -1.0f));

  const float a = dot(ray_dir, ray_dir);

  Vec3 color = background;
  float closest_t = 3.402823466e+38f;
  int closest_sphere = -1;

  #pragma unroll
  for (int sphere_idx = 0; sphere_idx < kMaxSpheres; ++sphere_idx) {
    if (sphere_idx >= sphere_count) {
      break;
    }

    const Vec3 sphere_center = load_vec3(sphere_centers_ptr + sphere_idx * 3);
    const float sphere_radius = sphere_radii_ptr[sphere_idx];
    const Vec3 oc = sub(ray_origin, sphere_center);
    const float b = 2.0f * dot(oc, ray_dir);
    const float c = dot(oc, oc) - sphere_radius * sphere_radius;
    const float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
      continue;
    }

    const float sqrt_disc = sqrtf(discriminant);
    const float inv_2a = 0.5f / a;
    float t = (-b - sqrt_disc) * inv_2a;
    if (t <= 1.0e-4f) {
      t = (-b + sqrt_disc) * inv_2a;
    }
    if (t > 1.0e-4f && t < closest_t) {
      closest_t = t;
      closest_sphere = sphere_idx;
    }
  }

  if (closest_sphere >= 0) {
    const Vec3 sphere_center = load_vec3(sphere_centers_ptr + closest_sphere * 3);
    const Vec3 sphere_color = load_vec3(sphere_colors_ptr + closest_sphere * 3);
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(sub(hit, sphere_center));
    const float shade = fmaxf(dot(normal, light_dir), 0.0f);
    const float ambient = 0.08f;
    color = mul(sphere_color, ambient + (1.0f - ambient) * shade);
  }

  const int offset = (y * width + x) * 3;
  image[offset + 0] = color.x;
  image[offset + 1] = color.y;
  image[offset + 2] = color.z;
}

}  // namespace

void render_spheres_cuda(
    torch::Tensor image,
    torch::Tensor camera_origin,
    torch::Tensor sphere_centers,
    torch::Tensor sphere_radii,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors) {
  const int height = static_cast<int>(image.size(0));
  const int width = static_cast<int>(image.size(1));
  const int sphere_count = static_cast<int>(sphere_centers.size(0));
  TORCH_CHECK(sphere_count <= kMaxSpheres, "render_spheres supports at most ", kMaxSpheres, " spheres");

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  render_spheres_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      image.data_ptr<float>(),
      width,
      height,
      camera_origin.data_ptr<float>(),
      sphere_centers.data_ptr<float>(),
      sphere_radii.data_ptr<float>(),
      sphere_count,
      light_dir.data_ptr<float>(),
      static_cast<float>(fov_degrees),
      background.data_ptr<float>(),
      sphere_colors.data_ptr<float>());

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
