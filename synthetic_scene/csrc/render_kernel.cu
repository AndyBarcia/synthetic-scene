#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

namespace {

constexpr int kMaxSpheres = 64;
constexpr int kMaxPlanes = 64;

struct Vec3 {
  float x;
  float y;
  float z;
};

struct CameraView {
  const float* origin;
  float fov_degrees;
};

struct RenderOptionsView {
  const float* light_dir;
  const float* background;
};

struct SphereView {
  const float* centers;
  const float* radii;
  const float* colors;
  int count;
};

struct PlaneView {
  const float* points;
  const float* normals;
  const float* colors;
  int count;
};

struct SceneView {
  SphereView spheres;
  PlaneView planes;
};

__host__ __device__ __forceinline__ Vec3 make_vec3(float x, float y, float z) {
  return Vec3{x, y, z};
}

__host__ __device__ __forceinline__ Vec3 load_vec3(const float* ptr) {
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

__global__ void render_scene_kernel(
    float* image,
    int width,
    int height,
    CameraView camera,
    SceneView scene,
    RenderOptionsView options) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const Vec3 camera_origin = load_vec3(camera.origin);
  const Vec3 light_dir = normalize(load_vec3(options.light_dir));
  const Vec3 background = load_vec3(options.background);

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float fov_radians = camera.fov_degrees * 0.017453292519943295f;
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
  int closest_plane = -1;

  #pragma unroll
  for (int sphere_idx = 0; sphere_idx < kMaxSpheres; ++sphere_idx) {
    if (sphere_idx >= scene.spheres.count) {
      break;
    }

    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_idx * 3);
    const float sphere_radius = scene.spheres.radii[sphere_idx];
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
      closest_plane = -1;
    }
  }

  #pragma unroll
  for (int plane_idx = 0; plane_idx < kMaxPlanes; ++plane_idx) {
    if (plane_idx >= scene.planes.count) {
      break;
    }

    const Vec3 plane_point = load_vec3(scene.planes.points + plane_idx * 3);
    const Vec3 plane_normal = normalize(load_vec3(scene.planes.normals + plane_idx * 3));
    const float denom = dot(ray_dir, plane_normal);
    if (fabsf(denom) < 1.0e-6f) {
      continue;
    }

    const float t = dot(sub(plane_point, ray_origin), plane_normal) / denom;
    if (t > 1.0e-4f && t < closest_t) {
      closest_t = t;
      closest_sphere = -1;
      closest_plane = plane_idx;
    }
  }

  if (closest_sphere >= 0) {
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + closest_sphere * 3);
    const Vec3 sphere_color = load_vec3(scene.spheres.colors + closest_sphere * 3);
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(sub(hit, sphere_center));
    const float shade = fmaxf(dot(normal, light_dir), 0.0f);
    const float ambient = 0.08f;
    color = mul(sphere_color, ambient + (1.0f - ambient) * shade);
  } else if (closest_plane >= 0) {
    Vec3 plane_normal = normalize(load_vec3(scene.planes.normals + closest_plane * 3));
    if (dot(plane_normal, ray_dir) > 0.0f) {
      plane_normal = mul(plane_normal, -1.0f);
    }
    const Vec3 plane_color = load_vec3(scene.planes.colors + closest_plane * 3);
    const float shade = fmaxf(dot(plane_normal, light_dir), 0.0f);
    const float ambient = 0.08f;
    color = mul(plane_color, ambient + (1.0f - ambient) * shade);
  }

  const int offset = (y * width + x) * 3;
  image[offset + 0] = color.x;
  image[offset + 1] = color.y;
  image[offset + 2] = color.z;
}

}  // namespace

void render_scene_cuda(
    torch::Tensor image,
    torch::Tensor camera_origin,
    torch::Tensor sphere_centers,
    torch::Tensor sphere_radii,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors,
    torch::Tensor plane_colors) {
  const int height = static_cast<int>(image.size(0));
  const int width = static_cast<int>(image.size(1));
  const int sphere_count = static_cast<int>(sphere_centers.size(0));
  const int plane_count = static_cast<int>(plane_points.size(0));
  TORCH_CHECK(sphere_count <= kMaxSpheres, "render_scene supports at most ", kMaxSpheres, " spheres");
  TORCH_CHECK(plane_count <= kMaxPlanes, "render_scene supports at most ", kMaxPlanes, " planes");

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  const CameraView camera{
      camera_origin.data_ptr<float>(),
      static_cast<float>(fov_degrees),
  };
  const SceneView scene{
      SphereView{
          sphere_centers.data_ptr<float>(),
          sphere_radii.data_ptr<float>(),
          sphere_colors.data_ptr<float>(),
          sphere_count,
      },
      PlaneView{
          plane_points.data_ptr<float>(),
          plane_normals.data_ptr<float>(),
          plane_colors.data_ptr<float>(),
          plane_count,
      },
  };
  const RenderOptionsView options{
      light_dir.data_ptr<float>(),
      background.data_ptr<float>(),
  };

  render_scene_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      image.data_ptr<float>(),
      width,
      height,
      camera,
      scene,
      options);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
