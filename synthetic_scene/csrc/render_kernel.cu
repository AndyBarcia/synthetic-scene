#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

namespace {

constexpr int kMaxSpheres = 64;
constexpr int kMaxPlanes = 64;
constexpr int kMaxBoxes = 64;

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

struct BoxView {
  const float* centers;
  const float* half_sizes;
  const float* axes;
  const float* colors;
  int count;
};

struct SceneView {
  SphereView spheres;
  PlaneView planes;
  BoxView boxes;
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

__device__ __forceinline__ bool intersect_box(
    Vec3 ray_origin,
    Vec3 ray_dir,
    Vec3 center,
    Vec3 half_size,
    const float* axes,
    float* out_t,
    Vec3* out_normal) {
  const Vec3 axis_x = normalize(load_vec3(axes + 0));
  const Vec3 axis_y = normalize(load_vec3(axes + 3));
  const Vec3 axis_z = normalize(load_vec3(axes + 6));
  const Vec3 local_origin_delta = sub(ray_origin, center);

  const float origin_local[3] = {
      dot(local_origin_delta, axis_x),
      dot(local_origin_delta, axis_y),
      dot(local_origin_delta, axis_z),
  };
  const float dir_local[3] = {
      dot(ray_dir, axis_x),
      dot(ray_dir, axis_y),
      dot(ray_dir, axis_z),
  };
  const float extents[3] = {half_size.x, half_size.y, half_size.z};
  const Vec3 world_axes[3] = {axis_x, axis_y, axis_z};

  float t_min = -3.402823466e+38f;
  float t_max = 3.402823466e+38f;
  Vec3 near_normal = make_vec3(0.0f, 0.0f, 0.0f);
  Vec3 far_normal = make_vec3(0.0f, 0.0f, 0.0f);

  #pragma unroll
  for (int axis_idx = 0; axis_idx < 3; ++axis_idx) {
    const float origin_axis = origin_local[axis_idx];
    const float dir_axis = dir_local[axis_idx];
    const float extent = extents[axis_idx];

    if (fabsf(dir_axis) < 1.0e-6f) {
      if (origin_axis < -extent || origin_axis > extent) {
        return false;
      }
      continue;
    }

    const float inv_dir = 1.0f / dir_axis;
    float t1 = (-extent - origin_axis) * inv_dir;
    float t2 = (extent - origin_axis) * inv_dir;
    Vec3 n1 = mul(world_axes[axis_idx], -1.0f);
    Vec3 n2 = world_axes[axis_idx];
    if (t1 > t2) {
      const float tmp_t = t1;
      t1 = t2;
      t2 = tmp_t;
      const Vec3 tmp_n = n1;
      n1 = n2;
      n2 = tmp_n;
    }

    if (t1 > t_min) {
      t_min = t1;
      near_normal = n1;
    }
    if (t2 < t_max) {
      t_max = t2;
      far_normal = n2;
    }
    if (t_min > t_max) {
      return false;
    }
  }

  float t = t_min;
  Vec3 normal = near_normal;
  if (t <= 1.0e-4f) {
    t = t_max;
    normal = far_normal;
  }
  if (t <= 1.0e-4f) {
    return false;
  }

  *out_t = t;
  *out_normal = normal;
  return true;
}

__global__ void render_scene_kernel(
    float* image,
    int* instance_map,
    int* semantic_map,
    int width,
    int height,
    CameraView camera,
    SceneView scene,
    RenderOptionsView options) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = blockIdx.z;
  if (x >= width || y >= height) {
    return;
  }

  const Vec3 camera_origin = load_vec3(camera.origin + batch_idx * 3);
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
  int closest_box = -1;
  int instance_id = 0;
  int semantic_id = 0;
  Vec3 closest_box_normal = make_vec3(0.0f, 0.0f, 0.0f);

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
      closest_box = -1;
    }
  }

  #pragma unroll
  for (int box_idx = 0; box_idx < kMaxBoxes; ++box_idx) {
    if (box_idx >= scene.boxes.count) {
      break;
    }

    const Vec3 box_center = load_vec3(scene.boxes.centers + box_idx * 3);
    const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_idx * 3);
    float t = 0.0f;
    Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
    if (intersect_box(ray_origin, ray_dir, box_center, box_half_size, scene.boxes.axes + box_idx * 9, &t, &normal) &&
        t < closest_t) {
      closest_t = t;
      closest_sphere = -1;
      closest_plane = -1;
      closest_box = box_idx;
      closest_box_normal = normal;
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
      closest_box = -1;
    }
  }

  if (closest_sphere >= 0) {
    instance_id = closest_sphere + 1;
    semantic_id = 1;
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + closest_sphere * 3);
    const Vec3 sphere_color = load_vec3(scene.spheres.colors + closest_sphere * 3);
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(sub(hit, sphere_center));
    const float shade = fmaxf(dot(normal, light_dir), 0.0f);
    const float ambient = 0.08f;
    color = mul(sphere_color, ambient + (1.0f - ambient) * shade);
  } else if (closest_box >= 0) {
    instance_id = scene.spheres.count + scene.planes.count + closest_box + 1;
    semantic_id = 3;
    Vec3 normal = normalize(closest_box_normal);
    if (dot(normal, ray_dir) > 0.0f) {
      normal = mul(normal, -1.0f);
    }
    const Vec3 box_color = load_vec3(scene.boxes.colors + closest_box * 3);
    const float shade = fmaxf(dot(normal, light_dir), 0.0f);
    const float ambient = 0.08f;
    color = mul(box_color, ambient + (1.0f - ambient) * shade);
  } else if (closest_plane >= 0) {
    instance_id = scene.spheres.count + closest_plane + 1;
    semantic_id = 2;
    Vec3 plane_normal = normalize(load_vec3(scene.planes.normals + closest_plane * 3));
    if (dot(plane_normal, ray_dir) > 0.0f) {
      plane_normal = mul(plane_normal, -1.0f);
    }
    const Vec3 plane_color = load_vec3(scene.planes.colors + closest_plane * 3);
    const float shade = fmaxf(dot(plane_normal, light_dir), 0.0f);
    const float ambient = 0.08f;
    color = mul(plane_color, ambient + (1.0f - ambient) * shade);
  }

  const int image_offset = ((batch_idx * 3 * height + y) * width) + x;
  image[image_offset + 0 * height * width] = color.x;
  image[image_offset + 1 * height * width] = color.y;
  image[image_offset + 2 * height * width] = color.z;

  const int map_offset = (batch_idx * height + y) * width + x;
  if (instance_map != nullptr) {
    instance_map[map_offset] = instance_id;
  }
  if (semantic_map != nullptr) {
    semantic_map[map_offset] = semantic_id;
  }
}

}  // namespace

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
    torch::Tensor box_colors) {
  const int batch_size = static_cast<int>(image.size(0));
  const int height = static_cast<int>(image.size(2));
  const int width = static_cast<int>(image.size(3));
  const int sphere_count = static_cast<int>(sphere_centers.size(0));
  const int plane_count = static_cast<int>(plane_points.size(0));
  const int box_count = static_cast<int>(box_centers.size(0));
  TORCH_CHECK(sphere_count <= kMaxSpheres, "render_scene supports at most ", kMaxSpheres, " spheres");
  TORCH_CHECK(plane_count <= kMaxPlanes, "render_scene supports at most ", kMaxPlanes, " planes");
  TORCH_CHECK(box_count <= kMaxBoxes, "render_scene supports at most ", kMaxBoxes, " boxes");

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);

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
      BoxView{
          box_centers.data_ptr<float>(),
          box_half_sizes.data_ptr<float>(),
          box_axes.data_ptr<float>(),
          box_colors.data_ptr<float>(),
          box_count,
      },
  };
  const RenderOptionsView options{
      light_dir.data_ptr<float>(),
      background.data_ptr<float>(),
  };

  render_scene_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      image.data_ptr<float>(),
      instance_map.numel() == 0 ? nullptr : instance_map.data_ptr<int>(),
      semantic_map.numel() == 0 ? nullptr : semantic_map.data_ptr<int>(),
      width,
      height,
      camera,
      scene,
      options);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
