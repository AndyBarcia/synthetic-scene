#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kMaxSpheres = 64;
constexpr int kMaxPlanes = 64;
constexpr int kMaxBoxes = 64;
constexpr float kRayTMin = 1.0e-4f;
constexpr float kParallelEpsilon = 1.0e-6f;
constexpr float kFloatMax = 3.402823466e+38f;
constexpr float kCameraNear = 1.0e-3f;

struct Vec3 {
  float x;
  float y;
  float z;
};

struct RenderOptionsView {
  const float* light_dir;
  const float* background;
  float fov_degrees;
  float ambient;
  int shadows;
  float shadow_strength;
};

struct SphereView {
  const float* centers;
  const float* radii;
  const float* colors;
  const int* counts;
  int count;
};

struct PlaneView {
  const float* points;
  const float* normals;
  const float* colors;
  const int* counts;
  int count;
};

struct BoxView {
  const float* centers;
  const float* half_sizes;
  const float* axes;
  const float* colors;
  const int* counts;
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

__device__ __forceinline__ bool intersect_sphere(
    Vec3 ray_origin,
    Vec3 ray_dir,
    Vec3 center,
    float radius,
    float* out_t) {
  const Vec3 oc = sub(ray_origin, center);
  const float a = dot(ray_dir, ray_dir);
  const float b = 2.0f * dot(oc, ray_dir);
  const float c = dot(oc, oc) - radius * radius;
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) {
    return false;
  }

  const float sqrt_disc = sqrtf(discriminant);
  const float inv_2a = 0.5f / a;
  float t = (-b - sqrt_disc) * inv_2a;
  if (t <= kRayTMin) {
    t = (-b + sqrt_disc) * inv_2a;
  }
  if (t <= kRayTMin) {
    return false;
  }

  *out_t = t;
  return true;
}

__device__ __forceinline__ bool intersect_plane(
    Vec3 ray_origin,
    Vec3 ray_dir,
    Vec3 point,
    Vec3 normal,
    float* out_t) {
  const float denom = dot(ray_dir, normal);
  if (fabsf(denom) < kParallelEpsilon) {
    return false;
  }

  const float t = dot(sub(point, ray_origin), normal) / denom;
  if (t <= kRayTMin) {
    return false;
  }

  *out_t = t;
  return true;
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

  float t_min = -kFloatMax;
  float t_max = kFloatMax;
  Vec3 near_normal = make_vec3(0.0f, 0.0f, 0.0f);
  Vec3 far_normal = make_vec3(0.0f, 0.0f, 0.0f);

  #pragma unroll
  for (int axis_idx = 0; axis_idx < 3; ++axis_idx) {
    const float origin_axis = origin_local[axis_idx];
    const float dir_axis = dir_local[axis_idx];
    const float extent = extents[axis_idx];

    if (fabsf(dir_axis) < kParallelEpsilon) {
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
  if (t <= kRayTMin) {
    t = t_max;
    normal = far_normal;
  }
  if (t <= kRayTMin) {
    return false;
  }

  *out_t = t;
  *out_normal = normal;
  return true;
}

__device__ __forceinline__ bool project_to_pixel(
    Vec3 point,
    float aspect,
    float image_plane_scale,
    int width,
    int height,
    float* out_x,
    float* out_y) {
  const float depth = -point.z;
  if (depth <= kCameraNear) {
    return false;
  }

  const float inv_depth = 1.0f / depth;
  const float ndc_x = point.x * inv_depth / (aspect * image_plane_scale);
  const float ndc_y = point.y * inv_depth / image_plane_scale;
  *out_x = (ndc_x + 1.0f) * 0.5f * static_cast<float>(width) - 0.5f;
  *out_y = (1.0f - ndc_y) * 0.5f * static_cast<float>(height) - 0.5f;
  return true;
}

__device__ __forceinline__ bool ranges_overlap(float min_a, float max_a, float min_b, float max_b) {
  return min_a <= max_b && max_a >= min_b;
}

__device__ __forceinline__ bool pixel_bounds_overlap(
    float min_x,
    float min_y,
    float max_x,
    float max_y,
    float block_min_x,
    float block_min_y,
    float block_max_x,
    float block_max_y) {
  return ranges_overlap(min_x, max_x, block_min_x, block_max_x) &&
      ranges_overlap(min_y, max_y, block_min_y, block_max_y);
}

__device__ __forceinline__ bool sphere_overlaps_block(
    Vec3 center,
    float radius,
    float aspect,
    float image_plane_scale,
    int width,
    int height,
    float block_min_x,
    float block_min_y,
    float block_max_x,
    float block_max_y) {
  const float nearest_depth = -(center.z + radius);
  if (nearest_depth <= kCameraNear) {
    return true;
  }

  float center_x = 0.0f;
  float center_y = 0.0f;
  if (!project_to_pixel(center, aspect, image_plane_scale, width, height, &center_x, &center_y)) {
    return false;
  }

  const float radius_pixels_x = radius / nearest_depth / (aspect * image_plane_scale) * 0.5f *
      static_cast<float>(width);
  const float radius_pixels_y = radius / nearest_depth / image_plane_scale * 0.5f * static_cast<float>(height);
  const float radius_pixels = fmaxf(radius_pixels_x, radius_pixels_y) + 1.0f;
  return pixel_bounds_overlap(
      center_x - radius_pixels,
      center_y - radius_pixels,
      center_x + radius_pixels,
      center_y + radius_pixels,
      block_min_x,
      block_min_y,
      block_max_x,
      block_max_y);
}

__device__ __forceinline__ bool box_overlaps_block(
    Vec3 center,
    Vec3 half_size,
    const float* axes,
    float aspect,
    float image_plane_scale,
    int width,
    int height,
    float block_min_x,
    float block_min_y,
    float block_max_x,
    float block_max_y) {
  const Vec3 axis_x = normalize(load_vec3(axes + 0));
  const Vec3 axis_y = normalize(load_vec3(axes + 3));
  const Vec3 axis_z = normalize(load_vec3(axes + 6));
  float min_x = kFloatMax;
  float min_y = kFloatMax;
  float max_x = -kFloatMax;
  float max_y = -kFloatMax;
  bool projected_any = false;

  #pragma unroll
  for (int corner_idx = 0; corner_idx < 8; ++corner_idx) {
    const float sx = (corner_idx & 1) ? 1.0f : -1.0f;
    const float sy = (corner_idx & 2) ? 1.0f : -1.0f;
    const float sz = (corner_idx & 4) ? 1.0f : -1.0f;
    const Vec3 corner = add(
        add(add(center, mul(axis_x, sx * half_size.x)), mul(axis_y, sy * half_size.y)),
        mul(axis_z, sz * half_size.z));
    if (corner.z >= -kCameraNear) {
      return true;
    }

    float pixel_x = 0.0f;
    float pixel_y = 0.0f;
    if (project_to_pixel(corner, aspect, image_plane_scale, width, height, &pixel_x, &pixel_y)) {
      projected_any = true;
      min_x = fminf(min_x, pixel_x);
      min_y = fminf(min_y, pixel_y);
      max_x = fmaxf(max_x, pixel_x);
      max_y = fmaxf(max_y, pixel_y);
    }
  }

  if (!projected_any) {
    return false;
  }
  return pixel_bounds_overlap(min_x - 1.0f, min_y - 1.0f, max_x + 1.0f, max_y + 1.0f, block_min_x, block_min_y,
      block_max_x, block_max_y);
}

__device__ __forceinline__ bool is_shadowed(
    Vec3 ray_origin,
    Vec3 light_dir,
    const SceneView& scene,
    int batch_idx,
    int sphere_count,
    int plane_count,
    int box_count,
    int skip_kind,
    int skip_idx) {
  #pragma unroll
  for (int sphere_idx = 0; sphere_idx < kMaxSpheres; ++sphere_idx) {
    if (sphere_idx >= sphere_count) {
      break;
    }
    if (skip_kind == 1 && sphere_idx == skip_idx) {
      continue;
    }

    const int sphere_offset = (batch_idx * scene.spheres.count + sphere_idx);
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
    const float sphere_radius = scene.spheres.radii[sphere_offset];
    float t = 0.0f;
    if (intersect_sphere(ray_origin, light_dir, sphere_center, sphere_radius, &t)) {
      return true;
    }
  }

  #pragma unroll
  for (int box_idx = 0; box_idx < kMaxBoxes; ++box_idx) {
    if (box_idx >= box_count) {
      break;
    }
    if (skip_kind == 3 && box_idx == skip_idx) {
      continue;
    }

    const int box_offset = (batch_idx * scene.boxes.count + box_idx);
    const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
    const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
    float t = 0.0f;
    Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
    if (intersect_box(ray_origin, light_dir, box_center, box_half_size, scene.boxes.axes + box_offset * 9, &t, &normal)) {
      return true;
    }
  }

  #pragma unroll
  for (int plane_idx = 0; plane_idx < kMaxPlanes; ++plane_idx) {
    if (plane_idx >= plane_count) {
      break;
    }
    if (skip_kind == 2 && plane_idx == skip_idx) {
      continue;
    }

    const int plane_offset = (batch_idx * scene.planes.count + plane_idx);
    const Vec3 plane_point = load_vec3(scene.planes.points + plane_offset * 3);
    const Vec3 plane_normal = normalize(load_vec3(scene.planes.normals + plane_offset * 3));
    float t = 0.0f;
    if (intersect_plane(ray_origin, light_dir, plane_point, plane_normal, &t)) {
      return true;
    }
  }

  return false;
}

__device__ __forceinline__ Vec3 apply_lighting(
    Vec3 base_color,
    Vec3 hit,
    Vec3 normal,
    Vec3 ray_dir,
    Vec3 light_dir,
    const SceneView& scene,
    const RenderOptionsView& options,
    int batch_idx,
    int sphere_count,
    int plane_count,
    int box_count,
    int skip_kind,
    int skip_idx) {
  if (dot(normal, ray_dir) > 0.0f) {
    normal = mul(normal, -1.0f);
  }

  const float ambient = options.ambient;
  const float shade = fmaxf(dot(normal, light_dir), 0.0f);
  float direct = (1.0f - ambient) * shade;
  if (options.shadows && direct > 0.0f) {
    const Vec3 shadow_origin = add(hit, mul(normal, 1.0e-3f));
    if (is_shadowed(shadow_origin, light_dir, scene, batch_idx, sphere_count, plane_count, box_count, skip_kind, skip_idx)) {
      direct *= (1.0f - options.shadow_strength);
    }
  }

  return mul(base_color, ambient + direct);
}

__global__ void render_scene_kernel(
    float* image,
    int* instance_map,
    int* semantic_map,
    int width,
    int height,
    SceneView scene,
    RenderOptionsView options) {
  __shared__ int visible_spheres[kMaxSpheres];
  __shared__ int visible_planes[kMaxPlanes];
  __shared__ int visible_boxes[kMaxBoxes];
  __shared__ int visible_sphere_count;
  __shared__ int visible_plane_count;
  __shared__ int visible_box_count;

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = blockIdx.z;
  const int sphere_count = scene.spheres.counts[batch_idx];
  const int plane_count = scene.planes.counts[batch_idx];
  const int box_count = scene.boxes.counts[batch_idx];

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float fov_radians = options.fov_degrees * 0.017453292519943295f;
  const float image_plane_scale = tanf(0.5f * fov_radians);
  const float block_min_x = static_cast<float>(blockIdx.x * blockDim.x);
  const float block_min_y = static_cast<float>(blockIdx.y * blockDim.y);
  const float block_max_x = static_cast<float>(min(width - 1, static_cast<int>((blockIdx.x + 1) * blockDim.x) - 1));
  const float block_max_y = static_cast<float>(min(height - 1, static_cast<int>((blockIdx.y + 1) * blockDim.y) - 1));
  const int thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
  const int threads_per_block = blockDim.x * blockDim.y;

  if (thread_linear == 0) {
    visible_sphere_count = 0;
    visible_plane_count = 0;
    visible_box_count = 0;
  }
  __syncthreads();

  for (int sphere_idx = thread_linear; sphere_idx < sphere_count; sphere_idx += threads_per_block) {
    const int sphere_offset = (batch_idx * scene.spheres.count + sphere_idx);
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
    const float sphere_radius = scene.spheres.radii[sphere_offset];
    if (sphere_overlaps_block(
            sphere_center,
            sphere_radius,
            aspect,
            image_plane_scale,
            width,
            height,
            block_min_x,
            block_min_y,
            block_max_x,
            block_max_y)) {
      const int list_idx = atomicAdd(&visible_sphere_count, 1);
      visible_spheres[list_idx] = sphere_idx;
    }
  }

  for (int box_idx = thread_linear; box_idx < box_count; box_idx += threads_per_block) {
    const int box_offset = (batch_idx * scene.boxes.count + box_idx);
    const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
    const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
    if (box_overlaps_block(
            box_center,
            box_half_size,
            scene.boxes.axes + box_offset * 9,
            aspect,
            image_plane_scale,
            width,
            height,
            block_min_x,
            block_min_y,
            block_max_x,
            block_max_y)) {
      const int list_idx = atomicAdd(&visible_box_count, 1);
      visible_boxes[list_idx] = box_idx;
    }
  }

  for (int plane_idx = thread_linear; plane_idx < plane_count; plane_idx += threads_per_block) {
    const int list_idx = atomicAdd(&visible_plane_count, 1);
    visible_planes[list_idx] = plane_idx;
  }

  __syncthreads();

  if (x >= width || y >= height) {
    return;
  }

  const Vec3 light_dir = normalize(load_vec3(options.light_dir));
  const Vec3 background = load_vec3(options.background);
  const float px = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f) *
      aspect * image_plane_scale;
  const float py = (1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2.0f) *
      image_plane_scale;

  const Vec3 ray_origin = make_vec3(0.0f, 0.0f, 0.0f);
  const Vec3 ray_dir = normalize(make_vec3(px, py, -1.0f));

  Vec3 color = background;
  float closest_t = kFloatMax;
  int closest_sphere = -1;
  int closest_plane = -1;
  int closest_box = -1;
  int instance_id = 0;
  int semantic_id = 0;
  Vec3 closest_box_normal = make_vec3(0.0f, 0.0f, 0.0f);

  for (int list_idx = 0; list_idx < visible_sphere_count; ++list_idx) {
    const int sphere_idx = visible_spheres[list_idx];

    const int sphere_offset = (batch_idx * scene.spheres.count + sphere_idx);
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
    const float sphere_radius = scene.spheres.radii[sphere_offset];
    float t = 0.0f;
    if (intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius, &t) && t < closest_t) {
      closest_t = t;
      closest_sphere = sphere_idx;
      closest_plane = -1;
      closest_box = -1;
    }
  }

  for (int list_idx = 0; list_idx < visible_box_count; ++list_idx) {
    const int box_idx = visible_boxes[list_idx];

    const int box_offset = (batch_idx * scene.boxes.count + box_idx);
    const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
    const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
    float t = 0.0f;
    Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
    if (intersect_box(ray_origin, ray_dir, box_center, box_half_size, scene.boxes.axes + box_offset * 9, &t, &normal) &&
        t < closest_t) {
      closest_t = t;
      closest_sphere = -1;
      closest_plane = -1;
      closest_box = box_idx;
      closest_box_normal = normal;
    }
  }

  for (int list_idx = 0; list_idx < visible_plane_count; ++list_idx) {
    const int plane_idx = visible_planes[list_idx];

    const int plane_offset = (batch_idx * scene.planes.count + plane_idx);
    const Vec3 plane_point = load_vec3(scene.planes.points + plane_offset * 3);
    const Vec3 plane_normal = normalize(load_vec3(scene.planes.normals + plane_offset * 3));
    float t = 0.0f;
    if (intersect_plane(ray_origin, ray_dir, plane_point, plane_normal, &t) && t < closest_t) {
      closest_t = t;
      closest_sphere = -1;
      closest_plane = plane_idx;
      closest_box = -1;
    }
  }

  if (closest_sphere >= 0) {
    const int sphere_offset = (batch_idx * scene.spheres.count + closest_sphere);
    instance_id = closest_sphere + 1;
    semantic_id = 1;
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
    const Vec3 sphere_color = load_vec3(scene.spheres.colors + sphere_offset * 3);
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(sub(hit, sphere_center));
    color = apply_lighting(
        sphere_color, hit, normal, ray_dir, light_dir, scene, options, batch_idx, sphere_count, plane_count, box_count, 1,
        closest_sphere);
  } else if (closest_box >= 0) {
    const int box_offset = (batch_idx * scene.boxes.count + closest_box);
    instance_id = sphere_count + plane_count + closest_box + 1;
    semantic_id = 3;
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(closest_box_normal);
    const Vec3 box_color = load_vec3(scene.boxes.colors + box_offset * 3);
    color = apply_lighting(
        box_color, hit, normal, ray_dir, light_dir, scene, options, batch_idx, sphere_count, plane_count, box_count, 3,
        closest_box);
  } else if (closest_plane >= 0) {
    const int plane_offset = (batch_idx * scene.planes.count + closest_plane);
    instance_id = sphere_count + closest_plane + 1;
    semantic_id = 2;
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 plane_normal = normalize(load_vec3(scene.planes.normals + plane_offset * 3));
    const Vec3 plane_color = load_vec3(scene.planes.colors + plane_offset * 3);
    color = apply_lighting(
        plane_color, hit, plane_normal, ray_dir, light_dir, scene, options, batch_idx, sphere_count, plane_count,
        box_count, 2, closest_plane);
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
    torch::Tensor sphere_centers,
    torch::Tensor sphere_radii,
    torch::Tensor sphere_counts,
    torch::Tensor plane_points,
    torch::Tensor plane_normals,
    torch::Tensor plane_counts,
    torch::Tensor box_centers,
    torch::Tensor box_half_sizes,
    torch::Tensor box_axes,
    torch::Tensor box_counts,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors,
    torch::Tensor plane_colors,
    torch::Tensor box_colors,
    double ambient,
    bool shadows,
    double shadow_strength) {
  const int batch_size = static_cast<int>(image.size(0));
  const int height = static_cast<int>(image.size(2));
  const int width = static_cast<int>(image.size(3));
  const int sphere_count = static_cast<int>(sphere_centers.size(1));
  const int plane_count = static_cast<int>(plane_points.size(1));
  const int box_count = static_cast<int>(box_centers.size(1));
  TORCH_CHECK(sphere_count <= kMaxSpheres, "render_scene supports at most ", kMaxSpheres, " spheres");
  TORCH_CHECK(plane_count <= kMaxPlanes, "render_scene supports at most ", kMaxPlanes, " planes");
  TORCH_CHECK(box_count <= kMaxBoxes, "render_scene supports at most ", kMaxBoxes, " boxes");

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);

  const SceneView scene{
      SphereView{
          sphere_centers.data_ptr<float>(),
          sphere_radii.data_ptr<float>(),
          sphere_colors.data_ptr<float>(),
          sphere_counts.data_ptr<int>(),
          sphere_count,
      },
      PlaneView{
          plane_points.data_ptr<float>(),
          plane_normals.data_ptr<float>(),
          plane_colors.data_ptr<float>(),
          plane_counts.data_ptr<int>(),
          plane_count,
      },
      BoxView{
          box_centers.data_ptr<float>(),
          box_half_sizes.data_ptr<float>(),
          box_axes.data_ptr<float>(),
          box_colors.data_ptr<float>(),
          box_counts.data_ptr<int>(),
          box_count,
      },
  };
  const RenderOptionsView options{
      light_dir.data_ptr<float>(),
      background.data_ptr<float>(),
      static_cast<float>(fov_degrees),
      static_cast<float>(ambient),
      shadows ? 1 : 0,
      static_cast<float>(shadow_strength),
  };

  render_scene_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      image.data_ptr<float>(),
      instance_map.numel() == 0 ? nullptr : instance_map.data_ptr<int>(),
      semantic_map.numel() == 0 ? nullptr : semantic_map.data_ptr<int>(),
      width,
      height,
      scene,
      options);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
