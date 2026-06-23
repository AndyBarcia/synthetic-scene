#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kMaxSpheres = 64;
constexpr int kMaxBoxes = 64;
constexpr int kMaxFinitePrimitives = kMaxSpheres + kMaxBoxes;
constexpr int kDepthBins = 8;
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
  const int* counts;
  int count;
};

struct TerrainView {
  const float* base_heights;
  const float* depth_limits;
  const float* phase_xs;
  const float* phase_zs;
  const float* dz;
  const float* dz_growth;
  const float* colors;
  const int* counts;
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
  TerrainView terrain;
  BoxView boxes;
};

struct Aabb {
  Vec3 min;
  Vec3 max;
};

struct LightBasis {
  Vec3 u;
  Vec3 v;
  Vec3 w;
};

struct LightBounds {
  float u_min;
  float u_max;
  float v_min;
  float v_max;
  float w_min;
  float w_max;
};

struct PrimitiveRef {
  int kind;
  int index;
};

struct HitRecord {
  float t;
  int sphere;
  int box;
  int terrain;
  Vec3 box_normal;
  Vec3 terrain_normal;
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

__device__ __forceinline__ Vec3 min_vec3(Vec3 a, Vec3 b) {
  return make_vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ Vec3 max_vec3(Vec3 a, Vec3 b) {
  return make_vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ Vec3 mul(Vec3 a, float s) {
  return make_vec3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ Vec3 cross(Vec3 a, Vec3 b) {
  return make_vec3(
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ Vec3 normalize(Vec3 v) {
  const float len2 = fmaxf(dot(v, v), 1.0e-20f);
  return mul(v, rsqrtf(len2));
}

__device__ __forceinline__ LightBasis make_light_basis(Vec3 light_dir) {
  const Vec3 w = normalize(light_dir);
  const Vec3 helper = fabsf(w.y) < 0.999f ? make_vec3(0.0f, 1.0f, 0.0f) : make_vec3(1.0f, 0.0f, 0.0f);
  const Vec3 u = normalize(cross(helper, w));
  const Vec3 v = cross(w, u);
  return LightBasis{u, v, w};
}

__device__ __forceinline__ LightBounds empty_light_bounds() {
  return LightBounds{kFloatMax, -kFloatMax, kFloatMax, -kFloatMax, kFloatMax, -kFloatMax};
}

__device__ __forceinline__ LightBounds extend_light_bounds_point(
    LightBounds bounds,
    Vec3 point,
    LightBasis basis) {
  const float u = dot(point, basis.u);
  const float v = dot(point, basis.v);
  const float w = dot(point, basis.w);
  bounds.u_min = fminf(bounds.u_min, u);
  bounds.u_max = fmaxf(bounds.u_max, u);
  bounds.v_min = fminf(bounds.v_min, v);
  bounds.v_max = fmaxf(bounds.v_max, v);
  bounds.w_min = fminf(bounds.w_min, w);
  bounds.w_max = fmaxf(bounds.w_max, w);
  return bounds;
}

__device__ __forceinline__ LightBounds project_aabb_to_light_bounds(Aabb aabb, LightBasis basis) {
  LightBounds bounds = empty_light_bounds();
  #pragma unroll
  for (int corner_idx = 0; corner_idx < 8; ++corner_idx) {
    const Vec3 corner = make_vec3(
        (corner_idx & 1) ? aabb.max.x : aabb.min.x,
        (corner_idx & 2) ? aabb.max.y : aabb.min.y,
        (corner_idx & 4) ? aabb.max.z : aabb.min.z);
    bounds = extend_light_bounds_point(bounds, corner, basis);
  }
  return bounds;
}

__device__ __forceinline__ bool light_ranges_overlap(float a_min, float a_max, float b_min, float b_max) {
  return a_min <= b_max && a_max >= b_min;
}

__device__ __forceinline__ bool aabb_can_shadow_block(
    Aabb caster_bounds,
    LightBounds receiver_bounds,
    LightBasis basis) {
  const LightBounds caster_bounds_ls = project_aabb_to_light_bounds(caster_bounds, basis);

  const bool uv_overlap =
      light_ranges_overlap(caster_bounds_ls.u_min, caster_bounds_ls.u_max, receiver_bounds.u_min, receiver_bounds.u_max) &&
      light_ranges_overlap(caster_bounds_ls.v_min, caster_bounds_ls.v_max, receiver_bounds.v_min, receiver_bounds.v_max);

  // Shadow rays travel in +basis.w. A caster entirely behind every possible receiver in this
  // screen block cannot affect the block. This is intentionally conservative: a caster far
  // beyond the receiver range is still kept and then rejected by the actual ray/AABB test.
  const bool in_front_of_some_receiver = caster_bounds_ls.w_max >= receiver_bounds.w_min + kRayTMin;
  return uv_overlap && in_front_of_some_receiver;
}

__device__ __forceinline__ LightBounds compute_block_receiver_light_bounds(
    float block_min_edge_x,
    float block_min_edge_y,
    float block_max_edge_x,
    float block_max_edge_y,
    float aspect,
    float image_plane_scale,
    int width,
    int height,
    float near_depth,
    float far_depth,
    LightBasis basis) {
  LightBounds bounds = empty_light_bounds();
  const float xs[2] = {block_min_edge_x, block_max_edge_x};
  const float ys[2] = {block_min_edge_y, block_max_edge_y};
  const float depths[2] = {near_depth, far_depth};

  #pragma unroll
  for (int xi = 0; xi < 2; ++xi) {
    #pragma unroll
    for (int yi = 0; yi < 2; ++yi) {
      const float ray_x = (xs[xi] / static_cast<float>(width) * 2.0f - 1.0f) * aspect * image_plane_scale;
      const float ray_y = (1.0f - ys[yi] / static_cast<float>(height) * 2.0f) * image_plane_scale;
      #pragma unroll
      for (int di = 0; di < 2; ++di) {
        const float depth = depths[di];
        const Vec3 point = make_vec3(ray_x * depth, ray_y * depth, -depth);
        bounds = extend_light_bounds_point(bounds, point, basis);
      }
    }
  }

  // Small numerical pad so a caster very close to the tile border is not dropped by the prefilter.
  constexpr float pad = 1.0e-3f;
  bounds.u_min -= pad;
  bounds.u_max += pad;
  bounds.v_min -= pad;
  bounds.v_max += pad;
  bounds.w_min -= pad;
  bounds.w_max += pad;
  return bounds;
}


__device__ __forceinline__ float depth_bin_edge(int edge, float far_depth) {
  const float near_depth = kCameraNear;
  const float safe_far_depth = fmaxf(far_depth, near_depth + 1.0e-3f);
  const float a = static_cast<float>(edge) / static_cast<float>(kDepthBins);
  return near_depth * powf(safe_far_depth / near_depth, a);
}

__device__ __forceinline__ int depth_to_bin_from_edges(float forward_depth, const float* edges) {
  const float depth = fmaxf(forward_depth, kCameraNear);
  #pragma unroll
  for (int bin = 0; bin < kDepthBins - 1; ++bin) {
    if (depth <= edges[bin + 1]) {
      return bin;
    }
  }
  return kDepthBins - 1;
}

__device__ __forceinline__ bool aabb_forward_depth_range(Aabb bounds, float* out_near_depth, float* out_far_depth) {
  // Camera convention in this renderer: visible objects are usually at negative Z, and
  // forward camera depth is -Z. The nearest point in forward depth is therefore -max.z.
  float near_depth = -bounds.max.z;
  float far_depth = -bounds.min.z;
  if (far_depth < kCameraNear) {
    return false;
  }
  near_depth = fmaxf(near_depth, kCameraNear);
  far_depth = fmaxf(far_depth, near_depth);
  *out_near_depth = near_depth;
  *out_far_depth = far_depth;
  return true;
}

__device__ __forceinline__ bool depth_ranges_overlap(
    float a_near,
    float a_far,
    float b_near,
    float b_far) {
  return a_near <= b_far && a_far >= b_near;
}

__device__ __forceinline__ float smooth_height(float x, float z, float phase_x, float phase_z) {
  const float forward_depth = fmaxf(-z, 0.0f);
  const float far_rise = 0.055f * forward_depth + 0.0011f * forward_depth * forward_depth;
  const float broad_undulation =
      1.20f * sinf(0.18f * x + 0.11f * z + phase_x) + 0.85f * cosf(0.13f * x - 0.20f * z + phase_z);
  const float foothills = 0.42f * sinf(0.46f * x + 0.34f * z + phase_x * 0.61f + phase_z * 0.23f);
  const float worn_detail = 0.12f * sinf(0.95f * x - 0.58f * z + phase_z * 1.37f);
  return far_rise + broad_undulation + foothills + worn_detail;
}

__device__ __forceinline__ float sample_terrain_height(const TerrainView& terrain, int batch_idx, float world_x, float world_z) {
  return terrain.base_heights[batch_idx] + smooth_height(world_x, world_z, terrain.phase_xs[batch_idx], terrain.phase_zs[batch_idx]);
}

__device__ __forceinline__ Vec3 terrain_normal_at_hit(const TerrainView& terrain, int batch_idx, Vec3 hit) {
  constexpr float eps = 0.04f;
  const float h_l = sample_terrain_height(terrain, batch_idx, hit.x - eps, hit.z);
  const float h_r = sample_terrain_height(terrain, batch_idx, hit.x + eps, hit.z);
  const float h_d = sample_terrain_height(terrain, batch_idx, hit.x, hit.z - eps);
  const float h_u = sample_terrain_height(terrain, batch_idx, hit.x, hit.z + eps);
  return normalize(make_vec3(h_l - h_r, 2.0f * eps, h_d - h_u));
}

__device__ __forceinline__ Aabb empty_aabb() {
  return Aabb{make_vec3(kFloatMax, kFloatMax, kFloatMax), make_vec3(-kFloatMax, -kFloatMax, -kFloatMax)};
}

__device__ __forceinline__ Aabb extend_aabb(Aabb bounds, Aabb other) {
  bounds.min = min_vec3(bounds.min, other.min);
  bounds.max = max_vec3(bounds.max, other.max);
  return bounds;
}

__device__ __forceinline__ Aabb expand_aabb(Aabb bounds, float amount) {
  const Vec3 pad = make_vec3(amount, amount, amount);
  return Aabb{sub(bounds.min, pad), add(bounds.max, pad)};
}

__device__ __forceinline__ Aabb sphere_aabb(Vec3 center, float radius) {
  const Vec3 extent = make_vec3(radius, radius, radius);
  return Aabb{sub(center, extent), add(center, extent)};
}

__device__ __forceinline__ Aabb box_aabb(Vec3 center, Vec3 half_size, const float* axes) {
  const Vec3 axis_x = normalize(load_vec3(axes + 0));
  const Vec3 axis_y = normalize(load_vec3(axes + 3));
  const Vec3 axis_z = normalize(load_vec3(axes + 6));
  const Vec3 extent = make_vec3(
      fabsf(axis_x.x) * half_size.x + fabsf(axis_y.x) * half_size.y + fabsf(axis_z.x) * half_size.z,
      fabsf(axis_x.y) * half_size.x + fabsf(axis_y.y) * half_size.y + fabsf(axis_z.y) * half_size.z,
      fabsf(axis_x.z) * half_size.x + fabsf(axis_y.z) * half_size.y + fabsf(axis_z.z) * half_size.z);
  return Aabb{sub(center, extent), add(center, extent)};
}

__device__ __forceinline__ bool ray_aabb_exit_t(Vec3 ray_origin, Vec3 ray_dir, Aabb bounds, float* out_t_exit) {
  float t_min = -kFloatMax;
  float t_max = kFloatMax;

  const float origin[3] = {ray_origin.x, ray_origin.y, ray_origin.z};
  const float dir[3] = {ray_dir.x, ray_dir.y, ray_dir.z};
  const float bounds_min[3] = {bounds.min.x, bounds.min.y, bounds.min.z};
  const float bounds_max[3] = {bounds.max.x, bounds.max.y, bounds.max.z};

  #pragma unroll
  for (int axis_idx = 0; axis_idx < 3; ++axis_idx) {
    if (fabsf(dir[axis_idx]) < kParallelEpsilon) {
      if (origin[axis_idx] < bounds_min[axis_idx] || origin[axis_idx] > bounds_max[axis_idx]) {
        return false;
      }
      continue;
    }

    const float inv_dir = 1.0f / dir[axis_idx];
    float t1 = (bounds_min[axis_idx] - origin[axis_idx]) * inv_dir;
    float t2 = (bounds_max[axis_idx] - origin[axis_idx]) * inv_dir;
    if (t1 > t2) {
      const float tmp = t1;
      t1 = t2;
      t2 = tmp;
    }
    t_min = fmaxf(t_min, t1);
    t_max = fminf(t_max, t2);
    if (t_min > t_max) {
      return false;
    }
  }

  if (t_max <= kRayTMin) {
    return false;
  }
  *out_t_exit = t_max;
  return true;
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


__global__ void init_terrain_depth_kernel(float* terrain_depth, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    terrain_depth[idx] = kFloatMax;
  }
}

__global__ void voxel_space_terrain_kernel(
    float* terrain_depth,
    int width,
    int height,
    SceneView scene,
    RenderOptionsView options) {
  const int screen_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (screen_x >= width || scene.terrain.counts[batch_idx] <= 0) {
    return;
  }

  // Existing camera convention: camera at the origin looking down -Z, with Y up.
  // Voxel Space marches positive forward_depth values and samples terrain Z as -forward_depth.
  float z = kCameraNear;
  const float z_end = scene.terrain.depth_limits[batch_idx];
  if (z_end <= kCameraNear || z > z_end) {
    return;
  }

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float fov_radians = options.fov_degrees * 0.017453292519943295f;
  const float image_plane_scale = tanf(0.5f * fov_radians);
  const float ray_x = ((static_cast<float>(screen_x) + 0.5f) / static_cast<float>(width) * 2.0f - 1.0f) *
      aspect * image_plane_scale;
  const float horizon = 0.5f * static_cast<float>(height - 1);
  const float scale_height = 0.5f * static_cast<float>(height) / image_plane_scale;

  int y_buffer = height;
  float dz = scene.terrain.dz[batch_idx];
  const float dz_growth = scene.terrain.dz_growth[batch_idx];
  if (dz <= 0.0f || dz_growth < 0.0f) {
    return;
  }

  while (z <= z_end && y_buffer > 0) {
    const float world_x = ray_x * z;
    const float world_z = -z;
    const float terrain_y = sample_terrain_height(scene.terrain, batch_idx, world_x, world_z);
    // Project terrain height onto this screen column. Camera height is currently 0 because the
    // rest of this renderer also assumes camera-space geometry.
    const float screen_y = horizon - terrain_y / z * scale_height;
    if (screen_y < static_cast<float>(y_buffer)) {
      const int y_top = max(static_cast<int>(ceilf(screen_y)), 0);
      const int y_bottom = min(y_buffer, height);
      if (y_top < y_bottom) {
        for (int py = y_top; py < y_bottom; ++py) {
          const float ray_y = (1.0f - (static_cast<float>(py) + 0.5f) /
              static_cast<float>(height) * 2.0f) * image_plane_scale;
          const float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + 1.0f);
          const float t = z * ray_len;
          const int depth_offset = (batch_idx * height + py) * width + screen_x;
          terrain_depth[depth_offset] = fminf(terrain_depth[depth_offset], t);
        }
      }
      y_buffer = y_top;
    }

    z += dz;
    dz += dz_growth;
  }
}

__device__ __forceinline__ bool is_shadowed_ref_list(
    Vec3 ray_origin,
    Vec3 light_dir,
    const SceneView& scene,
    const PrimitiveRef* primitives,
    int primitive_count,
    float max_t,
    int batch_idx,
    int skip_kind,
    int skip_idx) {
  const int count = min(primitive_count, kMaxFinitePrimitives);
  for (int item = 0; item < count; ++item) {
    const PrimitiveRef primitive = primitives[item];
    if (primitive.kind == skip_kind && primitive.index == skip_idx) {
      continue;
    }

    float t = 0.0f;
    if (primitive.kind == 1) {
      const int sphere_offset = (batch_idx * scene.spheres.count + primitive.index);
      const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
      const float sphere_radius = scene.spheres.radii[sphere_offset];
      if (intersect_sphere(ray_origin, light_dir, sphere_center, sphere_radius, &t) && t < max_t) {
        return true;
      }
    } else {
      const int box_offset = (batch_idx * scene.boxes.count + primitive.index);
      const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
      const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
      Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
      if (intersect_box(ray_origin, light_dir, box_center, box_half_size, scene.boxes.axes + box_offset * 9, &t, &normal) &&
          t < max_t) {
        return true;
      }
    }
  }

  // Terrain casting shadows is intentionally disabled in this rasterized-terrain path.
  return false;
}

__device__ __forceinline__ Vec3 apply_lighting_ref_list(
    Vec3 base_color,
    Vec3 hit,
    Vec3 normal,
    Vec3 ray_dir,
    Vec3 light_dir,
    const SceneView& scene,
    const PrimitiveRef* primitives,
    int primitive_count,
    const RenderOptionsView& options,
    Aabb shadow_scene_bounds,
    int shadow_scene_bounds_valid,
    int batch_idx,
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
    float shadow_max_t = kFloatMax;
    bool shadow_bounds_hit = true;
    if (shadow_scene_bounds_valid) {
      shadow_bounds_hit = ray_aabb_exit_t(shadow_origin, light_dir, shadow_scene_bounds, &shadow_max_t);
    }
    if (shadow_bounds_hit &&
        is_shadowed_ref_list(
            shadow_origin,
            light_dir,
            scene,
            primitives,
            primitive_count,
            shadow_max_t,
            batch_idx,
            skip_kind,
            skip_idx)) {
      direct *= (1.0f - options.shadow_strength);
    }
  }

  return mul(base_color, ambient + direct);
}

__device__ void intersect_finite_ref_list_depth_range(
    Vec3 ray_origin,
    Vec3 ray_dir,
    const SceneView& scene,
    const PrimitiveRef* primitives,
    int primitive_count,
    float min_t,
    float max_t,
    int batch_idx,
    HitRecord* hit) {
  const int count = min(primitive_count, kMaxFinitePrimitives);
  const float clipped_max_t = fminf(max_t, hit->t);
  if (clipped_max_t <= min_t) {
    return;
  }

  for (int item = 0; item < count; ++item) {
    const PrimitiveRef primitive = primitives[item];
    float t = 0.0f;
    if (primitive.kind == 1) {
      const int sphere_offset = (batch_idx * scene.spheres.count + primitive.index);
      const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
      const float sphere_radius = scene.spheres.radii[sphere_offset];
      if (intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius, &t) &&
          t >= min_t && t < clipped_max_t && t < hit->t) {
        hit->t = t;
        hit->sphere = primitive.index;
        hit->box = -1;
      }
    } else {
      const int box_offset = (batch_idx * scene.boxes.count + primitive.index);
      const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
      const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
      Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
      if (intersect_box(ray_origin, ray_dir, box_center, box_half_size, scene.boxes.axes + box_offset * 9, &t, &normal) &&
          t >= min_t && t < clipped_max_t && t < hit->t) {
        hit->t = t;
        hit->sphere = -1;
        hit->box = primitive.index;
        hit->box_normal = normal;
      }
    }
  }
}

__global__ void render_scene_kernel(
    float* image,
    int* instance_map,
    int* semantic_map,
    const float* terrain_depth,
    int width,
    int height,
    SceneView scene,
    RenderOptionsView options) {
  __shared__ PrimitiveRef primary_cell_primitives[kDepthBins][kMaxFinitePrimitives];
  __shared__ PrimitiveRef shadow_cell_primitives[kDepthBins][kMaxFinitePrimitives];
  __shared__ int primary_cell_counts[kDepthBins];
  __shared__ int shadow_cell_counts[kDepthBins];
  __shared__ LightBasis shadow_light_basis;
  __shared__ LightBounds shadow_receiver_bounds[kDepthBins];
  __shared__ Aabb shadow_scene_bounds;
  __shared__ int shadow_scene_bounds_valid;
  __shared__ float block_depth_edges[kDepthBins + 1];

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = blockIdx.z;
  const int sphere_count = scene.spheres.counts[batch_idx];
  const int plane_count = 0;
  const int terrain_count = scene.terrain.counts[batch_idx];
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

  for (int bin = thread_linear; bin < kDepthBins; bin += threads_per_block) {
    primary_cell_counts[bin] = 0;
    shadow_cell_counts[bin] = 0;
  }

  if (thread_linear == 0) {
    const Vec3 block_light_dir = normalize(load_vec3(options.light_dir));
    shadow_light_basis = make_light_basis(block_light_dir);

    Aabb finite_scene_bounds = empty_aabb();
    int finite_scene_primitive_count = 0;
    float receiver_far_depth = kCameraNear;

    for (int sphere_idx = 0; sphere_idx < sphere_count; ++sphere_idx) {
      const int sphere_offset = (batch_idx * scene.spheres.count + sphere_idx);
      const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
      const float sphere_radius = scene.spheres.radii[sphere_offset];
      const Aabb bounds = sphere_aabb(sphere_center, sphere_radius);
      finite_scene_bounds = extend_aabb(finite_scene_bounds, bounds);
      receiver_far_depth = fmaxf(receiver_far_depth, -bounds.min.z);
      ++finite_scene_primitive_count;
    }

    for (int box_idx = 0; box_idx < box_count; ++box_idx) {
      const int box_offset = (batch_idx * scene.boxes.count + box_idx);
      const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
      const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
      const Aabb bounds = box_aabb(box_center, box_half_size, scene.boxes.axes + box_offset * 9);
      finite_scene_bounds = extend_aabb(finite_scene_bounds, bounds);
      receiver_far_depth = fmaxf(receiver_far_depth, -bounds.min.z);
      ++finite_scene_primitive_count;
    }

    if (terrain_count > 0) {
      receiver_far_depth = fmaxf(receiver_far_depth, scene.terrain.depth_limits[batch_idx]);
    }
    receiver_far_depth = fmaxf(receiver_far_depth + 0.05f, kCameraNear + 0.05f);
    for (int edge = 0; edge <= kDepthBins; ++edge) {
      block_depth_edges[edge] = depth_bin_edge(edge, receiver_far_depth);
    }

    const float block_min_edge_x = static_cast<float>(blockIdx.x * blockDim.x);
    const float block_min_edge_y = static_cast<float>(blockIdx.y * blockDim.y);
    const float block_max_edge_x = static_cast<float>(min(width, static_cast<int>((blockIdx.x + 1) * blockDim.x)));
    const float block_max_edge_y = static_cast<float>(min(height, static_cast<int>((blockIdx.y + 1) * blockDim.y)));

    for (int bin = 0; bin < kDepthBins; ++bin) {
      const float bin_near_depth = block_depth_edges[bin];
      const float bin_far_depth = block_depth_edges[bin + 1];
      shadow_receiver_bounds[bin] = compute_block_receiver_light_bounds(
          block_min_edge_x,
          block_min_edge_y,
          block_max_edge_x,
          block_max_edge_y,
          aspect,
          image_plane_scale,
          width,
          height,
          bin_near_depth,
          bin_far_depth,
          shadow_light_basis);
    }

    shadow_scene_bounds_valid = finite_scene_primitive_count > 0 ? 1 : 0;
    shadow_scene_bounds = shadow_scene_bounds_valid ? expand_aabb(finite_scene_bounds, 1.0e-2f) : empty_aabb();
  }
  __syncthreads();

  for (int sphere_idx = thread_linear; sphere_idx < sphere_count; sphere_idx += threads_per_block) {
    const int sphere_offset = (batch_idx * scene.spheres.count + sphere_idx);
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
    const float sphere_radius = scene.spheres.radii[sphere_offset];
    const Aabb bounds = sphere_aabb(sphere_center, sphere_radius);

    float primitive_near_depth = kCameraNear;
    float primitive_far_depth = kCameraNear;
    const bool valid_depth_range = aabb_forward_depth_range(bounds, &primitive_near_depth, &primitive_far_depth);
    const bool primary_overlap = valid_depth_range && sphere_overlaps_block(
        sphere_center,
        sphere_radius,
        aspect,
        image_plane_scale,
        width,
        height,
        block_min_x,
        block_min_y,
        block_max_x,
        block_max_y);

    #pragma unroll
    for (int bin = 0; bin < kDepthBins; ++bin) {
      const float bin_near_depth = block_depth_edges[bin];
      const float bin_far_depth = block_depth_edges[bin + 1];
      if (primary_overlap && depth_ranges_overlap(primitive_near_depth, primitive_far_depth, bin_near_depth, bin_far_depth)) {
        const int list_idx = atomicAdd(&primary_cell_counts[bin], 1);
        if (list_idx < kMaxFinitePrimitives) {
          primary_cell_primitives[bin][list_idx] = PrimitiveRef{1, sphere_idx};
        }
      }

      if (aabb_can_shadow_block(bounds, shadow_receiver_bounds[bin], shadow_light_basis)) {
        const int list_idx = atomicAdd(&shadow_cell_counts[bin], 1);
        if (list_idx < kMaxFinitePrimitives) {
          shadow_cell_primitives[bin][list_idx] = PrimitiveRef{1, sphere_idx};
        }
      }
    }
  }

  for (int box_idx = thread_linear; box_idx < box_count; box_idx += threads_per_block) {
    const int box_offset = (batch_idx * scene.boxes.count + box_idx);
    const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
    const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
    const Aabb bounds = box_aabb(box_center, box_half_size, scene.boxes.axes + box_offset * 9);

    float primitive_near_depth = kCameraNear;
    float primitive_far_depth = kCameraNear;
    const bool valid_depth_range = aabb_forward_depth_range(bounds, &primitive_near_depth, &primitive_far_depth);
    const bool primary_overlap = valid_depth_range && box_overlaps_block(
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
        block_max_y);

    #pragma unroll
    for (int bin = 0; bin < kDepthBins; ++bin) {
      const float bin_near_depth = block_depth_edges[bin];
      const float bin_far_depth = block_depth_edges[bin + 1];
      if (primary_overlap && depth_ranges_overlap(primitive_near_depth, primitive_far_depth, bin_near_depth, bin_far_depth)) {
        const int list_idx = atomicAdd(&primary_cell_counts[bin], 1);
        if (list_idx < kMaxFinitePrimitives) {
          primary_cell_primitives[bin][list_idx] = PrimitiveRef{3, box_idx};
        }
      }

      if (aabb_can_shadow_block(bounds, shadow_receiver_bounds[bin], shadow_light_basis)) {
        const int list_idx = atomicAdd(&shadow_cell_counts[bin], 1);
        if (list_idx < kMaxFinitePrimitives) {
          shadow_cell_primitives[bin][list_idx] = PrimitiveRef{3, box_idx};
        }
      }
    }
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
  const float ray_forward_per_t = fmaxf(-ray_dir.z, 1.0e-8f);

  Vec3 color = background;
  const int map_offset = (batch_idx * height + y) * width + x;
  float terrain_t = kFloatMax;
  if (terrain_count > 0 && terrain_depth != nullptr) {
    terrain_t = terrain_depth[map_offset];
  }

  HitRecord finite_hit{terrain_t, -1, -1, -1, make_vec3(0.0f, 0.0f, 0.0f), make_vec3(0.0f, 1.0f, 0.0f)};
  for (int bin = 0; bin < kDepthBins; ++bin) {
    const float bin_near_depth = block_depth_edges[bin];
    const float bin_far_depth = block_depth_edges[bin + 1];
    const float bin_min_t = bin_near_depth / ray_forward_per_t;
    const float bin_max_t = bin_far_depth / ray_forward_per_t;
    if (bin_min_t >= finite_hit.t) {
      break;
    }

    const int previous_sphere = finite_hit.sphere;
    const int previous_box = finite_hit.box;
    intersect_finite_ref_list_depth_range(
        ray_origin,
        ray_dir,
        scene,
        primary_cell_primitives[bin],
        primary_cell_counts[bin],
        bin_min_t,
        bin_max_t,
        batch_idx,
        &finite_hit);

    if (finite_hit.sphere != previous_sphere || finite_hit.box != previous_box) {
      // Depth bins are processed front-to-back. If the first exact finite hit lies in this bin,
      // later bins cannot contain a closer finite hit on this same primary ray.
      break;
    }
  }

  float closest_t = (finite_hit.sphere >= 0 || finite_hit.box >= 0) ? finite_hit.t : kFloatMax;
  int closest_sphere = finite_hit.sphere;
  int closest_box = finite_hit.box;
  int closest_terrain = -1;
  int instance_id = 0;
  int semantic_id = 0;
  Vec3 closest_box_normal = finite_hit.box_normal;
  Vec3 closest_terrain_normal = finite_hit.terrain_normal;

  if (terrain_count > 0 && terrain_depth != nullptr) {
    if (terrain_t < closest_t) {
      closest_t = terrain_t;
      closest_sphere = -1;
      closest_box = -1;
      closest_terrain = 0;
      const Vec3 terrain_hit = add(ray_origin, mul(ray_dir, closest_t));
      closest_terrain_normal = terrain_normal_at_hit(scene.terrain, batch_idx, terrain_hit);
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
    const int receiver_bin = depth_to_bin_from_edges(fmaxf(-hit.z, kCameraNear), block_depth_edges);
    color = apply_lighting_ref_list(
        sphere_color,
        hit,
        normal,
        ray_dir,
        light_dir,
        scene,
        shadow_cell_primitives[receiver_bin],
        shadow_cell_counts[receiver_bin],
        options,
        shadow_scene_bounds,
        shadow_scene_bounds_valid,
        batch_idx,
        1,
        closest_sphere);
  } else if (closest_box >= 0) {
    const int box_offset = (batch_idx * scene.boxes.count + closest_box);
    instance_id = sphere_count + plane_count + terrain_count + closest_box + 1;
    semantic_id = 3;
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(closest_box_normal);
    const Vec3 box_color = load_vec3(scene.boxes.colors + box_offset * 3);
    const int receiver_bin = depth_to_bin_from_edges(fmaxf(-hit.z, kCameraNear), block_depth_edges);
    color = apply_lighting_ref_list(
        box_color,
        hit,
        normal,
        ray_dir,
        light_dir,
        scene,
        shadow_cell_primitives[receiver_bin],
        shadow_cell_counts[receiver_bin],
        options,
        shadow_scene_bounds,
        shadow_scene_bounds_valid,
        batch_idx,
        3,
        closest_box);
  } else if (closest_terrain >= 0) {
    instance_id = sphere_count + plane_count + 1;
    semantic_id = 2;
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 terrain_color = load_vec3(scene.terrain.colors + batch_idx * 3);
    const int receiver_bin = depth_to_bin_from_edges(fmaxf(-hit.z, kCameraNear), block_depth_edges);
    color = apply_lighting_ref_list(
        terrain_color,
        hit,
        closest_terrain_normal,
        ray_dir,
        light_dir,
        scene,
        shadow_cell_primitives[receiver_bin],
        shadow_cell_counts[receiver_bin],
        options,
        shadow_scene_bounds,
        shadow_scene_bounds_valid,
        batch_idx,
        4,
        0);
  }

  const int image_offset = ((batch_idx * 3 * height + y) * width) + x;
  image[image_offset + 0 * height * width] = color.x;
  image[image_offset + 1 * height * width] = color.y;
  image[image_offset + 2 * height * width] = color.z;

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
    torch::Tensor terrain_base_heights,
    torch::Tensor terrain_depth_limits,
    torch::Tensor terrain_phase_xs,
    torch::Tensor terrain_phase_zs,
    torch::Tensor terrain_dz,
    torch::Tensor terrain_dz_growth,
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
    double shadow_strength) {
  const int batch_size = static_cast<int>(image.size(0));
  const int height = static_cast<int>(image.size(2));
  const int width = static_cast<int>(image.size(3));
  const int sphere_count = static_cast<int>(sphere_centers.size(1));
  const int plane_count = static_cast<int>(plane_points.size(1));
  const int box_count = static_cast<int>(box_centers.size(1));
  TORCH_CHECK(sphere_count <= kMaxSpheres, "render_scene supports at most ", kMaxSpheres, " spheres");
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
          plane_counts.data_ptr<int>(),
          plane_count,
      },
      TerrainView{
          terrain_base_heights.data_ptr<float>(),
          terrain_depth_limits.data_ptr<float>(),
          terrain_phase_xs.data_ptr<float>(),
          terrain_phase_zs.data_ptr<float>(),
          terrain_dz.data_ptr<float>(),
          terrain_dz_growth.data_ptr<float>(),
          terrain_colors.data_ptr<float>(),
          terrain_counts.data_ptr<int>(),
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

  auto terrain_depth = torch::empty({batch_size, height, width}, image.options());
  const int terrain_depth_count = batch_size * height * width;
  const int init_threads = 256;
  const int init_blocks = (terrain_depth_count + init_threads - 1) / init_threads;
  init_terrain_depth_kernel<<<init_blocks, init_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      terrain_depth.data_ptr<float>(), terrain_depth_count);

  if (terrain_depth_limits.size(1) > 0) {
    const int voxel_threads = 128;
    const dim3 voxel_grid((width + voxel_threads - 1) / voxel_threads, batch_size);
    voxel_space_terrain_kernel<<<voxel_grid, voxel_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        terrain_depth.data_ptr<float>(), width, height, scene, options);
  }

  render_scene_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      image.data_ptr<float>(),
      instance_map.numel() == 0 ? nullptr : instance_map.data_ptr<int>(),
      semantic_map.numel() == 0 ? nullptr : semantic_map.data_ptr<int>(),
      terrain_depth.data_ptr<float>(),
      width,
      height,
      scene,
      options);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
