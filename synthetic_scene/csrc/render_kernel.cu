#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kMaxSpheres = 64;
constexpr int kMaxBoxes = 64;
constexpr int kMaxCylinders = 64;
constexpr int kMaxFinitePrimitives = kMaxSpheres + kMaxBoxes + kMaxCylinders;
constexpr int kDepthBins = 8;
constexpr int kTileWidth = 16;
constexpr int kTileHeight = 16;
constexpr int kPrimitiveMaskWords = (kMaxFinitePrimitives + 31) / 32;
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

struct CylinderView {
  const float* centers;
  const float* radii;
  const float* half_heights;
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
  CylinderView cylinders;
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

struct HitRecord {
  float t;
  int sphere;
  int box;
  int cylinder;
  int terrain;
  Vec3 box_normal;
  Vec3 cylinder_normal;
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

__device__ __forceinline__ Aabb cylinder_aabb(Vec3 center, float radius, float half_height, const float* axes) {
  const Vec3 axis_x = normalize(load_vec3(axes + 0));
  const Vec3 axis_y = normalize(load_vec3(axes + 3));
  const Vec3 axis_z = normalize(load_vec3(axes + 6));
  const Vec3 extent = make_vec3(
      fabsf(axis_x.x) * radius + fabsf(axis_y.x) * half_height + fabsf(axis_z.x) * radius,
      fabsf(axis_x.y) * radius + fabsf(axis_y.y) * half_height + fabsf(axis_z.y) * radius,
      fabsf(axis_x.z) * radius + fabsf(axis_y.z) * half_height + fabsf(axis_z.z) * radius);
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

__device__ __forceinline__ bool intersect_cylinder(
    Vec3 ray_origin,
    Vec3 ray_dir,
    Vec3 center,
    float radius,
    float half_height,
    const float* axes,
    float* out_t,
    Vec3* out_normal) {
  const Vec3 axis_x = normalize(load_vec3(axes + 0));
  const Vec3 axis_y = normalize(load_vec3(axes + 3));
  const Vec3 axis_z = normalize(load_vec3(axes + 6));
  const Vec3 local_origin_delta = sub(ray_origin, center);
  const float ox = dot(local_origin_delta, axis_x);
  const float oy = dot(local_origin_delta, axis_y);
  const float oz = dot(local_origin_delta, axis_z);
  const float dx = dot(ray_dir, axis_x);
  const float dy = dot(ray_dir, axis_y);
  const float dz = dot(ray_dir, axis_z);
  const float radius2 = radius * radius;

  float best_t = kFloatMax;
  Vec3 best_normal = make_vec3(0.0f, 0.0f, 0.0f);

  const float a = dx * dx + dz * dz;
  const float b = 2.0f * (ox * dx + oz * dz);
  const float c = ox * ox + oz * oz - radius2;
  if (fabsf(a) >= kParallelEpsilon) {
    const float discriminant = b * b - 4.0f * a * c;
    if (discriminant >= 0.0f) {
      const float sqrt_disc = sqrtf(discriminant);
      const float inv_2a = 0.5f / a;
      const float candidates[2] = {
          (-b - sqrt_disc) * inv_2a,
          (-b + sqrt_disc) * inv_2a,
      };
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const float t = candidates[i];
        const float y = oy + t * dy;
        if (t > kRayTMin && t < best_t && y >= -half_height && y <= half_height) {
          best_t = t;
          best_normal = normalize(add(mul(axis_x, ox + t * dx), mul(axis_z, oz + t * dz)));
        }
      }
    }
  }

  if (fabsf(dy) >= kParallelEpsilon) {
    const float cap_y[2] = {-half_height, half_height};
    const Vec3 cap_normal[2] = {mul(axis_y, -1.0f), axis_y};
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float t = (cap_y[i] - oy) / dy;
      const float x = ox + t * dx;
      const float z = oz + t * dz;
      if (t > kRayTMin && t < best_t && x * x + z * z <= radius2) {
        best_t = t;
        best_normal = cap_normal[i];
      }
    }
  }

  if (best_t == kFloatMax) {
    return false;
  }
  *out_t = best_t;
  *out_normal = best_normal;
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


__device__ __forceinline__ bool sphere_pixel_bounds(
    Vec3 center,
    float radius,
    float aspect,
    float image_plane_scale,
    int width,
    int height,
    float* out_min_x,
    float* out_min_y,
    float* out_max_x,
    float* out_max_y) {
  const float nearest_depth = -(center.z + radius);
  if (nearest_depth <= kCameraNear) {
    *out_min_x = -kFloatMax;
    *out_min_y = -kFloatMax;
    *out_max_x = kFloatMax;
    *out_max_y = kFloatMax;
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
  *out_min_x = center_x - radius_pixels;
  *out_min_y = center_y - radius_pixels;
  *out_max_x = center_x + radius_pixels;
  *out_max_y = center_y + radius_pixels;
  return true;
}

__device__ __forceinline__ bool box_pixel_bounds(
    Vec3 center,
    Vec3 half_size,
    const float* axes,
    float aspect,
    float image_plane_scale,
    int width,
    int height,
    float* out_min_x,
    float* out_min_y,
    float* out_max_x,
    float* out_max_y) {
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
      *out_min_x = -kFloatMax;
      *out_min_y = -kFloatMax;
      *out_max_x = kFloatMax;
      *out_max_y = kFloatMax;
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
  *out_min_x = min_x - 1.0f;
  *out_min_y = min_y - 1.0f;
  *out_max_x = max_x + 1.0f;
  *out_max_y = max_y + 1.0f;
  return true;
}

__device__ __forceinline__ int cluster_linear_index(
    int batch_idx,
    int tile_y,
    int tile_x,
    int bin,
    int tiles_x,
    int tiles_y) {
  return (((batch_idx * tiles_y + tile_y) * tiles_x + tile_x) * kDepthBins + bin);
}

__device__ __forceinline__ void mark_primitive_in_cluster_mask(
    int* masks,
    int batch_idx,
    int tile_y,
    int tile_x,
    int bin,
    int tiles_x,
    int tiles_y,
    int primitive_slot) {
  if (primitive_slot < 0 || primitive_slot >= kMaxFinitePrimitives) {
    return;
  }
  const int cluster_idx = cluster_linear_index(batch_idx, tile_y, tile_x, bin, tiles_x, tiles_y);
  const int word_idx = primitive_slot >> 5;
  const int bit_mask = static_cast<int>(1u << (primitive_slot & 31));
  atomicOr(&masks[cluster_idx * kPrimitiveMaskWords + word_idx], bit_mask);
}

__device__ __forceinline__ bool primitive_slot_to_kind_index(
    int primitive_slot,
    const SceneView& scene,
    int batch_idx,
    int* out_kind,
    int* out_index) {
  const int sphere_slot_count = scene.spheres.count;
  const int box_slot_count = scene.boxes.count;
  if (primitive_slot < sphere_slot_count) {
    if (primitive_slot >= scene.spheres.counts[batch_idx]) {
      return false;
    }
    *out_kind = 1;
    *out_index = primitive_slot;
    return true;
  }

  const int box_idx = primitive_slot - sphere_slot_count;
  if (box_idx < box_slot_count) {
    if (box_idx < 0 || box_idx >= scene.boxes.counts[batch_idx]) {
      return false;
    }
    *out_kind = 3;
    *out_index = box_idx;
    return true;
  }

  const int cylinder_idx = primitive_slot - sphere_slot_count - box_slot_count;
  if (cylinder_idx < 0 || cylinder_idx >= scene.cylinders.counts[batch_idx]) {
    return false;
  }
  *out_kind = 4;
  *out_index = cylinder_idx;
  return true;
}

__device__ __forceinline__ bool load_primitive_bounds_for_slot(
    const SceneView& scene,
    int batch_idx,
    int primitive_slot,
    int* out_kind,
    int* out_index,
    Aabb* out_bounds,
    Vec3* out_center,
    Vec3* out_half_size) {
  const int sphere_slot_count = scene.spheres.count;
  const int box_slot_count = scene.boxes.count;
  if (primitive_slot < sphere_slot_count) {
    const int sphere_idx = primitive_slot;
    if (sphere_idx >= scene.spheres.counts[batch_idx]) {
      return false;
    }
    const int sphere_offset = batch_idx * scene.spheres.count + sphere_idx;
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
    const float sphere_radius = scene.spheres.radii[sphere_offset];
    *out_kind = 1;
    *out_index = sphere_idx;
    *out_bounds = sphere_aabb(sphere_center, sphere_radius);
    *out_center = sphere_center;
    *out_half_size = make_vec3(sphere_radius, sphere_radius, sphere_radius);
    return true;
  }

  const int box_idx = primitive_slot - sphere_slot_count;
  if (box_idx < box_slot_count) {
    if (box_idx < 0 || box_idx >= scene.boxes.counts[batch_idx]) {
      return false;
    }
    const int box_offset = batch_idx * scene.boxes.count + box_idx;
    const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
    const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
    *out_kind = 3;
    *out_index = box_idx;
    *out_bounds = box_aabb(box_center, box_half_size, scene.boxes.axes + box_offset * 9);
    *out_center = box_center;
    *out_half_size = box_half_size;
    return true;
  }

  const int cylinder_idx = primitive_slot - sphere_slot_count - box_slot_count;
  if (cylinder_idx < 0 || cylinder_idx >= scene.cylinders.counts[batch_idx]) {
    return false;
  }
  const int cylinder_offset = batch_idx * scene.cylinders.count + cylinder_idx;
  const Vec3 cylinder_center = load_vec3(scene.cylinders.centers + cylinder_offset * 3);
  const float cylinder_radius = scene.cylinders.radii[cylinder_offset];
  const float cylinder_half_height = scene.cylinders.half_heights[cylinder_offset];
  *out_kind = 4;
  *out_index = cylinder_idx;
  *out_bounds = cylinder_aabb(cylinder_center, cylinder_radius, cylinder_half_height, scene.cylinders.axes + cylinder_offset * 9);
  *out_center = cylinder_center;
  *out_half_size = make_vec3(cylinder_radius, cylinder_half_height, cylinder_radius);
  return true;
}

__device__ __forceinline__ LightBounds load_light_bounds6(const float* bounds_ptr) {
  return LightBounds{
      bounds_ptr[0], bounds_ptr[1], bounds_ptr[2], bounds_ptr[3], bounds_ptr[4], bounds_ptr[5]};
}

__device__ __forceinline__ void store_light_bounds6(float* bounds_ptr, LightBounds bounds) {
  bounds_ptr[0] = bounds.u_min;
  bounds_ptr[1] = bounds.u_max;
  bounds_ptr[2] = bounds.v_min;
  bounds_ptr[3] = bounds.v_max;
  bounds_ptr[4] = bounds.w_min;
  bounds_ptr[5] = bounds.w_max;
}

__device__ __forceinline__ bool light_bounds_can_shadow_receiver(
    LightBounds caster_bounds_ls,
    LightBounds receiver_bounds) {
  const bool uv_overlap =
      light_ranges_overlap(caster_bounds_ls.u_min, caster_bounds_ls.u_max, receiver_bounds.u_min, receiver_bounds.u_max) &&
      light_ranges_overlap(caster_bounds_ls.v_min, caster_bounds_ls.v_max, receiver_bounds.v_min, receiver_bounds.v_max);
  const bool in_front_of_some_receiver = caster_bounds_ls.w_max >= receiver_bounds.w_min + kRayTMin;
  return uv_overlap && in_front_of_some_receiver;
}

__device__ __forceinline__ Aabb load_scene_bounds6(const float* scene_bounds, int batch_idx) {
  const float* ptr = scene_bounds + batch_idx * 6;
  return Aabb{make_vec3(ptr[0], ptr[1], ptr[2]), make_vec3(ptr[3], ptr[4], ptr[5])};
}

__global__ void compute_cluster_metadata_kernel(
    float* depth_edges,
    float* scene_bounds,
    int* scene_bounds_valid,
    int width,
    int height,
    SceneView scene) {
  const int batch_idx = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }

  const int sphere_count = scene.spheres.counts[batch_idx];
  const int box_count = scene.boxes.counts[batch_idx];
  const int cylinder_count = scene.cylinders.counts[batch_idx];
  const int terrain_count = scene.terrain.counts[batch_idx];

  Aabb finite_scene_bounds = empty_aabb();
  int finite_scene_primitive_count = 0;
  float receiver_far_depth = kCameraNear;

  for (int sphere_idx = 0; sphere_idx < sphere_count; ++sphere_idx) {
    const int sphere_offset = batch_idx * scene.spheres.count + sphere_idx;
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
    const float sphere_radius = scene.spheres.radii[sphere_offset];
    const Aabb bounds = sphere_aabb(sphere_center, sphere_radius);
    finite_scene_bounds = extend_aabb(finite_scene_bounds, bounds);
    receiver_far_depth = fmaxf(receiver_far_depth, -bounds.min.z);
    ++finite_scene_primitive_count;
  }

  for (int box_idx = 0; box_idx < box_count; ++box_idx) {
    const int box_offset = batch_idx * scene.boxes.count + box_idx;
    const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
    const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
    const Aabb bounds = box_aabb(box_center, box_half_size, scene.boxes.axes + box_offset * 9);
    finite_scene_bounds = extend_aabb(finite_scene_bounds, bounds);
    receiver_far_depth = fmaxf(receiver_far_depth, -bounds.min.z);
    ++finite_scene_primitive_count;
  }

  for (int cylinder_idx = 0; cylinder_idx < cylinder_count; ++cylinder_idx) {
    const int cylinder_offset = batch_idx * scene.cylinders.count + cylinder_idx;
    const Vec3 cylinder_center = load_vec3(scene.cylinders.centers + cylinder_offset * 3);
    const float cylinder_radius = scene.cylinders.radii[cylinder_offset];
    const float cylinder_half_height = scene.cylinders.half_heights[cylinder_offset];
    const Aabb bounds = cylinder_aabb(
        cylinder_center,
        cylinder_radius,
        cylinder_half_height,
        scene.cylinders.axes + cylinder_offset * 9);
    finite_scene_bounds = extend_aabb(finite_scene_bounds, bounds);
    receiver_far_depth = fmaxf(receiver_far_depth, -bounds.min.z);
    ++finite_scene_primitive_count;
  }

  if (terrain_count > 0) {
    receiver_far_depth = fmaxf(receiver_far_depth, scene.terrain.depth_limits[batch_idx]);
  }
  receiver_far_depth = fmaxf(receiver_far_depth + 0.05f, kCameraNear + 0.05f);

  float* batch_edges = depth_edges + batch_idx * (kDepthBins + 1);
  for (int edge = 0; edge <= kDepthBins; ++edge) {
    batch_edges[edge] = depth_bin_edge(edge, receiver_far_depth);
  }

  scene_bounds_valid[batch_idx] = finite_scene_primitive_count > 0 ? 1 : 0;
  Aabb expanded_bounds = finite_scene_primitive_count > 0 ? expand_aabb(finite_scene_bounds, 1.0e-2f) : empty_aabb();
  float* bounds_out = scene_bounds + batch_idx * 6;
  bounds_out[0] = expanded_bounds.min.x;
  bounds_out[1] = expanded_bounds.min.y;
  bounds_out[2] = expanded_bounds.min.z;
  bounds_out[3] = expanded_bounds.max.x;
  bounds_out[4] = expanded_bounds.max.y;
  bounds_out[5] = expanded_bounds.max.z;
}

__global__ void compute_receiver_light_bounds_kernel(
    float* receiver_light_bounds,
    const float* depth_edges,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    RenderOptionsView options) {
  const int total_clusters = tiles_x * tiles_y * kDepthBins;
  const int cluster_linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (cluster_linear >= total_clusters) {
    return;
  }

  const int bin = cluster_linear % kDepthBins;
  const int tile_linear = cluster_linear / kDepthBins;
  const int tile_x = tile_linear % tiles_x;
  const int tile_y = tile_linear / tiles_x;

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float fov_radians = options.fov_degrees * 0.017453292519943295f;
  const float image_plane_scale = tanf(0.5f * fov_radians);
  const Vec3 light_dir = normalize(load_vec3(options.light_dir));
  const LightBasis light_basis = make_light_basis(light_dir);

  const float* batch_edges = depth_edges + batch_idx * (kDepthBins + 1);
  const float block_min_edge_x = static_cast<float>(tile_x * kTileWidth);
  const float block_min_edge_y = static_cast<float>(tile_y * kTileHeight);
  const float block_max_edge_x = static_cast<float>(min(width, (tile_x + 1) * kTileWidth));
  const float block_max_edge_y = static_cast<float>(min(height, (tile_y + 1) * kTileHeight));

  const LightBounds bounds = compute_block_receiver_light_bounds(
      block_min_edge_x,
      block_min_edge_y,
      block_max_edge_x,
      block_max_edge_y,
      aspect,
      image_plane_scale,
      width,
      height,
      batch_edges[bin],
      batch_edges[bin + 1],
      light_basis);

  const int out_idx = (batch_idx * total_clusters + cluster_linear) * 6;
  store_light_bounds6(receiver_light_bounds + out_idx, bounds);
}

__device__ __forceinline__ int clamp_int_device(int value, int lo, int hi) {
  return max(lo, min(value, hi));
}

__device__ __forceinline__ bool pixel_bounds_to_tile_range(
    float min_x,
    float min_y,
    float max_x,
    float max_y,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    int* out_tile_x0,
    int* out_tile_x1,
    int* out_tile_y0,
    int* out_tile_y1) {
  if (max_x < 0.0f || max_y < 0.0f || min_x > static_cast<float>(width - 1) || min_y > static_cast<float>(height - 1)) {
    return false;
  }

  const float clamped_min_x = fmaxf(min_x, 0.0f);
  const float clamped_max_x = fminf(max_x, static_cast<float>(width - 1));
  const float clamped_min_y = fmaxf(min_y, 0.0f);
  const float clamped_max_y = fminf(max_y, static_cast<float>(height - 1));

  const int tile_x0 = clamp_int_device(static_cast<int>(floorf(clamped_min_x / static_cast<float>(kTileWidth))), 0, tiles_x - 1);
  const int tile_x1 = clamp_int_device(static_cast<int>(floorf(clamped_max_x / static_cast<float>(kTileWidth))), 0, tiles_x - 1);
  const int tile_y0 = clamp_int_device(static_cast<int>(floorf(clamped_min_y / static_cast<float>(kTileHeight))), 0, tiles_y - 1);
  const int tile_y1 = clamp_int_device(static_cast<int>(floorf(clamped_max_y / static_cast<float>(kTileHeight))), 0, tiles_y - 1);

  if (tile_x0 > tile_x1 || tile_y0 > tile_y1) {
    return false;
  }

  *out_tile_x0 = tile_x0;
  *out_tile_x1 = tile_x1;
  *out_tile_y0 = tile_y0;
  *out_tile_y1 = tile_y1;
  return true;
}

__global__ void build_primary_cluster_masks_object_driven_kernel(
    int* primary_cluster_masks,
    const float* depth_edges,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    SceneView scene,
    RenderOptionsView options) {
  __shared__ int shared_valid;
  __shared__ int shared_tile_x0;
  __shared__ int shared_tile_x1;
  __shared__ int shared_tile_y0;
  __shared__ int shared_tile_y1;
  __shared__ int shared_bin0;
  __shared__ int shared_bin1;

  const int primitive_slot = blockIdx.x;
  const int batch_idx = blockIdx.y;

  if (threadIdx.x == 0) {
    shared_valid = 0;
    shared_tile_x0 = 0;
    shared_tile_x1 = -1;
    shared_tile_y0 = 0;
    shared_tile_y1 = -1;
    shared_bin0 = 0;
    shared_bin1 = -1;

    Vec3 center = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 half_size = make_vec3(0.0f, 0.0f, 0.0f);
    Aabb bounds = empty_aabb();
    int kind = 0;
    int index = -1;
    if (load_primitive_bounds_for_slot(scene, batch_idx, primitive_slot, &kind, &index, &bounds, &center, &half_size)) {
      float near_depth = kCameraNear;
      float far_depth = kCameraNear;
      if (aabb_forward_depth_range(bounds, &near_depth, &far_depth)) {
        const float aspect = static_cast<float>(width) / static_cast<float>(height);
        const float fov_radians = options.fov_degrees * 0.017453292519943295f;
        const float image_plane_scale = tanf(0.5f * fov_radians);
        float min_x = kFloatMax;
        float min_y = kFloatMax;
        float max_x = -kFloatMax;
        float max_y = -kFloatMax;
        bool projected = false;
        if (kind == 1) {
          projected = sphere_pixel_bounds(
              center,
              half_size.x,
              aspect,
              image_plane_scale,
              width,
              height,
              &min_x,
              &min_y,
              &max_x,
              &max_y);
        } else if (kind == 3) {
          const int box_offset = batch_idx * scene.boxes.count + index;
          projected = box_pixel_bounds(
              center,
              half_size,
              scene.boxes.axes + box_offset * 9,
              aspect,
              image_plane_scale,
              width,
              height,
              &min_x,
              &min_y,
              &max_x,
              &max_y);
        } else {
          const int cylinder_offset = batch_idx * scene.cylinders.count + index;
          projected = box_pixel_bounds(
              center,
              half_size,
              scene.cylinders.axes + cylinder_offset * 9,
              aspect,
              image_plane_scale,
              width,
              height,
              &min_x,
              &min_y,
              &max_x,
              &max_y);
        }

        int tile_x0 = 0;
        int tile_x1 = -1;
        int tile_y0 = 0;
        int tile_y1 = -1;
        if (projected && pixel_bounds_to_tile_range(
                min_x,
                min_y,
                max_x,
                max_y,
                width,
                height,
                tiles_x,
                tiles_y,
                &tile_x0,
                &tile_x1,
                &tile_y0,
                &tile_y1)) {
          const float* batch_edges = depth_edges + batch_idx * (kDepthBins + 1);
          const int bin0 = depth_to_bin_from_edges(near_depth, batch_edges);
          const int bin1 = depth_to_bin_from_edges(far_depth, batch_edges);
          shared_valid = 1;
          shared_tile_x0 = tile_x0;
          shared_tile_x1 = tile_x1;
          shared_tile_y0 = tile_y0;
          shared_tile_y1 = tile_y1;
          shared_bin0 = min(bin0, bin1);
          shared_bin1 = max(bin0, bin1);
        }
      }
    }
  }
  __syncthreads();

  if (!shared_valid) {
    return;
  }

  const int tile_count_x = shared_tile_x1 - shared_tile_x0 + 1;
  const int tile_count_y = shared_tile_y1 - shared_tile_y0 + 1;
  const int bin_count = shared_bin1 - shared_bin0 + 1;
  const int mark_count = tile_count_x * tile_count_y * bin_count;

  for (int mark_idx = threadIdx.x; mark_idx < mark_count; mark_idx += blockDim.x) {
    const int bin_offset = mark_idx % bin_count;
    const int tile_offset = mark_idx / bin_count;
    const int local_tile_x = tile_offset % tile_count_x;
    const int local_tile_y = tile_offset / tile_count_x;
    const int tile_x = shared_tile_x0 + local_tile_x;
    const int tile_y = shared_tile_y0 + local_tile_y;
    const int bin = shared_bin0 + bin_offset;

    mark_primitive_in_cluster_mask(
        primary_cluster_masks,
        batch_idx,
        tile_y,
        tile_x,
        bin,
        tiles_x,
        tiles_y,
        primitive_slot);
  }
}

__global__ void build_shadow_cluster_masks_kernel(
    int* shadow_cluster_masks,
    const float* receiver_light_bounds,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    SceneView scene,
    RenderOptionsView options) {
  __shared__ int shared_valid;
  __shared__ LightBounds shared_caster_bounds_ls;

  const int cluster_linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int primitive_slot = blockIdx.y;
  const int batch_idx = blockIdx.z;
  const int total_clusters = tiles_x * tiles_y * kDepthBins;

  if (threadIdx.x == 0) {
    shared_valid = 0;
    shared_caster_bounds_ls = empty_light_bounds();

    Vec3 center = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 half_size = make_vec3(0.0f, 0.0f, 0.0f);
    Aabb bounds = empty_aabb();
    int kind = 0;
    int index = -1;
    if (load_primitive_bounds_for_slot(scene, batch_idx, primitive_slot, &kind, &index, &bounds, &center, &half_size)) {
      const Vec3 light_dir = normalize(load_vec3(options.light_dir));
      const LightBasis light_basis = make_light_basis(light_dir);
      shared_caster_bounds_ls = project_aabb_to_light_bounds(bounds, light_basis);
      shared_valid = 1;
    }
  }
  __syncthreads();

  if (!shared_valid || cluster_linear >= total_clusters) {
    return;
  }

  const int bin = cluster_linear % kDepthBins;
  const int tile_linear = cluster_linear / kDepthBins;
  const int tile_x = tile_linear % tiles_x;
  const int tile_y = tile_linear / tiles_x;

  const int bounds_idx = (batch_idx * total_clusters + cluster_linear) * 6;
  const LightBounds receiver_bounds = load_light_bounds6(receiver_light_bounds + bounds_idx);
  if (light_bounds_can_shadow_receiver(shared_caster_bounds_ls, receiver_bounds)) {
    mark_primitive_in_cluster_mask(
        shadow_cluster_masks,
        batch_idx,
        tile_y,
        tile_x,
        bin,
        tiles_x,
        tiles_y,
        primitive_slot);
  }
}

__device__ __forceinline__ bool is_shadowed_mask(
    Vec3 ray_origin,
    Vec3 light_dir,
    const SceneView& scene,
    const int* primitive_mask,
    float max_t,
    int batch_idx,
    int skip_kind,
    int skip_idx) {
  if (primitive_mask == nullptr) {
    return false;
  }

  #pragma unroll
  for (int word_idx = 0; word_idx < kPrimitiveMaskWords; ++word_idx) {
    unsigned int bits = static_cast<unsigned int>(primitive_mask[word_idx]);
    while (bits != 0u) {
      const int bit = __ffs(bits) - 1;
      const int primitive_slot = word_idx * 32 + bit;
      bits &= bits - 1u;

      int kind = 0;
      int index = -1;
      if (!primitive_slot_to_kind_index(primitive_slot, scene, batch_idx, &kind, &index)) {
        continue;
      }
      if (kind == skip_kind && index == skip_idx) {
        continue;
      }

      float t = 0.0f;
      if (kind == 1) {
        const int sphere_offset = batch_idx * scene.spheres.count + index;
        const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
        const float sphere_radius = scene.spheres.radii[sphere_offset];
        if (intersect_sphere(ray_origin, light_dir, sphere_center, sphere_radius, &t) && t < max_t) {
          return true;
        }
      } else if (kind == 3) {
        const int box_offset = batch_idx * scene.boxes.count + index;
        const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
        const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
        Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
        if (intersect_box(ray_origin, light_dir, box_center, box_half_size, scene.boxes.axes + box_offset * 9, &t, &normal) &&
            t < max_t) {
          return true;
        }
      } else {
        const int cylinder_offset = batch_idx * scene.cylinders.count + index;
        const Vec3 cylinder_center = load_vec3(scene.cylinders.centers + cylinder_offset * 3);
        const float cylinder_radius = scene.cylinders.radii[cylinder_offset];
        const float cylinder_half_height = scene.cylinders.half_heights[cylinder_offset];
        Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
        if (intersect_cylinder(
                ray_origin,
                light_dir,
                cylinder_center,
                cylinder_radius,
                cylinder_half_height,
                scene.cylinders.axes + cylinder_offset * 9,
                &t,
                &normal) &&
            t < max_t) {
          return true;
        }
      }
    }
  }

  // Terrain casting shadows is intentionally disabled in this rasterized-terrain path.
  return false;
}

__device__ __forceinline__ Vec3 apply_lighting_mask(
    Vec3 base_color,
    Vec3 hit,
    Vec3 normal,
    Vec3 ray_dir,
    Vec3 light_dir,
    const SceneView& scene,
    const int* primitive_mask,
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
  if (options.shadows && direct > 0.0f && primitive_mask != nullptr) {
    const Vec3 shadow_origin = add(hit, mul(normal, 1.0e-3f));
    float shadow_max_t = kFloatMax;
    bool shadow_bounds_hit = true;
    if (shadow_scene_bounds_valid) {
      shadow_bounds_hit = ray_aabb_exit_t(shadow_origin, light_dir, shadow_scene_bounds, &shadow_max_t);
    }
    if (shadow_bounds_hit &&
        is_shadowed_mask(
            shadow_origin,
            light_dir,
            scene,
            primitive_mask,
            shadow_max_t,
            batch_idx,
            skip_kind,
            skip_idx)) {
      direct *= (1.0f - options.shadow_strength);
    }
  }

  return mul(base_color, ambient + direct);
}

__device__ void intersect_finite_mask_depth_range(
    Vec3 ray_origin,
    Vec3 ray_dir,
    const SceneView& scene,
    const int* primitive_mask,
    float min_t,
    float max_t,
    int batch_idx,
    HitRecord* hit) {
  const float clipped_max_t = fminf(max_t, hit->t);
  if (clipped_max_t <= min_t || primitive_mask == nullptr) {
    return;
  }

  #pragma unroll
  for (int word_idx = 0; word_idx < kPrimitiveMaskWords; ++word_idx) {
    unsigned int bits = static_cast<unsigned int>(primitive_mask[word_idx]);
    while (bits != 0u) {
      const int bit = __ffs(bits) - 1;
      const int primitive_slot = word_idx * 32 + bit;
      bits &= bits - 1u;

      int kind = 0;
      int index = -1;
      if (!primitive_slot_to_kind_index(primitive_slot, scene, batch_idx, &kind, &index)) {
        continue;
      }

      float t = 0.0f;
      if (kind == 1) {
        const int sphere_offset = batch_idx * scene.spheres.count + index;
        const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
        const float sphere_radius = scene.spheres.radii[sphere_offset];
        if (intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius, &t) &&
            t >= min_t && t < clipped_max_t && t < hit->t) {
          hit->t = t;
          hit->sphere = index;
          hit->box = -1;
          hit->cylinder = -1;
        }
      } else if (kind == 3) {
        const int box_offset = batch_idx * scene.boxes.count + index;
        const Vec3 box_center = load_vec3(scene.boxes.centers + box_offset * 3);
        const Vec3 box_half_size = load_vec3(scene.boxes.half_sizes + box_offset * 3);
        Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
        if (intersect_box(ray_origin, ray_dir, box_center, box_half_size, scene.boxes.axes + box_offset * 9, &t, &normal) &&
            t >= min_t && t < clipped_max_t && t < hit->t) {
          hit->t = t;
          hit->sphere = -1;
          hit->box = index;
          hit->cylinder = -1;
          hit->box_normal = normal;
        }
      } else {
        const int cylinder_offset = batch_idx * scene.cylinders.count + index;
        const Vec3 cylinder_center = load_vec3(scene.cylinders.centers + cylinder_offset * 3);
        const float cylinder_radius = scene.cylinders.radii[cylinder_offset];
        const float cylinder_half_height = scene.cylinders.half_heights[cylinder_offset];
        Vec3 normal = make_vec3(0.0f, 0.0f, 0.0f);
        if (intersect_cylinder(
                ray_origin,
                ray_dir,
                cylinder_center,
                cylinder_radius,
                cylinder_half_height,
                scene.cylinders.axes + cylinder_offset * 9,
                &t,
                &normal) &&
            t >= min_t && t < clipped_max_t && t < hit->t) {
          hit->t = t;
          hit->sphere = -1;
          hit->box = -1;
          hit->cylinder = index;
          hit->cylinder_normal = normal;
        }
      }
    }
  }
}

__global__ void render_scene_kernel(
    float* image,
    int* instance_map,
    int* semantic_map,
    const float* terrain_depth,
    const int* primary_cluster_masks,
    const int* shadow_cluster_masks,
    const float* depth_edges,
    const float* scene_bounds,
    const int* scene_bounds_valid,
    int width,
    int height,
    int tiles_x,
    int tiles_y,
    SceneView scene,
    RenderOptionsView options) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int batch_idx = blockIdx.z;
  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int sphere_count = scene.spheres.counts[batch_idx];
  const int plane_count = 0;
  const int terrain_count = scene.terrain.counts[batch_idx];
  const int box_count = scene.boxes.counts[batch_idx];

  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float fov_radians = options.fov_degrees * 0.017453292519943295f;
  const float image_plane_scale = tanf(0.5f * fov_radians);
  const float* batch_depth_edges = depth_edges + batch_idx * (kDepthBins + 1);

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

  HitRecord finite_hit{
      terrain_t,
      -1,
      -1,
      -1,
      -1,
      make_vec3(0.0f, 0.0f, 0.0f),
      make_vec3(0.0f, 0.0f, 0.0f),
      make_vec3(0.0f, 1.0f, 0.0f)};
  for (int bin = 0; bin < kDepthBins; ++bin) {
    const float bin_near_depth = batch_depth_edges[bin];
    const float bin_far_depth = batch_depth_edges[bin + 1];
    const float bin_min_t = bin_near_depth / ray_forward_per_t;
    const float bin_max_t = bin_far_depth / ray_forward_per_t;
    if (bin_min_t >= finite_hit.t) {
      break;
    }

    const int previous_sphere = finite_hit.sphere;
    const int previous_box = finite_hit.box;
    const int previous_cylinder = finite_hit.cylinder;
    const int cluster_idx = cluster_linear_index(batch_idx, tile_y, tile_x, bin, tiles_x, tiles_y);
    const int* primary_mask = primary_cluster_masks + cluster_idx * kPrimitiveMaskWords;
    intersect_finite_mask_depth_range(
        ray_origin,
        ray_dir,
        scene,
        primary_mask,
        bin_min_t,
        bin_max_t,
        batch_idx,
        &finite_hit);

    if (finite_hit.sphere != previous_sphere || finite_hit.box != previous_box || finite_hit.cylinder != previous_cylinder) {
      // Depth bins are processed front-to-back. If the first exact finite hit lies in this bin,
      // later bins cannot contain a closer finite hit on this same primary ray.
      break;
    }
  }

  float closest_t = (finite_hit.sphere >= 0 || finite_hit.box >= 0 || finite_hit.cylinder >= 0) ? finite_hit.t : kFloatMax;
  int closest_sphere = finite_hit.sphere;
  int closest_box = finite_hit.box;
  int closest_cylinder = finite_hit.cylinder;
  int closest_terrain = -1;
  int instance_id = 0;
  int semantic_id = 0;
  Vec3 closest_box_normal = finite_hit.box_normal;
  Vec3 closest_cylinder_normal = finite_hit.cylinder_normal;
  Vec3 closest_terrain_normal = finite_hit.terrain_normal;

  if (terrain_count > 0 && terrain_depth != nullptr) {
    if (terrain_t < closest_t) {
      closest_t = terrain_t;
      closest_sphere = -1;
      closest_box = -1;
      closest_cylinder = -1;
      closest_terrain = 0;
      const Vec3 terrain_hit = add(ray_origin, mul(ray_dir, closest_t));
      closest_terrain_normal = terrain_normal_at_hit(scene.terrain, batch_idx, terrain_hit);
    }
  }

  const int shadow_scene_bounds_is_valid = scene_bounds_valid[batch_idx];
  const Aabb shadow_scene_bounds = shadow_scene_bounds_is_valid ? load_scene_bounds6(scene_bounds, batch_idx) : empty_aabb();

  if (closest_sphere >= 0) {
    const int sphere_offset = batch_idx * scene.spheres.count + closest_sphere;
    instance_id = closest_sphere + 1;
    semantic_id = 1;
    const Vec3 sphere_center = load_vec3(scene.spheres.centers + sphere_offset * 3);
    const Vec3 sphere_color = load_vec3(scene.spheres.colors + sphere_offset * 3);
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(sub(hit, sphere_center));
    const int receiver_bin = depth_to_bin_from_edges(fmaxf(-hit.z, kCameraNear), batch_depth_edges);
    const int shadow_cluster_idx = cluster_linear_index(batch_idx, tile_y, tile_x, receiver_bin, tiles_x, tiles_y);
    const int* shadow_mask = shadow_cluster_masks == nullptr ? nullptr : shadow_cluster_masks + shadow_cluster_idx * kPrimitiveMaskWords;
    color = apply_lighting_mask(
        sphere_color,
        hit,
        normal,
        ray_dir,
        light_dir,
        scene,
        shadow_mask,
        options,
        shadow_scene_bounds,
        shadow_scene_bounds_is_valid,
        batch_idx,
        1,
        closest_sphere);
  } else if (closest_box >= 0) {
    const int box_offset = batch_idx * scene.boxes.count + closest_box;
    instance_id = sphere_count + plane_count + terrain_count + closest_box + 1;
    semantic_id = 3;
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(closest_box_normal);
    const Vec3 box_color = load_vec3(scene.boxes.colors + box_offset * 3);
    const int receiver_bin = depth_to_bin_from_edges(fmaxf(-hit.z, kCameraNear), batch_depth_edges);
    const int shadow_cluster_idx = cluster_linear_index(batch_idx, tile_y, tile_x, receiver_bin, tiles_x, tiles_y);
    const int* shadow_mask = shadow_cluster_masks == nullptr ? nullptr : shadow_cluster_masks + shadow_cluster_idx * kPrimitiveMaskWords;
    color = apply_lighting_mask(
        box_color,
        hit,
        normal,
        ray_dir,
        light_dir,
        scene,
        shadow_mask,
        options,
        shadow_scene_bounds,
        shadow_scene_bounds_is_valid,
        batch_idx,
        3,
        closest_box);
  } else if (closest_cylinder >= 0) {
    const int cylinder_offset = batch_idx * scene.cylinders.count + closest_cylinder;
    instance_id = sphere_count + plane_count + terrain_count + box_count + closest_cylinder + 1;
    semantic_id = 4;
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 normal = normalize(closest_cylinder_normal);
    const Vec3 cylinder_color = load_vec3(scene.cylinders.colors + cylinder_offset * 3);
    const int receiver_bin = depth_to_bin_from_edges(fmaxf(-hit.z, kCameraNear), batch_depth_edges);
    const int shadow_cluster_idx = cluster_linear_index(batch_idx, tile_y, tile_x, receiver_bin, tiles_x, tiles_y);
    const int* shadow_mask = shadow_cluster_masks == nullptr ? nullptr : shadow_cluster_masks + shadow_cluster_idx * kPrimitiveMaskWords;
    color = apply_lighting_mask(
        cylinder_color,
        hit,
        normal,
        ray_dir,
        light_dir,
        scene,
        shadow_mask,
        options,
        shadow_scene_bounds,
        shadow_scene_bounds_is_valid,
        batch_idx,
        4,
        closest_cylinder);
  } else if (closest_terrain >= 0) {
    instance_id = sphere_count + plane_count + 1;
    semantic_id = 2;
    const Vec3 hit = add(ray_origin, mul(ray_dir, closest_t));
    const Vec3 terrain_color = load_vec3(scene.terrain.colors + batch_idx * 3);
    const int receiver_bin = depth_to_bin_from_edges(fmaxf(-hit.z, kCameraNear), batch_depth_edges);
    const int shadow_cluster_idx = cluster_linear_index(batch_idx, tile_y, tile_x, receiver_bin, tiles_x, tiles_y);
    const int* shadow_mask = shadow_cluster_masks == nullptr ? nullptr : shadow_cluster_masks + shadow_cluster_idx * kPrimitiveMaskWords;
    color = apply_lighting_mask(
        terrain_color,
        hit,
        closest_terrain_normal,
        ray_dir,
        light_dir,
        scene,
        shadow_mask,
        options,
        shadow_scene_bounds,
        shadow_scene_bounds_is_valid,
        batch_idx,
        5,
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
    torch::Tensor cylinder_centers,
    torch::Tensor cylinder_radii,
    torch::Tensor cylinder_half_heights,
    torch::Tensor cylinder_axes,
    torch::Tensor cylinder_counts,
    torch::Tensor light_dir,
    double fov_degrees,
    torch::Tensor background,
    torch::Tensor sphere_colors,
    torch::Tensor plane_colors,
    torch::Tensor terrain_colors,
    torch::Tensor box_colors,
    torch::Tensor cylinder_colors,
    double ambient,
    bool shadows,
    double shadow_strength) {
  const int batch_size = static_cast<int>(image.size(0));
  const int height = static_cast<int>(image.size(2));
  const int width = static_cast<int>(image.size(3));
  const int sphere_count = static_cast<int>(sphere_centers.size(1));
  const int plane_count = static_cast<int>(plane_points.size(1));
  const int box_count = static_cast<int>(box_centers.size(1));
  const int cylinder_count = static_cast<int>(cylinder_centers.size(1));
  TORCH_CHECK(sphere_count <= kMaxSpheres, "render_scene supports at most ", kMaxSpheres, " spheres");
  TORCH_CHECK(box_count <= kMaxBoxes, "render_scene supports at most ", kMaxBoxes, " boxes");
  TORCH_CHECK(cylinder_count <= kMaxCylinders, "render_scene supports at most ", kMaxCylinders, " cylinders");

  const dim3 block(kTileWidth, kTileHeight);
  const int tiles_x = (width + block.x - 1) / block.x;
  const int tiles_y = (height + block.y - 1) / block.y;
  const dim3 grid(tiles_x, tiles_y, batch_size);

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
      CylinderView{
          cylinder_centers.data_ptr<float>(),
          cylinder_radii.data_ptr<float>(),
          cylinder_half_heights.data_ptr<float>(),
          cylinder_axes.data_ptr<float>(),
          cylinder_colors.data_ptr<float>(),
          cylinder_counts.data_ptr<int>(),
          cylinder_count,
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

  const auto int_options = image.options().dtype(torch::kInt32);
  auto depth_edges = torch::empty({batch_size, kDepthBins + 1}, image.options());
  auto scene_bounds = torch::empty({batch_size, 6}, image.options());
  auto scene_bounds_valid = torch::empty({batch_size}, int_options);

  compute_cluster_metadata_kernel<<<batch_size, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      depth_edges.data_ptr<float>(),
      scene_bounds.data_ptr<float>(),
      scene_bounds_valid.data_ptr<int>(),
      width,
      height,
      scene);

  auto primary_cluster_masks = torch::zeros(
      {batch_size, tiles_y, tiles_x, kDepthBins, kPrimitiveMaskWords}, int_options);

  torch::Tensor shadow_cluster_masks;
  torch::Tensor receiver_light_bounds;
  int* shadow_cluster_masks_ptr = nullptr;

  const int primitive_slot_count = sphere_count + box_count + cylinder_count;
  const int total_tiles = tiles_x * tiles_y;
  const int total_clusters = total_tiles * kDepthBins;
  const int cluster_threads = 128;

  if (primitive_slot_count > 0) {
    const dim3 primary_build_grid(primitive_slot_count, batch_size);
    build_primary_cluster_masks_object_driven_kernel<<<primary_build_grid, cluster_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        primary_cluster_masks.data_ptr<int>(),
        depth_edges.data_ptr<float>(),
        width,
        height,
        tiles_x,
        tiles_y,
        scene,
        options);

    if (shadows) {
      shadow_cluster_masks = torch::zeros(
          {batch_size, tiles_y, tiles_x, kDepthBins, kPrimitiveMaskWords}, int_options);
      receiver_light_bounds = torch::empty(
          {batch_size, tiles_y, tiles_x, kDepthBins, 6}, image.options());

      const dim3 receiver_grid((total_clusters + cluster_threads - 1) / cluster_threads, batch_size);
      compute_receiver_light_bounds_kernel<<<receiver_grid, cluster_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          receiver_light_bounds.data_ptr<float>(),
          depth_edges.data_ptr<float>(),
          width,
          height,
          tiles_x,
          tiles_y,
          options);

      const dim3 shadow_build_grid((total_clusters + cluster_threads - 1) / cluster_threads, primitive_slot_count, batch_size);
      build_shadow_cluster_masks_kernel<<<shadow_build_grid, cluster_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          shadow_cluster_masks.data_ptr<int>(),
          receiver_light_bounds.data_ptr<float>(),
          width,
          height,
          tiles_x,
          tiles_y,
          scene,
          options);
      shadow_cluster_masks_ptr = shadow_cluster_masks.data_ptr<int>();
    }
  }

  render_scene_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      image.data_ptr<float>(),
      instance_map.numel() == 0 ? nullptr : instance_map.data_ptr<int>(),
      semantic_map.numel() == 0 ? nullptr : semantic_map.data_ptr<int>(),
      terrain_depth.data_ptr<float>(),
      primary_cluster_masks.data_ptr<int>(),
      shadow_cluster_masks_ptr,
      depth_edges.data_ptr<float>(),
      scene_bounds.data_ptr<float>(),
      scene_bounds_valid.data_ptr<int>(),
      width,
      height,
      tiles_x,
      tiles_y,
      scene,
      options);

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
