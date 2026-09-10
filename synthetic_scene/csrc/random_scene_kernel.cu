#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <array>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = 2.0f * kPi;
constexpr int kThreadsPerBlock = 64;
constexpr int kOutputCount = 36;
constexpr int kHouseClassId = 10;
constexpr int kTreeClassId = 11;
constexpr int kCloudClassId = 12;
constexpr int kCarClassId = 13;
constexpr int kPersonClassId = 14;

enum OutputIndex {
  kSphereCenters,
  kSphereRadii,
  kSphereColors,
  kSphereCounts,
  kSphereClassIds,
  kSphereInstanceIds,
  kTerrainBaseHeights,
  kTerrainDepthLimits,
  kTerrainPhaseXs,
  kTerrainPhaseZs,
  kTerrainDz,
  kTerrainDzGrowth,
  kTerrainColors,
  kTerrainCounts,
  kBoxCenters,
  kBoxHalfSizes,
  kBoxAxes,
  kBoxColors,
  kBoxCounts,
  kBoxClassIds,
  kBoxInstanceIds,
  kPrismCenters,
  kPrismHalfSizes,
  kPrismAxes,
  kPrismColors,
  kPrismCounts,
  kPrismClassIds,
  kPrismInstanceIds,
  kCylinderCenters,
  kCylinderRadii,
  kCylinderHalfHeights,
  kCylinderAxes,
  kCylinderColors,
  kCylinderCounts,
  kCylinderClassIds,
  kCylinderInstanceIds,
};

struct Vec3 {
  float x;
  float y;
  float z;
};

__device__ Vec3 add(Vec3 left, Vec3 right) {
  return {left.x + right.x, left.y + right.y, left.z + right.z};
}

__device__ Vec3 rotate_yaw(Vec3 vector, float cosine, float sine) {
  return {cosine * vector.x - sine * vector.z, vector.y, sine * vector.x + cosine * vector.z};
}

// SplitMix64 gives every batch element a small, independent deterministic RNG.
struct RandomGenerator {
  unsigned long long state;

  __device__ unsigned long long next() {
    state += 0x9e3779b97f4a7c15ULL;
    unsigned long long value = state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  __device__ float uniform(float low, float high) {
    constexpr float kInverse24BitRange = 1.0f / 16777216.0f;
    return low + (high - low) * static_cast<float>(next() >> 40) * kInverse24BitRange;
  }
};

struct SceneWriter {
  float* sphere_centers;
  float* sphere_radii;
  float* sphere_colors;
  int* sphere_class_ids;
  int* sphere_instance_ids;
  float* box_centers;
  float* box_half_sizes;
  float* box_axes;
  float* box_colors;
  int* box_class_ids;
  int* box_instance_ids;
  float* prism_centers;
  float* prism_half_sizes;
  float* prism_axes;
  float* prism_colors;
  int* prism_class_ids;
  int* prism_instance_ids;
  float* cylinder_centers;
  float* cylinder_radii;
  float* cylinder_half_heights;
  float* cylinder_axes;
  float* cylinder_colors;
  int* cylinder_class_ids;
  int* cylinder_instance_ids;
  int sphere_count = 0;
  int box_count = 0;
  int prism_count = 0;
  int cylinder_count = 0;

  __device__ static void write_vec3(float* output, int index, Vec3 value) {
    output[index * 3] = value.x;
    output[index * 3 + 1] = value.y;
    output[index * 3 + 2] = value.z;
  }

  __device__ static void write_axes(
      float* output, int index, float cosine, float sine, bool wheel_axes = false) {
    float* axes = output + index * 9;
    const float object_values[9] = {cosine, 0, -sine, 0, 1, 0, sine, 0, cosine};
    const float wheel_values[9] = {0, cosine, -sine, 1, 0, 0, 0, sine, cosine};
    const float* values = wheel_axes ? wheel_values : object_values;
    for (int element = 0; element < 9; ++element) {
      axes[element] = values[element];
    }
  }

  __device__ void add_sphere(Vec3 center, float radius, Vec3 color, int class_id, int instance_id) {
    write_vec3(sphere_centers, sphere_count, center);
    sphere_radii[sphere_count] = radius;
    write_vec3(sphere_colors, sphere_count, color);
    sphere_class_ids[sphere_count] = class_id;
    sphere_instance_ids[sphere_count] = instance_id;
    ++sphere_count;
  }

  __device__ void add_box(
      Vec3 center, Vec3 size, float cosine, float sine, Vec3 color, int class_id, int instance_id) {
    write_vec3(box_centers, box_count, center);
    write_vec3(box_half_sizes, box_count, size);
    write_axes(box_axes, box_count, cosine, sine);
    write_vec3(box_colors, box_count, color);
    box_class_ids[box_count] = class_id;
    box_instance_ids[box_count] = instance_id;
    ++box_count;
  }

  __device__ void add_prism(
      Vec3 center, Vec3 size, float cosine, float sine, Vec3 color, int class_id, int instance_id) {
    write_vec3(prism_centers, prism_count, center);
    write_vec3(prism_half_sizes, prism_count, size);
    write_axes(prism_axes, prism_count, cosine, sine);
    write_vec3(prism_colors, prism_count, color);
    prism_class_ids[prism_count] = class_id;
    prism_instance_ids[prism_count] = instance_id;
    ++prism_count;
  }

  __device__ void add_cylinder(
      Vec3 center, float radius, float height, float cosine, float sine, Vec3 color,
      int class_id, int instance_id, bool wheel_axes = false) {
    write_vec3(cylinder_centers, cylinder_count, center);
    cylinder_radii[cylinder_count] = radius;
    cylinder_half_heights[cylinder_count] = height;
    write_axes(cylinder_axes, cylinder_count, cosine, sine, wheel_axes);
    write_vec3(cylinder_colors, cylinder_count, color);
    cylinder_class_ids[cylinder_count] = class_id;
    cylinder_instance_ids[cylinder_count] = instance_id;
    ++cylinder_count;
  }
};

__device__ float smooth_terrain_height(float x, float z, float phase_x, float phase_z) {
  const float forward_depth = fmaxf(-z, 0.0f);
  const float hill_t = fminf(fmaxf((forward_depth - 24.0f) / 48.0f, 0.0f), 1.0f);
  const float hill_weight = hill_t * hill_t * (3.0f - 2.0f * hill_t);
  const float detail_weight = 0.06f + 0.94f * hill_weight;
  const float rise = hill_weight * (0.055f * forward_depth + 0.0011f * forward_depth * forward_depth);
  const float broad = 1.20f * sinf(0.18f * x + 0.11f * z + phase_x) +
                      0.85f * cosf(0.13f * x - 0.20f * z + phase_z);
  const float foothills = 0.42f * sinf(0.46f * x + 0.34f * z + 0.61f * phase_x + 0.23f * phase_z);
  const float detail = 0.12f * sinf(0.95f * x - 0.58f * z + 1.37f * phase_z);
  return rise + detail_weight * (broad + foothills + detail);
}

__device__ Vec3 random_frustum_point(
    RandomGenerator& random, float fov, float aspect, float near, float far,
    float minimum_y, float maximum_y) {
  const float scale = tanf(0.5f * fov * (kPi / 180.0f));
  const float image_x = random.uniform(-1.0f, 1.0f) * aspect * scale;
  const float image_y = random.uniform(minimum_y, maximum_y) * scale;
  const float inverse_length = rsqrtf(image_x * image_x + image_y * image_y + 1.0f);
  const float distance = random.uniform(near, far);
  return {image_x * inverse_length * distance, image_y * inverse_length * distance, -inverse_length * distance};
}

__device__ Vec3 random_ground_point(
    RandomGenerator& random, float fov, float aspect, float radius,
    float ground_y, float phase_x, float phase_z) {
  Vec3 point = random_frustum_point(random, fov, aspect, 2.0f, 2.0f + radius, -0.82f, -0.18f);
  point.y = ground_y + smooth_terrain_height(point.x, point.z, phase_x, phase_z);
  return point;
}

__device__ void random_yaw(RandomGenerator& random, float& cosine, float& sine) {
  const float yaw = random.uniform(-kPi, kPi);
  cosine = cosf(yaw);
  sine = sinf(yaw);
}

__device__ void add_houses(
    SceneWriter& writer, RandomGenerator& random, int count, float fov, float aspect,
    float radius, float ground_y, float phase_x, float phase_z, int& instance_id) {
  for (int object = 0; object < count; ++object) {
    const Vec3 position = random_ground_point(random, fov, aspect, radius, ground_y, phase_x, phase_z);
    float cosine;
    float sine;
    random_yaw(random, cosine, sine);
    const float width = random.uniform(0.75f, 1.65f);
    const float depth = random.uniform(0.65f, 1.35f);
    const float body_height = random.uniform(0.55f, 1.15f);
    const float roof_height = random.uniform(0.28f, 0.60f);
    writer.add_box(add(position, rotate_yaw({0, 0.5f * body_height, 0}, cosine, sine)),
                   {0.5f * width, 0.5f * body_height, 0.5f * depth}, cosine, sine,
                   {0.62f, 0.43f, 0.30f}, kHouseClassId, instance_id);
    writer.add_prism(add(position, rotate_yaw({0, body_height + 0.5f * roof_height, 0}, cosine, sine)),
                     {0.5f * width + 0.12f, 0.5f * roof_height, 0.5f * depth + 0.12f}, cosine, sine,
                     {0.72f, 0.14f, 0.10f}, kHouseClassId, instance_id++);
  }
}

__device__ void add_trees(
    SceneWriter& writer, RandomGenerator& random, int count, float fov, float aspect,
    float radius, float ground_y, float phase_x, float phase_z, int& instance_id) {
  for (int object = 0; object < count; ++object) {
    const Vec3 position = random_ground_point(random, fov, aspect, radius, ground_y, phase_x, phase_z);
    const float trunk_height = random.uniform(0.65f, 1.45f);
    const float trunk_radius = random.uniform(0.06f, 0.16f);
    const float crown_radius = random.uniform(0.28f, 0.62f);
    writer.add_sphere(add(position, {0, trunk_height + 0.55f * crown_radius, 0}), crown_radius,
                      {0.16f, 0.48f, 0.18f}, kTreeClassId, instance_id);
    writer.add_cylinder(add(position, {0, 0.5f * trunk_height, 0}), trunk_radius, 0.5f * trunk_height,
                        1.0f, 0.0f, {0.42f, 0.25f, 0.12f}, kTreeClassId, instance_id++);
  }
}

__device__ void add_clouds(
    SceneWriter& writer, RandomGenerator& random, int count, float fov, float aspect,
    float radius, int& instance_id) {
  for (int object = 0; object < count; ++object) {
    Vec3 position = random_frustum_point(random, fov, aspect, 5.0f, 5.0f + radius, 0.24f, 0.88f);
    position.y += 1.8f;
    float cosine;
    float sine;
    random_yaw(random, cosine, sine);
    const float scale = random.uniform(0.75f, 1.35f);
    const Vec3 color{0.93f, 0.95f, 0.96f};
    writer.add_sphere(add(position, rotate_yaw({-0.32f * scale, 0, 0}, cosine, sine)), 0.34f * scale, color, kCloudClassId, instance_id);
    writer.add_sphere(add(position, rotate_yaw({0.08f * scale, 0.10f * scale, 0.02f * scale}, cosine, sine)), 0.42f * scale, color, kCloudClassId, instance_id);
    writer.add_sphere(add(position, rotate_yaw({0.46f * scale, -0.02f * scale, 0.03f * scale}, cosine, sine)), 0.31f * scale, color, kCloudClassId, instance_id++);
  }
}

__device__ void add_cars(
    SceneWriter& writer, RandomGenerator& random, int count, float fov, float aspect,
    float radius, float ground_y, float phase_x, float phase_z, int& instance_id) {
  for (int object = 0; object < count; ++object) {
    const Vec3 position = random_ground_point(random, fov, aspect, radius, ground_y, phase_x, phase_z);
    float cosine;
    float sine;
    random_yaw(random, cosine, sine);
    const float length = random.uniform(0.85f, 1.35f);
    const float width = random.uniform(0.42f, 0.62f);
    const float height = random.uniform(0.26f, 0.42f);
    const float wheel_radius = 0.13f * length;
    const Vec3 body_color{random.uniform(0.18f, 0.85f), random.uniform(0.12f, 0.55f), random.uniform(0.12f, 0.45f)};
    writer.add_box(add(position, rotate_yaw({0, wheel_radius + 0.5f * height, 0}, cosine, sine)),
                   {0.5f * length, 0.5f * height, 0.5f * width}, cosine, sine,
                   body_color, kCarClassId, instance_id);
    for (int x_sign = -1; x_sign <= 1; x_sign += 2) {
      for (int z_sign = -1; z_sign <= 1; z_sign += 2) {
        const Vec3 offset{x_sign * 0.32f * length, wheel_radius, z_sign * 0.58f * width};
        writer.add_cylinder(add(position, rotate_yaw(offset, cosine, sine)), wheel_radius, 0.08f * width,
                            cosine, sine, {0.04f, 0.04f, 0.04f}, kCarClassId, instance_id, true);
      }
    }
    ++instance_id;
  }
}

__device__ void add_people(
    SceneWriter& writer, RandomGenerator& random, int count, float fov, float aspect,
    float radius, float ground_y, float phase_x, float phase_z, int& instance_id) {
  for (int object = 0; object < count; ++object) {
    const Vec3 position = random_ground_point(random, fov, aspect, radius, ground_y, phase_x, phase_z);
    float cosine;
    float sine;
    random_yaw(random, cosine, sine);
    const float height = random.uniform(0.95f, 1.45f);
    const float leg_height = 0.34f * height;
    const float body_height = 0.34f * height;
    const float arm_height = 0.31f * height;
    const float head_radius = 0.105f * height;
    writer.add_box(add(position, rotate_yaw({0, leg_height + 0.5f * body_height, 0}, cosine, sine)),
                   {0.15f * height, 0.5f * body_height, 0.055f * height}, cosine, sine,
                   {0.18f, 0.34f, 0.82f}, kPersonClassId, instance_id);
    for (int side = -1; side <= 1; side += 2) {
      writer.add_box(add(position, rotate_yaw({side * 0.055f * height, 0.5f * leg_height, 0}, cosine, sine)),
                     {0.04f * height, 0.5f * leg_height, 0.04f * height}, cosine, sine,
                     {0.10f, 0.12f, 0.18f}, kPersonClassId, instance_id);
    }
    for (int side = -1; side <= 1; side += 2) {
      writer.add_box(add(position, rotate_yaw({side * 0.18f * height, leg_height + 0.47f * body_height, 0}, cosine, sine)),
                     {0.035f * height, 0.5f * arm_height, 0.035f * height}, cosine, sine,
                     {0.10f, 0.12f, 0.18f}, kPersonClassId, instance_id);
    }
    writer.add_sphere(add(position, rotate_yaw({0, leg_height + body_height + 1.08f * head_radius, 0}, cosine, sine)),
                      head_radius, {0.78f, 0.56f, 0.40f}, kPersonClassId, instance_id++);
  }
}

__global__ void generate_random_scenes(
    float** float_outputs,
    int** integer_outputs,
    long long seed,
    int batch_size,
    float scatter_radius,
    float ground_y,
    float depth_limit,
    float terrain_dz,
    float terrain_dz_growth,
    float fov_degrees,
    float aspect_ratio,
    int house_count,
    int tree_count,
    int cloud_count,
    int car_count,
    int person_count,
    int sphere_capacity,
    int box_capacity,
    int prism_capacity,
    int cylinder_capacity) {
  const int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  // A thread owns one complete batch slice, so writing primitives requires no atomics.
  SceneWriter writer{
      float_outputs[kSphereCenters] + batch_index * sphere_capacity * 3,
      float_outputs[kSphereRadii] + batch_index * sphere_capacity,
      float_outputs[kSphereColors] + batch_index * sphere_capacity * 3,
      integer_outputs[kSphereClassIds] + batch_index * sphere_capacity,
      integer_outputs[kSphereInstanceIds] + batch_index * sphere_capacity,
      float_outputs[kBoxCenters] + batch_index * box_capacity * 3,
      float_outputs[kBoxHalfSizes] + batch_index * box_capacity * 3,
      float_outputs[kBoxAxes] + batch_index * box_capacity * 9,
      float_outputs[kBoxColors] + batch_index * box_capacity * 3,
      integer_outputs[kBoxClassIds] + batch_index * box_capacity,
      integer_outputs[kBoxInstanceIds] + batch_index * box_capacity,
      float_outputs[kPrismCenters] + batch_index * prism_capacity * 3,
      float_outputs[kPrismHalfSizes] + batch_index * prism_capacity * 3,
      float_outputs[kPrismAxes] + batch_index * prism_capacity * 9,
      float_outputs[kPrismColors] + batch_index * prism_capacity * 3,
      integer_outputs[kPrismClassIds] + batch_index * prism_capacity,
      integer_outputs[kPrismInstanceIds] + batch_index * prism_capacity,
      float_outputs[kCylinderCenters] + batch_index * cylinder_capacity * 3,
      float_outputs[kCylinderRadii] + batch_index * cylinder_capacity,
      float_outputs[kCylinderHalfHeights] + batch_index * cylinder_capacity,
      float_outputs[kCylinderAxes] + batch_index * cylinder_capacity * 9,
      float_outputs[kCylinderColors] + batch_index * cylinder_capacity * 3,
      integer_outputs[kCylinderClassIds] + batch_index * cylinder_capacity,
      integer_outputs[kCylinderInstanceIds] + batch_index * cylinder_capacity,
  };

  RandomGenerator random{
      static_cast<unsigned long long>(seed) ^ (0xd1b54a32d192ed03ULL * (batch_index + 1))};
  const float phase_x = random.uniform(0.0f, kTau);
  const float phase_z = random.uniform(0.0f, kTau);

  float_outputs[kTerrainBaseHeights][batch_index] = ground_y;
  float_outputs[kTerrainDepthLimits][batch_index] = depth_limit;
  float_outputs[kTerrainPhaseXs][batch_index] = phase_x;
  float_outputs[kTerrainPhaseZs][batch_index] = phase_z;
  float_outputs[kTerrainDz][batch_index] = terrain_dz;
  float_outputs[kTerrainDzGrowth][batch_index] = terrain_dz_growth;
  SceneWriter::write_vec3(float_outputs[kTerrainColors], batch_index, {0.34f, 0.46f, 0.28f});
  integer_outputs[kTerrainCounts][batch_index] = 1;

  int instance_id = 1;
  add_houses(writer, random, house_count, fov_degrees, aspect_ratio, scatter_radius,
             ground_y, phase_x, phase_z, instance_id);
  add_trees(writer, random, tree_count, fov_degrees, aspect_ratio, scatter_radius,
            ground_y, phase_x, phase_z, instance_id);
  add_clouds(writer, random, cloud_count, fov_degrees, aspect_ratio, scatter_radius, instance_id);
  add_cars(writer, random, car_count, fov_degrees, aspect_ratio, scatter_radius,
           ground_y, phase_x, phase_z, instance_id);
  add_people(writer, random, person_count, fov_degrees, aspect_ratio, scatter_radius,
             ground_y, phase_x, phase_z, instance_id);

  integer_outputs[kSphereCounts][batch_index] = writer.sphere_count;
  integer_outputs[kBoxCounts][batch_index] = writer.box_count;
  integer_outputs[kPrismCounts][batch_index] = writer.prism_count;
  integer_outputs[kCylinderCounts][batch_index] = writer.cylinder_count;
}

}  // namespace

void random_scene_cuda(
    const std::vector<torch::Tensor>& outputs,
    int64_t seed,
    float scatter_radius,
    float ground_y,
    float depth_limit,
    float terrain_dz,
    float terrain_dz_growth,
    float fov_degrees,
    float aspect_ratio,
    int house_count,
    int tree_count,
    int cloud_count,
    int car_count,
    int person_count) {
  std::array<float*, kOutputCount> host_float_pointers{};
  std::array<int*, kOutputCount> host_integer_pointers{};
  for (int index = 0; index < kOutputCount; ++index) {
    if (outputs[index].scalar_type() == torch::kFloat32) {
      host_float_pointers[index] = outputs[index].data_ptr<float>();
    } else {
      host_integer_pointers[index] = outputs[index].data_ptr<int>();
    }
  }

  // The kernel receives two small pointer tables instead of a 36-pointer signature.
  const auto pointer_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  auto device_float_pointers = torch::empty({kOutputCount}, pointer_options);
  auto device_integer_pointers = torch::empty({kOutputCount}, pointer_options);
  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(
      device_float_pointers.data_ptr(),
      host_float_pointers.data(),
      sizeof(float*) * kOutputCount,
      cudaMemcpyHostToDevice,
      stream);
  cudaMemcpyAsync(
      device_integer_pointers.data_ptr(),
      host_integer_pointers.data(),
      sizeof(int*) * kOutputCount,
      cudaMemcpyHostToDevice,
      stream);

  const int batch_size = outputs[kSphereCounts].size(0);
  const int sphere_capacity = outputs[kSphereRadii].size(1);
  const int box_capacity = outputs[kBoxHalfSizes].size(1);
  const int prism_capacity = outputs[kPrismHalfSizes].size(1);
  const int cylinder_capacity = outputs[kCylinderRadii].size(1);
  const int block_count = (batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock;

  generate_random_scenes<<<block_count, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<float**>(device_float_pointers.data_ptr()),
      reinterpret_cast<int**>(device_integer_pointers.data_ptr()),
      seed,
      batch_size,
      scatter_radius,
      ground_y,
      depth_limit,
      terrain_dz,
      terrain_dz_growth,
      fov_degrees,
      aspect_ratio,
      house_count,
      tree_count,
      cloud_count,
      car_count,
      person_count,
      sphere_capacity,
      box_capacity,
      prism_capacity,
      cylinder_capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
