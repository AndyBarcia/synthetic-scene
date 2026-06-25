#pragma once

#include <cstdint>
#include <vector>

namespace synthetic_scene {

constexpr int kHouseClassId = 10;
constexpr int kTreeClassId = 11;
constexpr int kCloudClassId = 12;
constexpr int kCarClassId = 13;
constexpr int kPersonClassId = 14;

struct Vec3 {
  float x;
  float y;
  float z;
};

struct Mat3 {
  Vec3 rows[3];
};

inline Vec3 add3(Vec3 a, Vec3 b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Mat3 multiply3(Mat3 a, Mat3 b) {
  Mat3 result{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const Vec3& ar = a.rows[row];
      const Vec3 bc{b.rows[0].x, b.rows[1].x, b.rows[2].x};
      const Vec3 by{b.rows[0].y, b.rows[1].y, b.rows[2].y};
      const Vec3 bz{b.rows[0].z, b.rows[1].z, b.rows[2].z};
      const Vec3 axis = col == 0 ? bc : (col == 1 ? by : bz);
      const float value = ar.x * axis.x + ar.y * axis.y + ar.z * axis.z;
      if (col == 0) {
        result.rows[row].x = value;
      } else if (col == 1) {
        result.rows[row].y = value;
      } else {
        result.rows[row].z = value;
      }
    }
  }
  return result;
}

inline Vec3 rotate3(Mat3 rotation, Vec3 vector) {
  return Vec3{
      rotation.rows[0].x * vector.x + rotation.rows[0].y * vector.y + rotation.rows[0].z * vector.z,
      rotation.rows[1].x * vector.x + rotation.rows[1].y * vector.y + rotation.rows[1].z * vector.z,
      rotation.rows[2].x * vector.x + rotation.rows[2].y * vector.y + rotation.rows[2].z * vector.z,
  };
}

inline void append_vec3(std::vector<float>& values, Vec3 vector) {
  values.push_back(vector.x);
  values.push_back(vector.y);
  values.push_back(vector.z);
}

inline void append_mat3(std::vector<float>& values, Mat3 matrix) {
  for (const Vec3& row : matrix.rows) {
    append_vec3(values, row);
  }
}

struct RandomPrimitiveWriter {
  std::vector<float>& sphere_centers;
  std::vector<float>& sphere_radii;
  std::vector<float>& sphere_colors;
  std::vector<int32_t>& sphere_class_ids;
  std::vector<int32_t>& sphere_instance_ids;
  int& scene_spheres;

  std::vector<float>& box_centers;
  std::vector<float>& box_half_sizes;
  std::vector<float>& box_axes;
  std::vector<float>& box_colors;
  std::vector<int32_t>& box_class_ids;
  std::vector<int32_t>& box_instance_ids;
  int& scene_boxes;

  std::vector<float>& prism_centers;
  std::vector<float>& prism_half_sizes;
  std::vector<float>& prism_axes;
  std::vector<float>& prism_colors;
  std::vector<int32_t>& prism_class_ids;
  std::vector<int32_t>& prism_instance_ids;
  int& scene_prisms;

  std::vector<float>& cylinder_centers;
  std::vector<float>& cylinder_radii;
  std::vector<float>& cylinder_half_heights;
  std::vector<float>& cylinder_axes;
  std::vector<float>& cylinder_colors;
  std::vector<int32_t>& cylinder_class_ids;
  std::vector<int32_t>& cylinder_instance_ids;
  int& scene_cylinders;

  void add_sphere(Vec3 center, float radius, Vec3 color, int32_t class_id, int32_t instance_id) {
    append_vec3(sphere_centers, center);
    sphere_radii.push_back(radius);
    append_vec3(sphere_colors, color);
    sphere_class_ids.push_back(class_id);
    sphere_instance_ids.push_back(instance_id);
    ++scene_spheres;
  }

  void add_box(Vec3 center, Vec3 half_size, Mat3 axes, Vec3 color, int32_t class_id, int32_t instance_id) {
    append_vec3(box_centers, center);
    append_vec3(box_half_sizes, half_size);
    append_mat3(box_axes, axes);
    append_vec3(box_colors, color);
    box_class_ids.push_back(class_id);
    box_instance_ids.push_back(instance_id);
    ++scene_boxes;
  }

  void add_prism(Vec3 center, Vec3 half_size, Mat3 axes, Vec3 color, int32_t class_id, int32_t instance_id) {
    append_vec3(prism_centers, center);
    append_vec3(prism_half_sizes, half_size);
    append_mat3(prism_axes, axes);
    append_vec3(prism_colors, color);
    prism_class_ids.push_back(class_id);
    prism_instance_ids.push_back(instance_id);
    ++scene_prisms;
  }

  void add_cylinder(Vec3 center, float radius, float half_height, Mat3 axes, Vec3 color, int32_t class_id, int32_t instance_id) {
    append_vec3(cylinder_centers, center);
    cylinder_radii.push_back(radius);
    cylinder_half_heights.push_back(half_height);
    append_mat3(cylinder_axes, axes);
    append_vec3(cylinder_colors, color);
    cylinder_class_ids.push_back(class_id);
    cylinder_instance_ids.push_back(instance_id);
    ++scene_cylinders;
  }
};

template <typename RandFloat>
void add_random_house(RandomPrimitiveWriter& writer, Vec3 position, Mat3 rotation, int32_t instance_id, RandFloat&& rand_float) {
  const float width = rand_float(0.75f, 1.65f);
  const float depth = rand_float(0.65f, 1.35f);
  const float body_height = rand_float(0.55f, 1.15f);
  const float roof_height = rand_float(0.28f, 0.60f);
  const float roof_overhang = 0.12f;
  writer.add_box(
      add3(position, rotate3(rotation, Vec3{0.0f, 0.5f * body_height, 0.0f})),
      Vec3{0.5f * width, 0.5f * body_height, 0.5f * depth},
      rotation,
      Vec3{0.62f, 0.43f, 0.30f},
      kHouseClassId,
      instance_id);
  writer.add_prism(
      add3(position, rotate3(rotation, Vec3{0.0f, body_height + 0.5f * roof_height, 0.0f})),
      Vec3{0.5f * width + roof_overhang, 0.5f * roof_height, 0.5f * depth + roof_overhang},
      rotation,
      Vec3{0.72f, 0.14f, 0.10f},
      kHouseClassId,
      instance_id);
}

template <typename RandFloat>
void add_random_tree(RandomPrimitiveWriter& writer, Vec3 position, Mat3 axes, int32_t instance_id, RandFloat&& rand_float) {
  const float trunk_height = rand_float(0.65f, 1.45f);
  const float trunk_radius = rand_float(0.06f, 0.16f);
  const float crown_radius = rand_float(0.28f, 0.62f);
  const float crown_center_height = trunk_height + 0.55f * crown_radius;
  writer.add_sphere(
      add3(position, Vec3{0.0f, crown_center_height, 0.0f}),
      crown_radius,
      Vec3{0.16f, 0.48f, 0.18f},
      kTreeClassId,
      instance_id);
  writer.add_cylinder(
      add3(position, Vec3{0.0f, 0.5f * trunk_height, 0.0f}),
      trunk_radius,
      0.5f * trunk_height,
      axes,
      Vec3{0.42f, 0.25f, 0.12f},
      kTreeClassId,
      instance_id);
}

template <typename RandFloat>
void add_random_cloud(RandomPrimitiveWriter& writer, Vec3 position, Mat3 rotation, int32_t instance_id, RandFloat&& rand_float) {
  const Vec3 color{0.93f, 0.95f, 0.96f};
  const float scale = rand_float(0.75f, 1.35f);
  writer.add_sphere(add3(position, rotate3(rotation, Vec3{-0.32f * scale, 0.0f, 0.0f})), 0.34f * scale, color, kCloudClassId, instance_id);
  writer.add_sphere(add3(position, rotate3(rotation, Vec3{0.08f * scale, 0.10f * scale, 0.02f * scale})), 0.42f * scale, color, kCloudClassId, instance_id);
  writer.add_sphere(add3(position, rotate3(rotation, Vec3{0.46f * scale, -0.02f * scale, 0.03f * scale})), 0.31f * scale, color, kCloudClassId, instance_id);
}

template <typename RandFloat>
void add_random_car(RandomPrimitiveWriter& writer, Vec3 position, Mat3 rotation, int32_t instance_id, RandFloat&& rand_float) {
  const float length = rand_float(0.85f, 1.35f);
  const float width = rand_float(0.42f, 0.62f);
  const float height = rand_float(0.26f, 0.42f);
  const float wheel_radius = 0.13f * length;
  const float wheel_half_width = 0.08f * width;
  const Vec3 body_color{rand_float(0.18f, 0.85f), rand_float(0.12f, 0.55f), rand_float(0.12f, 0.45f)};
  const Vec3 wheel_color{0.04f, 0.04f, 0.04f};
  writer.add_box(
      add3(position, rotate3(rotation, Vec3{0.0f, wheel_radius + 0.5f * height, 0.0f})),
      Vec3{0.5f * length, 0.5f * height, 0.5f * width},
      rotation,
      body_color,
      kCarClassId,
      instance_id);
  const Mat3 wheel_axes = multiply3(rotation, Mat3{{Vec3{0.0f, 1.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, 1.0f}}});
  for (const float x : {-0.32f * length, 0.32f * length}) {
    for (const float z : {-0.58f * width, 0.58f * width}) {
      writer.add_cylinder(
          add3(position, rotate3(rotation, Vec3{x, wheel_radius, z})),
          wheel_radius,
          wheel_half_width,
          wheel_axes,
          wheel_color,
          kCarClassId,
          instance_id);
    }
  }
}

template <typename RandFloat>
void add_random_person(RandomPrimitiveWriter& writer, Vec3 position, Mat3 rotation, int32_t instance_id, RandFloat&& rand_float) {
  const float height = rand_float(0.95f, 1.45f);
  const float leg_height = 0.34f * height;
  const float body_height = 0.34f * height;
  const float arm_height = 0.31f * height;
  const float head_radius = 0.105f * height;
  const float body_width = 0.15f * height;
  const Vec3 body_color{0.18f, 0.34f, 0.82f};
  const Vec3 limb_color{0.10f, 0.12f, 0.18f};
  const Vec3 skin_color{0.78f, 0.56f, 0.40f};
  writer.add_box(add3(position, rotate3(rotation, Vec3{0.0f, leg_height + 0.5f * body_height, 0.0f})), Vec3{body_width, 0.5f * body_height, 0.055f * height}, rotation, body_color, kPersonClassId, instance_id);
  writer.add_box(add3(position, rotate3(rotation, Vec3{-0.055f * height, 0.5f * leg_height, 0.0f})), Vec3{0.04f * height, 0.5f * leg_height, 0.04f * height}, rotation, limb_color, kPersonClassId, instance_id);
  writer.add_box(add3(position, rotate3(rotation, Vec3{0.055f * height, 0.5f * leg_height, 0.0f})), Vec3{0.04f * height, 0.5f * leg_height, 0.04f * height}, rotation, limb_color, kPersonClassId, instance_id);
  writer.add_box(add3(position, rotate3(rotation, Vec3{-0.18f * height, leg_height + 0.47f * body_height, 0.0f})), Vec3{0.035f * height, 0.5f * arm_height, 0.035f * height}, rotation, limb_color, kPersonClassId, instance_id);
  writer.add_box(add3(position, rotate3(rotation, Vec3{0.18f * height, leg_height + 0.47f * body_height, 0.0f})), Vec3{0.035f * height, 0.5f * arm_height, 0.035f * height}, rotation, limb_color, kPersonClassId, instance_id);
  writer.add_sphere(add3(position, rotate3(rotation, Vec3{0.0f, leg_height + body_height + head_radius * 1.08f, 0.0f})), head_radius, skin_color, kPersonClassId, instance_id);
}

}  // namespace synthetic_scene
