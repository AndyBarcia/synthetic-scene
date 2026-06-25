#pragma once

#include <cstdint>
#include <vector>

namespace synthetic_scene {

constexpr int kHouseClassId = 10;
constexpr int kTreeClassId = 11;

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

}  // namespace synthetic_scene
