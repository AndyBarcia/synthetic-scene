from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import torch

from .renderer import CompositeObject, Cylinders, OrientedBoxes, Prisms, Scene, Spheres, Terrain, Vec3


HOUSE_CLASS_ID = 10
TREE_CLASS_ID = 11
CLOUD_CLASS_ID = 12
CAR_CLASS_ID = 13
PERSON_CLASS_ID = 14

Axes = Union[Sequence[Sequence[Vec3]], torch.Tensor]


IDENTITY_AXES: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def empty_scene() -> Scene:
    """Return a scene with no implicit default primitives."""
    return Scene(
        spheres=Spheres(centers=(), radii=(), colors=()),
        terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(), phase_zs=(), dz=(), dz_growth=(), colors=()),
    )


@dataclass(frozen=True)
class HouseConfig:
    width: float = 1.2
    depth: float = 1.0
    body_height: float = 0.8
    roof_height: float = 0.45
    roof_overhang: float = 0.12
    body_color: Vec3 = (0.62, 0.43, 0.30)
    roof_color: Vec3 = (0.72, 0.14, 0.10)


@dataclass(frozen=True)
class TreeConfig:
    trunk_height: float = 0.9
    trunk_radius: float = 0.12
    crown_radius: float = 0.42
    crown_center_height: float | None = None
    trunk_color: Vec3 = (0.42, 0.25, 0.12)
    crown_color: Vec3 = (0.16, 0.48, 0.18)


@dataclass(frozen=True)
class CloudConfig:
    scale: float = 1.0
    color: Vec3 = (0.93, 0.95, 0.96)


@dataclass(frozen=True)
class CarConfig:
    length: float = 1.2
    width: float = 0.55
    body_height: float = 0.34
    wheel_radius: float = 0.15
    wheel_half_width: float = 0.045
    body_color: Vec3 = (0.75, 0.12, 0.10)
    wheel_color: Vec3 = (0.04, 0.04, 0.04)


@dataclass(frozen=True)
class PersonConfig:
    height: float = 1.2
    body_color: Vec3 = (0.18, 0.34, 0.82)
    limb_color: Vec3 = (0.10, 0.12, 0.18)
    head_color: Vec3 = (0.78, 0.56, 0.40)


def composite_object(
    scene: Scene,
    *,
    class_id: int,
    instance_id: int,
    position: Vec3 = (0.0, 0.0, 0.0),
    rotation: Axes = IDENTITY_AXES,
) -> CompositeObject:
    return CompositeObject(
        scene=scene,
        class_id=class_id,
        instance_id=instance_id,
        position=position,
        rotation=rotation,
    )


def house_scene(config: HouseConfig = HouseConfig()) -> Scene:
    """Build a local-origin house from a box body and triangular prism roof."""
    body_center_y = 0.5 * config.body_height
    roof_center_y = config.body_height + 0.5 * config.roof_height
    return Scene(
        spheres=Spheres(centers=(), radii=(), colors=()),
        terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(), phase_zs=(), dz=(), dz_growth=(), colors=()),
        boxes=OrientedBoxes(
            centers=[(0.0, body_center_y, 0.0)],
            half_sizes=[(0.5 * config.width, 0.5 * config.body_height, 0.5 * config.depth)],
            axes=[IDENTITY_AXES],
            colors=[config.body_color],
        ),
        prisms=Prisms(
            centers=[(0.0, roof_center_y, 0.0)],
            half_sizes=[
                (
                    0.5 * config.width + config.roof_overhang,
                    0.5 * config.roof_height,
                    0.5 * config.depth + config.roof_overhang,
                )
            ],
            axes=[IDENTITY_AXES],
            colors=[config.roof_color],
        ),
    )


def tree_scene(config: TreeConfig = TreeConfig()) -> Scene:
    """Build a local-origin tree from a cylinder trunk and sphere crown."""
    crown_center_height = config.crown_center_height
    if crown_center_height is None:
        crown_center_height = config.trunk_height + 0.55 * config.crown_radius
    return Scene(
        spheres=Spheres(
            centers=[(0.0, crown_center_height, 0.0)],
            radii=[config.crown_radius],
            colors=[config.crown_color],
        ),
        terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(), phase_zs=(), dz=(), dz_growth=(), colors=()),
        cylinders=Cylinders(
            centers=[(0.0, 0.5 * config.trunk_height, 0.0)],
            radii=[config.trunk_radius],
            half_heights=[0.5 * config.trunk_height],
            axes=[IDENTITY_AXES],
            colors=[config.trunk_color],
        ),
    )


def cloud_scene(config: CloudConfig = CloudConfig()) -> Scene:
    """Build a local-origin cloud from a small cluster of white spheres."""
    scale = config.scale
    return Scene(
        spheres=Spheres(
            centers=[
                (-0.32 * scale, 0.0, 0.0),
                (0.08 * scale, 0.10 * scale, 0.02 * scale),
                (0.46 * scale, -0.02 * scale, 0.03 * scale),
            ],
            radii=[0.34 * scale, 0.42 * scale, 0.31 * scale],
            colors=[config.color, config.color, config.color],
        ),
        terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(), phase_zs=(), dz=(), dz_growth=(), colors=()),
    )


def car_scene(config: CarConfig = CarConfig()) -> Scene:
    """Build a local-origin car from a body box and four cylindrical wheels."""
    wheel_axes = (
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    wheel_centers = [
        (-0.32 * config.length, config.wheel_radius, -0.58 * config.width),
        (-0.32 * config.length, config.wheel_radius, 0.58 * config.width),
        (0.32 * config.length, config.wheel_radius, -0.58 * config.width),
        (0.32 * config.length, config.wheel_radius, 0.58 * config.width),
    ]
    return Scene(
        spheres=Spheres(centers=(), radii=(), colors=()),
        terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(), phase_zs=(), dz=(), dz_growth=(), colors=()),
        boxes=OrientedBoxes(
            centers=[(0.0, config.wheel_radius + 0.5 * config.body_height, 0.0)],
            half_sizes=[(0.5 * config.length, 0.5 * config.body_height, 0.5 * config.width)],
            axes=[IDENTITY_AXES],
            colors=[config.body_color],
        ),
        cylinders=Cylinders(
            centers=wheel_centers,
            radii=[config.wheel_radius] * 4,
            half_heights=[config.wheel_half_width] * 4,
            axes=[wheel_axes] * 4,
            colors=[config.wheel_color] * 4,
        ),
    )


def person_scene(config: PersonConfig = PersonConfig()) -> Scene:
    """Build a local-origin person from boxes for body/limbs and a spherical head."""
    height = config.height
    leg_height = 0.34 * height
    body_height = 0.34 * height
    arm_height = 0.31 * height
    head_radius = 0.105 * height
    body_width = 0.15 * height
    return Scene(
        spheres=Spheres(
            centers=[(0.0, leg_height + body_height + head_radius * 1.08, 0.0)],
            radii=[head_radius],
            colors=[config.head_color],
        ),
        terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(), phase_zs=(), dz=(), dz_growth=(), colors=()),
        boxes=OrientedBoxes(
            centers=[
                (0.0, leg_height + 0.5 * body_height, 0.0),
                (-0.055 * height, 0.5 * leg_height, 0.0),
                (0.055 * height, 0.5 * leg_height, 0.0),
                (-0.18 * height, leg_height + 0.47 * body_height, 0.0),
                (0.18 * height, leg_height + 0.47 * body_height, 0.0),
            ],
            half_sizes=[
                (body_width, 0.5 * body_height, 0.055 * height),
                (0.04 * height, 0.5 * leg_height, 0.04 * height),
                (0.04 * height, 0.5 * leg_height, 0.04 * height),
                (0.035 * height, 0.5 * arm_height, 0.035 * height),
                (0.035 * height, 0.5 * arm_height, 0.035 * height),
            ],
            axes=[IDENTITY_AXES] * 5,
            colors=[config.body_color, config.limb_color, config.limb_color, config.limb_color, config.limb_color],
        ),
    )


def make_house(
    *,
    instance_id: int,
    class_id: int = HOUSE_CLASS_ID,
    config: HouseConfig = HouseConfig(),
    position: Vec3 = (0.0, 0.0, 0.0),
    rotation: Axes = IDENTITY_AXES,
) -> CompositeObject:
    return composite_object(
        house_scene(config),
        class_id=class_id,
        instance_id=instance_id,
        position=position,
        rotation=rotation,
    )


def make_tree(
    *,
    instance_id: int,
    class_id: int = TREE_CLASS_ID,
    config: TreeConfig = TreeConfig(),
    position: Vec3 = (0.0, 0.0, 0.0),
    rotation: Axes = IDENTITY_AXES,
) -> CompositeObject:
    return composite_object(
        tree_scene(config),
        class_id=class_id,
        instance_id=instance_id,
        position=position,
        rotation=rotation,
    )


def make_cloud(
    *,
    instance_id: int,
    class_id: int = CLOUD_CLASS_ID,
    config: CloudConfig = CloudConfig(),
    position: Vec3 = (0.0, 0.0, 0.0),
    rotation: Axes = IDENTITY_AXES,
) -> CompositeObject:
    return composite_object(
        cloud_scene(config),
        class_id=class_id,
        instance_id=instance_id,
        position=position,
        rotation=rotation,
    )


def make_car(
    *,
    instance_id: int,
    class_id: int = CAR_CLASS_ID,
    config: CarConfig = CarConfig(),
    position: Vec3 = (0.0, 0.0, 0.0),
    rotation: Axes = IDENTITY_AXES,
) -> CompositeObject:
    return composite_object(
        car_scene(config),
        class_id=class_id,
        instance_id=instance_id,
        position=position,
        rotation=rotation,
    )


def make_person(
    *,
    instance_id: int,
    class_id: int = PERSON_CLASS_ID,
    config: PersonConfig = PersonConfig(),
    position: Vec3 = (0.0, 0.0, 0.0),
    rotation: Axes = IDENTITY_AXES,
) -> CompositeObject:
    return composite_object(
        person_scene(config),
        class_id=class_id,
        instance_id=instance_id,
        position=position,
        rotation=rotation,
    )
