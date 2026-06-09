from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Sequence, Tuple, Union, overload
from PIL import Image

import torch

from . import _cuda_renderer


Vec3 = Union[Sequence[float], torch.Tensor]
Vec3List = Union[Sequence[Vec3], torch.Tensor]


@dataclass(frozen=True)
class Camera:
    origin: Vec3 = (0.0, 0.0, 0.0)
    orientation: Union[Sequence[Vec3], torch.Tensor] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    fov_degrees: float = 45.0


@dataclass(frozen=True)
class RenderOptions:
    light_dir: Vec3 = (-0.6, 0.7, 0.5)
    background: Vec3 = (0.02, 0.03, 0.04)


@dataclass(frozen=True)
class Spheres:
    centers: Vec3List = ((0.0, 0.0, -3.0),)
    radii: Sequence[float] | torch.Tensor = (1.0,)
    colors: Vec3List = ((0.9, 0.35, 0.18),)


@dataclass(frozen=True)
class Planes:
    points: Vec3List = ((0.0, -1.0, 0.0), (0.0, 0.0, -6.0))
    normals: Vec3List = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    colors: Vec3List = ((0.52, 0.55, 0.58), (0.12, 0.14, 0.18))


@dataclass(frozen=True)
class OrientedBoxes:
    centers: Vec3List = ()
    half_sizes: Vec3List = ()
    axes: Union[Sequence[Sequence[Vec3]], torch.Tensor] = ()
    colors: Vec3List = ()


@dataclass(frozen=True)
class Scene:
    spheres: Spheres = field(default_factory=Spheres)
    planes: Planes = field(default_factory=Planes)
    boxes: OrientedBoxes = field(default_factory=OrientedBoxes)


@dataclass(frozen=True)
class RandomScene:
    scene: Scene
    cameras: tuple[Camera, ...]


@dataclass(frozen=True)
class RenderResult:
    image: torch.Tensor
    visible_count: torch.Tensor
    visible_classes: torch.Tensor
    instance_map: torch.Tensor
    semantic_map: torch.Tensor


def _compact_visible_instances(
    instance_map: torch.Tensor,
    *,
    sphere_count: int,
    plane_count: int,
    box_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-image visible classes and maps with compact 1-based IDs."""
    batch_size = instance_map.shape[0]
    max_gt = sphere_count + plane_count + box_count
    visible_count = torch.empty((batch_size,), dtype=torch.int32, device=instance_map.device)
    visible_classes = torch.zeros((batch_size, max_gt), dtype=torch.int32, device=instance_map.device)
    compact_map = torch.empty_like(instance_map)

    class_lookup = torch.zeros((max_gt + 1,), dtype=torch.int32, device=instance_map.device)
    if sphere_count:
        class_lookup[1 : sphere_count + 1] = 1
    if plane_count:
        plane_start = sphere_count + 1
        class_lookup[plane_start : plane_start + plane_count] = 2
    if box_count:
        box_start = sphere_count + plane_count + 1
        class_lookup[box_start : box_start + box_count] = 3

    for batch_idx in range(batch_size):
        labels = torch.unique(instance_map[batch_idx])
        labels = labels[labels > 0]
        count = int(labels.numel())
        visible_count[batch_idx] = count
        if count == 0:
            compact_map[batch_idx].zero_()
            continue

        remap = torch.zeros((max_gt + 1,), dtype=torch.int32, device=instance_map.device)
        remap[labels.to(torch.long)] = torch.arange(1, count + 1, dtype=torch.int32, device=instance_map.device)
        compact_map[batch_idx] = remap[instance_map[batch_idx].to(torch.long)]
        visible_classes[batch_idx, :count] = class_lookup[labels.to(torch.long)]

    return visible_count, visible_classes, compact_map


def _vec3(value: Vec3, *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() != 3:
        raise ValueError("expected a 3D vector")
    return tensor.reshape(3).contiguous()


def _camera_origins(cameras: Camera | Sequence[Camera], *, device: torch.device | str = "cuda") -> torch.Tensor:
    if isinstance(cameras, Camera):
        return _vec3(cameras.origin, device=device).reshape(1, 3)
    if len(cameras) == 0:
        raise ValueError("at least one camera is required")
    origins = [_vec3(camera.origin, device=device) for camera in cameras]
    return torch.stack(origins, dim=0).contiguous()


def _camera_orientations(cameras: Camera | Sequence[Camera], *, device: torch.device | str = "cuda") -> torch.Tensor:
    if isinstance(cameras, Camera):
        return _mat3(cameras.orientation, device=device).reshape(1, 3, 3)
    if len(cameras) == 0:
        raise ValueError("at least one camera is required")
    orientations = [_mat3(camera.orientation, device=device) for camera in cameras]
    return torch.stack(orientations, dim=0).contiguous()


def _camera_fov_degrees(cameras: Camera | Sequence[Camera]) -> float:
    if isinstance(cameras, Camera):
        return float(cameras.fov_degrees)
    if len(cameras) == 0:
        raise ValueError("at least one camera is required")
    fov_degrees = float(cameras[0].fov_degrees)
    if any(float(camera.fov_degrees) != fov_degrees for camera in cameras):
        raise ValueError("batched cameras must share the same fov_degrees")
    return fov_degrees


def _mat3(value: Union[Sequence[Vec3], torch.Tensor], *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() != 9:
        raise ValueError("expected a 3 x 3 matrix")
    tensor = tensor.reshape(3, 3)
    if bool((tensor.norm(dim=1) <= 1.0e-8).any().item()):
        raise ValueError("camera orientation rows must be non-zero")
    return tensor.contiguous()


def _vec3_list(value: Vec3List, *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0, 3).contiguous()
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError("expected vectors with shape N x 3")
    return tensor.contiguous()


def _mat3_list(value: Union[Sequence[Sequence[Vec3]], torch.Tensor], *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0, 3, 3).contiguous()
    if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
        raise ValueError("expected matrices with shape N x 3 x 3")
    return tensor.contiguous()


def _as_random_range(name: str, value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    low = float(value[0])
    high = float(value[1])
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return low, high


def _rand_float(generator: torch.Generator, low: float, high: float) -> float:
    if low == high:
        return low
    return float(torch.empty((), dtype=torch.float32).uniform_(low, high, generator=generator).item())


def _rand_int(generator: torch.Generator, low: int, high: int) -> int:
    if low == high:
        return low
    return int(torch.randint(low, high + 1, (), generator=generator).item())


def _normalize3(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if length <= 1.0e-8:
        raise ValueError("expected a non-zero vector")
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


Vec3Tuple = Tuple[float, float, float]
Mat3Tuple = Tuple[Vec3Tuple, Vec3Tuple, Vec3Tuple]


def _look_at_orientation(origin: Vec3Tuple, target: Vec3Tuple) -> Mat3Tuple:
    forward = _normalize3((target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]))
    back = (-forward[0], -forward[1], -forward[2])
    world_up = (0.0, 1.0, 0.0)
    right = _cross3(world_up, back)
    if right[0] * right[0] + right[1] * right[1] + right[2] * right[2] <= 1.0e-8:
        right = (1.0, 0.0, 0.0)
    else:
        right = _normalize3(right)
    up = _normalize3(_cross3(back, right))
    return (right, up, back)


def _random_color(generator: torch.Generator) -> tuple[float, float, float]:
    hue = _rand_float(generator, 0.0, 1.0)
    saturation = _rand_float(generator, 0.55, 0.95)
    value = _rand_float(generator, 0.55, 1.0)
    sector = int(hue * 6.0)
    frac = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - frac * saturation)
    t = value * (1.0 - (1.0 - frac) * saturation)
    sector %= 6
    if sector == 0:
        return (value, t, p)
    if sector == 1:
        return (q, value, p)
    if sector == 2:
        return (p, value, t)
    if sector == 3:
        return (p, q, value)
    if sector == 4:
        return (t, p, value)
    return (value, p, q)


def _random_ground_xz(generator: torch.Generator, scatter_radius: float) -> tuple[float, float]:
    radius = scatter_radius * math.sqrt(_rand_float(generator, 0.0, 1.0))
    angle = _rand_float(generator, 0.0, math.tau)
    return radius * math.cos(angle), radius * math.sin(angle)


def _yaw_axes(yaw: float) -> Mat3Tuple:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        (cos_yaw, 0.0, -sin_yaw),
        (0.0, 1.0, 0.0),
        (sin_yaw, 0.0, cos_yaw),
    )


def random_scene(
    seed: int,
    *,
    ground_objects: int = 10,
    floating_objects: int = 5,
    cameras: int = 4,
    scatter_radius: float = 3.0,
    ground_y: float = -1.0,
    camera_distance: tuple[float, float] = (2.4, 5.0),
    camera_height: tuple[float, float] = (0.35, 2.4),
    fov_degrees: float = 50.0,
) -> RandomScene:
    """Generate a deterministic random scene and cameras from a seed."""
    if ground_objects < 0 or floating_objects < 0:
        raise ValueError("object counts must be non-negative")
    if ground_objects + floating_objects <= 0:
        raise ValueError("at least one non-plane object is required")
    if cameras <= 0:
        raise ValueError("cameras must be positive")
    if scatter_radius <= 0.0:
        raise ValueError("scatter_radius must be positive")
    if fov_degrees <= 0.0 or fov_degrees >= 180.0:
        raise ValueError("fov_degrees must be in the open interval (0, 180)")
    camera_distance_min, camera_distance_max = _as_random_range("camera_distance", camera_distance)
    camera_height_min, camera_height_max = _as_random_range("camera_height", camera_height)
    if camera_distance_min <= 0.0:
        raise ValueError("camera_distance values must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    sphere_centers: list[tuple[float, float, float]] = []
    sphere_radii: list[float] = []
    sphere_colors: list[tuple[float, float, float]] = []
    box_centers: list[tuple[float, float, float]] = []
    box_half_sizes: list[tuple[float, float, float]] = []
    box_axes: list[Mat3Tuple] = []
    box_colors: list[tuple[float, float, float]] = []
    targets: list[tuple[float, float, float]] = []

    for _ in range(ground_objects):
        x, z = _random_ground_xz(generator, scatter_radius)
        if _rand_float(generator, 0.0, 1.0) < 0.55:
            radius = _rand_float(generator, 0.18, 0.65)
            center = (x, ground_y + radius, z)
            sphere_centers.append(center)
            sphere_radii.append(radius)
            sphere_colors.append(_random_color(generator))
            targets.append(center)
        else:
            half_size = (
                _rand_float(generator, 0.18, 0.6),
                _rand_float(generator, 0.18, 0.75),
                _rand_float(generator, 0.18, 0.6),
            )
            center = (x, ground_y + half_size[1], z)
            box_centers.append(center)
            box_half_sizes.append(half_size)
            box_axes.append(_yaw_axes(_rand_float(generator, 0.0, math.tau)))
            box_colors.append(_random_color(generator))
            targets.append(center)

    for _ in range(floating_objects):
        x, z = _random_ground_xz(generator, scatter_radius * 0.9)
        if _rand_float(generator, 0.0, 1.0) < 0.65:
            radius = _rand_float(generator, 0.15, 0.5)
            center = (x, _rand_float(generator, ground_y + 1.0, ground_y + 3.2), z)
            sphere_centers.append(center)
            sphere_radii.append(radius)
            sphere_colors.append(_random_color(generator))
            targets.append(center)
        else:
            half_size = (
                _rand_float(generator, 0.14, 0.45),
                _rand_float(generator, 0.14, 0.45),
                _rand_float(generator, 0.14, 0.45),
            )
            center = (x, _rand_float(generator, ground_y + 1.1, ground_y + 3.4), z)
            box_centers.append(center)
            box_half_sizes.append(half_size)
            box_axes.append(_yaw_axes(_rand_float(generator, 0.0, math.tau)))
            box_colors.append(_random_color(generator))
            targets.append(center)

    if len(sphere_centers) > 64 or len(box_centers) > 64:
        raise ValueError("random_scene generated more primitives than the renderer supports")

    generated_cameras = []
    for _ in range(cameras):
        target = targets[_rand_int(generator, 0, len(targets) - 1)]
        distance = _rand_float(generator, camera_distance_min, camera_distance_max)
        angle = _rand_float(generator, 0.0, math.tau)
        height = _rand_float(generator, camera_height_min, camera_height_max)
        origin = (
            target[0] + distance * math.cos(angle),
            target[1] + height,
            target[2] + distance * math.sin(angle),
        )
        generated_cameras.append(
            Camera(
                origin=origin,
                orientation=_look_at_orientation(origin, target),
                fov_degrees=fov_degrees,
            )
        )

    return RandomScene(
        scene=Scene(
            spheres=Spheres(centers=sphere_centers, radii=sphere_radii, colors=sphere_colors),
            planes=Planes(
                points=[(0.0, ground_y, 0.0)],
                normals=[(0.0, 1.0, 0.0)],
                colors=[(0.45, 0.48, 0.43)],
            ),
            boxes=OrientedBoxes(centers=box_centers, half_sizes=box_half_sizes, axes=box_axes, colors=box_colors),
        ),
        cameras=tuple(generated_cameras),
    )


@overload
def render_scene(
    width: int = 512,
    height: int = 512,
    *,
    scene: Scene | None = None,
    camera: Camera | Sequence[Camera] | None = None,
    options: RenderOptions | None = None,
    return_maps: bool = False,
) -> torch.Tensor: ...


@overload
def render_scene(
    width: int = 512,
    height: int = 512,
    *,
    scene: Scene | None = None,
    camera: Camera | Sequence[Camera] | None = None,
    options: RenderOptions | None = None,
    return_maps: bool,
) -> torch.Tensor | RenderResult: ...


def render_scene(
    width: int = 512,
    height: int = 512,
    *,
    scene: Scene | None = None,
    camera: Camera | Sequence[Camera] | None = None,
    options: RenderOptions | None = None,
    return_maps: bool = False,
) -> torch.Tensor | RenderResult:
    """Render a scene into batched RGB, and optionally instance and primitive-class label maps."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to render with this extension")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    camera_data = camera or Camera()
    options_data = options or RenderOptions()
    scene_data = scene or Scene()

    device = torch.device("cuda")
    centers = _vec3_list(scene_data.spheres.centers, device=device)
    radii = torch.as_tensor(scene_data.spheres.radii, dtype=torch.float32, device=device).reshape(-1).contiguous()
    colors = _vec3_list(scene_data.spheres.colors, device=device)
    points = _vec3_list(scene_data.planes.points, device=device)
    normals = _vec3_list(scene_data.planes.normals, device=device)
    plane_colors_tensor = _vec3_list(scene_data.planes.colors, device=device)
    box_centers = _vec3_list(scene_data.boxes.centers, device=device)
    box_half_sizes = _vec3_list(scene_data.boxes.half_sizes, device=device)
    box_axes = _mat3_list(scene_data.boxes.axes, device=device)
    box_colors = _vec3_list(scene_data.boxes.colors, device=device)
    sphere_count = centers.shape[0]
    plane_count = points.shape[0]
    box_count = box_centers.shape[0]
    if sphere_count == 0 and plane_count == 0 and box_count == 0:
        raise ValueError("at least one object is required")
    if radii.shape[0] != sphere_count or colors.shape[0] != sphere_count:
        raise ValueError("sphere_centers, sphere_radii, and sphere_colors must have matching lengths")
    if normals.shape[0] != plane_count or plane_colors_tensor.shape[0] != plane_count:
        raise ValueError("plane_points, plane_normals, and plane_colors must have matching lengths")
    if box_half_sizes.shape[0] != box_count or box_axes.shape[0] != box_count or box_colors.shape[0] != box_count:
        raise ValueError("box_centers, box_half_sizes, box_axes, and box_colors must have matching lengths")
    if bool((radii <= 0).any().item()):
        raise ValueError("sphere_radii must all be positive")
    if bool((normals.norm(dim=1) <= 1.0e-8).any().item()):
        raise ValueError("plane_normals must be non-zero")
    if bool((box_half_sizes <= 0).any().item()):
        raise ValueError("box_half_sizes must all be positive")
    if bool((box_axes.norm(dim=2) <= 1.0e-8).any().item()):
        raise ValueError("box_axes must contain non-zero axis vectors")

    camera_origins = _camera_origins(camera_data, device=device)
    camera_orientations = _camera_orientations(camera_data, device=device)
    batch_size = camera_origins.shape[0]

    image = torch.empty((batch_size, 3, height, width), dtype=torch.float32, device=device)
    instance_map = torch.empty((batch_size, height, width), dtype=torch.int32, device=device) if return_maps else torch.empty((0,), dtype=torch.int32, device=device)
    semantic_map = torch.empty((batch_size, height, width), dtype=torch.int32, device=device) if return_maps else torch.empty((0,), dtype=torch.int32, device=device)
    _cuda_renderer.render_scene(
        image,
        instance_map,
        semantic_map,
        {
            "origin": camera_origins,
            "orientation": camera_orientations,
            "fov_degrees": _camera_fov_degrees(camera_data),
        },
        {
            "spheres": {
                "centers": centers,
                "radii": radii,
                "colors": colors,
            },
            "planes": {
                "points": points,
                "normals": normals,
                "colors": plane_colors_tensor,
            },
            "boxes": {
                "centers": box_centers,
                "half_sizes": box_half_sizes,
                "axes": box_axes,
                "colors": box_colors,
            },
        },
        {
            "light_dir": _vec3(options_data.light_dir, device=device),
            "background": _vec3(options_data.background, device=device),
        },
    )
    if return_maps:
        visible_count, visible_classes, instance_map = _compact_visible_instances(
            instance_map,
            sphere_count=sphere_count,
            plane_count=plane_count,
            box_count=box_count,
        )
        return RenderResult(
            image=image,
            visible_count=visible_count,
            visible_classes=visible_classes,
            instance_map=instance_map,
            semantic_map=semantic_map,
        )
    return image


def save_image(image: torch.Tensor, path: str | Path) -> None:
    """Save an H x W x 3 float image tensor as an 8-bit PNG."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("expected image with shape H x W x 3")
    image_u8 = (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    Image.fromarray(image_u8, mode="RGB").save(path)


def colorize_label_map(label_map: torch.Tensor, *, seed: int = 17) -> torch.Tensor:
    """Map integer labels to deterministic RGB colors for visualization."""
    if label_map.ndim != 2:
        raise ValueError("expected label map with shape H x W")
    if label_map.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("expected an integer label map")

    labels = label_map.detach().to(torch.int64).cpu()
    unique_labels = torch.unique(labels)
    colors = torch.zeros((int(unique_labels.max().item()) + 1, 3), dtype=torch.uint8)
    for label in unique_labels.tolist():
        if label == 0:
            continue
        generator = torch.Generator()
        generator.manual_seed(seed + int(label) * 1009)
        colors[int(label)] = torch.randint(48, 256, (3,), generator=generator, dtype=torch.uint8)
    return colors[labels]


def save_label_map_visualization(label_map: torch.Tensor, path: str | Path, *, seed: int = 17) -> None:
    """Save an H x W integer label map as a colorized 8-bit RGB PNG."""
    colors = colorize_label_map(label_map, seed=seed).numpy()
    Image.fromarray(colors, mode="RGB").save(path)
