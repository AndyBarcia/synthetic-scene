from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union, overload

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
class RenderResult:
    image: torch.Tensor
    instance_map: torch.Tensor
    semantic_map: torch.Tensor


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
        return RenderResult(image=image, instance_map=instance_map, semantic_map=semantic_map)
    return image


def save_image(image: torch.Tensor, path: str | Path) -> None:
    """Save an H x W x 3 float image tensor as an 8-bit PNG."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for save_image; install pillow") from exc

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
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for save_label_map_visualization; install pillow") from exc

    colors = colorize_label_map(label_map, seed=seed).numpy()
    Image.fromarray(colors, mode="RGB").save(path)
