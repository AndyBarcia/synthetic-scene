from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

import torch

from . import _cuda_renderer


Vec3 = Union[Sequence[float], torch.Tensor]
Vec3List = Union[Sequence[Vec3], torch.Tensor]


@dataclass(frozen=True)
class Camera:
    origin: Vec3 = (0.0, 0.0, 0.0)
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


def _vec3(value: Vec3, *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() != 3:
        raise ValueError("expected a 3D vector")
    return tensor.reshape(3).contiguous()


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


def render_scene(
    width: int = 512,
    height: int = 512,
    *,
    scene: Scene | None = None,
    camera: Camera | None = None,
    options: RenderOptions | None = None,
) -> torch.Tensor:
    """Render Lambert-shaded spheres, oriented boxes, and infinite planes into an H x W x 3 CUDA tensor."""
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

    image = torch.empty((height, width, 3), dtype=torch.float32, device=device)
    _cuda_renderer.render_scene(
        image,
        {
            "origin": _vec3(camera_data.origin, device=device),
            "fov_degrees": float(camera_data.fov_degrees),
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
