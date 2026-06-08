from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import torch

from . import _cuda_renderer


Vec3 = Union[Sequence[float], torch.Tensor]
Vec3List = Union[Sequence[Vec3], torch.Tensor]


def _vec3(value: Vec3, *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() != 3:
        raise ValueError("expected a 3D vector")
    return tensor.reshape(3).contiguous()


def _vec3_list(value: Vec3List, *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError("expected vectors with shape N x 3")
    return tensor.contiguous()


def render_spheres(
    width: int = 512,
    height: int = 512,
    *,
    camera_origin: Vec3 = (0.0, 0.0, 0.0),
    sphere_centers: Vec3List = ((0.0, 0.0, -3.0),),
    sphere_radii: Sequence[float] | torch.Tensor = (1.0,),
    light_dir: Vec3 = (-0.6, 0.7, 0.5),
    fov_degrees: float = 45.0,
    background: Vec3 = (0.02, 0.03, 0.04),
    sphere_colors: Vec3List = ((0.9, 0.35, 0.18),),
) -> torch.Tensor:
    """Render Lambert-shaded spheres into an H x W x 3 CUDA tensor."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to render with this extension")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    device = torch.device("cuda")
    centers = _vec3_list(sphere_centers, device=device)
    radii = torch.as_tensor(sphere_radii, dtype=torch.float32, device=device).reshape(-1).contiguous()
    colors = _vec3_list(sphere_colors, device=device)
    sphere_count = centers.shape[0]
    if sphere_count == 0:
        raise ValueError("at least one sphere is required")
    if radii.shape[0] != sphere_count or colors.shape[0] != sphere_count:
        raise ValueError("sphere_centers, sphere_radii, and sphere_colors must have matching lengths")
    if bool((radii <= 0).any().item()):
        raise ValueError("sphere_radii must all be positive")

    image = torch.empty((height, width, 3), dtype=torch.float32, device=device)
    _cuda_renderer.render_spheres(
        image,
        _vec3(camera_origin, device=device),
        centers,
        radii,
        _vec3(light_dir, device=device),
        float(fov_degrees),
        _vec3(background, device=device),
        colors,
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
