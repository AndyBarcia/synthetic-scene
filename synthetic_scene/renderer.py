from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union, overload
from PIL import Image

import torch

from . import _cuda_renderer


Vec3 = Union[Sequence[float], torch.Tensor]
Vec3List = Union[Sequence[Vec3], torch.Tensor]


@dataclass(frozen=True)
class RenderOptions:
    light_dir: Vec3 = (-0.6, 0.7, 0.5)
    background: Vec3 = (0.02, 0.03, 0.04)
    fov_degrees: float = 45.0
    ambient: float = 0.2
    shadows: bool = True
    shadow_strength: float = 1.0


@dataclass(frozen=True)
class Spheres:
    centers: Vec3List = ((0.0, 0.0, -3.0),)
    radii: Sequence[float] | torch.Tensor = (1.0,)
    colors: Vec3List = ((0.9, 0.35, 0.18),)
    counts: Sequence[int] | torch.Tensor | None = None


@dataclass(frozen=True)
class Terrain:
    base_heights: Sequence[float] | torch.Tensor = (-1.0,)
    depth_limits: Sequence[float] | torch.Tensor = (7.0,)
    phase_xs: Sequence[float] | torch.Tensor = (0.0,)
    phase_zs: Sequence[float] | torch.Tensor = (0.0,)
    dz: Sequence[float] | torch.Tensor = (0.05,)
    dz_growth: Sequence[float] | torch.Tensor = (0.0001,)
    colors: Vec3List = ((0.36, 0.46, 0.30),)
    counts: Sequence[int] | torch.Tensor | None = None


@dataclass(frozen=True)
class OrientedBoxes:
    centers: Vec3List = ()
    half_sizes: Vec3List = ()
    axes: Union[Sequence[Sequence[Vec3]], torch.Tensor] = ()
    colors: Vec3List = ()
    counts: Sequence[int] | torch.Tensor | None = None


@dataclass(frozen=True)
class Prisms:
    centers: Vec3List = ()
    half_sizes: Vec3List = ()
    axes: Union[Sequence[Sequence[Vec3]], torch.Tensor] = ()
    colors: Vec3List = ()
    counts: Sequence[int] | torch.Tensor | None = None


@dataclass(frozen=True)
class Cylinders:
    centers: Vec3List = ()
    radii: Sequence[float] | torch.Tensor = ()
    half_heights: Sequence[float] | torch.Tensor = ()
    axes: Union[Sequence[Sequence[Vec3]], torch.Tensor] = ()
    colors: Vec3List = ()
    counts: Sequence[int] | torch.Tensor | None = None


@dataclass(frozen=True)
class Scene:
    spheres: Spheres = field(default_factory=Spheres)
    terrain: Terrain = field(default_factory=Terrain)
    boxes: OrientedBoxes = field(default_factory=OrientedBoxes)
    prisms: Prisms = field(default_factory=Prisms)
    cylinders: Cylinders = field(default_factory=Cylinders)


@dataclass(frozen=True)
class RandomScene:
    scene: Scene


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
    sphere_counts: torch.Tensor,
    terrain_counts: torch.Tensor,
    box_counts: torch.Tensor,
    prism_counts: torch.Tensor,
    cylinder_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-image visible classes and maps with compact 1-based IDs."""
    batch_size = instance_map.shape[0]
    max_gt = int((sphere_counts + terrain_counts + box_counts + prism_counts + cylinder_counts).max().item())
    visible_count = torch.empty((batch_size,), dtype=torch.int32, device=instance_map.device)
    visible_classes = torch.zeros((batch_size, max_gt), dtype=torch.int32, device=instance_map.device)
    compact_map = torch.empty_like(instance_map)

    for batch_idx in range(batch_size):
        sphere_count = int(sphere_counts[batch_idx].item())
        terrain_count = int(terrain_counts[batch_idx].item())
        box_count = int(box_counts[batch_idx].item())
        prism_count = int(prism_counts[batch_idx].item())
        cylinder_count = int(cylinder_counts[batch_idx].item())
        class_lookup = torch.zeros((max_gt + 1,), dtype=torch.int32, device=instance_map.device)
        if sphere_count:
            class_lookup[1 : sphere_count + 1] = 1
        if terrain_count:
            terrain_start = sphere_count + 1
            class_lookup[terrain_start : terrain_start + terrain_count] = 2
        if box_count:
            box_start = sphere_count + terrain_count + 1
            class_lookup[box_start : box_start + box_count] = 3
        if prism_count:
            prism_start = sphere_count + terrain_count + box_count + 1
            class_lookup[prism_start : prism_start + prism_count] = 5
        if cylinder_count:
            cylinder_start = sphere_count + terrain_count + box_count + prism_count + 1
            class_lookup[cylinder_start : cylinder_start + cylinder_count] = 4

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


def _vec3_list(value: Vec3List, *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0, 3).contiguous()
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError("expected vectors with shape N x 3")
    return tensor.contiguous()


def _vec3_batch(value: Vec3List, *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(1, 0, 3).contiguous()
    if tensor.ndim == 2 and tensor.shape[1] == 3:
        return tensor.reshape(1, tensor.shape[0], 3).contiguous()
    if tensor.ndim == 3 and tensor.shape[2] == 3:
        return tensor.contiguous()
    raise ValueError("expected vectors with shape N x 3 or B x N x 3")


def _scalar_batch(value: Sequence[float] | torch.Tensor, *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(1, 0).contiguous()
    if tensor.ndim == 1:
        return tensor.reshape(1, tensor.shape[0]).contiguous()
    if tensor.ndim == 2:
        return tensor.contiguous()
    raise ValueError("expected scalars with shape N or B x N")


def _mat3_batch(value: Union[Sequence[Sequence[Vec3]], torch.Tensor], *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(1, 0, 3, 3).contiguous()
    if tensor.ndim == 3 and tensor.shape[1:] == (3, 3):
        return tensor.reshape(1, tensor.shape[0], 3, 3).contiguous()
    if tensor.ndim == 4 and tensor.shape[2:] == (3, 3):
        return tensor.contiguous()
    raise ValueError("expected matrices with shape N x 3 x 3 or B x N x 3 x 3")


def _counts(value: Sequence[int] | torch.Tensor | None, *, batch_size: int, count: int, device: torch.device | str = "cuda") -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), count, dtype=torch.int32, device=device)
    tensor = torch.as_tensor(value, dtype=torch.int32, device=device).reshape(-1).contiguous()
    if tensor.shape[0] != batch_size:
        raise ValueError("primitive counts must have shape B")
    if bool(((tensor < 0) | (tensor > count)).any().item()):
        raise ValueError("primitive counts must be in range [0, N]")
    return tensor


def _broadcast_batch(*tensors: torch.Tensor) -> int:
    batch_size = max(tensor.shape[0] for tensor in tensors)
    if any(tensor.shape[0] not in (1, batch_size) for tensor in tensors):
        raise ValueError("batched scene tensors must use the same B dimension")
    return batch_size


def _expand_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor.contiguous()
    return tensor.expand((batch_size, *tensor.shape[1:])).contiguous()


def _mat3_list(value: Union[Sequence[Sequence[Vec3]], torch.Tensor], *, device: torch.device | str = "cuda") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(0, 3, 3).contiguous()
    if tensor.ndim != 3 or tensor.shape[1:] != (3, 3):
        raise ValueError("expected matrices with shape N x 3 x 3")
    return tensor.contiguous()


def random_scene(
    seed: int,
    *,
    ground_objects: int = 60,
    floating_objects: int = 4,
    batch_size: int = 4,
    scatter_radius: float = 50.0,
    ground_y: float = -1.0,
    depth_limit: float = 50.0,
    terrain_dz: float = 0.005,
    terrain_dz_growth: float = 0.0001,
    fov_degrees: float = 50.0,
    aspect_ratio: float = 1.5,
) -> RandomScene:
    """Generate deterministic random camera-space scenes from a seed."""
    native = _cuda_renderer.random_scene(
        int(seed),
        int(ground_objects),
        int(floating_objects),
        int(batch_size),
        float(scatter_radius),
        float(ground_y),
        float(depth_limit),
        float(terrain_dz),
        float(terrain_dz_growth),
        float(fov_degrees),
        float(aspect_ratio),
    )

    return RandomScene(
        scene=Scene(
            spheres=Spheres(
                centers=native["sphere_centers"],
                radii=native["sphere_radii"],
                colors=native["sphere_colors"],
                counts=native["sphere_counts"],
            ),
            terrain=Terrain(
                base_heights=native["terrain_base_heights"],
                depth_limits=native["terrain_depth_limits"],
                phase_xs=native["terrain_phase_xs"],
                phase_zs=native["terrain_phase_zs"],
                dz=native["terrain_dz"],
                dz_growth=native["terrain_dz_growth"],
                colors=native["terrain_colors"],
                counts=native["terrain_counts"],
            ),
            boxes=OrientedBoxes(
                centers=native["box_centers"],
                half_sizes=native["box_half_sizes"],
                axes=native["box_axes"],
                colors=native["box_colors"],
                counts=native["box_counts"],
            ),
            prisms=Prisms(
                centers=native["prism_centers"],
                half_sizes=native["prism_half_sizes"],
                axes=native["prism_axes"],
                colors=native["prism_colors"],
                counts=native["prism_counts"],
            ),
            cylinders=Cylinders(
                centers=native["cylinder_centers"],
                radii=native["cylinder_radii"],
                half_heights=native["cylinder_half_heights"],
                axes=native["cylinder_axes"],
                colors=native["cylinder_colors"],
                counts=native["cylinder_counts"],
            ),
        ),
    )


@overload
def render_scene(
    width: int = 512,
    height: int = 512,
    *,
    scene: Scene | None = None,
    options: RenderOptions | None = None,
    return_maps: bool = False,
) -> torch.Tensor: ...


@overload
def render_scene(
    width: int = 512,
    height: int = 512,
    *,
    scene: Scene | None = None,
    options: RenderOptions | None = None,
    return_maps: bool,
) -> torch.Tensor | RenderResult: ...


def render_scene(
    width: int = 512,
    height: int = 512,
    *,
    scene: Scene | None = None,
    options: RenderOptions | None = None,
    return_maps: bool = False,
) -> torch.Tensor | RenderResult:
    """Render camera-space scenes into batched RGB, and optionally label maps."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to render with this extension")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    options_data = options or RenderOptions()
    scene_data = scene or Scene()
    if options_data.fov_degrees <= 0.0 or options_data.fov_degrees >= 180.0:
        raise ValueError("fov_degrees must be in the open interval (0, 180)")
    if options_data.ambient < 0.0 or options_data.ambient > 1.0:
        raise ValueError("ambient must be in the range [0, 1]")
    if options_data.shadow_strength < 0.0 or options_data.shadow_strength > 1.0:
        raise ValueError("shadow_strength must be in the range [0, 1]")

    device = torch.device("cuda")
    centers = _vec3_batch(scene_data.spheres.centers, device=device)
    radii = _scalar_batch(scene_data.spheres.radii, device=device)
    colors = _vec3_batch(scene_data.spheres.colors, device=device)
    terrain_base_heights = _scalar_batch(scene_data.terrain.base_heights, device=device)
    terrain_depth_limits = _scalar_batch(scene_data.terrain.depth_limits, device=device)
    terrain_phase_xs = _scalar_batch(scene_data.terrain.phase_xs, device=device)
    terrain_phase_zs = _scalar_batch(scene_data.terrain.phase_zs, device=device)
    terrain_dz = _scalar_batch(scene_data.terrain.dz, device=device)
    terrain_dz_growth = _scalar_batch(scene_data.terrain.dz_growth, device=device)
    terrain_colors = _vec3_batch(scene_data.terrain.colors, device=device)
    box_centers = _vec3_batch(scene_data.boxes.centers, device=device)
    box_half_sizes = _vec3_batch(scene_data.boxes.half_sizes, device=device)
    box_axes = _mat3_batch(scene_data.boxes.axes, device=device)
    box_colors = _vec3_batch(scene_data.boxes.colors, device=device)
    prism_centers = _vec3_batch(scene_data.prisms.centers, device=device)
    prism_half_sizes = _vec3_batch(scene_data.prisms.half_sizes, device=device)
    prism_axes = _mat3_batch(scene_data.prisms.axes, device=device)
    prism_colors = _vec3_batch(scene_data.prisms.colors, device=device)
    cylinder_centers = _vec3_batch(scene_data.cylinders.centers, device=device)
    cylinder_radii = _scalar_batch(scene_data.cylinders.radii, device=device)
    cylinder_half_heights = _scalar_batch(scene_data.cylinders.half_heights, device=device)
    cylinder_axes = _mat3_batch(scene_data.cylinders.axes, device=device)
    cylinder_colors = _vec3_batch(scene_data.cylinders.colors, device=device)
    batch_size = _broadcast_batch(
        centers,
        radii,
        colors,
        terrain_base_heights,
        terrain_depth_limits,
        terrain_phase_xs,
        terrain_phase_zs,
        terrain_dz,
        terrain_dz_growth,
        terrain_colors,
        box_centers,
        box_half_sizes,
        box_axes,
        box_colors,
        prism_centers,
        prism_half_sizes,
        prism_axes,
        prism_colors,
        cylinder_centers,
        cylinder_radii,
        cylinder_half_heights,
        cylinder_axes,
        cylinder_colors,
    )
    centers = _expand_batch(centers, batch_size)
    radii = _expand_batch(radii, batch_size)
    colors = _expand_batch(colors, batch_size)
    terrain_base_heights = _expand_batch(terrain_base_heights, batch_size)
    terrain_depth_limits = _expand_batch(terrain_depth_limits, batch_size)
    terrain_phase_xs = _expand_batch(terrain_phase_xs, batch_size)
    terrain_phase_zs = _expand_batch(terrain_phase_zs, batch_size)
    terrain_dz = _expand_batch(terrain_dz, batch_size)
    terrain_dz_growth = _expand_batch(terrain_dz_growth, batch_size)
    terrain_colors = _expand_batch(terrain_colors, batch_size)
    box_centers = _expand_batch(box_centers, batch_size)
    box_half_sizes = _expand_batch(box_half_sizes, batch_size)
    box_axes = _expand_batch(box_axes, batch_size)
    box_colors = _expand_batch(box_colors, batch_size)
    prism_centers = _expand_batch(prism_centers, batch_size)
    prism_half_sizes = _expand_batch(prism_half_sizes, batch_size)
    prism_axes = _expand_batch(prism_axes, batch_size)
    prism_colors = _expand_batch(prism_colors, batch_size)
    cylinder_centers = _expand_batch(cylinder_centers, batch_size)
    cylinder_radii = _expand_batch(cylinder_radii, batch_size)
    cylinder_half_heights = _expand_batch(cylinder_half_heights, batch_size)
    cylinder_axes = _expand_batch(cylinder_axes, batch_size)
    cylinder_colors = _expand_batch(cylinder_colors, batch_size)
    sphere_count = centers.shape[1]
    terrain_count = 1 if terrain_depth_limits.shape[1] > 0 else 0
    box_count = box_centers.shape[1]
    prism_count = prism_centers.shape[1]
    cylinder_count = cylinder_centers.shape[1]
    sphere_counts = _counts(scene_data.spheres.counts, batch_size=batch_size, count=sphere_count, device=device)
    points = torch.empty((batch_size, 0, 3), dtype=torch.float32, device=device)
    normals = torch.empty((batch_size, 0, 3), dtype=torch.float32, device=device)
    plane_colors_tensor = torch.empty((batch_size, 0, 3), dtype=torch.float32, device=device)
    plane_counts = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    terrain_counts = _counts(scene_data.terrain.counts, batch_size=batch_size, count=terrain_count, device=device)
    box_counts = _counts(scene_data.boxes.counts, batch_size=batch_size, count=box_count, device=device)
    prism_counts = _counts(scene_data.prisms.counts, batch_size=batch_size, count=prism_count, device=device)
    cylinder_counts = _counts(scene_data.cylinders.counts, batch_size=batch_size, count=cylinder_count, device=device)
    if bool(((sphere_counts + plane_counts + terrain_counts + box_counts + prism_counts + cylinder_counts) <= 0).any().item()):
        raise ValueError("at least one object is required")
    if radii.shape[1] != sphere_count or colors.shape[1] != sphere_count:
        raise ValueError("sphere_centers, sphere_radii, and sphere_colors must have matching lengths")
    if (
        terrain_base_heights.shape[1] != 1
        or terrain_depth_limits.shape[1] != 1
        or terrain_phase_xs.shape[1] != 1
        or terrain_phase_zs.shape[1] != 1
        or terrain_dz.shape[1] != 1
        or terrain_dz_growth.shape[1] != 1
        or terrain_colors.shape[1] != 1
    ):
        raise ValueError("terrain base_heights, depth_limits, phase_xs, phase_zs, dz, dz_growth, and colors must each contain one entry")
    if box_half_sizes.shape[1] != box_count or box_axes.shape[1] != box_count or box_colors.shape[1] != box_count:
        raise ValueError("box_centers, box_half_sizes, box_axes, and box_colors must have matching lengths")
    if prism_half_sizes.shape[1] != prism_count or prism_axes.shape[1] != prism_count or prism_colors.shape[1] != prism_count:
        raise ValueError("prism_centers, prism_half_sizes, prism_axes, and prism_colors must have matching lengths")
    if (
        cylinder_radii.shape[1] != cylinder_count
        or cylinder_half_heights.shape[1] != cylinder_count
        or cylinder_axes.shape[1] != cylinder_count
        or cylinder_colors.shape[1] != cylinder_count
    ):
        raise ValueError("cylinder_centers, cylinder_radii, cylinder_half_heights, cylinder_axes, and cylinder_colors must have matching lengths")
    if bool((radii <= 0).any().item()):
        raise ValueError("sphere_radii must all be positive")
    if bool((terrain_depth_limits <= 0).any().item()):
        raise ValueError("terrain depth_limits must be positive")
    if bool((terrain_dz <= 0).any().item()):
        raise ValueError("terrain dz must be positive")
    if bool((terrain_dz_growth < 0).any().item()):
        raise ValueError("terrain dz_growth must be non-negative")
    if bool((box_half_sizes <= 0).any().item()):
        raise ValueError("box_half_sizes must all be positive")
    if bool((box_axes.norm(dim=3) <= 1.0e-8).any().item()):
        raise ValueError("box_axes must contain non-zero axis vectors")
    if bool((prism_half_sizes <= 0).any().item()):
        raise ValueError("prism_half_sizes must all be positive")
    if bool((prism_axes.norm(dim=3) <= 1.0e-8).any().item()):
        raise ValueError("prism_axes must contain non-zero axis vectors")
    if bool((cylinder_radii <= 0).any().item()):
        raise ValueError("cylinder_radii must all be positive")
    if bool((cylinder_half_heights <= 0).any().item()):
        raise ValueError("cylinder_half_heights must all be positive")
    if bool((cylinder_axes.norm(dim=3) <= 1.0e-8).any().item()):
        raise ValueError("cylinder_axes must contain non-zero axis vectors")

    image = torch.empty((batch_size, 3, height, width), dtype=torch.float32, device=device)
    instance_map = torch.empty((batch_size, height, width), dtype=torch.int32, device=device) if return_maps else torch.empty((0,), dtype=torch.int32, device=device)
    semantic_map = torch.empty((batch_size, height, width), dtype=torch.int32, device=device) if return_maps else torch.empty((0,), dtype=torch.int32, device=device)
    _cuda_renderer.render_scene(
        image,
        instance_map,
        semantic_map,
        {
            "spheres": {
                "centers": centers,
                "radii": radii,
                "colors": colors,
                "counts": sphere_counts,
            },
            "planes": {
                "points": points,
                "normals": normals,
                "colors": plane_colors_tensor,
                "counts": plane_counts,
            },
            "terrain": {
                "base_heights": terrain_base_heights,
                "depth_limits": terrain_depth_limits,
                "phase_xs": terrain_phase_xs,
                "phase_zs": terrain_phase_zs,
                "dz": terrain_dz,
                "dz_growth": terrain_dz_growth,
                "colors": terrain_colors,
                "counts": terrain_counts,
            },
            "boxes": {
                "centers": box_centers,
                "half_sizes": box_half_sizes,
                "axes": box_axes,
                "colors": box_colors,
                "counts": box_counts,
            },
            "prisms": {
                "centers": prism_centers,
                "half_sizes": prism_half_sizes,
                "axes": prism_axes,
                "colors": prism_colors,
                "counts": prism_counts,
            },
            "cylinders": {
                "centers": cylinder_centers,
                "radii": cylinder_radii,
                "half_heights": cylinder_half_heights,
                "axes": cylinder_axes,
                "colors": cylinder_colors,
                "counts": cylinder_counts,
            },
        },
        {
            "light_dir": _vec3(options_data.light_dir, device=device),
            "background": _vec3(options_data.background, device=device),
            "fov_degrees": float(options_data.fov_degrees),
            "ambient": float(options_data.ambient),
            "shadows": bool(options_data.shadows),
            "shadow_strength": float(options_data.shadow_strength),
        },
    )
    if return_maps:
        visible_count, visible_classes, instance_map = _compact_visible_instances(
            instance_map,
            sphere_counts=sphere_counts,
            terrain_counts=terrain_counts,
            box_counts=box_counts,
            prism_counts=prism_counts,
            cylinder_counts=cylinder_counts,
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
