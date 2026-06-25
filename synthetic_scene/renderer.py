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
    class_ids: Sequence[int] | torch.Tensor | None = None
    instance_ids: Sequence[int] | torch.Tensor | None = None


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
    class_id: int = 2
    instance_id: int | None = None


@dataclass(frozen=True)
class OrientedBoxes:
    centers: Vec3List = ()
    half_sizes: Vec3List = ()
    axes: Union[Sequence[Sequence[Vec3]], torch.Tensor] = ()
    colors: Vec3List = ()
    counts: Sequence[int] | torch.Tensor | None = None
    class_ids: Sequence[int] | torch.Tensor | None = None
    instance_ids: Sequence[int] | torch.Tensor | None = None


@dataclass(frozen=True)
class Prisms:
    centers: Vec3List = ()
    half_sizes: Vec3List = ()
    axes: Union[Sequence[Sequence[Vec3]], torch.Tensor] = ()
    colors: Vec3List = ()
    counts: Sequence[int] | torch.Tensor | None = None
    class_ids: Sequence[int] | torch.Tensor | None = None
    instance_ids: Sequence[int] | torch.Tensor | None = None


@dataclass(frozen=True)
class Cylinders:
    centers: Vec3List = ()
    radii: Sequence[float] | torch.Tensor = ()
    half_heights: Sequence[float] | torch.Tensor = ()
    axes: Union[Sequence[Sequence[Vec3]], torch.Tensor] = ()
    colors: Vec3List = ()
    counts: Sequence[int] | torch.Tensor | None = None
    class_ids: Sequence[int] | torch.Tensor | None = None
    instance_ids: Sequence[int] | torch.Tensor | None = None


@dataclass(frozen=True)
class Scene:
    spheres: Spheres = field(default_factory=Spheres)
    terrain: Terrain = field(default_factory=Terrain)
    boxes: OrientedBoxes = field(default_factory=OrientedBoxes)
    prisms: Prisms = field(default_factory=Prisms)
    cylinders: Cylinders = field(default_factory=Cylinders)


@dataclass(frozen=True)
class CompositeObject:
    scene: Scene
    class_id: int
    instance_id: int
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Union[Sequence[Vec3], torch.Tensor] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class RandomScene:
    scene: Scene


@dataclass(frozen=True)
class RenderResult:
    image: torch.Tensor
    visible_count: torch.Tensor
    visible_classes: torch.Tensor
    visible_instance_ids: torch.Tensor
    instance_map: torch.Tensor
    semantic_map: torch.Tensor


def _compact_visible_instances(
    instance_map: torch.Tensor,
    semantic_map: torch.Tensor,
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
        labels = torch.unique(instance_map[batch_idx])
        labels = labels[labels > 0]
        count = int(labels.numel())
        visible_count[batch_idx] = count
        if count == 0:
            compact_map[batch_idx].zero_()
            continue

        remap_size = int(labels.max().item()) + 1
        remap = torch.zeros((remap_size,), dtype=torch.int32, device=instance_map.device)
        remap[labels.to(torch.long)] = torch.arange(1, count + 1, dtype=torch.int32, device=instance_map.device)
        compact_map[batch_idx] = remap[instance_map[batch_idx].clamp(max=remap_size - 1).to(torch.long)]
        for label_idx, label in enumerate(labels):
            semantic_values = torch.unique(semantic_map[batch_idx][instance_map[batch_idx] == label])
            semantic_values = semantic_values[semantic_values > 0]
            if semantic_values.numel() > 0:
                visible_classes[batch_idx, label_idx] = semantic_values[0].to(torch.int32)

    return visible_count, visible_classes, compact_map


def _visible_custom_instances(
    instance_map: torch.Tensor,
    semantic_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return visible custom instance IDs and their classes without remapping the map."""
    batch_size = instance_map.shape[0]
    per_batch_labels = []
    max_visible = 0
    for batch_idx in range(batch_size):
        labels = torch.unique(instance_map[batch_idx])
        labels = labels[labels > 0]
        per_batch_labels.append(labels)
        max_visible = max(max_visible, int(labels.numel()))

    visible_count = torch.empty((batch_size,), dtype=torch.int32, device=instance_map.device)
    visible_classes = torch.zeros((batch_size, max_visible), dtype=torch.int32, device=instance_map.device)
    visible_instance_ids = torch.zeros((batch_size, max_visible), dtype=torch.int32, device=instance_map.device)
    for batch_idx, labels in enumerate(per_batch_labels):
        count = int(labels.numel())
        visible_count[batch_idx] = count
        if count == 0:
            continue
        visible_instance_ids[batch_idx, :count] = labels.to(torch.int32)
        for label_idx, label in enumerate(labels):
            semantic_values = torch.unique(semantic_map[batch_idx][instance_map[batch_idx] == label])
            semantic_values = semantic_values[semantic_values > 0]
            if semantic_values.numel() > 0:
                visible_classes[batch_idx, label_idx] = semantic_values[0].to(torch.int32)
    return visible_count, visible_classes, visible_instance_ids


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


def _metadata(
    value: Sequence[int] | torch.Tensor | None,
    *,
    batch_size: int,
    count: int,
    default_start: int,
    default_value: int | None,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    if value is None:
        if default_value is not None:
            tensor = torch.full((batch_size, count), default_value, dtype=torch.int32, device=device)
        else:
            tensor = torch.arange(default_start, default_start + count, dtype=torch.int32, device=device).reshape(1, count)
            tensor = tensor.expand(batch_size, count)
        return tensor.contiguous()
    tensor = torch.as_tensor(value, dtype=torch.int32, device=device)
    if tensor.numel() == 0:
        return tensor.reshape(1, 0).expand(batch_size, 0).contiguous()
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, tensor.shape[0])
    if tensor.ndim != 2:
        raise ValueError("primitive metadata must have shape N or B x N")
    if tensor.shape[1] != count:
        raise ValueError("primitive metadata length must match primitive slots")
    if tensor.shape[0] not in (1, batch_size):
        raise ValueError("primitive metadata batch size must be 1 or B")
    if tensor.shape[0] == 1:
        tensor = tensor.expand(batch_size, count)
    if bool((tensor < 0).any().item()):
        raise ValueError("primitive metadata IDs must be non-negative")
    return tensor.contiguous()


def _terrain_metadata(
    class_id: int,
    instance_id: int | None,
    *,
    batch_size: int,
    count: int,
    default_instance_id: int,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(class_id) < 0:
        raise ValueError("terrain class_id must be non-negative")
    if instance_id is not None and int(instance_id) < 0:
        raise ValueError("terrain instance_id must be non-negative")
    class_ids = torch.full((batch_size, count), int(class_id), dtype=torch.int32, device=device)
    resolved_instance_id = default_instance_id if instance_id is None else int(instance_id)
    instance_ids = torch.full((batch_size, count), resolved_instance_id, dtype=torch.int32, device=device)
    return class_ids.contiguous(), instance_ids.contiguous()


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


def _concat_sequence_field(values: list[object]) -> object:
    kept = [value for value in values if torch.as_tensor(value).numel() > 0]
    if not kept:
        return ()
    if any(isinstance(value, torch.Tensor) for value in kept):
        return torch.cat([torch.as_tensor(value) for value in kept], dim=0)
    result: list[object] = []
    for value in kept:
        result.extend(list(value))  # type: ignore[arg-type]
    return tuple(result)


def _concat_scalar_field(values: list[object]) -> object:
    kept = [value for value in values if torch.as_tensor(value).numel() > 0]
    if not kept:
        return ()
    if any(isinstance(value, torch.Tensor) for value in kept):
        return torch.cat([torch.as_tensor(value).reshape(-1) for value in kept], dim=0)
    result: list[object] = []
    for value in kept:
        result.extend(list(value))  # type: ignore[arg-type]
    return tuple(result)


def _override_metadata(value: object, count: int, metadata_id: int) -> object:
    if count == 0:
        return ()
    if isinstance(value, torch.Tensor):
        return torch.full((count,), metadata_id, dtype=torch.int32, device=value.device)
    return tuple([metadata_id] * count)


def _slot_count(value: object) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        return 0
    if tensor.ndim == 0:
        return 1
    return int(tensor.shape[0])


def _optional_field(value: object | None) -> object:
    return () if value is None else value


def _empty_scene() -> Scene:
    return Scene(
        spheres=Spheres(centers=(), radii=(), colors=()),
        terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(), phase_zs=(), dz=(), dz_growth=(), colors=()),
    )


def _rotation_matrix(value: Union[Sequence[Vec3], torch.Tensor]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (3, 3):
        raise ValueError("composite rotation must have shape 3 x 3")
    return tensor


def _transform_centers(value: Vec3List, rotation: torch.Tensor, position: torch.Tensor) -> object:
    centers = torch.as_tensor(value, dtype=torch.float32)
    if centers.numel() == 0:
        return ()
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("composite primitive centers must have shape N x 3")
    return centers.matmul(rotation.T).add(position)


def _transform_axes(value: Union[Sequence[Sequence[Vec3]], torch.Tensor], rotation: torch.Tensor) -> object:
    axes = torch.as_tensor(value, dtype=torch.float32)
    if axes.numel() == 0:
        return ()
    if axes.ndim != 3 or axes.shape[1:] != (3, 3):
        raise ValueError("composite primitive axes must have shape N x 3 x 3")
    return axes.matmul(rotation.T)


def _has_terrain(scene: Scene) -> bool:
    count = _slot_count(scene.terrain.depth_limits)
    if scene.terrain.counts is None:
        return count > 0
    counts = torch.as_tensor(scene.terrain.counts)
    return bool((counts > 0).any().item())


def _with_default_metadata(scene: Scene) -> Scene:
    sphere_count = _slot_count(scene.spheres.radii)
    terrain_count = _slot_count(scene.terrain.depth_limits)
    box_count = _slot_count(scene.boxes.centers)
    prism_count = _slot_count(scene.prisms.centers)
    cylinder_count = _slot_count(scene.cylinders.centers)
    terrain_start = sphere_count + 1
    box_start = sphere_count + terrain_count + 1
    prism_start = sphere_count + terrain_count + box_count + 1
    cylinder_start = sphere_count + terrain_count + box_count + prism_count + 1
    return Scene(
        spheres=Spheres(
            centers=scene.spheres.centers,
            radii=scene.spheres.radii,
            colors=scene.spheres.colors,
            counts=scene.spheres.counts,
            class_ids=scene.spheres.class_ids if scene.spheres.class_ids is not None else _override_metadata(scene.spheres.radii, sphere_count, 1),
            instance_ids=scene.spheres.instance_ids if scene.spheres.instance_ids is not None else tuple(range(1, sphere_count + 1)),
        ),
        terrain=Terrain(
            base_heights=scene.terrain.base_heights,
            depth_limits=scene.terrain.depth_limits,
            phase_xs=scene.terrain.phase_xs,
            phase_zs=scene.terrain.phase_zs,
            dz=scene.terrain.dz,
            dz_growth=scene.terrain.dz_growth,
            colors=scene.terrain.colors,
            counts=scene.terrain.counts,
            class_id=scene.terrain.class_id,
            instance_id=scene.terrain.instance_id if scene.terrain.instance_id is not None else (terrain_start if terrain_count else None),
        ),
        boxes=OrientedBoxes(
            centers=scene.boxes.centers,
            half_sizes=scene.boxes.half_sizes,
            axes=scene.boxes.axes,
            colors=scene.boxes.colors,
            counts=scene.boxes.counts,
            class_ids=scene.boxes.class_ids if scene.boxes.class_ids is not None else _override_metadata(scene.boxes.centers, box_count, 3),
            instance_ids=scene.boxes.instance_ids if scene.boxes.instance_ids is not None else tuple(range(box_start, box_start + box_count)),
        ),
        prisms=Prisms(
            centers=scene.prisms.centers,
            half_sizes=scene.prisms.half_sizes,
            axes=scene.prisms.axes,
            colors=scene.prisms.colors,
            counts=scene.prisms.counts,
            class_ids=scene.prisms.class_ids if scene.prisms.class_ids is not None else _override_metadata(scene.prisms.centers, prism_count, 5),
            instance_ids=scene.prisms.instance_ids if scene.prisms.instance_ids is not None else tuple(range(prism_start, prism_start + prism_count)),
        ),
        cylinders=Cylinders(
            centers=scene.cylinders.centers,
            radii=scene.cylinders.radii,
            half_heights=scene.cylinders.half_heights,
            axes=scene.cylinders.axes,
            colors=scene.cylinders.colors,
            counts=scene.cylinders.counts,
            class_ids=scene.cylinders.class_ids if scene.cylinders.class_ids is not None else _override_metadata(scene.cylinders.centers, cylinder_count, 4),
            instance_ids=scene.cylinders.instance_ids if scene.cylinders.instance_ids is not None else tuple(range(cylinder_start, cylinder_start + cylinder_count)),
        ),
    )


def flatten_composite_objects(
    composites: Sequence[CompositeObject],
    *,
    base_scene: Scene | None = None,
) -> Scene:
    """Expand composite objects into raw primitives with shared class and instance IDs."""
    scenes = [_with_default_metadata(base_scene) if base_scene is not None else _empty_scene()]
    for composite in composites:
        scene = composite.scene
        rotation = _rotation_matrix(composite.rotation)
        position = torch.as_tensor(composite.position, dtype=torch.float32)
        if position.numel() != 3:
            raise ValueError("composite position must be a 3D vector")
        position = position.reshape(3)
        if _has_terrain(scene):
            raise ValueError("composite objects cannot contain terrain; pass terrain through base_scene instead")
        sphere_count = _slot_count(scene.spheres.radii)
        box_count = _slot_count(scene.boxes.centers)
        prism_count = _slot_count(scene.prisms.centers)
        cylinder_count = _slot_count(scene.cylinders.centers)
        scenes.append(
            Scene(
                spheres=Spheres(
                    centers=_transform_centers(scene.spheres.centers, rotation, position),
                    radii=scene.spheres.radii,
                    colors=scene.spheres.colors,
                    class_ids=_override_metadata(scene.spheres.radii, sphere_count, composite.class_id),
                    instance_ids=_override_metadata(scene.spheres.radii, sphere_count, composite.instance_id),
                ),
                terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(), phase_zs=(), dz=(), dz_growth=(), colors=()),
                boxes=OrientedBoxes(
                    centers=_transform_centers(scene.boxes.centers, rotation, position),
                    half_sizes=scene.boxes.half_sizes,
                    axes=_transform_axes(scene.boxes.axes, rotation),
                    colors=scene.boxes.colors,
                    class_ids=_override_metadata(scene.boxes.centers, box_count, composite.class_id),
                    instance_ids=_override_metadata(scene.boxes.centers, box_count, composite.instance_id),
                ),
                prisms=Prisms(
                    centers=_transform_centers(scene.prisms.centers, rotation, position),
                    half_sizes=scene.prisms.half_sizes,
                    axes=_transform_axes(scene.prisms.axes, rotation),
                    colors=scene.prisms.colors,
                    class_ids=_override_metadata(scene.prisms.centers, prism_count, composite.class_id),
                    instance_ids=_override_metadata(scene.prisms.centers, prism_count, composite.instance_id),
                ),
                cylinders=Cylinders(
                    centers=_transform_centers(scene.cylinders.centers, rotation, position),
                    radii=scene.cylinders.radii,
                    half_heights=scene.cylinders.half_heights,
                    axes=_transform_axes(scene.cylinders.axes, rotation),
                    colors=scene.cylinders.colors,
                    class_ids=_override_metadata(scene.cylinders.centers, cylinder_count, composite.class_id),
                    instance_ids=_override_metadata(scene.cylinders.centers, cylinder_count, composite.instance_id),
                ),
            )
        )

    return Scene(
        spheres=Spheres(
            centers=_concat_sequence_field([scene.spheres.centers for scene in scenes]),
            radii=_concat_scalar_field([scene.spheres.radii for scene in scenes]),
            colors=_concat_sequence_field([scene.spheres.colors for scene in scenes]),
            class_ids=_concat_scalar_field([_optional_field(scene.spheres.class_ids) for scene in scenes]),
            instance_ids=_concat_scalar_field([_optional_field(scene.spheres.instance_ids) for scene in scenes]),
        ),
        terrain=Terrain(
            base_heights=_concat_scalar_field([scene.terrain.base_heights for scene in scenes]),
            depth_limits=_concat_scalar_field([scene.terrain.depth_limits for scene in scenes]),
            phase_xs=_concat_scalar_field([scene.terrain.phase_xs for scene in scenes]),
            phase_zs=_concat_scalar_field([scene.terrain.phase_zs for scene in scenes]),
            dz=_concat_scalar_field([scene.terrain.dz for scene in scenes]),
            dz_growth=_concat_scalar_field([scene.terrain.dz_growth for scene in scenes]),
            colors=_concat_sequence_field([scene.terrain.colors for scene in scenes]),
            class_id=scenes[0].terrain.class_id,
            instance_id=scenes[0].terrain.instance_id,
        ),
        boxes=OrientedBoxes(
            centers=_concat_sequence_field([scene.boxes.centers for scene in scenes]),
            half_sizes=_concat_sequence_field([scene.boxes.half_sizes for scene in scenes]),
            axes=_concat_sequence_field([scene.boxes.axes for scene in scenes]),
            colors=_concat_sequence_field([scene.boxes.colors for scene in scenes]),
            class_ids=_concat_scalar_field([_optional_field(scene.boxes.class_ids) for scene in scenes]),
            instance_ids=_concat_scalar_field([_optional_field(scene.boxes.instance_ids) for scene in scenes]),
        ),
        prisms=Prisms(
            centers=_concat_sequence_field([scene.prisms.centers for scene in scenes]),
            half_sizes=_concat_sequence_field([scene.prisms.half_sizes for scene in scenes]),
            axes=_concat_sequence_field([scene.prisms.axes for scene in scenes]),
            colors=_concat_sequence_field([scene.prisms.colors for scene in scenes]),
            class_ids=_concat_scalar_field([_optional_field(scene.prisms.class_ids) for scene in scenes]),
            instance_ids=_concat_scalar_field([_optional_field(scene.prisms.instance_ids) for scene in scenes]),
        ),
        cylinders=Cylinders(
            centers=_concat_sequence_field([scene.cylinders.centers for scene in scenes]),
            radii=_concat_scalar_field([scene.cylinders.radii for scene in scenes]),
            half_heights=_concat_scalar_field([scene.cylinders.half_heights for scene in scenes]),
            axes=_concat_sequence_field([scene.cylinders.axes for scene in scenes]),
            colors=_concat_sequence_field([scene.cylinders.colors for scene in scenes]),
            class_ids=_concat_scalar_field([_optional_field(scene.cylinders.class_ids) for scene in scenes]),
            instance_ids=_concat_scalar_field([_optional_field(scene.cylinders.instance_ids) for scene in scenes]),
        ),
    )


def random_scene(
    seed: int,
    *,
    house_count: int = 10,
    tree_count: int = 50,
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
    if house_count < 0 or tree_count < 0:
        raise ValueError("house_count and tree_count must be non-negative")
    if house_count + tree_count <= 0:
        raise ValueError("at least one composite object is required")
    try:
        native = _cuda_renderer.random_scene(
            int(seed),
            int(batch_size),
            float(scatter_radius),
            float(ground_y),
            float(depth_limit),
            float(terrain_dz),
            float(terrain_dz_growth),
            float(fov_degrees),
            float(aspect_ratio),
            int(house_count),
            int(tree_count),
        )
    except TypeError as error:
        if "ground_objects" not in str(error) or "floating_objects" not in str(error):
            raise
        native = _cuda_renderer.random_scene(
            int(seed),
            0,
            0,
            int(batch_size),
            float(scatter_radius),
            float(ground_y),
            float(depth_limit),
            float(terrain_dz),
            float(terrain_dz_growth),
            float(fov_degrees),
            float(aspect_ratio),
            int(house_count),
            int(tree_count),
        )

    scene = Scene(
        spheres=Spheres(
            centers=native["sphere_centers"],
            radii=native["sphere_radii"],
            colors=native["sphere_colors"],
            counts=native["sphere_counts"],
            class_ids=native["sphere_class_ids"],
            instance_ids=native["sphere_instance_ids"],
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
            class_ids=native["box_class_ids"],
            instance_ids=native["box_instance_ids"],
        ),
        prisms=Prisms(
            centers=native["prism_centers"],
            half_sizes=native["prism_half_sizes"],
            axes=native["prism_axes"],
            colors=native["prism_colors"],
            counts=native["prism_counts"],
            class_ids=native["prism_class_ids"],
            instance_ids=native["prism_instance_ids"],
        ),
        cylinders=Cylinders(
            centers=native["cylinder_centers"],
            radii=native["cylinder_radii"],
            half_heights=native["cylinder_half_heights"],
            axes=native["cylinder_axes"],
            colors=native["cylinder_colors"],
            counts=native["cylinder_counts"],
            class_ids=native["cylinder_class_ids"],
            instance_ids=native["cylinder_instance_ids"],
        ),
    )
    return RandomScene(
        scene=scene,
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
    sphere_class_ids = _metadata(scene_data.spheres.class_ids, batch_size=batch_size, count=sphere_count, default_start=1, default_value=1, device=device)
    terrain_class_ids, terrain_instance_ids = _terrain_metadata(
        scene_data.terrain.class_id,
        scene_data.terrain.instance_id,
        batch_size=batch_size,
        count=terrain_count,
        default_instance_id=sphere_count + 1,
        device=device,
    )
    box_class_ids = _metadata(scene_data.boxes.class_ids, batch_size=batch_size, count=box_count, default_start=1, default_value=3, device=device)
    prism_class_ids = _metadata(scene_data.prisms.class_ids, batch_size=batch_size, count=prism_count, default_start=1, default_value=5, device=device)
    cylinder_class_ids = _metadata(scene_data.cylinders.class_ids, batch_size=batch_size, count=cylinder_count, default_start=1, default_value=4, device=device)
    sphere_instance_ids = _metadata(scene_data.spheres.instance_ids, batch_size=batch_size, count=sphere_count, default_start=1, default_value=None, device=device)
    box_instance_ids = _metadata(
        scene_data.boxes.instance_ids,
        batch_size=batch_size,
        count=box_count,
        default_start=sphere_count + terrain_count + 1,
        default_value=None,
        device=device,
    )
    prism_instance_ids = _metadata(
        scene_data.prisms.instance_ids,
        batch_size=batch_size,
        count=prism_count,
        default_start=sphere_count + terrain_count + box_count + 1,
        default_value=None,
        device=device,
    )
    cylinder_instance_ids = _metadata(
        scene_data.cylinders.instance_ids,
        batch_size=batch_size,
        count=cylinder_count,
        default_start=sphere_count + terrain_count + box_count + prism_count + 1,
        default_value=None,
        device=device,
    )
    has_custom_metadata = any(
        value is not None
        for value in (
            scene_data.spheres.class_ids,
            scene_data.spheres.instance_ids,
            None if scene_data.terrain.class_id == 2 else scene_data.terrain.class_id,
            scene_data.terrain.instance_id,
            scene_data.boxes.class_ids,
            scene_data.boxes.instance_ids,
            scene_data.prisms.class_ids,
            scene_data.prisms.instance_ids,
            scene_data.cylinders.class_ids,
            scene_data.cylinders.instance_ids,
        )
    )
    if bool(((sphere_counts + plane_counts + terrain_counts + box_counts + prism_counts + cylinder_counts) <= 0).any().item()):
        raise ValueError("at least one object is required")
    if radii.shape[1] != sphere_count or colors.shape[1] != sphere_count:
        raise ValueError("sphere_centers, sphere_radii, and sphere_colors must have matching lengths")
    terrain_slots = terrain_depth_limits.shape[1]
    if terrain_slots not in (0, 1):
        raise ValueError("terrain may contain zero or one entry")
    if (
        terrain_base_heights.shape[1] != terrain_slots
        or terrain_phase_xs.shape[1] != terrain_slots
        or terrain_phase_zs.shape[1] != terrain_slots
        or terrain_dz.shape[1] != terrain_slots
        or terrain_dz_growth.shape[1] != terrain_slots
        or terrain_colors.shape[1] != terrain_slots
    ):
        raise ValueError("terrain base_heights, depth_limits, phase_xs, phase_zs, dz, dz_growth, and colors must have matching lengths")
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
                "class_ids": sphere_class_ids,
                "instance_ids": sphere_instance_ids,
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
                "class_ids": terrain_class_ids,
                "instance_ids": terrain_instance_ids,
            },
            "boxes": {
                "centers": box_centers,
                "half_sizes": box_half_sizes,
                "axes": box_axes,
                "colors": box_colors,
                "counts": box_counts,
                "class_ids": box_class_ids,
                "instance_ids": box_instance_ids,
            },
            "prisms": {
                "centers": prism_centers,
                "half_sizes": prism_half_sizes,
                "axes": prism_axes,
                "colors": prism_colors,
                "counts": prism_counts,
                "class_ids": prism_class_ids,
                "instance_ids": prism_instance_ids,
            },
            "cylinders": {
                "centers": cylinder_centers,
                "radii": cylinder_radii,
                "half_heights": cylinder_half_heights,
                "axes": cylinder_axes,
                "colors": cylinder_colors,
                "counts": cylinder_counts,
                "class_ids": cylinder_class_ids,
                "instance_ids": cylinder_instance_ids,
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
        if has_custom_metadata:
            visible_count, visible_classes, visible_instance_ids = _visible_custom_instances(instance_map, semantic_map)
        else:
            visible_count, visible_classes, instance_map = _compact_visible_instances(
                instance_map,
                semantic_map,
                sphere_counts=sphere_counts,
                terrain_counts=terrain_counts,
                box_counts=box_counts,
                prism_counts=prism_counts,
                cylinder_counts=cylinder_counts,
            )
            visible_instance_ids = torch.zeros_like(visible_classes)
            for batch_idx in range(batch_size):
                count = int(visible_count[batch_idx].item())
                if count > 0:
                    visible_instance_ids[batch_idx, :count] = torch.arange(1, count + 1, dtype=torch.int32, device=device)
        return RenderResult(
            image=image,
            visible_count=visible_count,
            visible_classes=visible_classes,
            visible_instance_ids=visible_instance_ids,
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
