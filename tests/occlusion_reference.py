"""Slow isolated-render oracle for CUDA annotation regression tests only."""
from __future__ import annotations
from dataclasses import fields, replace
import numpy as np
import torch
from synthetic_scene.renderer import (Scene, PackedScene, Spheres, Terrain, OrientedBoxes, Prisms, Cylinders,
    RenderOptions, RenderResult, render_scene, _with_default_metadata, _vec3_batch, _scalar_batch, _mat3_batch)
from synthetic_scene import AmodalAnnotations

def _unpack(scene: PackedScene) -> list[Scene]:
    """Decode the layout shared with bindings.cpp's packed_scene_views."""
    offsets = [0, 0]
    batch = scene.batch_size

    def take(shape, integer=False):
        index = int(integer)
        size = int(np.prod(shape))
        buffer = scene.integer_data if integer else scene.float_data
        value = buffer.narrow(0, offsets[index], size).view(shape)
        offsets[index] += size
        return value

    families = {}
    for name, cls, count in (
        ('spheres', Spheres, scene.sphere_count), ('terrain', Terrain, 1),
        ('boxes', OrientedBoxes, scene.box_count), ('prisms', Prisms, scene.prism_count),
        ('cylinders', Cylinders, scene.cylinder_count),
    ):
        data = {}
        for field in fields(cls):
            if field.name in ('class_id', 'instance_id'):
                continue
            if field.name == 'counts':
                data[field.name] = take((batch,), True)
            elif field.name in ('class_ids', 'instance_ids'):
                data[field.name] = take((batch, count), True)
            else:
                tail = (3, 3) if field.name == 'axes' else (3,) if field.name in ('centers', 'colors', 'half_sizes') else ()
                data[field.name] = take((batch, count, *tail))
        families[name] = (cls, data)
    take((batch,), True)  # unused legacy plane counts
    terrain_classes = take((batch, 1), True)
    terrain_ids = take((batch, 1), True)
    if offsets != [scene.float_data.numel(), scene.integer_data.numel()]:
        raise ValueError('packed scene size does not match its capacities')
    output = []
    for b in range(batch):
        values = {name: cls(**{key: value[b] for key, value in data.items()})
                  for name, (cls, data) in families.items()}
        values['terrain'] = replace(values['terrain'], class_id=int(terrain_classes[b, 0]),
                                    instance_id=int(terrain_ids[b, 0]))
        output.append(Scene(**values))
    return output


def _split(scene: Scene, batch: int) -> list[Scene]:
    scene = _with_default_metadata(scene)
    families = {}
    for name in ('spheres', 'terrain', 'boxes', 'prisms', 'cylinders'):
        family = getattr(scene, name)
        data = {}
        for field in fields(family):
            key = field.name
            value = getattr(family, key)
            if key in ('counts', 'class_id', 'instance_id'):
                data[key] = value
            else:
                convert = _mat3_batch if key == 'axes' else _vec3_batch if key in ('centers', 'colors', 'half_sizes') else _scalar_batch
                # Metadata must retain integer precision.
                data[key] = (torch.as_tensor(value, dtype=torch.int32, device='cuda')
                             if key in ('class_ids', 'instance_ids') else convert(value))
                if key in ('class_ids', 'instance_ids') and data[key].ndim == 1:
                    data[key] = data[key].unsqueeze(0)
        families[name] = (type(family), data)
    output = []
    for b in range(batch):
        values = {}
        for name, (cls, data) in families.items():
            selected = {}
            count = data['counts']
            if count is not None:
                count_tensor = torch.as_tensor(count).reshape(-1)
                count = int(count_tensor[0 if count_tensor.numel() == 1 else b])
            for key, value in data.items():
                if key == 'counts':
                    selected[key] = None
                elif key in ('class_id', 'instance_id'):
                    selected[key] = value
                else:
                    selected[key] = value[0 if value.shape[0] == 1 else b][:count]
            values[name] = cls(**selected)
        output.append(Scene(**values))
    return output


def _isolate(scene: Scene, instance_id: int) -> Scene:
    values = {}
    for name in ('spheres', 'terrain', 'boxes', 'prisms', 'cylinders'):
        family = getattr(scene, name)
        count = len(family.colors)
        if family.counts is not None:
            count = int(torch.as_tensor(family.counts).reshape(-1)[0])
        mask = (torch.full((count,), family.instance_id == instance_id, device='cuda')
                if name == 'terrain' else family.instance_ids[:count] == instance_id)
        data = {}
        for field in fields(family):
            key = field.name
            value = getattr(family, key)
            data[key] = None if key == 'counts' else value if key in ('class_id', 'instance_id') else value[:count][mask]
        values[name] = type(family)(**data)
    return Scene(**values)


def reference_occlusion(
    result: RenderResult, source: Scene | PackedScene,
    width: int, height: int, options: RenderOptions,
) -> RenderResult:
    scenes = _unpack(source) if isinstance(source, PackedScene) else _split(source, result.image.shape[0])
    annotations = []
    for b, single in enumerate(scenes):
        candidates = {}
        for name in ('spheres', 'terrain', 'boxes', 'prisms', 'cylinders'):
            family = getattr(single, name)
            count = len(family.colors) if family.counts is None else int(torch.as_tensor(family.counts).reshape(-1)[0])
            pairs = ([(family.instance_id, family.class_id)] if count else []) if name == 'terrain' else zip(family.instance_ids[:count].tolist(), family.class_ids[:count].tolist())
            for instance_id, class_id in pairs:
                if instance_id <= 0:
                    raise ValueError('occlusion ground truth requires positive instance IDs')
                if instance_id in candidates and candidates[instance_id] != class_id:
                    raise ValueError('parts sharing an instance ID must share a class ID')
                candidates[instance_id] = class_id
        visible_ids = set(result.instance_map[b].unique().tolist())
        ids = sorted(i for i in candidates if i in visible_ids)
        amodal = []
        for instance_id in ids:
            isolated = render_scene(width, height, scene=_isolate(single, instance_id),
                                    options=replace(options or RenderOptions(), shadows=False), return_maps=True)
            amodal.append(isolated.instance_map[0] == instance_id)
        device = result.image.device
        ids_tensor = torch.tensor(ids, device=device, dtype=torch.int32)
        masks = torch.stack(amodal) if amodal else torch.empty((0, height, width), device=device, dtype=torch.bool)
        visible = result.instance_map[b].unsqueeze(0) == ids_tensor[:, None, None]
        hidden = masks & ~visible
        area = masks.sum(dim=(1, 2))
        fraction = hidden.sum(dim=(1, 2)).float() / area
        # Rows are frontmost covering IDs; columns are hidden target IDs.
        relationships = torch.stack([
            (hidden & (result.instance_map[b] == i).unsqueeze(0)).flatten(1).any(dim=1)
            for i in ids
        ]) if ids else torch.empty((0, 0), device=device, dtype=torch.bool)
        annotations.append(AmodalAnnotations(
            ids_tensor, torch.tensor([candidates[i] for i in ids], device=device, dtype=torch.int32),
            masks, visible, fraction, relationships,
        ))
    return replace(result, amodal=annotations)


