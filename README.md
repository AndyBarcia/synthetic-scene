# synthetic-scene

Tiny CUDA/PyTorch synthetic scene renderer.

![Example render](outputs/render.png)

The renderer draws Lambert-shaded geometric objects entirely in CUDA and can
return batched RGB plus panoptic-friendly segmentation tensors. It supports up
to 64 spheres, 64 oriented boxes, 64 oriented triangular prisms, 64 oriented cylinders, and one procedural terrain per scene. Batched
renders draw different camera-space scenes.

## Build

```bash
conda run -n clipdino-cu117 python setup.py build_ext --inplace
```

## Render

```bash
conda run -n clipdino-cu117 python -m examples.render
```

The example writes `outputs/render.png` plus side-by-side colorized instance and
semantic label-map visualizations. Random renders include a generated rolling
procedural terrain, and grounded objects are placed on that terrain.

You can render a scene directly from Python:

```python
from synthetic_scene import Cylinders, OrientedBoxes, Prisms, RenderOptions, Scene, Spheres, Terrain, render_scene

image = render_scene(
    width=768,
    height=512,
    options=RenderOptions(shadows=True, shadow_strength=1.0, ambient=0.16),
    scene=Scene(
        spheres=Spheres(
            centers=[(-0.8, 0.0, -3.0), (0.8, 0.0, -3.2)],
            radii=[0.55, 0.65],
            colors=[(0.9, 0.25, 0.18), (0.2, 0.65, 0.95)],
        ),
        terrain=Terrain(
            base_heights=[-1.0],
            depth_limits=[7.0],
            phase_xs=[0.4],
            phase_zs=[1.2],
            dz=[0.05],
            dz_growth=[0.0001],
            colors=[(0.36, 0.46, 0.30)],
        ),
        boxes=OrientedBoxes(
            centers=[(0.0, -0.35, -2.6)],
            half_sizes=[(0.38, 0.42, 0.55)],
            axes=[
                (
                    (0.866, 0.0, -0.5),
                    (0.0, 1.0, 0.0),
                    (0.5, 0.0, 0.866),
                ),
            ],
            colors=[(0.45, 0.9, 0.48)],
        ),
        prisms=Prisms(
            centers=[(-0.95, -0.35, -2.7)],
            half_sizes=[(0.42, 0.46, 0.50)],
            axes=[
                (
                    (0.707, 0.0, -0.707),
                    (0.0, 1.0, 0.0),
                    (0.707, 0.0, 0.707),
                ),
            ],
            colors=[(0.80, 0.55, 0.95)],
        ),
        cylinders=Cylinders(
            centers=[(0.95, 0.05, -2.55)],
            radii=[0.22],
            half_heights=[0.70],
            axes=[
                (
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            ],
            colors=[(0.95, 0.48, 0.22)],
        ),
    ),
)
```

RGB tensors are returned as `B x 3 x H x W`. Unbatched scene tensors return
`B = 1`; scene tensors shaped as `B x N x ...` render one independent scene per
batch item. Object coordinates are camera-space, with the camera at the origin
looking down `-Z`.

Directional-light hard shadows are enabled by default. Set
`RenderOptions(shadows=False)` to render direct Lambert shading without shadow
rays, or tune `shadow_strength` in `[0, 1]` to control how much blocked direct
light is removed. The default `1.0` leaves shadowed pixels at the same ambient
floor as surfaces facing away from the light. Increase `ambient` to soften both
cast shadows and unlit object sides together.

To request segmentation ground truth, pass `return_maps=True`:

```python
from synthetic_scene import render_scene, save_image, save_label_map_visualization

result = render_scene(width=768, height=512, scene=scene, return_maps=True)
save_image(result.image[0].permute(1, 2, 0), "outputs/render.png")
save_label_map_visualization(result.instance_map[0], "outputs/instance_map.png")
save_label_map_visualization(result.semantic_map[0], "outputs/semantic_map.png")
```

`instance_map` is a `B x H x W int32` tensor with `0` for background and one
sequential ID for each visible object in that image. `visible_count` is a
`B int32` tensor with the number of visible objects per image. `visible_classes`
and `visible_instance_ids` are `B x MAX_GT int32` tensors where columns
`0:visible_count[b]` contain the class labels and instance IDs for visible
objects. `MAX_GT` is the total number of objects in the scene. Class labels are
`1 = sphere`, `2 = terrain`, `3 = box`, `4 = cylinder`, `5 = prism`; unused
class slots and background pixels are `0`.

`semantic_map` is also returned as a `B x H x W int32` compatibility tensor with
primitive class labels per pixel: `0 = background`, `1 = sphere`,
`2 = terrain`, `3 = box`, `4 = cylinder`, `5 = prism`.
Raw label maps are saved as 16-bit PNGs so the numeric IDs are preserved. The
visualization helper maps each consecutive integer label to a deterministic
random RGB color, keeping background label `0` black.

Finite primitive families also accept optional `class_ids` and `instance_ids`
metadata with shape `N` or `B x N`. Terrain is scene-level and has scalar
`class_id` and `instance_id` fields instead. When custom metadata is present,
`instance_map` preserves those instance IDs instead of compacting to sequential
IDs, and `semantic_map` uses the provided classes. Composite objects can be
positioned, rotated, and flattened into raw primitives before rendering. Terrain
cannot be part of a composite object; pass it through `base_scene` if needed.

```python
from synthetic_scene import HouseConfig, Scene, Spheres, Terrain, TreeConfig, flatten_composite_objects, make_house, make_tree

base_scene = Scene(
    spheres=Spheres(centers=(), radii=(), colors=()),
    terrain=Terrain(
        base_heights=[-1.0],
        depth_limits=[7.0],
        phase_xs=[0.4],
        phase_zs=[1.2],
        dz=[0.05],
        dz_growth=[0.0001],
        colors=[(0.36, 0.46, 0.30)],
    ),
)

scene = flatten_composite_objects([
    make_house(
        instance_id=1001,
        config=HouseConfig(width=1.4, depth=1.0, body_height=0.9, roof_height=0.45),
        position=(-1.0, -1.0, -4.0),
        rotation=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ),
    make_house(
        instance_id=1002,
        config=HouseConfig(width=0.9, depth=0.8, body_height=0.65, roof_height=0.35),
        position=(0.8, -1.0, -3.4),
    ),
    make_tree(
        instance_id=2001,
        config=TreeConfig(trunk_height=0.9, trunk_radius=0.08, crown_radius=0.36),
        position=(-2.0, -1.0, -3.6),
    ),
    make_tree(
        instance_id=2002,
        config=TreeConfig(trunk_height=1.25, trunk_radius=0.12, crown_radius=0.52),
        position=(1.8, -1.0, -4.2),
    ),
], base_scene=base_scene)
```

## Random Scenes

For synthetic data generation, create seeded random camera-space scenes:

```python
from synthetic_scene import random_scene, render_scene

width = 768
height = 512
generated = random_scene(
    seed=1234,
    batch_size=8,
    house_count=3,
    tree_count=8,
    aspect_ratio=width / height,
    depth_limit=7.0,
    terrain_dz=0.05,
    terrain_dz_growth=0.0001,
)
result = render_scene(
    width=width,
    height=height,
    scene=generated.scene,
    return_maps=True,
)
```

Random scenes always include a generated procedural terrain and random
composite objects. `house_count` and `tree_count` create randomized houses and
trees with per-object instance IDs and semantic classes `10` and `11`; they are
placed with camera-frustum sampling and grounded against the generated terrain
height.
The native random scene generator expands composite templates into ordinary
primitive tensors, so rendering still receives the same sphere, box, prism, and
cylinder arrays.

## Benchmark

```bash
conda run -n clipdino-cu117 python -m examples.benchmark
```

The benchmark renders the same seeded random scene style as `examples.render`
with a batch size of 8 by default. You can sweep the image size, batch size,
sample count, and random seed:

```bash
conda run -n clipdino-cu117 python -m examples.benchmark --width 1920 --height 1080 --batch-size 16 --iterations 200 --seed 5678
```

The benchmark reports CUDA event timing for the render kernel, synchronized host
wall time, output tensor size, and PyTorch CUDA allocated/reserved memory peaks.
