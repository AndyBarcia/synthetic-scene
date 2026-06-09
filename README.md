# synthetic-scene

Tiny CUDA/PyTorch synthetic scene renderer.

![Example render](outputs/render.png)

The renderer draws Lambert-shaded geometric objects entirely in CUDA and can
return batched RGB, instance, and semantic label tensors. It supports up to 64
spheres, 64 oriented boxes, and 64 planes per render.

## Build

```bash
conda run -n clipdino-cu117 python setup.py build_ext --inplace
```

## Render

```bash
conda run -n clipdino-cu117 python -m examples.render
```

The example writes `outputs/render.png` plus side-by-side colorized instance and
semantic label-map visualizations. By default, renders include a floor plane and
a rear background plane.

You can render a scene directly from Python:

```python
from synthetic_scene import Camera, OrientedBoxes, Planes, Scene, Spheres, render_scene

image = render_scene(
    width=768,
    height=512,
    camera=[
        Camera(origin=(-0.2, 0.0, 0.0)),
        Camera(
            origin=(0.2, 1.0, 0.0),
            orientation=(
                (1.0, 0.0, 0.0),
                (0.0, 0.94, -0.34),
                (0.0, 0.34, 0.94),
            ),
        ),
    ],
    scene=Scene(
        spheres=Spheres(
            centers=[(-0.8, 0.0, -3.0), (0.8, 0.0, -3.2)],
            radii=[0.55, 0.65],
            colors=[(0.9, 0.25, 0.18), (0.2, 0.65, 0.95)],
        ),
        planes=Planes(
            points=[(0.0, -1.0, 0.0), (0.0, 0.0, -6.0)],
            normals=[(0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            colors=[(0.52, 0.55, 0.58), (0.12, 0.14, 0.18)],
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
    ),
)
```

RGB tensors are returned as `B x 3 x H x W`. Passing a single camera, or leaving
`camera` unset, returns a batch with `B = 1`.
`Camera.orientation` is a `3 x 3` camera-to-world rotation matrix whose rows are
the world-space right, up, and back axes; the default identity orientation looks
down world `-Z`.

To request segmentation ground truth, pass `return_maps=True`:

```python
from synthetic_scene import render_scene, save_image, save_label_map_visualization

result = render_scene(width=768, height=512, scene=scene, return_maps=True)
save_image(result.image[0].permute(1, 2, 0), "outputs/render.png")
save_label_map_visualization(result.instance_map[0], "outputs/instance_map.png")
save_label_map_visualization(result.semantic_map[0], "outputs/semantic_map.png")
```

`instance_map` is a `B x H x W int32` tensor with `0` for background and one
unique ID for each visible object. IDs are assigned in scene order: spheres
first, then planes, then boxes. `semantic_map` is a `B x H x W int32` tensor
with primitive class labels: `0 = background`, `1 = sphere`, `2 = plane`,
`3 = box`.
Raw label maps are saved as 16-bit PNGs so the numeric IDs are preserved. The
visualization helper maps each consecutive integer label to a deterministic
random RGB color, keeping background label `0` black.

## Benchmark

```bash
conda run -n clipdino-cu117 python -m examples.benchmark
```

The benchmark renders the same scene as `examples.render` from 8 camera
positions by default. You can sweep the image size, camera count, and sample
count:

```bash
conda run -n clipdino-cu117 python -m examples.benchmark --width 1920 --height 1080 --cameras 16 --iterations 200
```

The benchmark reports CUDA event timing for the render kernel, synchronized host
wall time, output tensor size, and PyTorch CUDA allocated/reserved memory peaks.
