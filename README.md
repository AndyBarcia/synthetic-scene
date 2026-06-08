# synthetic-scene

Tiny CUDA/PyTorch synthetic scene renderer.

The renderer draws Lambert-shaded geometric objects entirely in CUDA and returns
an `H x W x 3` float32 CUDA tensor that can be saved from Python. It supports up
to 64 spheres, 64 oriented boxes, and 64 planes per render.

## Build

```bash
conda run -n clipdino-cu117 python setup.py build_ext --inplace
```

## Render

```bash
conda run -n clipdino-cu117 python -m examples.render
```

The example writes `outputs/render.png`. By default, renders include a floor
plane and a rear background plane.

You can render a scene directly from Python:

```python
from synthetic_scene import OrientedBoxes, Planes, Scene, Spheres, render_scene

image = render_scene(
    width=768,
    height=512,
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

## Benchmark

```bash
conda run -n clipdino-cu117 python -m examples.benchmark
```

The benchmark renders the same scene as `examples.render`. You can sweep the
image size and sample count:

```bash
conda run -n clipdino-cu117 python -m examples.benchmark --width 1920 --height 1080 --iterations 200
```

The benchmark reports CUDA event timing for the render kernel, synchronized host
wall time, output tensor size, and PyTorch CUDA allocated/reserved memory peaks.
