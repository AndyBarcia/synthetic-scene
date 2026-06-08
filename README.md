# synthetic-scene

Tiny CUDA/PyTorch synthetic scene renderer.

The renderer draws Lambert-shaded spheres entirely in CUDA and returns an
`H x W x 3` float32 CUDA tensor that can be saved from Python. It supports up to
64 spheres per render.

## Build

```bash
conda run -n clipdino-cu117 python setup.py build_ext --inplace
```

## Render

```bash
conda run -n clipdino-cu117 python -m examples.render_sphere
```

The example writes `outputs/sphere.png`.

You can render several spheres directly from Python:

```python
from synthetic_scene import render_spheres

image = render_spheres(
    width=768,
    height=512,
    sphere_centers=[(-0.8, 0.0, -3.0), (0.8, 0.0, -3.2)],
    sphere_radii=[0.55, 0.65],
    sphere_colors=[(0.9, 0.25, 0.18), (0.2, 0.65, 0.95)],
)
```

## Benchmark

```bash
conda run -n clipdino-cu117 python -m examples.benchmark_sphere
```

By default the benchmark renders one sphere through `render_spheres`. You can
sweep the image size, sample count, and number of spheres:

```bash
conda run -n clipdino-cu117 python -m examples.benchmark_sphere --width 1920 --height 1080 --iterations 200 --spheres 16
```

The benchmark reports CUDA event timing for the render kernel, synchronized host
wall time, output tensor size, and PyTorch CUDA allocated/reserved memory peaks.
