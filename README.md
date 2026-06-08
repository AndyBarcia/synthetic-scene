# synthetic-scene

Tiny CUDA/PyTorch synthetic scene renderer.

The first pass renders one Lambert-shaded sphere entirely in CUDA and returns an
`H x W x 3` float32 CUDA tensor that can be saved from Python.

## Build

```bash
conda run -n clipdino-cu117 python setup.py build_ext --inplace
```

## Render

```bash
conda run -n clipdino-cu117 python -m examples.render_sphere
```

The example writes `outputs/sphere.png`.

## Benchmark

```bash
conda run -n clipdino-cu117 python -m examples.benchmark_sphere
```

You can sweep the image size and sample count:

```bash
conda run -n clipdino-cu117 python -m examples.benchmark_sphere --width 1920 --height 1080 --iterations 200
```

The benchmark reports CUDA event timing for the render kernel, synchronized host
wall time, output tensor size, and PyTorch CUDA allocated/reserved memory peaks.
