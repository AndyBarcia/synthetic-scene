from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Iterable

import torch

from synthetic_scene import RenderOptions, RenderResult, random_scene, render_scene


RANDOM_SCENE_SEED = 1234
STAGE_KERNELS = {
    "terrain": ("init_terrain_depth_kernel", "voxel_space_terrain_kernel"),
    "mask construction": (
        "compute_cluster_metadata_kernel",
        "build_primary_cluster_masks_object_driven_kernel",
        "compute_receiver_light_bounds_kernel",
        "build_shadow_cluster_masks_kernel",
    ),
    "rendering": ("render_scene_kernel",),
}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def fmean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return float(sum(values) / len(values))


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def render_benchmark_scene(width: int, height: int, seed: int, batch_size: int, shadows: bool = True) -> RenderResult:
    generated = random_scene(seed=seed, batch_size=batch_size, aspect_ratio=width / height)
    result = render_scene(
        width=width,
        height=height,
        scene=generated.scene,
        options=RenderOptions(shadows=shadows),
        return_maps=True,
    )
    assert isinstance(result, RenderResult)
    return result


def cuda_time_us(event: object) -> float:
    # PyTorch renamed this field after 1.8; support both names.
    return float(getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.0)))


def total_cuda_time_us(event: object) -> float:
    return float(getattr(event, "cuda_time_total", getattr(event, "device_time_total", 0.0)))


def matching_cuda_time_ms(events: Iterable[object], names: tuple[str, ...]) -> float:
    return sum(cuda_time_us(event) for event in events if any(name in str(event.key) for name in names)) / 1000.0


def profile_stages(
    width: int,
    height: int,
    batch_size: int,
    iterations: int,
    seed: int,
    shadows: bool = True,
) -> dict[str, float]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        for iteration in range(iterations):
            with torch.autograd.profiler.record_function("synthetic_scene::benchmark_iteration"):
                render_benchmark_scene(width, height, seed + iteration, batch_size, shadows)
        torch.cuda.synchronize()

    events = profile.key_averages()
    stage_times = {
        stage: matching_cuda_time_ms(events, kernel_names) / iterations
        for stage, kernel_names in STAGE_KERNELS.items()
    }
    render_pipeline_ms = sum(stage_times.values())
    segmentation_ms = sum(
        total_cuda_time_us(event) for event in events if str(event.key) == "synthetic_scene::segmentation"
    ) / 1000.0 / iterations
    total_cuda_ms = sum(
        total_cuda_time_us(event) for event in events if str(event.key) == "synthetic_scene::benchmark_iteration"
    ) / 1000.0 / iterations
    stage_times["segmentation"] = segmentation_ms
    stage_times["preparation"] = max(0.0, total_cuda_ms - render_pipeline_ms - segmentation_ms)
    return stage_times


def print_timing_summary(label: str, values: list[float]) -> None:
    print(f"{label}:")
    print(f"  mean: {fmean(values):.4f} ms")
    print(f"  median: {statistics.median(values):.4f} ms")
    print(f"  p95: {percentile(values, 95):.4f} ms")
    print(f"  min / max: {min(values):.4f} / {max(values):.4f} ms")


def benchmark(
    width: int,
    height: int,
    batch_size: int,
    warmup: int,
    iterations: int,
    profile_iterations: int,
    seed: int,
    shadows: bool = True,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark this renderer")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if profile_iterations <= 0:
        raise ValueError("profile_iterations must be positive")

    device_index = 0
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()

    result = None
    for iteration in range(warmup):
        result = render_benchmark_scene(width, height, seed + iteration, batch_size, shadows)
    torch.cuda.synchronize(device)
    del result

    torch.cuda.reset_peak_memory_stats(device)
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    host_times_ms: list[float] = []
    cuda_times_ms: list[float] = []

    for iteration in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start_event.record()
        result = render_benchmark_scene(width, height, seed + iteration, batch_size, shadows)
        end_event.record()
        torch.cuda.synchronize(device)
        host_times_ms.append((time.perf_counter() - host_start) * 1000.0)
        cuda_times_ms.append(start_event.elapsed_time(end_event))

    assert result is not None
    outputs = (result.image, result.visible_count, result.visible_classes, result.instance_map, result.semantic_map)
    after_allocated = torch.cuda.memory_allocated(device)
    after_reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    output_bytes = sum(tensor.numel() * tensor.element_size() for tensor in outputs)

    print("profiling CUDA stages (excluded from end-to-end timing)...")
    stage_times_ms = profile_stages(width, height, batch_size, profile_iterations, seed, shadows)

    pixels_per_scene = width * height
    pixels = batch_size * pixels_per_scene
    mean_cuda = fmean(cuda_times_ms)
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"resolution: {width} x {height} ({pixels_per_scene:,} pixels per scene)")
    print(f"batch size: {batch_size} ({pixels:,} total pixels)")
    print(f"scene seeds: {seed} through {seed + iterations - 1}")
    print(f"shadows: {'enabled' if shadows else 'disabled'}")
    print(f"warmup / measured / profiled iterations: {warmup} / {iterations} / {profile_iterations}")
    print("outputs: RGB, visible instances/classes, instance map, semantic map")
    print()
    print("mean CUDA stage time:")
    for stage in ("preparation", "terrain", "mask construction", "rendering", "segmentation"):
        print(f"  {stage}: {stage_times_ms[stage]:.4f} ms")
    print()
    print_timing_summary("end-to-end CUDA time", cuda_times_ms)
    print(f"  throughput: {pixels / (mean_cuda / 1000.0) / 1_000_000.0:.2f} Mpixels/s")
    print()
    print_timing_summary("synchronized host wall time", host_times_ms)
    print()
    print("cuda memory:")
    print(f"  output tensors: {format_bytes(output_bytes)}")
    print(f"  allocated before / after: {format_bytes(before_allocated)} / {format_bytes(after_allocated)}")
    print(f"  reserved before / after: {format_bytes(before_reserved)} / {format_bytes(after_reserved)}")
    print(f"  peak allocated / reserved: {format_bytes(peak_allocated)} / {format_bytes(peak_reserved)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end CUDA scene generation and rendering.")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_SCENE_SEED)
    parser.add_argument("--no-shadows", action="store_true", help="disable shadow rays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark(
        args.width,
        args.height,
        args.batch_size,
        args.warmup,
        args.iterations,
        args.profile_iterations,
        args.seed,
        not args.no_shadows,
    )


if __name__ == "__main__":
    main()
