from __future__ import annotations

import argparse
import statistics
import time

import torch

from synthetic_scene import render_scene

from .scene import default_scene


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


def render_benchmark_scene(width: int, height: int) -> torch.Tensor:
    return render_scene(
        width=width,
        height=height,
        scene=default_scene(),
    )


def benchmark(width: int, height: int, warmup: int, iterations: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark this renderer")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    device_index = 0
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    image = None
    for _ in range(warmup):
        image = render_benchmark_scene(width, height)
    torch.cuda.synchronize(device)
    del image

    torch.cuda.reset_peak_memory_stats(device)
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)

    host_times_ms: list[float] = []
    kernel_times_ms: list[float] = []

    for _ in range(iterations):
        image = None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        host_start = time.perf_counter()
        start_event.record()
        image = render_benchmark_scene(width, height)
        end_event.record()
        torch.cuda.synchronize(device)
        host_end = time.perf_counter()

        kernel_times_ms.append(start_event.elapsed_time(end_event))
        host_times_ms.append((host_end - host_start) * 1000.0)

    # Keep the last tensor alive so "after allocated" includes the output image.
    _ = image

    after_allocated = torch.cuda.memory_allocated(device)
    after_reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    image_bytes = width * height * 3 * torch.finfo(torch.float32).bits // 8

    pixels = width * height
    mean_kernel = fmean(kernel_times_ms)
    mean_host = fmean(host_times_ms)

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"resolution: {width} x {height} ({pixels:,} pixels)")
    print("scene: default")
    print(f"warmup / iterations: {warmup} / {iterations}")
    print()
    print("render kernel time:")
    print(f"  mean: {mean_kernel:.4f} ms")
    print(f"  median: {statistics.median(kernel_times_ms):.4f} ms")
    print(f"  p95: {percentile(kernel_times_ms, 95):.4f} ms")
    print(f"  min / max: {min(kernel_times_ms):.4f} / {max(kernel_times_ms):.4f} ms")
    print(f"  throughput: {pixels / (mean_kernel / 1000.0) / 1_000_000.0:.2f} Mpixels/s")
    print()
    print("host wall time:")
    print(f"  mean: {mean_host:.4f} ms")
    print(f"  median: {statistics.median(host_times_ms):.4f} ms")
    print(f"  p95: {percentile(host_times_ms, 95):.4f} ms")
    print(f"  min / max: {min(host_times_ms):.4f} / {max(host_times_ms):.4f} ms")
    print()
    print("cuda memory:")
    print(f"  output tensor: {format_bytes(image_bytes)}")
    print(f"  allocated before / after: {format_bytes(before_allocated)} / {format_bytes(after_allocated)}")
    print(f"  reserved before / after: {format_bytes(before_reserved)} / {format_bytes(after_reserved)}")
    print(f"  peak allocated / reserved: {format_bytes(peak_allocated)} / {format_bytes(peak_reserved)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the CUDA scene renderer.")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark(
        width=args.width,
        height=args.height,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
