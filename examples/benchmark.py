from __future__ import annotations

import argparse
import json
import random
from dataclasses import fields
import statistics
import time
import torch

from synthetic_scene import RandomSceneOptions, RenderOptions, RenderResult, generate_random_scene, render_scene, save_amodal_visualization


RANDOM_SCENE_SEED = 1234
STAGE_RANGES = {
    "scene generation": "synthetic_scene::scene_generation",
    "input preparation": "synthetic_scene::input_preparation",
    "terrain": "synthetic_scene::terrain",
    "mask construction": "synthetic_scene::mask_construction",
    "rendering": "synthetic_scene::rendering",
    "occlusion": "synthetic_scene::occlusion",
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


def render_benchmark_scene(
    width: int,
    height: int,
    seed: int,
    batch_size: int,
    shadows: bool = True,
    terrain: bool = True,
    occlusion: bool = False,
    parts: bool = False,
) -> RenderResult:
    with torch.autograd.profiler.record_function("synthetic_scene::scene_generation"):
        scene_options = RandomSceneOptions(aspect_ratio=width / height)
        scene = generate_random_scene(seed=seed, batch_size=batch_size, options=scene_options)
        if not terrain:
            # Packed integer fields start with sphere counts, classes and IDs,
            # followed by one terrain count per scene. Keep the generated
            # geometry grounded, but disable terrain intersections.
            terrain_offset = batch_size * (1 + 2 * scene.sphere_count)
            scene.integer_data.narrow(0, terrain_offset, batch_size).zero_()
    result = render_scene(
        width=width,
        height=height,
        scene=scene,
        options=RenderOptions(shadows=shadows),
        return_maps=True,
        return_occlusion=occlusion,
        return_parts=parts,
    )
    assert isinstance(result, RenderResult)
    return result


def total_cuda_time_us(event: object) -> float:
    # PyTorch renamed this field after 1.8; support both names.
    return float(getattr(event, "cuda_time_total", getattr(event, "device_time_total", 0.0)))


def profile_stages(
    width: int,
    height: int,
    batch_size: int,
    iterations: int,
    seed: int,
    shadows: bool = True,
    terrain: bool = True,
    occlusion: bool = False,
    parts: bool = False,
) -> tuple[dict[str, float], float]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profile:
        for iteration in range(iterations):
            with torch.autograd.profiler.record_function("synthetic_scene::benchmark_iteration"):
                render_benchmark_scene(width, height, seed + iteration, batch_size, shadows, terrain, occlusion, parts)
        torch.cuda.synchronize()

    events = profile.key_averages()
    stage_times = {
        stage: sum(total_cuda_time_us(event) for event in events if str(event.key) == range_name)
        / 1000.0
        / iterations
        for stage, range_name in STAGE_RANGES.items()
    }
    attributed_ms = sum(stage_times.values())
    segmentation_ms = sum(
        total_cuda_time_us(event) for event in events if str(event.key) == "synthetic_scene::segmentation"
    ) / 1000.0 / iterations
    total_cuda_ms = sum(
        total_cuda_time_us(event) for event in events if str(event.key) == "synthetic_scene::benchmark_iteration"
    ) / 1000.0 / iterations
    stage_times["segmentation"] = segmentation_ms
    stage_times["unattributed / profiler overhead"] = max(
        0.0, total_cuda_ms - attributed_ms - segmentation_ms
    )
    return stage_times, total_cuda_ms


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
    terrain: bool = True,
    occlusion: bool = False,
    amodal_path: str | None = None,
    parts: bool = False,
) -> None:
    if amodal_path is not None and not occlusion:
        raise ValueError("amodal export requires occlusion outputs")
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
        result = render_benchmark_scene(width, height, seed + iteration, batch_size, shadows, terrain, occlusion, parts)
    torch.cuda.synchronize(device)
    del result
    result = None

    torch.cuda.reset_peak_memory_stats(device)
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    host_times_ms: list[float] = []
    cuda_times_ms: list[float] = []

    for iteration in range(iterations):
        result = None  # Release the previous frame before measuring its replacement.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start_event.record()
        result = render_benchmark_scene(width, height, seed + iteration, batch_size, shadows, terrain, occlusion, parts)
        end_event.record()
        torch.cuda.synchronize(device)
        host_times_ms.append((time.perf_counter() - host_start) * 1000.0)
        cuda_times_ms.append(start_event.elapsed_time(end_event))

    assert result is not None
    outputs = [getattr(result, entry.name) for entry in fields(result)
               if isinstance(getattr(result, entry.name), torch.Tensor)]
    if result.amodal is not None:
        outputs.extend(getattr(gt, entry.name) for gt in result.amodal for entry in fields(gt))
    after_allocated = torch.cuda.memory_allocated(device)
    after_reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    output_bytes = sum(tensor.numel() * tensor.element_size() for tensor in outputs)

    print("profiling CUDA stages (excluded from end-to-end timing)...")
    stage_times_ms, profiled_total_ms = profile_stages(
        width, height, batch_size, profile_iterations, seed, shadows, terrain, occlusion, parts
    )

    pixels_per_scene = width * height
    pixels = batch_size * pixels_per_scene
    mean_cuda = fmean(cuda_times_ms)
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"resolution: {width} x {height} ({pixels_per_scene:,} pixels per scene)")
    print(f"batch size: {batch_size} ({pixels:,} total pixels)")
    print(f"scene seeds: {seed} through {seed + iterations - 1}")
    print(f"shadows: {'enabled' if shadows else 'disabled'}")
    print(f"terrain: {'enabled' if terrain else 'disabled'}")
    print(f"warmup / measured / profiled iterations: {warmup} / {iterations} / {profile_iterations}")
    print("outputs: RGB, visible instances/classes/IDs, instance map, semantic map"
          + (", amodal and occlusion ground truth" if occlusion else "")
          + (", modal parts and membership" if parts else ""))
    if result.amodal is not None:
        print(f"annotated instances in last batch: {[len(gt.instance_ids) for gt in result.amodal]}")
    print()
    print("mean CUDA stage time (profiled pass):")
    for stage in (
        "scene generation",
        "input preparation",
        "terrain",
        "mask construction",
        "rendering",
        "occlusion",
        "segmentation",
        "unattributed / profiler overhead",
    ):
        if stage != "occlusion" or occlusion:
            print(f"  {stage}: {stage_times_ms[stage]:.4f} ms")
    print(f"  profiled total: {profiled_total_ms:.4f} ms")
    print()
    print_timing_summary("end-to-end CUDA time", cuda_times_ms)
    print(f"  throughput: {pixels / (mean_cuda / 1000.0) / 1_000_000.0:.2f} Mpixels/s")
    print()
    print_timing_summary("synchronized host wall time", host_times_ms)
    print(f"  throughput: {batch_size * 1000. / fmean(host_times_ms):.2f} images/s")
    print()
    print("cuda memory:")
    print(f"  output tensors: {format_bytes(output_bytes)}")
    print(f"  allocated before / after: {format_bytes(before_allocated)} / {format_bytes(after_allocated)}")
    print(f"  reserved before / after: {format_bytes(before_reserved)} / {format_bytes(after_reserved)}")
    print(f"  peak allocated / reserved: {format_bytes(peak_allocated)} / {format_bytes(peak_reserved)}")

    if amodal_path is not None:
        export_times_ms = []
        for _ in range(5):
            started = time.perf_counter()
            save_amodal_visualization(result, amodal_path)
            export_times_ms.append((time.perf_counter() - started) * 1000.)
        print()
        print_timing_summary("amodal PNG export wall time (excluded from rendering timing)", export_times_ms)
        print(f"  wrote {amodal_path}")


def compare_outputs(args: argparse.Namespace) -> None:
    modes = {'rgb': {}, 'maps': {'return_maps': True}, 'parts': {'return_parts': True}}
    if args.occlusion:
        modes.update(amodal={'return_occlusion': True},
                     parts_amodal={'return_parts': True, 'return_occlusion': True})
    rng = random.Random(42)
    report = {'device': torch.cuda.get_device_name(), 'iterations': args.iterations,
              'warmup': args.warmup, 'seed': args.seed, 'shadows': not args.no_shadows,
              'scene_generation_included': False, 'results': []}
    options = RenderOptions(shadows=not args.no_shadows)
    for resolution in args.resolutions:
        width, height = map(int, resolution.lower().split('x'))
        for batch in args.batch_sizes:
            scene = generate_random_scene(args.seed, batch_size=batch,
                                          options=RandomSceneOptions(aspect_ratio=width / height))
            def render(mode):
                return render_scene(width, height, scene=scene, options=options, **modes[mode])

            for _ in range(args.warmup):
                for mode in modes:
                    result = render(mode)
                    del result
            torch.cuda.synchronize()
            timings = {mode: {'wall': [], 'cuda': []} for mode in modes}
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(args.iterations):
                order = list(modes)
                rng.shuffle(order)
                for mode in order:
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    start.record()
                    result = render(mode)
                    end.record()
                    end.synchronize()
                    timings[mode]['wall'].append((time.perf_counter() - before) * 1000)
                    timings[mode]['cuda'].append(start.elapsed_time(end))
                    del result
            for mode in modes:
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result = render(mode)
                torch.cuda.synchronize()
                tensors = ([result] if isinstance(result, torch.Tensor) else
                           [getattr(result, f.name) for f in fields(result)
                            if isinstance(getattr(result, f.name), torch.Tensor)])
                if not isinstance(result, torch.Tensor) and result.amodal is not None:
                    tensors.extend(getattr(gt, entry.name) for gt in result.amodal for entry in fields(gt))
                row = dict(resolution=resolution, batch=batch, mode=mode,
                           wall_median_ms=statistics.median(timings[mode]['wall']),
                           cuda_median_ms=statistics.median(timings[mode]['cuda']),
                           wall_p95_ms=percentile(timings[mode]['wall'], 95),
                           output_mib=sum(t.numel() * t.element_size() for t in tensors) / 2**20,
                           peak_extra_mib=(torch.cuda.max_memory_allocated() - baseline) / 2**20)
                if mode in ('parts', 'parts_amodal'):
                    row['visible_parts'] = result.part_count.tolist()
                report['results'].append(row)
                print(json.dumps(row), flush=True)
                del result, tensors
            del scene
    print(json.dumps(report, indent=2))



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
    parser.add_argument("--no-terrain", action="store_true", help="render generated scenes without terrain")
    parser.add_argument("--occlusion", action="store_true", help="compare baseline rendering with amodal and occlusion outputs")
    parser.add_argument("--export-amodal", metavar="PATH", help="also time PNG overlay export (requires --occlusion)")
    parser.add_argument("--parts", action="store_true", help="include modal part masks in the end-to-end benchmark")
    parser.add_argument("--compare-outputs", action="store_true",
                        help="compare RGB, object maps, and parts on identical pre-generated scenes")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8],
                        help="batch sizes for --compare-outputs")
    parser.add_argument("--resolutions", nargs="+", default=["256x256", "768x512"],
                        help="WIDTHxHEIGHT values for --compare-outputs")
    args = parser.parse_args()
    if args.export_amodal and not args.occlusion:
        parser.error("--export-amodal requires --occlusion")
    if args.iterations <= 0 or args.warmup < 0 or min(args.batch_sizes) <= 0:
        parser.error("iterations and batch sizes must be positive; warmup must be nonnegative")
    if args.compare_outputs and args.no_terrain:
        parser.error("packed comparison scenes always include terrain")
    for resolution in args.resolutions:
        try:
            width, height = map(int, resolution.lower().split("x"))
            if width <= 0 or height <= 0:
                raise ValueError
        except ValueError:
            parser.error("resolutions must use positive WIDTHxHEIGHT values")
    if args.compare_outputs and args.export_amodal:
        parser.error("--export-amodal is supported by the end-to-end benchmark")
    return args


def main() -> None:
    args = parse_args()
    if args.compare_outputs:
        compare_outputs(args)
        return
    for occlusion in ((False, True) if args.occlusion else (False,)):
        if args.occlusion:
            print("\nBaseline render:" if not occlusion else "\nRender with amodal and occlusion ground truth:")
        benchmark(
            args.width,
            args.height,
            args.batch_size,
            args.warmup,
            args.iterations,
            args.profile_iterations,
            args.seed,
            not args.no_shadows,
            not args.no_terrain,
            occlusion=occlusion,
            amodal_path=args.export_amodal if occlusion else None,
            parts=args.parts,
        )


if __name__ == "__main__":
    main()
