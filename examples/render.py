from pathlib import Path

from synthetic_scene import render_scene, save_image, save_label_map_visualization

from .scene import default_scene


def main() -> None:
    result = render_scene(
        width=768,
        height=512,
        scene=default_scene(),
        return_maps=True,
    )
    output = Path("outputs/render.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(result.image, output)
    save_label_map_visualization(result.instance_map, output.with_name("instance_map.png"))
    save_label_map_visualization(result.semantic_map, output.with_name("semantic_map.png"))
    print(f"wrote {output}")
    print(f"wrote {output.with_name('instance_map.png')}")
    print(f"wrote {output.with_name('semantic_map.png')}")
    print(f"wrote {output.with_name('instance_map_visualization.png')}")
    print(f"wrote {output.with_name('semantic_map_visualization.png')}")


if __name__ == "__main__":
    main()
