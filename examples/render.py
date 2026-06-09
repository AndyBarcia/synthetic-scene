from pathlib import Path

import torch

from synthetic_scene import Camera, colorize_label_map, render_scene, save_image

from .scene import default_scene


def main() -> None:
    result = render_scene(
        width=768,
        height=512,
        scene=default_scene(),
        camera=[
            Camera(origin=(-0.22, 0.0, 0.0)),
            Camera(origin=(0.22, 1.0, 0.0)),
        ],
        return_maps=True,
    )
    output = Path("outputs/render.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image_grid = torch.cat(result.image.permute(0, 2, 3, 1).unbind(0), dim=1)
    save_image(image_grid, output)

    instance_grid = torch.cat([colorize_label_map(label_map) for label_map in result.instance_map], dim=1)
    semantic_grid = torch.cat([colorize_label_map(label_map) for label_map in result.semantic_map], dim=1)
    save_image(instance_grid.to(torch.float32).div(255.0), output.with_name("instance_map.png"))
    save_image(semantic_grid.to(torch.float32).div(255.0), output.with_name("semantic_map.png"))
    print(f"wrote {output}")
    print(f"wrote {output.with_name('instance_map.png')}")
    print(f"wrote {output.with_name('semantic_map.png')}")


if __name__ == "__main__":
    main()
