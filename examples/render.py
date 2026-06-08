from pathlib import Path

from synthetic_scene import render_scene, save_image

from .scene import default_scene


def main() -> None:
    image = render_scene(
        width=768,
        height=512,
        scene=default_scene(),
    )
    output = Path("outputs/render.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
