from pathlib import Path

from synthetic_scene import render_sphere, save_image


def main() -> None:
    image = render_sphere(width=768, height=512)
    output = Path("outputs/sphere.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
