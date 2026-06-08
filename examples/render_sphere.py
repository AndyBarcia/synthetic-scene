from pathlib import Path

from synthetic_scene import render_spheres, save_image


def main() -> None:
    image = render_spheres(
        width=768,
        height=512,
        sphere_centers=[
            (-0.85, 0.0, -3.0),
            (0.85, 0.05, -3.25),
            (0.0, -0.65, -2.35),
        ],
        sphere_radii=[0.58, 0.68, 0.36],
        sphere_colors=[
            (0.9, 0.25, 0.18),
            (0.2, 0.62, 0.95),
            (0.95, 0.82, 0.22),
        ],
    )
    output = Path("outputs/sphere.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
