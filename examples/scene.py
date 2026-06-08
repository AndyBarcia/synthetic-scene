from synthetic_scene import Scene, Spheres


def default_scene() -> Scene:
    return Scene(
        spheres=Spheres(
            centers=[
                (-0.85, 0.0, -3.0),
                (0.85, 0.05, -3.25),
                (0.0, -0.65, -2.35),
            ],
            radii=[0.58, 0.68, 0.36],
            colors=[
                (0.9, 0.25, 0.18),
                (0.2, 0.62, 0.95),
                (0.95, 0.82, 0.22),
            ],
        ),
    )
