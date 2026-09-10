"""GPU regression checks: python -m unittest discover -s tests."""

import unittest

import torch

from synthetic_scene import RandomSceneOptions, generate_random_scene, render_scene


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RandomGenerationTests(unittest.TestCase):
    def test_slot_ownership_and_determinism(self):
        # Include absent families, maximum capacities, and more than 64 objects.
        cases = [(10, 10, 5, 5, 5), (64, 0, 0, 0, 0), (0, 64, 0, 0, 0),
                 (0, 0, 21, 0, 0), (0, 0, 0, 16, 0), (0, 0, 0, 0, 12),
                 (32, 32, 10, 0, 0)]
        for houses, trees, clouds, cars, people in cases:
            options = RandomSceneOptions(house_count=houses, tree_count=trees,
                                         cloud_count=clouds, car_count=cars,
                                         person_count=people)
            for batch in (1, 8, 65):
                with self.subTest(options=options, batch=batch):
                    scene = generate_random_scene(1234, batch_size=batch, options=options)
                    again = generate_random_scene(1234, batch_size=batch, options=options)
                    self.assertTrue(torch.equal(scene.float_data, again.float_data))
                    self.assertTrue(torch.equal(scene.integer_data, again.integer_data))
                    self.assertTrue(torch.isfinite(scene.float_data).all())
                    # Decode the public packed metadata and check every primitive slot.
                    fields = scene.integer_data.cpu()
                    offset = 0

                    def take(width):
                        nonlocal offset
                        result = fields[offset:offset + batch * width].reshape(batch, width)
                        offset += batch * width
                        return result

                    start = 1
                    ids = []
                    for count in (houses, trees, clouds, cars, people):
                        ids.append(list(range(start, start + count)))
                        start += count
                    h, t, c, a, p = ids
                    families = [(t + [i for i in c for _ in range(3)] + p,
                                 [11] * trees + [12] * (3 * clouds) + [14] * people),
                                (h + a + [i for i in p for _ in range(5)],
                                 [10] * houses + [13] * cars + [14] * (5 * people)),
                                (h, [10] * houses),
                                (t + [i for i in a for _ in range(4)],
                                 [11] * trees + [13] * (4 * cars))]
                    for index, (expected_ids, expected_classes) in enumerate(families):
                        if index == 1:
                            self.assertTrue((take(1) == 1).all())  # terrain counts
                        self.assertTrue((take(1) == len(expected_ids)).all())
                        for expected in (expected_classes, expected_ids):
                            actual = take(len(expected))
                            reference = torch.tensor(expected, dtype=torch.int32).expand(batch, -1)
                            self.assertTrue(torch.equal(actual, reference))

    def test_render_and_seed_variation(self):
        first = generate_random_scene(42, batch_size=8)
        second = generate_random_scene(43, batch_size=8)
        self.assertFalse(torch.equal(first.float_data, second.float_data))
        image = render_scene(65, 49, scene=first)
        self.assertEqual(tuple(image.shape), (8, 3, 49, 65))
        self.assertTrue(torch.isfinite(image).all())
