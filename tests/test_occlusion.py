import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch
from synthetic_scene import (
    Scene, Spheres, Terrain, RandomSceneOptions, generate_random_scene,
    render_scene, save_amodal_visualization,
)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class OcclusionTests(unittest.TestCase):
    def test_overlap_and_export(self):
        # Front sphere completely covers the rear sphere; third sphere is offscreen.
        scene = Scene(spheres=Spheres(
            centers=[(0, 0, -2), (0, 0, -4), (100, 0, -3)],
            radii=[.7, .7, .2], colors=[(1, 0, 0)] * 3,
            instance_ids=[11, 22, 33]),
            terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(),
                            phase_zs=(), dz=(), dz_growth=(), colors=()))
        result = render_scene(64, 48, scene=scene, return_occlusion=True)
        self.assertTrue(torch.equal(result.panoptic_map[..., 0], result.semantic_map))
        self.assertTrue(torch.equal(result.panoptic_map[..., 1], result.instance_map))
        gt = result.amodal[0]
        self.assertEqual(gt.instance_ids.tolist(), [11])
        self.assertEqual(gt.occlusion_fraction.tolist(), [0])
        self.assertEqual(gt.amodal_masks.shape, (1, 48, 64))
        self.assertFalse((gt.visible_masks & ~gt.amodal_masks).any())
        self.assertEqual(gt.occlusion_matrix.dtype, torch.bool)
        self.assertFalse(gt.occlusion_matrix.diagonal().any())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'amodal.png'
            save_amodal_visualization(result, target)
            with Image.open(target) as overlay:
                self.assertEqual(overlay.size, (64, 48))
                self.assertEqual(overlay.mode, 'RGB')
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['amodal.png'])

    def test_partial_composite_and_counts(self):
        # Two overlapping parts share an ID; padding is inactive in batch item 1.
        scene = Scene(spheres=Spheres(
            centers=[[(0, 0, -3), (.4, 0, -3), (-.35, 0, -2)],
                     [(0, 0, -3), (.4, 0, -3), (-.35, 0, -2)]],
            radii=[[.6, .6, .35], [.6, .6, .35]],
            colors=[(1, 0, 0)] * 3, counts=[3, 2],
            instance_ids=[7, 7, 8], class_ids=[10, 10, 11]),
            terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(),
                            phase_zs=(), dz=(), dz_growth=(), colors=()))
        result = render_scene(64, 48, scene=scene, return_occlusion=True)
        front, alone = result.amodal
        self.assertEqual(front.instance_ids.tolist(), [7, 8])
        self.assertGreater(float(front.occlusion_fraction[0]), 0)
        self.assertLess(float(front.occlusion_fraction[0]), 1)
        self.assertEqual(alone.instance_ids.tolist(), [7])
        self.assertEqual(float(alone.occlusion_fraction[0]), 0)
        self.assertTrue(torch.equal(front.amodal_masks[0], alone.amodal_masks[0]))
        self.assertEqual(front.occlusion_matrix.tolist(), [[False, False], [True, False]])

    def test_cuda_matches_isolated_render_oracle(self):
        from occlusion_reference import reference_occlusion
        from synthetic_scene import RenderOptions
        for seed in (7, 44, 1234):
            scene = generate_random_scene(seed, batch_size=2, options=RandomSceneOptions(
                house_count=2, tree_count=2, cloud_count=1, car_count=1, person_count=2))
            options = RenderOptions(fov_degrees=57)
            actual = render_scene(97, 65, scene=scene, options=options, return_occlusion=True)
            expected = reference_occlusion(actual, scene, 97, 65, options)
            baseline = render_scene(97, 65, scene=scene, options=options, return_maps=True)
            self.assertTrue(torch.equal(actual.image, baseline.image))
            self.assertTrue(torch.equal(actual.instance_map, baseline.instance_map))
            self.assertTrue(torch.equal(actual.semantic_map, baseline.semantic_map))
            self.assertTrue(torch.equal(actual.visible_count, baseline.visible_count))
            for b, count in enumerate(actual.visible_count.tolist()):
                actual_pairs = sorted(zip(actual.visible_instance_ids[b, :count].tolist(),
                                          actual.visible_classes[b, :count].tolist()))
                baseline_pairs = sorted(zip(baseline.visible_instance_ids[b, :count].tolist(),
                                            baseline.visible_classes[b, :count].tolist()))
                self.assertEqual(actual_pairs, baseline_pairs)
                self.assertTrue((actual.visible_instance_ids[b, count:] == 0).all())
                self.assertTrue((actual.visible_classes[b, count:] == 0).all())
            for a, e in zip(actual.amodal, expected.amodal):
                for field in ('instance_ids', 'class_ids', 'amodal_masks', 'visible_masks',
                              'occlusion_matrix'):
                    self.assertTrue(torch.equal(getattr(a, field), getattr(e, field)),
                                    (seed, field, int((getattr(a, field) != getattr(e, field)).sum())))
                torch.testing.assert_close(a.occlusion_fraction, e.occlusion_fraction, equal_nan=True)

    def test_all_primitive_mask_words(self):
        from occlusion_reference import reference_occlusion
        from synthetic_scene import RenderOptions
        scene = generate_random_scene(1234, batch_size=1, options=RandomSceneOptions(
            house_count=64, tree_count=64, cloud_count=0, car_count=0, person_count=0))
        actual = render_scene(65, 49, scene=scene, return_occlusion=True)
        expected = reference_occlusion(actual, scene, 65, 49, RenderOptions())
        self.assertEqual(set(actual.amodal[0].instance_ids.tolist()),
                         set(actual.instance_map.unique().tolist()) - {0})
        self.assertTrue(torch.equal(actual.amodal[0].amodal_masks,
                                    expected.amodal[0].amodal_masks))

    def test_more_than_32_visible_instances(self):
        from occlusion_reference import reference_occlusion
        from synthetic_scene import RenderOptions
        centers = [(x * .225, y * .3, -3.) for y in range(-2, 3) for x in range(-4, 5)]
        # Two hidden parts exercise target columns and source rows beyond bit 31.
        centers += [(-1.8, -1.2, -6.), (.45, .6, -6.)]
        scene = Scene(spheres=Spheres(centers=centers, radii=[.06] * 45 + [.1, .1],
            colors=[(1, 0, 0)] * 47, instance_ids=list(range(1, 46)) + [45, 1]),
            terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(),
                            phase_zs=(), dz=(), dz_growth=(), colors=()))
        actual = render_scene(129, 97, scene=scene, return_occlusion=True)
        self.assertEqual(actual.visible_count.tolist(), [45])
        relationships = actual.amodal[0].occlusion_matrix
        self.assertTrue(relationships[0, 44])
        self.assertTrue(relationships[32, 0])
        self.assertEqual(int(relationships.sum()), 2)
        expected = reference_occlusion(actual, scene, 129, 97, RenderOptions())
        for field in ('amodal_masks', 'visible_masks', 'occlusion_matrix'):
            self.assertTrue(torch.equal(getattr(actual.amodal[0], field),
                                        getattr(expected.amodal[0], field)))
        torch.testing.assert_close(actual.amodal[0].occlusion_fraction,
                                   expected.amodal[0].occlusion_fraction)

    def test_terrain_only_and_camera_crossing(self):
        from occlusion_reference import reference_occlusion
        from synthetic_scene import RenderOptions
        scenes = [Scene(spheres=Spheres(centers=(), radii=(), colors=())),
                  Scene(spheres=Spheres(centers=[(0, 0, -.15), (.2, 0, -1), (0, 0, 2)],
                      radii=[.3, .25, .3], colors=[(1, 0, 0)] * 3,
                      instance_ids=[101, 203, 401]),
                      terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(),
                                      phase_zs=(), dz=(), dz_growth=(), colors=()))]
        for scene in scenes:
            actual = render_scene(65, 49, scene=scene, return_occlusion=True)
            expected = reference_occlusion(actual, scene, 65, 49, RenderOptions())
            for a, e in zip(actual.amodal, expected.amodal):
                self.assertTrue(torch.equal(a.amodal_masks, e.amodal_masks))
                torch.testing.assert_close(a.occlusion_fraction, e.occlusion_fraction, equal_nan=True)

    def test_hidden_composite_parts_and_empty_batch_item(self):
        scene = Scene(spheres=Spheres(
            centers=[[(0, 0, -2), (0, 0, -4), (.7, 0, -3)],
                     [(100, 0, -2), (100, 0, -4), (100, 0, -3)]],
            radii=[.5, .5, .25], colors=[(1, 0, 0)] * 3,
            instance_ids=[11, 22, 22]),
            terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(),
                            phase_zs=(), dz=(), dz_growth=(), colors=()))
        result = render_scene(65, 49, scene=scene, return_occlusion=True)
        gt, empty = result.amodal
        self.assertEqual(gt.instance_ids.tolist(), [11, 22])
        self.assertTrue(gt.amodal_masks[1, 24, 32])  # entirely hidden part of ID 22
        self.assertFalse(gt.visible_masks[1, 24, 32])
        self.assertEqual(gt.occlusion_matrix.tolist(), [[False, True], [False, False]])
        self.assertEqual(empty.occlusion_matrix.shape, (0, 0))
        self.assertEqual(empty.amodal_masks.shape, (0, 49, 65))
        self.assertEqual(empty.instance_ids.numel(), 0)
        self.assertEqual(result.visible_count.tolist(), [2, 0])
        self.assertTrue((result.visible_instance_ids[1] == 0).all())
        self.assertTrue((result.visible_classes[1] == 0).all())
        # Also exercise the native allocation path when every image is empty.
        offscreen = Scene(spheres=Spheres(centers=[(100, 0, -2)], radii=[.5],
            colors=[(1, 0, 0)]), terrain=scene.terrain)
        empty_result = render_scene(65, 49, scene=offscreen, return_occlusion=True)
        self.assertEqual(empty_result.amodal[0].amodal_masks.shape, (0, 49, 65))

    def test_pairwise_direction_can_differ_across_pixels(self):
        scene = Scene(spheres=Spheres(
            centers=[(-.3, 0, -2), (.3, 0, -4), (.3, 0, -2), (-.3, 0, -4)],
            radii=[.3, .45, .3, .45], colors=[(1, 0, 0)] * 4,
            instance_ids=[7, 7, 19, 19]),
            terrain=Terrain(base_heights=(), depth_limits=(), phase_xs=(),
                            phase_zs=(), dz=(), dz_growth=(), colors=()))
        annotations = render_scene(97, 65, scene=scene, return_occlusion=True).amodal[0]
        self.assertEqual(annotations.instance_ids.tolist(), [7, 19])
        self.assertEqual(annotations.occlusion_matrix.tolist(), [[False, True], [True, False]])

    def test_metadata_validation(self):
        for ids, classes, message in [([0, 1], [1, 1], 'positive instance'),
                                       ([7, 7], [1, 2], 'share a class')]:
            scene = Scene(spheres=Spheres(centers=[(0, 0, -2), (0, 0, -3)],
                radii=[.5, .5], colors=[(1, 0, 0)] * 2, instance_ids=ids, class_ids=classes))
            with self.assertRaisesRegex(RuntimeError, message):
                render_scene(32, 24, scene=scene, return_occlusion=True)

    def test_packed_batch(self):
        scene = generate_random_scene(44, batch_size=2, options=RandomSceneOptions(
            house_count=1, tree_count=1, cloud_count=1, car_count=1, person_count=1))
        result = render_scene(64, 48, scene=scene, return_occlusion=True)
        self.assertEqual(len(result.amodal), 2)
        for gt in result.amodal:
            self.assertGreater(len(gt.instance_ids), 0)
            self.assertTrue(gt.visible_masks.flatten(1).any(dim=1).all())
            self.assertFalse((gt.visible_masks & ~gt.amodal_masks).any())
            self.assertEqual(gt.occlusion_matrix.dtype, torch.bool)
            self.assertFalse(gt.occlusion_matrix.diagonal().any())
            hidden = gt.amodal_masks & ~gt.visible_masks
            expected_fraction = hidden.sum(dim=(1, 2)).float() / gt.amodal_masks.sum(dim=(1, 2))
            torch.testing.assert_close(gt.occlusion_fraction, expected_fraction)


if __name__ == '__main__':
    unittest.main()
