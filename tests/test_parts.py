"""CUDA checks for modal primitive parts and their object membership."""
import unittest

import torch

from synthetic_scene import (Scene, Spheres, empty_scene, render_scene,
                             generate_random_scene, render_random_scene, RenderOptions)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PartTests(unittest.TestCase):
    def check_parts(self, result):
        for b in range(result.image.shape[0]):
            count = int(result.part_count[b])
            masks = result.part_masks[b]
            self.assertTrue(masks[:count].flatten(1).any(1).all())
            self.assertFalse(masks[count:].any())
            self.assertTrue((masks.sum(0) <= 1).all())
            self.assertTrue(torch.equal(masks.any(0), result.instance_map[b] > 0))
            self.assertTrue((result.part_belongs_to[b, :count].sum(1) == 1).all())
            self.assertFalse(result.part_belongs_to[b, count:].any())
            for p in range(count):
                parent = result.visible_instance_ids[b][result.part_belongs_to[b, p]]
                self.assertTrue((result.instance_map[b][masks[p]] == parent.item()).all())

    def test_occlusion_and_shared_parent(self):
        # Two separate visible spheres belong to one object; a third is hidden.
        scene = Scene(spheres=Spheres(
            centers=((-0.65, 0, -4), (0.65, 0, -4), (-0.65, 0, -7)),
            radii=(0.5, 0.5, 0.2), colors=((1, 0, 0),) * 3,
            instance_ids=(71, 71, 92), class_ids=(14, 14, 14)),
            terrain=empty_scene().terrain)
        result = render_scene(96, 64, scene=scene, return_parts=True)
        self.assertEqual(result.part_count.item(), 2)
        self.assertEqual(result.visible_count.item(), 1)
        self.check_parts(result)
        plain = render_scene(96, 64, scene=scene, return_maps=True)
        self.assertTrue(torch.equal(result.image, plain.image))
        self.assertIsNone(plain.part_masks)

    def test_packed_batch_and_empty(self):
        self.check_parts(render_scene(65, 49, scene=generate_random_scene(1234, batch_size=3),
                                     return_parts=True, options=RenderOptions(shadows=False)))
        scene = Scene(spheres=Spheres(centers=((100, 0, -4),)),
                      terrain=empty_scene().terrain)
        result = render_scene(16, 16, scene=scene, return_parts=True)
        self.assertEqual(result.part_masks.shape, (1, 0, 16, 16))
        self.assertEqual(result.part_count.item(), 0)
        self.assertFalse(result.part_map.any())

    def test_default_ids(self):
        self.check_parts(render_scene(32, 32, return_parts=True))

    def test_cuda_expansion_matches_tensor_reference(self):
        from synthetic_scene import _cuda_renderer
        for parts in (0, 1, 7, 8, 9, 257):
            raw = torch.randint(0, 258, (2, 17, 19), device='cuda', dtype=torch.int32)
            remap = torch.randint(0, parts + 1, (2, 258), device='cuda', dtype=torch.int32)
            part_map = torch.empty_like(raw)
            masks = torch.empty((2, parts, 17, 19), device='cuda', dtype=torch.bool)
            _cuda_renderer.expand_parts(raw, remap, part_map, masks)
            expected = remap.gather(1, raw.flatten(1).long()).reshape_as(raw)
            self.assertTrue(torch.equal(part_map, expected))
            reference = expected[:, None] == torch.arange(1, parts + 1, device='cuda')[None, :, None, None]
            self.assertTrue(torch.equal(masks, reference))

    def test_cuda_metadata_matches_reference(self):
        from synthetic_scene import _cuda_renderer
        for slots in (0, 1, 31, 32, 33, 257):
            for compact_default in (False, True):
                ids = torch.arange(1, slots + 1, device='cuda', dtype=torch.int32).flip(0).repeat(2, 1)
                if not compact_default:
                    ids = (ids % 7) * 100  # duplicate, sparse and unlabeled owners
                visibility = torch.full((2, 9), -1, dtype=torch.int32, device='cuda')
                visibility[1, ::2] = 0
                count, remap, owners = _cuda_renderer.prepare_parts(visibility, ids, compact_default)
                indices = torch.arange(slots, device='cuda')
                seen = (((visibility[:, indices // 32] >> (indices % 32)) & 1) > 0) & (ids > 0)
                rank = seen.cumsum(1, dtype=torch.int32)
                self.assertTrue(torch.equal(count, seen.sum(1).int()))
                self.assertFalse(remap[:, 0].any())
                self.assertTrue(torch.equal(remap[:, 1:], rank * seen))
                for b in range(2):
                    expected = ids[b, seen[b]]
                    if compact_default:
                        expected = (expected[:, None] >= expected[None, :]).sum(1).int()
                    self.assertTrue(torch.equal(owners[b, :count[b]], expected))
                    self.assertTrue((owners[b, count[b]:] == -1).all())

    def test_parts_and_amodal_together(self):
        # Covers packed metadata decoding, variable counts, shared parents,
        # default metadata normalization, and a batch with no visible geometry.
        custom = Scene(spheres=Spheres(
            centers=[[(0, 0, -3), (.4, 0, -3), (-.35, 0, -2)],
                     [(0, 0, -3), (.4, 0, -3), (-.35, 0, -2)]],
            radii=[[.6, .6, .35], [.6, .6, .35]],
            colors=[(1, 0, 0)] * 3, counts=[3, 2],
            instance_ids=[7, 7, 8], class_ids=[10, 10, 11]),
            terrain=empty_scene().terrain)
        offscreen = Scene(spheres=Spheres(centers=((100, 0, -4),)),
                          terrain=empty_scene().terrain)
        for scene in (generate_random_scene(44, batch_size=2), custom, Scene(), offscreen):
            together = render_scene(65, 49, scene=scene, return_parts=True, return_occlusion=True)
            parts_only = render_scene(65, 49, scene=scene, return_parts=True)
            amodal_only = render_scene(65, 49, scene=scene, return_occlusion=True)
            self.check_parts(together)
            for field in ('image', 'part_count', 'part_map', 'part_masks'):
                self.assertTrue(torch.equal(getattr(together, field), getattr(parts_only, field)), field)
            for field in ('instance_map', 'semantic_map', 'visible_count', 'visible_instance_ids'):
                self.assertTrue(torch.equal(getattr(together, field), getattr(amodal_only, field)), field)
            for b, (actual, expected) in enumerate(zip(together.amodal, amodal_only.amodal)):
                for field in ('instance_ids', 'class_ids', 'amodal_masks', 'visible_masks',
                              'occlusion_fraction', 'occlusion_matrix'):
                    self.assertTrue(torch.equal(getattr(actual, field), getattr(expected, field)), field)
                for i, instance in enumerate(actual.instance_ids):
                    column = (together.visible_instance_ids[b] == instance).nonzero().item()
                    children = together.part_belongs_to[b, :, column]
                    union = together.part_masks[b, children].any(0)
                    self.assertTrue(torch.equal(union, actual.visible_masks[i]))
                    self.assertFalse((union & ~actual.amodal_masks[i]).any())

    def test_random_scene_combined_flags(self):
        result = render_random_scene(44, 65, 49, batch_size=2,
                                     return_parts=True, return_occlusion=True)
        self.check_parts(result)
        self.assertEqual(len(result.amodal), 2)
