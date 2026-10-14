# -*- coding: utf-8 -*-
"""Tests for human_renderer procedural human sprites (no golden-image matching)."""
import math
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

import human_renderer as hr


def alpha_bbox(img):
    a = np.array(img)[:, :, 3]
    ys, xs = np.where(a > 0)
    return xs.min(), ys.min(), xs.max(), ys.max(), a


class TestSprites(unittest.TestCase):
    def test_sprite_rgba_transparent_nonempty(self):
        img, (ax, ay) = hr.render_sprite(height_px=300)
        self.assertEqual(img.mode, "RGBA")
        xs0, ys0, xs1, ys1, a = alpha_bbox(img)
        self.assertGreater((a > 0).sum(), 500)
        # background corners transparent
        self.assertEqual(a[0, 0], 0)
        self.assertEqual(a[-1, -1], 0)
        # anchor inside the image and near the bottom (ground/foot point)
        self.assertTrue(0 <= ax <= img.width)
        self.assertTrue(0 <= ay <= img.height)
        self.assertLess(abs(ay - ys1), 12)

    def test_repeat_byte_identical(self):
        a1, _ = hr.render_sprite(phase=1.0, person_id=2, heading=(0, -1), height_px=200)
        a2, _ = hr.render_sprite(phase=1.0, person_id=2, heading=(0, -1), height_px=200)
        self.assertEqual(a1.tobytes(), a2.tobytes())

    def test_walk_phases_differ(self):
        p1, _ = hr.render_sprite(phase=0.7, height_px=200)
        p2, _ = hr.render_sprite(phase=0.7 + math.pi, height_px=200)
        self.assertNotEqual(p1.tobytes(), p2.tobytes())

    def test_back_vs_left_profile_differ(self):
        back, _ = hr.render_sprite(heading=(0, -1), height_px=200)
        left, _ = hr.render_sprite(heading=(-1, 0), height_px=200)
        self.assertNotEqual(back.tobytes(), left.tobytes())

    def test_seated_shorter_than_standing(self):
        st, (ax_s, ay_st) = hr.render_sprite(state="seated", height_px=200)
        wk, (ax_w, ay_w) = hr.render_sprite(state="walking", height_px=200)
        _, ys0_seat, _, ys1_seat, _ = alpha_bbox(st)
        _, ys0_walk, _, ys1_walk, _ = alpha_bbox(wk)
        seat_h = ys1_seat - ys0_seat
        walk_h = ys1_walk - ys0_walk
        self.assertLess(seat_h, walk_h)

    def test_offering_progress_distinct(self):
        p0, _ = hr.render_sprite(state="offering", offer_progress=0.0, height_px=200)
        p1, _ = hr.render_sprite(state="offering", offer_progress=1.0, height_px=200)
        self.assertNotEqual(p0.tobytes(), p1.tobytes())

    def test_no_flower_primitive_when_not_holding(self):
        prims_f, _ = hr.build_person(state="walking", variant=0, yaw=0.0,
                                   phase=0.0, offer_progress=0.0, flower=True)
        prims_nf, _ = hr.build_person(state="walking", variant=0, yaw=0.0,
                                    phase=0.0, offer_progress=0.0, flower=False)
        self.assertTrue(any(p.mat == hr.MAT_FLOWER for p in prims_f))
        self.assertFalse(any(p.mat == hr.MAT_FLOWER for p in prims_nf))

    def test_variants_differ(self):
        imgs = [hr.render_sprite(person_id=i, height_px=160)[0].tobytes() for i in range(4)]
        self.assertEqual(len(set(imgs)), 4)

    def test_cache_hits(self):
        canvas = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
        before = hr.scaled_cache_info()
        for _ in range(5):
            hr.draw_human(canvas, 100, 100, 60, person_id=3, phase=2.5)
        after = hr.scaled_cache_info()
        self.assertLessEqual(after.misses, before.misses + 1)
        self.assertGreaterEqual(after.hits, before.hits + 4)

    def test_anchor_offsets_correctly(self):
        canvas = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
        hr.draw_human(canvas, 200, 200, 100, person_id=0, phase=0.0)
        a = np.array(canvas)[:, :, 3]
        self.assertGreater(a[140:230, 160:240].sum(), 0)
        ys, xs = np.where(a > 0)
        self.assertGreater(xs.min(), 140)
        self.assertLess(xs.max(), 270)
        # second person at another location doesn't move the first
        hr.draw_human(canvas, 500, 400, 100, person_id=1, phase=0.0)
        a2 = np.array(canvas)[:, :, 3]
        ys2, xs2 = np.where(a2 > 0)
        self.assertGreater(xs2.max(), 400)

    def test_sprite_not_cropped(self):
        img, _ = hr.render_sprite(height_px=300)
        xs0, ys0, xs1, ys1, a = alpha_bbox(img)
        self.assertGreaterEqual(xs0, 1)
        self.assertGreaterEqual(ys0, 1)
        self.assertLessEqual(xs1, img.width - 2)
        self.assertLessEqual(ys1, img.height - 2)

    def test_ground_shadow_present(self):
        img, (ax, ay) = hr.render_sprite(state="walking", height_px=200)
        a = np.array(img)[:, :, 3]
        # some semi-transparent shadow pixels near/below the anchor row
        band = a[int(ay) - 2:, :]
        self.assertGreater(((band > 0) & (band < 255)).sum(), 5)


class TestRendererMath(unittest.TestCase):
    def _mpp(self, camera, body_px):
        e = math.radians(hr._CAM_ELEV[camera])
        proj = hr._BODY_M * (math.cos(e) if camera != "topdown" else 1.0)
        return proj / body_px

    def test_depth_near_occludes_far(self):
        # (a) two same-projection ellipsoids: nearer along the view ray wins
        v, right, up = hr._camera_vecs("close")
        c = np.array([0.0, 0.0, 1.0])
        near = hr._ell(c - 0.20 * v, (0.15, 0.15, 0.15), hr.MAT_FLOWER)
        far = hr._ell(c + 0.20 * v, (0.15, 0.15, 0.15), hr.MAT_SKIN)
        # material list order must match MAT_* ids
        mats = [hr._SKIN_TONES[0], hr._HAIR_COLORS[0], hr._CLOTH_RGB, hr._CLOTH2_RGB,
                hr._SHOE_RGB, hr._CHAIR_RGB, hr._FLOWER_RGB, hr._DARK_RGB, hr._WOOD_RGB]
        body_px = 200
        img_both, (ax, ay) = hr._render([near, far], mats, "close", body_px=body_px)
        img_near, _ = hr._render([near], mats, "close", body_px=body_px)
        img_far, _ = hr._render([far], mats, "close", body_px=body_px)
        mpp = self._mpp("close", body_px)
        px = int(round(ax + (c @ right) / mpp))
        py = int(round(ay - (c @ up) / mpp))
        self.assertTrue(np.array_equal(np.array(img_both)[py, px], np.array(img_near)[py, px]))
        self.assertFalse(np.array_equal(np.array(img_both)[py, px], np.array(img_far)[py, px]))

    def test_oriented_limb_projects_along_axis(self):
        # (b) a limb from a to b must render along the projected a->b direction
        a, b = np.array([-0.2, -0.2, 0.7]), np.array([0.2, 0.2, 1.3])
        limb = hr._limb(a, b, 0.05, 0.05, hr.MAT_SKIN)
        mats = [hr._SKIN_TONES[0]] * 9
        img, (ax, ay) = hr._render([limb], mats, "close", body_px=200)
        al = np.array(img)[:, :, 3]
        ys, xs = np.where(al > 128)
        self.assertGreater(len(xs), 20)
        pts = np.stack([xs - ax, ay - ys], axis=1).astype(float)  # screen x right, y up
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        v, right, up = hr._camera_vecs("close")
        pa = np.array([a @ right, a @ up])
        pb = np.array([b @ right, b @ up])
        d = pb - pa
        d /= np.linalg.norm(d)
        self.assertGreater(abs(float(axis @ d)), 0.98)
        mid_px = 0.5 * (pa + pb) / self._mpp("close", 200)
        self.assertLess(np.linalg.norm(pts.mean(0) - mid_px), 15.0)

    def test_back_view_shows_hair_not_face(self):
        # (c) isolated head prims: front view shows more skin than back view
        def head_prims(heading):
            yaw = hr._heading_yaw(heading, "close")
            prims, _mats = hr.build_person("idle", 0, yaw, 0.0, 0.0, True)
            return [p for p in prims if p.c[2] > 1.55]
        mats = [hr._SKIN_TONES[0], hr._HAIR_COLORS[0], hr._CLOTH_RGB, hr._CLOTH2_RGB,
                hr._SHOE_RGB, hr._CHAIR_RGB, hr._FLOWER_RGB, hr._DARK_RGB, hr._WOOD_RGB]
        # heading (0,1) = walking toward camera on screen -> front; (0,-1) -> back
        front, _ = hr._render(head_prims((0.0, 1.0)), mats, "close", body_px=200)
        back, _ = hr._render(head_prims((0.0, -1.0)), mats, "close", body_px=200)
        def bright(img):
            a = np.array(img)
            m = a[:, :, 3] > 128
            return int(((a[:, :, 0] > 100) & m).sum())
        self.assertGreater(bright(front), bright(back) * 1.3)

    def test_stationary_phase_canonical(self):
        # (d) idle/offering ignore phase; walking does not
        i0, _ = hr.render_sprite(state="idle", phase=0.0, height_px=200)
        i1, _ = hr.render_sprite(state="idle", phase=2.3, height_px=200)
        self.assertEqual(i0.tobytes(), i1.tobytes())
        o0, _ = hr.render_sprite(state="offering", phase=0.0, offer_progress=0.5, height_px=200)
        o1, _ = hr.render_sprite(state="offering", phase=2.3, offer_progress=0.5, height_px=200)
        self.assertEqual(o0.tobytes(), o1.tobytes())

    def test_feet_grounded(self):
        # (e) shoe bottoms >= 0 and at least one stance foot at ~0
        for state, phases in (("idle", [0.0]), ("walking", [0.0, 0.7, 1.5]),
                              ("offering", [0.0])):
            for ph in phases:
                prims, _ = hr.build_person(state, 0, 0.0, ph, 0.5, True)
                bottoms = []
                for p in prims:
                    if p.mat == hr.MAT_SHOE:
                        bottoms.append(p.c[2] - float(np.linalg.norm(p.rot[2, :] * p.r)))
                self.assertTrue(all(b >= -1e-6 for b in bottoms), f"{state} ph{ph}: {bottoms}")
                self.assertLessEqual(min(bottoms), 0.01)

    def test_all_cameras_interior_bounds(self):
        # (f) every camera x 4 headings x states: opaque interior, anchor valid
        for cam in ("close", "iso", "topdown"):
            for hd in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                for st, op in (("walking", 0.0), ("seated", 0.0), ("offering", 1.0)):
                    img, (ax, ay) = hr.render_sprite(camera=cam, heading=hd, state=st,
                                                   offer_progress=op, height_px=100)
                    a = np.array(img)[:, :, 3]
                    self.assertGreater((a > 128).sum(), 30, f"{cam} {hd} {st}")
                    x0, y0, x1, y1, _ = alpha_bbox(img)
                    self.assertGreaterEqual(x0, 1)
                    self.assertGreaterEqual(y0, 1)
                    self.assertLessEqual(x1, img.width - 2)
                    self.assertLessEqual(y1, img.height - 2)
                    self.assertTrue(0 <= ax <= img.width and 0 <= ay <= img.height)

    def test_scaled_cache_hits(self):
        # (g) scaled sprite cache: first call misses, repeats hit, positions not keyed
        hr.clear_caches()
        canvas = Image.new("RGBA", (900, 700), (0, 0, 0, 0))
        info0 = hr.scaled_cache_info()
        hr.draw_human(canvas, 100, 300, 60, person_id=2)
        info1 = hr.scaled_cache_info()
        self.assertEqual(info1.misses, info0.misses + 1)
        hr.draw_human(canvas, 700, 500, 60, person_id=2)
        info2 = hr.scaled_cache_info()
        self.assertEqual(info2.hits, info1.hits + 1)
        hr.draw_human(canvas, 300, 400, 61, person_id=2)
        hr.draw_human(canvas, 320, 420, 61, person_id=2)
        info3 = hr.scaled_cache_info()
        self.assertEqual(info3.misses, info2.misses + 1)
        self.assertEqual(info3.hits, info2.hits + 1)


class TestRoutesUnaffected(unittest.TestCase):
    """Human rendering must not alter route/path/cohort outputs."""

    def setUp(self):
        import generate_video as gv
        gv.BASE_DIR = str(ROOT)
        gv.PIC_DIR = str(ROOT / "pic")
        self.gv = gv

    def test_cohort_positions_unchanged(self):
        s4 = self.gv.Scene4CloseUpOffering.__new__(self.gv.Scene4CloseUpOffering)
        s4.exit_start_x = [558.0, 600.0, 644.0, 688.0]
        s4.exit_speed = 106.25
        rows0 = [r for r in s4.exit_people_native(3.2) if r[0] == 0]
        self.assertEqual(len(rows0), 4)
        xs = [r[2] for r in s4.exit_people_native(3.6) if r[0] == 0]
        expected = [s - 106.25 * 0.4 for s in (558.0, 600.0, 644.0, 688.0)]
        for got, exp in zip(xs, expected):
            self.assertAlmostEqual(got, exp, places=6)

    def test_exit_rows_still_four(self):
        try:
            s1 = self.gv.Scene1Plan()
        except Exception:
            self.skipTest("assets unavailable")
        for d, pts in s1.sample_exit_rows(0.0):
            self.assertEqual(len(pts), 4)


if __name__ == "__main__":
    unittest.main()
