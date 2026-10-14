# -*- coding: utf-8 -*-
"""Regression tests for the walking routes in generate_video.py.

Portable: resolves the repo root from this file; if the repo lives somewhere other
than the renderer's hardcoded BASE_DIR, gv.BASE_DIR / gv.PIC_DIR are redirected so
backgrounds load from the cloned tree. Fonts still require Windows Tahoma.

Run:  python -X utf8 -m unittest discover -s tests -v
"""
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_video as gv  # noqa: E402

# Redirect asset root if the checkout is not at the hardcoded location
if not Path(gv.PIC_DIR).is_dir() or str(ROOT) != gv.BASE_DIR:
    gv.BASE_DIR = str(ROOT)
    gv.PIC_DIR = str(ROOT / "pic")


def _s1_native(s1, pt):
    return (pt[0] / s1.sx + s1.crop_box[0], pt[1] / s1.sy + s1.crop_box[1])


class TestScene1DirectLeftExit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1 = gv.Scene1Plan()

    def test_exit_chain_native(self):
        expected = [(1552, 763), (1552, 700), (1552, 643), (1675, 643), (1675, 1045),
                    (540, 1045), (490, 1100), (490, 310), (380, 310)]
        got = [_s1_native(self.s1, p) for p in self.s1.exit_pts]
        self.assertEqual(len(got), len(expected))
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g[0], e[0], delta=2.0)
            self.assertAlmostEqual(g[1], e[1], delta=2.0)

    def test_first_two_segments_north_only(self):
        # Sample the first two segments densely: x stays 1552, y decreases 763 -> 643
        s1 = self.s1
        seg_len = s1.exit_path.seg_lengths[0] + s1.exit_path.seg_lengths[1]
        prev_y = None
        for i in range(100):
            d = (i / 99.0) * seg_len
            pos, _, _ = s1.exit_path.sample(d)
            x, y = _s1_native(s1, pos)
            self.assertAlmostEqual(x, 1552.0, delta=2.0)
            if prev_y is not None:
                self.assertLessEqual(y, prev_y)
            prev_y = y
        self.assertAlmostEqual(prev_y, 643.0, delta=3.0)

    def test_exit_tangent_perpendicular_not_backward(self):
        # Approach tangent (+x) dot initial exit tangent (-y) must be ~0 (not backward)
        s1 = self.s1
        _, app_t, _ = s1.suffix_path.sample(s1.suffix_path.total_length)
        _, ex_t, _ = s1.exit_path.sample(0.0)
        dot = float(np.dot(app_t / np.linalg.norm(app_t), ex_t / np.linalg.norm(ex_t)))
        self.assertAlmostEqual(dot, 0.0, places=6)

    def test_no_footprint_back_into_approach(self):
        # On the northward departure leg (before reaching tent-front line y=643),
        # samples within +/-20 native px of the approach lane y=763 never go x<1530
        s1 = self.s1
        seg_len = s1.exit_path.seg_lengths[0] + s1.exit_path.seg_lengths[1]
        for i in range(200):
            d = (i / 199.0) * seg_len
            pos, _, _ = s1.exit_path.sample(d)
            x, y = _s1_native(s1, pos)
            if abs(y - 763) <= 20:
                self.assertGreaterEqual(x, 1530, f"exit backtracks into approach at ({x:.0f},{y:.0f})")

    def test_walker_footprints_in_direct_exit_band(self):
        # Actual 4-abreast walker footprints (pos + norm*offset) through the first
        # 20 native px of the northward exit band: all stay inside x=1552+/-10
        s1 = self.s1
        for i in range(100):
            d = (i / 99.0) * (20.0 * s1.sy)
            pos, _, norm = s1.exit_path.sample(d)
            for lane in range(4):
                pt = pos + norm * ((lane - 1.5) * 4.5)
                x, y = _s1_native(s1, pt)
                self.assertGreaterEqual(x, 1530, f"walker into approach at ({x:.0f},{y:.0f})")
                self.assertLessEqual(y, 783, f"walker beyond approach edge at ({x:.0f},{y:.0f})")
                self.assertLessEqual(abs(x - 1552), 10, f"walker outside direct exit band at ({x:.0f},{y:.0f})")

    def test_arrows_and_walkers_share_path(self):
        s1 = self.s1
        for pt, native in zip(s1.exit_pts,
                              [(1552, 763), (1552, 700), (1552, 643), (1675, 643), (1675, 1045),
                               (540, 1045), (490, 1100), (490, 310), (380, 310)]):
            x, y = _s1_native(s1, pt)
            self.assertAlmostEqual(x, native[0], delta=2.0)
            self.assertAlmostEqual(y, native[1], delta=2.0)
        # exit_path geometry equals the transformed exit_pts (same data drives arrows + people)
        self.assertEqual(len(s1.exit_path.points), len(s1.exit_pts))
        for p_pt, p_path in zip(s1.exit_pts, s1.exit_path.points):
            np.testing.assert_allclose(np.asarray(p_pt), p_path, atol=1e-6)
        for t in (0.0, 6.0, 15.9):
            rows = s1.sample_exit_rows(t)
            self.assertTrue(rows)
            for d, pts in rows:
                self.assertEqual(len(pts), 4)
                # each participant within a plausible lane width of the exit centerline
                pos, _, _ = s1.exit_path.sample(d)
                for pt in pts:
                    self.assertLess(float(np.linalg.norm(pt - pos)), 15.0)


class TestScene2FrozenExit(unittest.TestCase):
    def test_exit_chain_unchanged(self):
        s2 = gv.Scene2IsoOverview()
        expected = [(744, 343), (711, 325), (708, 317), (751, 300), (800, 336), (849, 372),
                    (883, 399), (723, 498), (573, 590), (435, 675), (397, 699), (177, 429)]
        self.assertEqual(len(s2.exit_pts), len(expected))
        for pt, (ex, ey) in zip(s2.exit_pts, expected):
            self.assertAlmostEqual(pt[0], ex * s2.sx, delta=1.0)
            self.assertAlmostEqual(pt[1], ey * s2.sy, delta=1.0)
        # exit starts exactly where the suffix ends
        self.assertEqual(s2.suffix_pts[-1], s2.exit_pts[0])


class TestScene3LeftExit(unittest.TestCase):
    FULL_CHAIN = [(638, 272), (535, 272), (500, 255), (510, 242),
                  (750, 242), (774, 280), (820, 350), (882, 450),
                  (918, 500), (983, 600), (1060, 715)]

    def test_first_segment_left_and_chain(self):
        s3 = gv.Scene3AxialView()
        self.assertEqual(s3.exit_native, self.FULL_CHAIN)
        # initial LEFT departure unchanged
        self.assertEqual(s3.exit_native[:4], [(638, 272), (535, 272), (500, 255), (510, 242)])
        x0, y0 = s3.exit_native[0]
        x1, y1 = s3.exit_native[1]
        self.assertLess(x1, x0)
        self.assertEqual(y0, y1)
        # return leg terminates at the right-corridor bottom
        self.assertEqual(s3.exit_native[-1], (1060, 715))
        # from (774,280) onward the chain runs down-right: y and x strictly increase
        i774 = self.FULL_CHAIN.index((774, 280))
        for i in range(i774, len(self.FULL_CHAIN) - 1):
            (xa, ya), (xb, yb) = self.FULL_CHAIN[i], self.FULL_CHAIN[i + 1]
            self.assertGreater(yb, ya, f"segment {i} not descending")
            self.assertGreater(xb, xa, f"segment {i} not rightward")

    def test_right_return_populated(self):
        s3 = gv.Scene3AxialView()
        for t in (0.0, 5.0, 13.9):
            rows = s3.sample_exit_rows(t)
            right = []
            for d, pts in rows:
                self.assertEqual(len(pts), 4)  # 4-abreast rows down the right corridor
                for pt, tang in pts:
                    nx, ny = pt[0] / s3.sx, pt[1] / s3.sy
                    if nx > 760 and ny > 300:
                        right.append((nx, ny, tang))
            self.assertGreaterEqual(len(right), 6,
                                    f"t={t}: only {len(right)} walkers in right-return zone")
            for nx, ny, tang in right:
                self.assertGreater(tang[1], 0, "right-return walker not heading down-camera")

    def test_exit_rows_all_four_abreast(self):
        s1 = gv.Scene1Plan()
        s2 = gv.Scene2IsoOverview()
        s3 = gv.Scene3AxialView()
        for s in (s1, s2, s3):
            for t in (0.0, 5.0, 7.9):
                for d, pts in s.sample_exit_rows(t):
                    self.assertEqual(len(pts), 4)
                    poss = [p[0] for p in pts] if isinstance(pts[0], tuple) else list(pts)
                    self.assertEqual(len({(round(p[0], 4), round(p[1], 4)) for p in poss}), 4)
        # S3 lane offsets are exactly (lane-1.5)*7*sc along the normal, none along-path
        for t in (0.0, 5.0):
            for d, pts in s3.sample_exit_rows(t):
                pos, tang, norm = s3.exit_path.sample(d)
                sc = 0.65 + (pos[1] / gv.H) * 0.55
                for lane, (pt, tg) in enumerate(pts):
                    off = pt - pos
                    self.assertAlmostEqual(float(off @ tg), 0.0, delta=1e-6)
                    self.assertAlmostEqual(float(off @ norm), (lane - 1.5) * 7.0 * sc, delta=1e-6)

    def test_render_calls_right_return_walkers(self):
        s3 = gv.Scene3AxialView()
        calls = []
        orig = gv.draw_person_3d_iso
        gv.draw_person_3d_iso = lambda *a, **k: calls.append((a, k))
        try:
            s3.render(5.0, 0.6)
        finally:
            gv.draw_person_3d_iso = orig
        # screen-space: right corridor is x>1140 output px, y>450, no flower, facing down
        right = [(a, k) for a, k in calls
                 if a[1] > 1140 and a[2] > 450
                 and k.get("holding_flower") is False
                 and k.get("facing_dir", (0, -1))[1] > 0]
        self.assertGreaterEqual(len(right), 6,
                                f"only {len(right)} right-return draw calls at t=5")
        # 4-abreast rows use distinct ids 400 + row*4 + lane
        pids = {k.get("person_id") for a, k in right}
        self.assertGreaterEqual(len(pids), 6)
        groups = {}
        for a, k in right:
            groups.setdefault(k.get("person_id") // 4, set()).add(k.get("person_id") % 4)
        self.assertTrue(any(len(lanes) == 4 for lanes in groups.values()),
                        f"no full 4-abreast row seen: {groups}")


class TestScene4LeftExitCohorts(unittest.TestCase):
    def test_cohort0_monotonic_left(self):
        s4 = gv.Scene4CloseUpOffering()
        for lane in range(4):
            xs = []
            for t in (3.2, 3.6, 4.001, 4.8):
                row = {c: (x, y) for c, l, x, y in s4.exit_people_native(t) if l == lane}
                self.assertIn(0, row, f"cohort0 lane{lane} missing at t={t}")
                xs.append(row[0][0])
            self.assertTrue(all(xs[i] > xs[i + 1] for i in range(len(xs) - 1)),
                            f"lane {lane} not monotonic left: {xs}")
        # all 4 lanes of cohort0 present through t=4.8
        self.assertEqual(len([p for p in s4.exit_people_native(4.8) if p[0] == 0]), 4)
        # walker band y matches the arrow chain y=438 (no backward toward camera)
        for _, _, x, y in s4.exit_people_native(4.0):
            self.assertTrue(435.0 <= y <= 441.0)
        self.assertTrue(all(p[1] == 438 for p in s4.exit_native))


class TestFortySecondTiming(unittest.TestCase):
    def test_scene_durations_and_frame_count(self):
        self.assertEqual(gv.SCENE_DURATIONS, (10.0, 8.0, 10.0, 12.0))
        self.assertEqual(gv.TOTAL_DURATION, 40.0)
        self.assertEqual(int(gv.TOTAL_DURATION * gv.FPS), 1200)
        src = inspect.getsource(gv.main)
        self.assertIn("SCENE_DURATIONS", src)
        self.assertIn("TOTAL_DURATION", src)

    def test_html_chapter_times(self):
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        for t in ("10.1", "18.1", "28.1", "31.2"):
            self.assertIn(f"jumpToChapter({t}", html)
        for thr in ("t < 10.0", "t < 18.0", "t < 28.0"):
            self.assertIn(thr, html)
        for lbl in ("0:10", "0:18", "0:28"):
            self.assertIn(lbl, html)
        for v in ("four-abreast-40s-20261014",):
            self.assertEqual(html.count(v), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
