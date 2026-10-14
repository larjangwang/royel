import os
import math
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from human_renderer import draw_human

BASE_DIR = r"C:\xampp\htdocs\royal"
PIC_DIR = os.path.join(BASE_DIR, "pic")
FONT_REG = r"C:\Windows\Fonts\tahoma.ttf"
FONT_BOLD = r"C:\Windows\Fonts\tahomabd.ttf"

W, H = 1920, 1080
FPS = 30
# Scene lengths in seconds: S1 plan [0,10), S2 iso [10,18), S3 axial [18,28), S4 close [28,40)
SCENE_DURATIONS = (10.0, 8.0, 10.0, 12.0)
TOTAL_DURATION = sum(SCENE_DURATIONS)

def get_font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)

F_TITLE = get_font(34, True)
F_SUB = get_font(23, False)
F_SUB_B = get_font(24, True)
F_LABEL = get_font(20, True)
F_SMALL = get_font(17, False)
F_SMALL_B = get_font(18, True)
F_HUGE = get_font(42, True)
F_BADGE = get_font(21, True)

def fit_image_1080p(img_path, bg_color=(15, 23, 42), crop_box=None):
    im = Image.open(img_path).convert("RGB")
    if crop_box:
        im = im.crop(crop_box)
    iw, ih = im.size
    scale = min(W / iw, H / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (W, H), bg_color)
    ox = (W - nw) // 2
    oy = (H - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy

def fill_image_1080p(img_path):
    im = Image.open(img_path).convert("RGB")
    return im.resize((W, H), Image.Resampling.LANCZOS), W / im.width, H / im.height

class PolylinePath:
    def __init__(self, points):
        self.points = [np.array(p, dtype=float) for p in points]
        self.seg_lengths = []
        self.cum_lengths = [0.0]
        total = 0.0
        for i in range(len(self.points) - 1):
            d = float(np.linalg.norm(self.points[i+1] - self.points[i]))
            self.seg_lengths.append(d)
            total += d
            self.cum_lengths.append(total)
        self.total_length = max(total, 1e-6)

    def sample(self, dist):
        dist = max(0.0, min(dist, self.total_length))
        for i, seg_len in enumerate(self.seg_lengths):
            if dist <= self.cum_lengths[i+1] or i == len(self.seg_lengths) - 1:
                if seg_len < 1e-6:
                    t = 0.0
                else:
                    t = (dist - self.cum_lengths[i]) / seg_len
                pos = self.points[i] * (1.0 - t) + self.points[i+1] * t
                tangent = self.points[i+1] - self.points[i]
                norm = np.linalg.norm(tangent)
                if norm > 1e-6:
                    tangent = tangent / norm
                else:
                    tangent = np.array([1.0, 0.0])
                normal = np.array([-tangent[1], tangent[0]])
                return pos, tangent, normal
        return self.points[-1], np.array([1.0, 0.0]), np.array([0.0, 1.0])

def draw_red_arrow_polyline(draw, pts, width=6, color=(225, 29, 46, 220), arrow_spacing=75, phase_offset=0.0):
    if len(pts) < 2:
        return
    # Outer white/gold glow for contrast
    draw.line(pts, fill=(255, 255, 255, 190), width=width + 4, joint="curve")
    draw.line(pts, fill=color, width=width, joint="curve")
    path = PolylinePath(pts)
    d = (phase_offset % arrow_spacing)
    while d < path.total_length - 10:
        pos, tang, norm = path.sample(d)
        tip = pos + tang * (width * 1.45)
        left = pos - tang * (width * 1.0) + norm * (width * 1.15)
        right = pos - tang * (width * 1.0) - norm * (width * 1.15)
        draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=(255, 255, 255, 240), outline=color[:3] + (255,))
        tip2 = pos + tang * (width * 1.15)
        left2 = pos - tang * (width * 0.75) + norm * (width * 0.85)
        right2 = pos - tang * (width * 0.75) - norm * (width * 0.85)
        draw.polygon([tuple(tip2), tuple(left2), tuple(right2)], fill=color[:3] + (255,))
        d += arrow_spacing

def draw_badge(draw, x, y, text, bg=(18, 26, 48, 230), border=(250, 204, 21, 255), text_color=(255, 255, 255, 255), font=F_LABEL, pad_x=14, pad_y=8):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    rect = [x, y, x + tw + pad_x * 2, y + th + pad_y * 2]
    draw.rounded_rectangle(rect, radius=10, fill=bg, outline=border, width=2)
    draw.text((x + pad_x, y + pad_y - bbox[1]), text, font=font, fill=text_color)
    return rect[2] - rect[0], rect[3] - rect[1]

def draw_top_header(draw, title, subtitle, scene_badge, progress=0.0):
    # Top translucent dark bar
    draw.rectangle([0, 0, W, 88], fill=(12, 18, 34, 235))
    draw.line([(0, 88), (W, 88)], fill=(234, 179, 8, 255), width=3)
    # Gold accent bar on left
    draw.rectangle([24, 16, 32, 72], fill=(250, 204, 21, 255))
    draw.text((46, 13), title, font=F_TITLE, fill=(254, 240, 138, 255))
    draw.text((46, 52), subtitle, font=F_SUB, fill=(226, 232, 240, 255))
    # Scene badge on right
    bw, bh = draw_badge(draw, W - 430, 20, scene_badge, bg=(185, 28, 28, 240), border=(253, 224, 71, 255), font=F_SUB_B, pad_x=18, pad_y=10)
    # Bottom progress bar
    draw.rectangle([0, H - 8, W, H], fill=(30, 41, 59, 220))
    draw.rectangle([0, H - 8, int(W * progress), H], fill=(250, 204, 21, 255))

def draw_person_topdown(canvas, x, y, radius=5.5, holding_flower=True, is_seated=False,
                        highlight=False, heading=(0.0, -1.0), person_id=0, phase=0.0):
    # Top-down plan view person; (x, y) stays the ground foot anchor
    state = "seated" if is_seated else ("offering" if highlight else "walking")
    draw_human(canvas, x, y, radius * 4.4, heading=heading, phase=phase, state=state,
               person_id=person_id, holding_flower=holding_flower, camera="topdown",
               offer_progress=0.7 if highlight else 0.0)

def draw_person_3d_iso(canvas, x, y, scale=1.0, holding_flower=True, facing_dir=(1.0, -0.5),
                       bowing=False, person_id=0, phase=0.0, state=None):
    # Isometric person; heading follows the actual path tangent so bodies really turn
    st = state or ("offering" if bowing else "walking")
    draw_human(canvas, x, y, 28.0 * scale, heading=facing_dir, phase=phase, state=st,
               person_id=person_id, holding_flower=holding_flower, camera="iso",
               offer_progress=0.6 if bowing else 0.0)

# ==============================================================================
# SCENE 1: 2D Architectural Plan from PDF (plan_clean.png) + Red Arrow + 4-Abreast
# ==============================================================================
class Scene1Plan:
    def __init__(self):
        # Crop plan_clean.png (2339x1654) to focus on the ceremony ground while keeping full context
        # Crop box: (220, 120, 2120, 1190) -> 1900 x 1070 (~16:9 ratio!)
        self.crop_box = (240, 130, 2080, 1165)
        im = Image.open(os.path.join(PIC_DIR, "plan_clean.png")).convert("RGB")
        cropped = im.crop(self.crop_box)
        self.bg = cropped.resize((W, H), Image.Resampling.LANCZOS)
        self.sx = W / (self.crop_box[2] - self.crop_box[0])
        self.sy = H / (self.crop_box[3] - self.crop_box[1])

        def tr(px, py):
            return ((px - self.crop_box[0]) * self.sx, (py - self.crop_box[1]) * self.sy)
        self.tr = tr

        # Red arrow routes in original plan_clean.png coordinates:
        # 1) Entry from rear screening point (1760,435) up and along behind the tents
        self.entry_pts = [tr(x, y) for x, y in [
            (1760, 435), (1760, 295), (1465, 295), (1140, 295), (815, 295), (725, 295)
        ]]
        # Short branches dropping into the tent blocks (seated wait)
        self.entry_branches = [
            [tr(x, y) for x, y in [(815, 295), (815, 350)]],
            [tr(x, y) for x, y in [(1465, 295), (1465, 350)]],
        ]
        # Feeder along the tent fronts (below the front chairs at y=643) toward the left collection corner
        self.dome_right_feeder = [tr(x, y) for x, y in [
            (1640, 643), (1375, 643), (1140, 643), (970, 643), (815, 643)
        ]]
        self.feeder_path = PolylinePath(self.dome_right_feeder)
        # Main procession: prefix -> split 2+2 around the fountain (1140,763) -> suffix to ซุ้มถวายดอกไม้จันทน์
        self.prefix_pts = [tr(x, y) for x, y in [(815, 643), (815, 763), (1085, 763)]]
        self.upper_pts = [tr(x, y) for x, y in [
            (1085, 763), (1095, 738), (1115, 727), (1140, 722), (1165, 727), (1185, 738), (1195, 763)
        ]]
        self.lower_pts = [tr(x, y) for x, y in [
            (1085, 763), (1095, 788), (1115, 799), (1140, 804), (1165, 799), (1185, 788), (1195, 763)
        ]]
        self.suffix_pts = [tr(x, y) for x, y in [(1195, 763), (1552, 763)]]
        self.prefix_path = PolylinePath(self.prefix_pts)
        self.upper_path = PolylinePath(self.upper_pts)
        self.lower_path = PolylinePath(self.lower_pts)
        self.suffix_path = PolylinePath(self.suffix_pts)
        self.branch_len = max(self.upper_path.total_length, self.lower_path.total_length)
        self.main_total = self.prefix_path.total_length + self.branch_len + self.suffix_path.total_length
        # Lane-position continuity corrections at the branch boundaries (computed for 4 lanes)
        self._branch_start_fix, self._branch_end_fix = self._compute_branch_fixes(9.0, 4.0)
        # Exit route: LEFT of the pavilion, around the stage edge and along the front boundary to the left exit.
        # Left relative to facing the pavilion is north in the plan; do not step back onto approach.
        self.exit_pts = [tr(x, y) for x, y in [
            (1552, 763), (1552, 700), (1552, 643), (1675, 643), (1675, 1045),
            (540, 1045), (490, 1100), (490, 310), (380, 310)
        ]]
        self.exit_path = PolylinePath(self.exit_pts)

        # Chair blocks inside the 4 domes in plan_clean.png coordinates
        self.dome_columns_x = [
            660, 705, 750, 795,   # Dome 1 (Far Left)
            838, 882, 926, 970,   # Dome 2 (Mid Left)
            1308, 1352, 1396, 1440, # Dome 3 (Mid Right)
            1482, 1526, 1570, 1615  # Dome 4 (Far Right)
        ]
        self.dome_rows_y = list(range(350, 575, 18))

    def _sample_main(self, d, branch=None):
        # branch: "up" (lanes 0,1) or "low" (lanes 2,3) inside the fountain split section
        if d <= self.prefix_path.total_length:
            return self.prefix_path.sample(d)
        d -= self.prefix_path.total_length
        if d <= self.branch_len:
            path = self.upper_path if branch == "up" else self.lower_path
            return path.sample(d / self.branch_len * path.total_length)
        return self.suffix_path.sample(d - self.branch_len)

    def _compute_branch_fixes(self, lane_spacing, pair_lat):
        # Per-lane deltas that make lane positions continuous across the branch boundaries
        pre_pos, _, pre_n = self.prefix_path.sample(self.prefix_path.total_length)
        suf_pos, _, suf_n = self.suffix_path.sample(0.0)
        fixes_s, fixes_e = [], []
        for lane in range(4):
            path = self.upper_path if lane < 2 else self.lower_path
            lat = (-pair_lat if lane % 2 == 0 else pair_lat) * self.sx
            b0, _, n0 = path.sample(0.0)
            b1, _, n1 = path.sample(path.total_length)
            target_s = pre_pos + pre_n * ((lane - 1.5) * lane_spacing * self.sx)
            target_e = suf_pos + suf_n * ((lane - 1.5) * lane_spacing * self.sx)
            fixes_s.append(target_s - (b0 + n0 * lat))
            fixes_e.append(target_e - (b1 + n1 * lat))
        return fixes_s, fixes_e

    @staticmethod
    def _smoothstep(u):
        u = max(0.0, min(1.0, u))
        return u * u * (3.0 - 2.0 * u)

    def sample_4abreast_main(self, d, lane_idx):
        # lane_idx in {0, 1, 2, 3} -> lateral offsets for 4-abreast (แถวตอนเรียง 4)
        in_branch = self.prefix_path.total_length < d <= self.prefix_path.total_length + self.branch_len
        if in_branch:
            branch_d = d - self.prefix_path.total_length
            pos, tang, norm = self._sample_main(d, "up" if lane_idx < 2 else "low")
            lat = (-4.0 if lane_idx % 2 == 0 else 4.0) * self.sx
            pt = pos + norm * lat
            # Taper lane positions into/out of the split over 20 screen px so lanes don't snap sideways
            pt = pt + self._branch_start_fix[lane_idx] * (1.0 - self._smoothstep(branch_d / 20.0))
            pt = pt + self._branch_end_fix[lane_idx] * self._smoothstep(1.0 - (self.branch_len - branch_d) / 20.0)
            return pt
        pos, tang, norm = self._sample_main(d, "up")
        lat = (lane_idx - 1.5) * 9.0 * self.sx
        return pos + norm * lat

    def sample_exit_rows(self, t):
        # Illustrative steady-state queue of completed-offering rows spread over the whole exit loop
        count = max(1, int(self.exit_path.total_length / 50.0))
        rows = []
        for r in range(count):
            d = (r / count * self.exit_path.total_length + t * 58.0) % self.exit_path.total_length
            pos, tang, norm = self.exit_path.sample(d)
            pts = [pos + norm * ((lane - 1.5) * 4.5) for lane in range(4)]
            rows.append((d, pts))
        return rows

    def render(self, t, total_progress):
        # t in [0, 16] seconds
        frame = self.bg.copy().convert("RGBA")
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Highlight the 4 Domes with subtle golden/blue tint
        for (x1, y1, x2, y2, label) in [
            (640, 325, 995, 585, "เต็นท์โดมผู้ร่วมพิธี 1-2 (นั่งรอตั้งแถว)"),
            (1285, 325, 1640, 585, "เต็นท์โดมผู้ร่วมพิธี 3-4 (นั่งรอตั้งแถว)")
        ]:
            p1 = self.tr(x1, y1)
            p2 = self.tr(x2, y2)
            draw.rounded_rectangle([p1[0], p1[1], p2[0], p2[1]], radius=8,
                                   fill=(59, 130, 246, 38), outline=(37, 99, 235, 210), width=3)

        # Draw Red Arrow Paths (animated marching arrows)
        phase = t * 48.0
        draw_red_arrow_polyline(draw, self.entry_pts, width=5, color=(220, 38, 38, 210), arrow_spacing=70, phase_offset=phase)
        for br in self.entry_branches:
            draw_red_arrow_polyline(draw, br, width=4, color=(220, 38, 38, 200), arrow_spacing=60, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.dome_right_feeder, width=5, color=(220, 38, 38, 210), arrow_spacing=70, phase_offset=phase)
        # Main path split into prefix + curved fountain branches + suffix (no straight line over the fountain)
        draw_red_arrow_polyline(draw, self.prefix_pts, width=7, color=(220, 38, 38, 235), arrow_spacing=65, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.upper_pts, width=5, color=(220, 38, 38, 230), arrow_spacing=60, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.lower_pts, width=5, color=(220, 38, 38, 230), arrow_spacing=60, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.suffix_pts, width=7, color=(220, 38, 38, 235), arrow_spacing=65, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.exit_pts, width=6, color=(220, 38, 38, 215), arrow_spacing=75, phase_offset=phase)

        # Draw seated people inside the 4 domes; bottom rows gradually empty as they walk out!
        rows_walked_out = int(min(6, t * 0.45))
        for col_idx, cx in enumerate(self.dome_columns_x):
            for row_idx, ry in enumerate(self.dome_rows_y):
                # Bottom rows (highest ry) walk out first
                rev_idx = len(self.dome_rows_y) - 1 - row_idx
                if rev_idx < rows_walked_out:
                    continue
                for sub_i, sub_x in enumerate((-12, -4, 4, 12)):
                    px, py = self.tr(cx + sub_x, ry)
                    draw_person_topdown(overlay, px, py, radius=3.5, holding_flower=False, is_seated=True,
                                        person_id=col_idx * 100 + row_idx * 4 + sub_i)

        # Draw people walking along the rear entry path (จุดคัดกรอง -> เต็นท์โดม)
        entry_path = PolylinePath(self.entry_pts)
        for k in range(14):
            d = (k * 45.0 + t * 42.0) % entry_path.total_length
            pos, tang, norm = entry_path.sample(d)
            pid = 200 + k
            draw_person_topdown(overlay, pos[0], pos[1], radius=4.8, holding_flower=True,
                                heading=(tang[0], tang[1]), person_id=pid, phase=t * 7.5 + pid * 0.6)

        # Draw 4-abreast rows walking along the tent-front feeder toward the left collection corner
        for k in range(10):
            d = (k * 55.0 + t * 58.0) % self.feeder_path.total_length
            pos, tang, norm = self.feeder_path.sample(d)
            for lane in range(4):
                pt = pos + norm * ((lane - 1.5) * 7.0 * self.sx)
                pid = 300 + k * 4 + lane
                draw_person_topdown(overlay, pt[0], pt[1], radius=4.8, holding_flower=True,
                                    heading=(tang[0], tang[1]), person_id=pid, phase=t * 7.5 + pid * 0.6)

        # Draw rows of 4 people (แถวตอนเรียง 4) walking from the domes along the main walkway to ซุ้มถวายดอกไม้จันทน์!
        row_spacing = 24.0
        head_dist = min(self.main_total + 420.0, 140.0 + t * 72.0)
        num_rows = 36
        pavilion_pos = self.tr(1560, 763)

        for r in range(num_rows):
            d = head_dist - r * row_spacing
            if d < 0:
                continue
            if d <= self.main_total:
                # Walking in 4-abreast formation along the main path (2+2 around the fountain)
                is_front_offering = (self.main_total - d) < 18.0
                for lane in range(4):
                    pt = self.sample_4abreast_main(d, lane)
                    _, tang, _ = self._sample_main(d, "up" if lane < 2 else "low")
                    pid = 400 + r * 4 + lane
                    draw_person_topdown(overlay, pt[0], pt[1], radius=5.2, holding_flower=True,
                                        highlight=is_front_offering, heading=(tang[0], tang[1]),
                                        person_id=pid, phase=t * 7.5 + pid * 0.6)

        # Representative completed-offering queue covering the entire exit loop (stage edge -> front boundary -> left exit)
        for r_i, (d, pts) in enumerate(self.sample_exit_rows(t)):
            _, tang, _ = self.exit_path.sample(d)
            for lane, pt in enumerate(pts):
                pid = 600 + r_i * 4 + lane
                draw_person_topdown(overlay, pt[0], pt[1], radius=4.8, holding_flower=False,
                                    heading=(tang[0], tang[1]), person_id=pid, phase=t * 7.5 + pid * 0.6)

        # Glowing pulse at the ซุ้มถวายดอกไม้จันทน์ / พานพุ่ม when rows arrive
        pulse = 0.5 + 0.5 * math.sin(t * 6.0)
        px, py = self.tr(1560, 763)
        pr = int(22 + 8 * pulse)
        draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(250, 204, 21, 75), outline=(250, 204, 21, 240), width=3)

        # Callout labels on the plan
        sx_e, sy_e = self.tr(1760, 435)
        draw_badge(draw, int(sx_e - 90), int(sy_e + 14), "1. ทางเข้า / จุดคัดกรอง", bg=(22, 101, 52, 235), border=(134, 239, 172, 255))

        sx_d, sy_d = self.tr(815, 325)
        draw_badge(draw, int(sx_d - 160), int(sy_d - 48), "2. นั่งพักรอในเต็นท์โดม (4 หลัง)", bg=(30, 58, 138, 235), border=(147, 197, 253, 255))

        sx_w, sy_w = self.tr(950, 643)
        draw_badge(draw, int(sx_w - 135), int(sy_w + 60), "3. เดินออกเป็นแถวตอนเรียง 4", bg=(185, 28, 28, 240), border=(253, 224, 71, 255))

        sx_r, sy_r = self.tr(1140, 763)
        draw_badge(draw, int(sx_r - 135), int(sy_r + 55), "แยกอ้อมวงเวียนตามลูกศร แล้วรวมแถวเรียง 4", bg=(15, 23, 42, 230), border=(250, 204, 21, 255), font=F_SMALL_B)

        draw_badge(draw, 1400, 480, "4. ถวายดอกไม้จันทน์ที่พานพุ่ม\n   เสร็จแล้วเดินออกด้านซ้ายทันที", bg=(146, 64, 14, 245), border=(253, 224, 71, 255))

        sx_out, sy_out = self.tr(380, 310)
        draw_badge(draw, int(sx_out - 20), int(sy_out - 58), "5. ทางออก: ออกด้านซ้ายของซุ้ม วนเลียบขอบเวทีและแนวด้านหน้า", bg=(185, 28, 28, 240), border=(255, 255, 255, 255))

        # Zoomed-in Live Diagram Box (Bottom-Right) showing "แถวตอนเรียง 4 (4-Abreast Formation)" clearly!
        bx1, by1, bx2, by2 = W - 520, H - 310, W - 24, H - 24
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=14, fill=(15, 23, 42, 242), outline=(250, 204, 21, 255), width=3)
        draw.text((bx1 + 18, by1 + 12), "แถวตอนเรียง 4 (4-Abreast)", font=F_SUB_B, fill=(250, 204, 21, 255))
        draw.text((bx1 + 18, by1 + 42), "ประชาชนลุกจากเต็นท์โดม เดินเรียง 4 คนต่อแถว มุ่งหน้าสู่พานพุ่ม", font=F_SMALL, fill=(203, 213, 225, 255))
        # Draw visual walkway inside diagram box
        wy1, wy2 = by1 + 82, by2 - 22
        wx1, wx2 = bx1 + 24, bx2 - 115
        draw.rounded_rectangle([wx1, wy1, wx2, wy2], radius=8, fill=(51, 65, 85, 255), outline=(148, 163, 184, 255), width=2)
        # Draw 4 lane lines
        for lane in range(4):
            ly = wy1 + 26 + lane * 42
            draw.line([(wx1 + 10, ly), (wx2 - 10, ly)], fill=(100, 116, 139, 140), width=1)
            draw.text((wx1 + 8, ly - 10), f"แถว {lane+1}", font=F_SMALL_B, fill=(147, 197, 253, 255))
        # Draw animated rows of 4 moving right toward the 4 พานพุ่ม icons on the right of the box
        for r_idx in range(5):
            rx = wx1 + 80 + ((r_idx * 54 + int(t * 45)) % (wx2 - wx1 - 95))
            for lane in range(4):
                ly = wy1 + 26 + lane * 42
                pid = 800 + r_idx * 4 + lane
                draw_person_topdown(overlay, rx, ly, radius=8.5, holding_flower=True,
                                    highlight=(rx > wx2 - 55), heading=(1.0, 0.0),
                                    person_id=pid, phase=t * 7.5 + pid * 0.6)
        # Draw 4 Golden Trays (พานพุ่ม) at the right side of the mini-diagram
        draw.rounded_rectangle([wx2 + 10, wy1, bx2 - 16, wy2], radius=8, fill=(120, 53, 15, 220), outline=(250, 204, 21, 255), width=2)
        draw.text((wx2 + 18, wy1 + 8), "พานพุ่ม", font=F_SMALL_B, fill=(254, 240, 138, 255))
        for lane in range(4):
            ly = wy1 + 32 + lane * 38
            tx = wx2 + 48
            draw.ellipse([tx - 12, ly - 12, tx + 12, ly + 12], fill=(250, 204, 21, 255), outline=(255, 255, 255, 255), width=2)

        draw_top_header(
            draw,
            "วีดีโอจำลองแนวการเดินไปถวายดอกไม้จันทน์ (แถวตอนเรียง 4)",
            "อ้างอิงผังบริเวณงานใหม่ และจำลองแนวการเดินตามลูกศรสีแดงจากเต็นท์โดมสู่ซุ้มถวายดอกไม้จันทน์",
            "ส่วนที่ 1/4 : ผังรวมและแนวเดิน",
            total_progress
        )
        return Image.alpha_composite(frame, overlay).convert("RGB")


# ==============================================================================
# SCENE 2: 3D Isometric Overview (B-01.png) - Same camera as the new annotated route (walking_route_new.jpg)!
# ==============================================================================
class Scene2IsoOverview:
    def __init__(self):
        # B-01.png shares the exact camera/registration of the annotated route reference (pic/walking_route_new.jpg)
        self.bg, self.sx, self.sy = fill_image_1080p(os.path.join(PIC_DIR, "B-01.png"))
        def tr(x, y):
            return (x * self.sx, y * self.sy)
        self.tr = tr

        # All coordinates below are in B-01.png space (1280x720)

        # 1) Rear screening entry chain + short branches into the tent rows (seated wait)
        self.entry_pts = [tr(x, y) for x, y in [
            (711, 199), (640, 235), (558, 275), (441, 332), (329, 380)
        ]]
        self.entry_branches = [
            [tr(x, y) for x, y in [(558, 275), (587, 304)]],
            [tr(x, y) for x, y in [(329, 380), (363, 411)]],
        ]

        # 2) Feeder along the tent fronts travelling LEFT/DOWN into the collection corner
        self.feeder_pts = [tr(x, y) for x, y in [
            (620, 373), (560, 409), (487, 447), (408, 477)
        ]]
        self.feeder_path = PolylinePath(self.feeder_pts)

        # 3) Main route: prefix -> split 2+2 around the fountain (587,439) -> suffix to the offering trays
        self.prefix_pts = [tr(x, y) for x, y in [(408, 477), (455, 513), (561, 454)]]
        self.upper_pts = [tr(x, y) for x, y in [
            (561, 454), (558, 439), (568, 421), (586, 412), (605, 416), (613, 424)
        ]]
        self.lower_pts = [tr(x, y) for x, y in [
            (561, 454), (570, 464), (590, 466), (609, 452), (617, 435), (613, 424)
        ]]
        self.suffix_pts = [tr(x, y) for x, y in [(613, 424), (694, 377), (744, 343)]]
        self.prefix_path = PolylinePath(self.prefix_pts)
        self.upper_path = PolylinePath(self.upper_pts)
        self.lower_path = PolylinePath(self.lower_pts)
        self.suffix_path = PolylinePath(self.suffix_pts)
        # Branch section is sampled by normalized phase so rows stay synchronized through the split
        self.branch_len = max(self.upper_path.total_length, self.lower_path.total_length)
        self.main_total = self.prefix_path.total_length + self.branch_len + self.suffix_path.total_length
        # Lane-position continuity corrections at the branch boundaries (computed for 4 lanes)
        self._branch_start_fix, self._branch_end_fix = self._compute_branch_fixes(7.0, 3.5)

        # 4) Exit chain: LEFT/tent-side around the upper-left flowerbed, along the backdrop edge
        #    to the far right corner, back down-left along the front boundary, out the left exit
        self.exit_pts = [tr(x, y) for x, y in [
            (744, 343), (711, 325), (708, 317), (751, 300), (800, 336), (849, 372),
            (883, 399), (723, 498), (573, 590), (435, 675), (397, 699), (177, 429)
        ]]
        self.exit_path = PolylinePath(self.exit_pts)

    def _sample_main(self, d, branch=None):
        # branch: "up" (lanes 0,1) or "low" (lanes 2,3) inside the fountain split section
        if d <= self.prefix_path.total_length:
            return self.prefix_path.sample(d)
        d -= self.prefix_path.total_length
        if d <= self.branch_len:
            path = self.upper_path if branch == "up" else self.lower_path
            return path.sample(d / self.branch_len * path.total_length)
        return self.suffix_path.sample(d - self.branch_len)

    def _compute_branch_fixes(self, lane_spacing, pair_lat):
        # Per-lane deltas that make lane positions continuous across the branch boundaries
        pre_pos, _, pre_n = self.prefix_path.sample(self.prefix_path.total_length)
        suf_pos, _, suf_n = self.suffix_path.sample(0.0)
        fixes_s, fixes_e = [], []
        for lane in range(4):
            path = self.upper_path if lane < 2 else self.lower_path
            lat = (-pair_lat if lane % 2 == 0 else pair_lat) * self.sx
            b0, _, n0 = path.sample(0.0)
            b1, _, n1 = path.sample(path.total_length)
            target_s = pre_pos + pre_n * ((lane - 1.5) * lane_spacing * self.sx)
            target_e = suf_pos + suf_n * ((lane - 1.5) * lane_spacing * self.sx)
            fixes_s.append(target_s - (b0 + n0 * lat))
            fixes_e.append(target_e - (b1 + n1 * lat))
        return fixes_s, fixes_e

    @staticmethod
    def _smoothstep(u):
        u = max(0.0, min(1.0, u))
        return u * u * (3.0 - 2.0 * u)

    def sample_4abreast(self, d, lane_idx):
        # 7px native lane spacing outside the split; +/-3.5 native around each branch center
        in_branch = self.prefix_path.total_length < d <= self.prefix_path.total_length + self.branch_len
        if in_branch:
            branch_d = d - self.prefix_path.total_length
            pos, tang, norm = self._sample_main(d, "up" if lane_idx < 2 else "low")
            lat = (-3.5 if lane_idx % 2 == 0 else 3.5) * self.sx
            pt = pos + norm * lat
            # Taper lane positions into/out of the split over 20 screen px so lanes don't snap sideways
            pt = pt + self._branch_start_fix[lane_idx] * (1.0 - self._smoothstep(branch_d / 20.0))
            pt = pt + self._branch_end_fix[lane_idx] * self._smoothstep(1.0 - (self.branch_len - branch_d) / 20.0)
            return pt, tang
        pos, tang, norm = self._sample_main(d, "up")
        lat = (lane_idx - 1.5) * 7.0 * self.sx
        return pos + norm * lat, tang

    def sample_exit_rows(self, t):
        # Illustrative steady-state queue of completed-offering rows spread over the whole exit loop
        count = max(1, int(self.exit_path.total_length / 50.0))
        rows = []
        for r in range(count):
            d = (r / count * self.exit_path.total_length + t * 58.0) % self.exit_path.total_length
            pos, tang, norm = self.exit_path.sample(d)
            pts = [(pos + norm * ((lane - 1.5) * 4.5), tang) for lane in range(4)]
            rows.append((d, pts))
        return rows

    def render(self, t, total_progress):
        # t in [0, 14] seconds
        frame = self.bg.copy().convert("RGBA")
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        phase = t * 52.0
        draw_red_arrow_polyline(draw, self.entry_pts, width=6, color=(220, 38, 38, 215), arrow_spacing=75, phase_offset=phase)
        for br in self.entry_branches:
            draw_red_arrow_polyline(draw, br, width=4, color=(220, 38, 38, 200), arrow_spacing=60, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.feeder_pts, width=6, color=(220, 38, 38, 215), arrow_spacing=70, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.prefix_pts, width=8, color=(220, 38, 38, 240), arrow_spacing=65, phase_offset=phase)
        # Only the curved branches get arrows at the fountain - no straight line through it
        draw_red_arrow_polyline(draw, self.upper_pts, width=6, color=(220, 38, 38, 235), arrow_spacing=60, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.lower_pts, width=6, color=(220, 38, 38, 235), arrow_spacing=60, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.suffix_pts, width=8, color=(220, 38, 38, 240), arrow_spacing=65, phase_offset=phase)
        draw_red_arrow_polyline(draw, self.exit_pts, width=6, color=(220, 38, 38, 220), arrow_spacing=75, phase_offset=phase)

        # Draw seated people clusters at the tent fronts (near the entry branches and feeder start)
        for g_i, (dx, dy) in enumerate([(600, 318), (640, 340), (360, 415), (400, 432), (555, 355), (660, 285)]):
            bx, by = self.tr(dx, dy)
            for r in range(3):
                for c in range(4):
                    draw_person_3d_iso(overlay, bx + c*11 - r*8, by + r*9 + c*5, scale=0.72,
                                       holding_flower=True, state="seated",
                                       person_id=g_i * 20 + r * 4 + c)

        # Collect all walkers for a far-to-near depth sort (overlapping bodies composite correctly)
        people = []

        # Feeder rows: 4-abreast walking along the tent fronts toward the left collection corner
        for k in range(12):
            d = (k * 40.0 + t * 58.0) % self.feeder_path.total_length
            pos, tang, norm = self.feeder_path.sample(d)
            for lane in range(4):
                pt = pos + norm * ((lane - 1.5) * 7.0 * self.sx)
                pid = 200 + k * 4 + lane
                people.append((pt[1], pt[0], 0.9, True, (tang[0], tang[1]), False, pid))

        # Draw 4-abreast procession: prefix -> 2+2 around fountain -> offering -> exit chain
        row_spacing = 23.0
        total_cycle = self.main_total + self.exit_path.total_length
        for r in range(48):
            d = (r * row_spacing + t * 64.0) % total_cycle
            if d <= self.main_total:
                near_offering = (self.main_total - d) < 22.0
                for lane in range(4):
                    pt, tang = self.sample_4abreast(d, lane)
                    pid = 100 + r * 4 + lane
                    people.append((pt[1], pt[0], 0.95, True, (tang[0], tang[1]), near_offering, pid))

        # Representative completed-offering queue covering the entire exit loop
        # (left of shrine -> backdrop edge -> front boundary -> left exit)
        for r_i, (d, pts) in enumerate(self.sample_exit_rows(t)):
            for lane, (pt, tang) in enumerate(pts):
                pid = 300 + r_i * 4 + lane
                people.append((pt[1], pt[0], 0.88, False, (tang[0], tang[1]), False, pid))

        people.sort(key=lambda p: p[0])
        for py_f, px_f, sc, flower, facing, bowing, pid in people:
            draw_person_3d_iso(overlay, px_f, py_f, scale=sc, holding_flower=flower,
                               facing_dir=facing, bowing=bowing,
                               person_id=pid, phase=t * 7.5 + pid * 0.6)

        # Golden glow at the ซุ้มถวายดอกไม้จันทน์ offering point (744, 343)
        px, py = self.tr(744, 343)
        pulse = 0.5 + 0.5 * math.sin(t * 6.0)
        r_glow = int(26 + 8 * pulse)
        draw.ellipse([px - r_glow, py - r_glow, px + r_glow, py + r_glow],
                     fill=(250, 204, 21, 80), outline=(253, 224, 71, 245), width=3)

        # Callout Badges on 3D Isometric View (placed away from route corners)
        p_in = self.tr(711, 199)
        draw_badge(draw, int(p_in[0] - 330), int(p_in[1] - 48), "1. เข้าจุดคัดกรองด้านหลังเต็นท์", bg=(22, 101, 52, 235), border=(134, 239, 172, 255))

        draw_badge(draw, 50, 560, "2. นั่งรอในเต็นท์โดม\n   ออกมารวมแถวทางซ้าย", bg=(30, 58, 138, 240), border=(147, 197, 253, 255))

        draw_badge(draw, 1130, 655, "3. แยกอ้อมวงเวียน ซ้าย 2 / ขวา 2\n   แล้วรวมแถวตอนเรียง 4", bg=(185, 28, 28, 240), border=(253, 224, 71, 255))

        draw_badge(draw, 1400, 370, "4. ถวายดอกไม้จันทน์ที่พานพุ่ม\n   หน้าซุ้มฯ แล้วออกด้านซ้าย", bg=(146, 64, 14, 245), border=(253, 224, 71, 255))

        p_ex = self.tr(700, 560)
        draw_badge(draw, int(p_ex[0] - 60), int(p_ex[1] + 40), "5. วนเลียบขอบเวทีและแนวด้านหน้า ไปทางออกซ้าย", bg=(15, 23, 42, 235), border=(248, 113, 113, 255))

        draw_top_header(
            draw,
            "มุมมอง 3 มิติ ภาพรวมบริเวณงาน (3D Isometric Overview)",
            "จำลองการเคลื่อนแถวตอนเรียง 4 จากเต็นท์โดมผู้ร่วมพิธี สู่ซุ้มถวายดอกไม้จันทน์ และแนวเดินออกตามลูกศรสีแดง",
            "ส่วนที่ 2/4 : มุมสูง 3 มิติ",
            total_progress
        )
        return Image.alpha_composite(frame, overlay).convert("RGB")


# ==============================================================================
# SCENE 3: 3D Axial Walkway View (B-13.png) - Domes -> Central Walkway -> Pavilion
# ==============================================================================
class Scene3AxialView:
    def __init__(self):
        self.bg, self.sx, self.sy = fill_image_1080p(os.path.join(PIC_DIR, "B-13.png"))
        def tr(x, y):
            return (x * self.sx, y * self.sy)
        self.tr = tr
        # Tent-front feeder toward the near/bottom-left collection corner (turn is off-camera)
        self.feeder_native = [(430, 315), (390, 410), (345, 535), (300, 660), (276, 715)]
        # Full exit chain: initial LEFT past the upper-left flowerbed, then behind the
        # shrine along the stage front and DOWN the right white walkway toward camera
        self.exit_native = [(638, 272), (535, 272), (500, 255), (510, 242),
                            (750, 242), (774, 280), (820, 350), (882, 450),
                            (918, 500), (983, 600), (1060, 715)]
        self.exit_path = PolylinePath([self.tr(x, y) for x, y in self.exit_native])

    def sample_exit_rows(self, t):
        # Completed-offering walkers returning 4-abreast down the right corridor,
        # populated over the whole loop from t=0 (illustrative, not a count model)
        count = max(16, int(self.exit_path.total_length / 55.0))
        rows = []
        for r in range(count):
            d = (r / count * self.exit_path.total_length + t * 58.0) % self.exit_path.total_length
            pos, tang, norm = self.exit_path.sample(d)
            sc = 0.65 + (pos[1] / H) * 0.55
            pts = [(pos + norm * ((lane - 1.5) * 7.0 * sc), tang) for lane in range(4)]
            rows.append((d, pts))
        return rows

    def render(self, t, total_progress):
        # t in [0, 14] seconds
        frame = self.bg.copy().convert("RGBA")
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        phase = t * 55.0
        # Feeder arrow along the tent fronts toward the near/bottom-left collection corner
        # (the corner itself and the turn into the central path are outside this camera)
        feeder_pts = [self.tr(x, y) for x, y in self.feeder_native]
        draw_red_arrow_polyline(draw, feeder_pts, width=7, color=(220, 38, 38, 225), arrow_spacing=75, phase_offset=phase)

        # Central walkway red arrows (left & right around roundabout at (638, 468))
        left_walk = [self.tr(x, y) for x, y in [
            (612, 715), (615, 522), (562, 492), (554, 464), (580, 434), (622, 418), (626, 272)
        ]]
        right_walk = [self.tr(x, y) for x, y in [
            (656, 715), (656, 522), (710, 492), (716, 464), (690, 434), (648, 418), (646, 272)
        ]]
        # Exit: LEFT of shrine, behind it, then down the right white walkway toward camera
        exit_full = [self.tr(x, y) for x, y in self.exit_native]
        draw_red_arrow_polyline(draw, left_walk, width=6, color=(220, 38, 38, 230), arrow_spacing=70, phase_offset=phase)
        draw_red_arrow_polyline(draw, right_walk, width=6, color=(220, 38, 38, 230), arrow_spacing=70, phase_offset=phase)
        draw_red_arrow_polyline(draw, exit_full, width=6, color=(220, 38, 38, 225), arrow_spacing=75, phase_offset=phase)

        # Seated people under the left domes (x=40..330, y=380..505)
        for r in range(5):
            for c in range(7):
                px, py = self.tr(65 + c * 34 - r * 12, 395 + r * 22)
                draw_person_3d_iso(overlay, px, py, scale=0.85 + r * 0.05, holding_flower=True,
                                   facing_dir=(0.6, 0.5), state="seated", person_id=r * 7 + c)

        # Collect walkers for a far-to-near depth sort so overlapping bodies composite correctly
        people = []

        # People walking 4-abreast along the tent-front feeder path
        feeder_path = PolylinePath(feeder_pts)
        for k in range(8):
            d = (k * 65.0 + t * 62.0) % feeder_path.total_length
            pos, tang, norm = feeder_path.sample(d)
            for lane in range(4):
                pt = pos + norm * ((lane - 1.5) * 8.5)
                sc = 0.95 + (pt[1] / H) * 0.45
                pid = 200 + k * 4 + lane
                people.append((pt[1], pt[0], sc, True, (tang[0], tang[1]), False, pid))

        # People marching up the central walkway in 4-abreast formation (แถวตอนเรียง 4)
        # Follow left_walk for lanes 0,1 and right_walk for lanes 2,3 so no one steps on the roundabout flowerbed!
        left_path = PolylinePath(left_walk)
        right_path = PolylinePath(right_walk)
        num_rows = 18
        for r in range(num_rows):
            u = ((r / num_rows) + t * 0.065) % 1.0
            s = u ** 0.78
            sc = 1.35 * (1.0 - s * 0.55)
            near_top = (s > 0.93)
            for lane in range(4):
                if lane in (0, 1):
                    pos, tang, norm = left_path.sample(s * left_path.total_length)
                    offset = (-8.5 if lane == 0 else 8.5) * (1.0 - s * 0.45)
                else:
                    pos, tang, norm = right_path.sample(s * right_path.total_length)
                    offset = (-8.5 if lane == 2 else 8.5) * (1.0 - s * 0.45)
                pt = pos + norm * offset
                pid = 300 + r * 4 + lane
                # Bodies turn with the actual walkway tangent (was fixed north before)
                people.append((pt[1], pt[0], sc, True, (tang[0], tang[1]), near_top, pid))

        # Completed-offering walkers exiting LEFT of the shrine then returning DOWN
        # the right white corridor (full chain populated from t=0)
        for r_i, (d, pts) in enumerate(self.sample_exit_rows(t)):
            sc = 0.65 + (pts[0][0][1] / H) * 0.55
            for lane, (pt, tang) in enumerate(pts):
                pid = 400 + r_i * 4 + lane
                people.append((pt[1], pt[0], sc, False, (tang[0], tang[1]), False, pid))

        people.sort(key=lambda p: p[0])
        for py_f, px_f, sc, flower, facing, bowing, pid in people:
            draw_person_3d_iso(overlay, px_f, py_f, scale=sc, holding_flower=flower,
                               facing_dir=facing, bowing=bowing,
                               person_id=pid, phase=t * 7.5 + pid * 0.6)

        # Foreground occlusion seam: the shrine monument itself stays in front of
        # the behind-shrine return segment — clear overlay (people+arrows) inside
        # the monument silhouette so the original B-13 statue/steps show through
        draw.polygon([self.tr(x, y) for x, y in [
            (624, 178), (648, 178), (666, 213), (667, 246),
            (676, 268), (604, 268), (612, 245), (612, 213)
        ]], fill=(0, 0, 0, 0))

        # Callout badges
        b1 = self.tr(180, 355)
        draw_badge(draw, int(b1[0] - 110), int(b1[1] - 40), "1. ประชาชนนั่งรอในเต็นท์โดม", bg=(30, 58, 138, 240), border=(147, 197, 253, 255))

        draw_badge(draw, 24, 720, "2. รวมแถวด้านซ้ายของเต็นท์\n   จุดเลี้ยวอยู่นอกมุมภาพ\n   เดินแถวตอนเรียง 4", bg=(185, 28, 28, 240), border=(253, 224, 71, 255))

        draw_badge(draw, 1470, 710, "3. อ้อมวงเวียน 2+2\n   แล้วรวมแถวเรียง 4", bg=(15, 23, 42, 235), border=(250, 204, 21, 255))

        b4 = self.tr(640, 220)
        draw_badge(draw, int(b4[0] - 155), int(b4[1] - 95), "4. วางดอกไม้จันทน์ที่พานพุ่มหน้าซุ้มฯ (ครั้งละ 4 คน)", bg=(146, 64, 14, 245), border=(253, 224, 71, 255))

        draw_badge(draw, 1260, 200, "5. ออกด้านซ้าย แล้ววนกลับฝั่งขวา\n   เดินกลับเป็นแถวตอนเรียง 4", bg=(22, 101, 52, 235), border=(134, 239, 172, 255))

        draw_badge(draw, 1500, 830, "เดินกลับเรียง 4 หลังถวายฯ", bg=(185, 28, 28, 240), border=(253, 224, 71, 255), font=F_SMALL_B)

        draw_top_header(
            draw,
            "มุมมอง 3 มิติ แนวแกนทางเดินกลาง (Axial Walkway Simulation)",
            "แสดงการเดินออกจากเต็นท์โดมด้านซ้าย ตั้งแถวตอนเรียง 4 เดินผ่านวงเวียนกลาง มุ่งตรงสู่พานพุ่มหน้าซุ้มฯ",
            "ส่วนที่ 3/4 : ทางเดินกลางเรียง 4",
            total_progress
        )
        return Image.alpha_composite(frame, overlay).convert("RGB")


# ==============================================================================
# SCENE 4: Close-Up Offering at ซุ้มถวายดอกไม้จันทน์ & พานพุ่ม (B-06.png + Inset)
# ==============================================================================
class Scene4CloseUpOffering:
    def __init__(self):
        self.bg, self.sx, self.sy = fill_image_1080p(os.path.join(PIC_DIR, "B-06.png"))
        def tr(x, y):
            return (x * self.sx, y * self.sy)
        self.tr = tr
        # Load reference photo 1791968512941.jpg for picture-in-picture inset of the ซุ้ม & พานพุ่ม
        ref_im = Image.open(os.path.join(PIC_DIR, "1791968512941.jpg")).convert("RGB")
        self.ref_inset = ref_im.resize((410, 290), Image.Resampling.LANCZOS)
        # Visible LEFT exit chain in B-06 coords (perimeter continuation is outside this close-up)
        self.exit_native = [(638, 438), (535, 438), (350, 438), (160, 438), (0, 438)]
        # Deterministic exited-walker model: cohort k departs at t = 4k + 3.2 (after offering sub-phase)
        self.exit_start_x = [558.0, 600.0, 644.0, 688.0]
        self.exit_speed = 106.25  # native px/sec

    def exit_people_native(self, t):
        # Returns (cohort_id, lane_idx, x_native, y_native) for every departed walker still on screen
        people = []
        for k in range(-2, int(t // 4.0) + 1):
            exit_start = 4.0 * k + 3.2
            if t < exit_start:
                continue
            dx = self.exit_speed * (t - exit_start)
            for lane in range(4):
                x = self.exit_start_x[lane] - dx
                if x >= -30:
                    people.append((k, lane, x, 438.0 + (lane - 1.5) * 2.0))
        return people

    def draw_close_person(self, canvas, x, y, scale, state="walking", walk_phase=0.0,
                          person_id=0, offer_progress=0.0):
        """
        state: 'walking' (back view walking toward pavilion),
               'offering' (bowing & extending arms to place sandalwood flower on พานพุ่ม),
               'exiting' (walking left toward exit), 'idle' (queued, standing)
        """
        heading = (-1.0, 0.0) if state == "exiting" else (0.0, -1.0)
        st = "walking" if state == "exiting" else state
        draw_human(canvas, x, y, 116.0 * scale, heading=heading, phase=walk_phase,
                   state=st, person_id=person_id, holding_flower=(state != "exiting"),
                   camera="close", offer_progress=offer_progress)

    def render(self, t, total_progress):
        # t in [0, 16] seconds
        # Each cycle of a 4-abreast row stepping up, bowing/placing flower, and turning left takes 4.0 seconds!
        frame = self.bg.copy().convert("RGBA")
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cycle_len = 4.0
        cycle_idx = int(t // cycle_len)
        cycle_u = (t % cycle_len) / cycle_len  # 0.0 -> 1.0 within current row's offering cycle
        # Sub-phases within each 4.0s cycle:
        # 0.00 .. 0.45: All rows march forward 1 row-step along the walkway
        # 0.45 .. 0.80: Front row of 4 pauses at the พานพุ่ม, bows & places sandalwood flowers (วางดอกไม้จันทน์ที่พานพุ่ม)
        # 0.80 .. 1.00: Front row turns left and walks out along the left walkway, next row prepares to advance

        if cycle_u < 0.45:
            advance = cycle_u / 0.45
            front_state = "walking"
        elif cycle_u < 0.80:
            advance = 1.0
            front_state = "offering"
        else:
            advance = 1.0
            front_state = "exiting"

        # Draw perspective red arrows on the stone walkway
        phase = t * 65.0
        for col_x_bot, col_x_top in [(335, 558), (518, 600), (695, 644), (875, 688)]:
            pts = [self.tr(col_x_bot, 715), self.tr(col_x_top, 438)]
            draw_red_arrow_polyline(draw, pts, width=5, color=(220, 38, 38, 185), arrow_spacing=85, phase_offset=phase)

        # Exit red arrow turning LEFT in front of the ซุ้มถวายดอกไม้จันทน์
        # (only the visible initial portion; the perimeter continuation is outside this close-up)
        exit_pts = [self.tr(x, y) for x, y in self.exit_native]
        draw_red_arrow_polyline(draw, exit_pts, width=7, color=(220, 38, 38, 230), arrow_spacing=75, phase_offset=phase)

        # Highlight the 4 Golden Offering Trays (พานพุ่มวางดอกไม้จันทน์) in front of the ซุ้มฯ
        tray_coords = [(558, 396), (600, 396), (644, 396), (688, 396)]
        for idx, (tx, ty) in enumerate(tray_coords):
            px, py = self.tr(tx, ty)
            # Pedestal & Golden Tray (พานพุ่มสีทองสำหรับวางดอกไม้จันทน์)
            glow_r = 22 if front_state == "offering" else 14
            draw.ellipse([px - glow_r, py - glow_r, px + glow_r, py + glow_r],
                         fill=(250, 204, 21, 145 if front_state == "offering" else 70))
            # Golden tray base & bowl
            draw.polygon([(px - 10, py + 12), (px + 10, py + 12), (px + 4, py + 2), (px - 4, py + 2)],
                         fill=(217, 119, 6, 255), outline=(254, 240, 138, 255))
            draw.ellipse([px - 16, py - 6, px + 16, py + 5],
                         fill=(250, 204, 21, 255), outline=(255, 255, 255, 240), width=2)
            # Accumulated sandalwood flowers on the tray (increases each cycle!)
            flowers_count = min(8, 2 + cycle_idx + (1 if front_state in ("offering", "exiting") else 0))
            for f_i in range(flowers_count):
                fx = px + (f_i - flowers_count / 2.0) * 3.2
                fy = py - 3.0 - (f_i % 2) * 2.5
                draw.ellipse([fx - 3, fy - 3, fx + 3, fy + 3], fill=(254, 249, 195, 255), outline=(180, 83, 9, 255))

        # Draw departed cohorts walking off to the LEFT along (y=438) - continuous across cycle boundaries
        for coh, lane, ex, ey in self.exit_people_native(t):
            px, py = self.tr(ex, ey)
            self.draw_close_person(overlay, px, py, scale=0.52, state="exiting",
                                   walk_phase=t * 9.0 + lane, person_id=coh * 4 + lane)

        # Draw the rows of 4 people (แถวตอนเรียง 4) from back to front (painter's algorithm: far y first, near y last!)
        # Row 0 is the front-most row at the offering line (y_orig = 438), Row 5 is at the bottom (y_orig = 700)
        num_rows = 6
        rows_to_draw = []
        for r in range(num_rows):
            # Effective row position (moves from r+1 -> r as advance goes 0 -> 1)
            eff_r = (r + 1.0 - advance) / num_rows  # 0.0 at front (y=438), 1.0 at bottom (y=715)
            rows_to_draw.append((r, eff_r))

        # Sort from smallest eff_r (furthest, near ซุ้ม) to largest eff_r (closest to camera)
        rows_to_draw.sort(key=lambda item: item[1])

        for r, eff_r in rows_to_draw:
            if r == 0 and front_state == "exiting":
                continue  # departing cohort is rendered by exit_people_native (no teleport/double-draw)
            # Perspective curve
            p_s = eff_r ** 1.15
            y_orig = 438.0 + p_s * (705.0 - 438.0)
            scale = 0.52 + p_s * 0.72
            cols_bot = [335.0, 518.0, 695.0, 875.0]
            cols_top = [558.0, 600.0, 644.0, 688.0]

            for lane in range(4):
                x_orig = cols_top[lane] * (1.0 - p_s) + cols_bot[lane] * p_s
                st = "walking"
                offer_prog = 0.0
                if r == 0:
                    if front_state == "offering":
                        st = "offering"
                        offer_prog = (cycle_u - 0.45) / 0.35
                        # Small cream sandalwood flower drifting from hands onto the tray
                        px_t, py_t = self.tr(cols_top[lane], 396)
                        px_h, py_h = self.tr(x_orig, y_orig)
                        offer_sub = offer_prog
                        fx = px_h * (1.0 - offer_sub) + px_t * offer_sub
                        fy = (py_h - 42*scale) * (1.0 - offer_sub) + py_t * offer_sub
                        for pet in range(5):
                            ang = pet * (2 * math.pi / 5) - math.pi / 2
                            pxf, pyf = fx + 2.2 * math.cos(ang), fy + 2.2 * math.sin(ang)
                            draw.ellipse([pxf - 1.4, pyf - 1.4, pxf + 1.4, pyf + 1.4],
                                         fill=(254, 244, 200, 255))
                        draw.ellipse([fx - 1.6, fy - 1.6, fx + 1.6, fy + 1.6],
                                     fill=(217, 119, 6, 255))
                    elif front_state == "exiting":
                        st = "exiting"
                elif front_state != "walking":
                    st = "idle"  # queued rows stand still while the front row offers/exits
                px, py = self.tr(x_orig, y_orig)
                walk_ph = (t * 8.5 + lane * 0.4 + r * 0.7) if (cycle_u < 0.45 or st == "exiting") else 0.0
                pid = (cycle_idx + r) * 4 + lane
                self.draw_close_person(overlay, px, py, scale=scale, state=st, walk_phase=walk_ph,
                                       person_id=pid, offer_progress=offer_prog)

        # Picture-in-Picture Reference Box (Top-Right) showing 1791968512941.jpg (แบบซุ้มถวายดอกไม้จันทน์และพานพุ่ม)
        ix, iy = W - 446, 108
        draw.rounded_rectangle([ix - 8, iy - 8, ix + 418, iy + 340], radius=12,
                               fill=(15, 23, 42, 245), outline=(250, 204, 21, 255), width=3)
        overlay.paste(self.ref_inset.convert("RGBA"), (ix, iy))
        draw.text((ix + 6, iy + 298), "แบบซุ้มถวายดอกไม้จันทน์ และจุดตั้งพานพุ่ม", font=F_SMALL_B, fill=(254, 240, 138, 255))

        # Dynamic Status Banner at Bottom-Left showing current step of the 4-abreast row
        status_labels = {
            "walking": ("ขั้นตอนที่ 1 : ก้าวเข้าสู่จุดถวายฯ (แถวตอนเรียง 4)", "ประชาชนแถวหน้า 4 คน ก้าวขึ้นพร้อมกันสู่หน้าพานพุ่มหน้าซุ้มถวายดอกไม้จันทน์", (30, 58, 138, 240)),
            "offering": ("ขั้นตอนที่ 2 : หยุดทำความเคารพและวางดอกไม้จันทน์ที่พานพุ่ม", "ประชาชนทั้ง 4 คนในแถว วางดอกไม้จันทน์บนพานพุ่มพร้อมกันอย่างเป็นระเบียบ", (146, 64, 14, 245)),
            "exiting": ("ขั้นตอนที่ 3 : ออกด้านซ้ายของซุ้ม แล้ววนไปตามขอบเวที", "เดินเลียบแนวด้านหน้าบริเวณงานไปยังทางออกซ้ายตามภาพใหม่", (22, 101, 52, 240)),
        }
        st_title, st_desc, st_bg = status_labels[front_state]
        draw.rounded_rectangle([28, H - 132, 860, H - 22], radius=14, fill=st_bg, outline=(253, 224, 71, 255), width=3)
        draw.text((48, H - 118), st_title, font=F_SUB_B, fill=(254, 240, 138, 255))
        draw.text((48, H - 78), st_desc, font=F_SUB, fill=(255, 255, 255, 255))

        # Callout on the offering trays
        pt_tray = self.tr(558, 396)
        draw_badge(draw, int(pt_tray[0] - 320), int(pt_tray[1] - 48), "จุดวางดอกไม้จันทน์บนพานพุ่ม (ครั้งละ 4 คน)",
                   bg=(146, 64, 14, 240), border=(253, 224, 71, 255))

        draw_badge(draw, 100, 740, "<- ออกด้านซ้ายของซุ้ม ตามลูกศรสีแดง",
                   bg=(185, 28, 28, 240), border=(255, 255, 255, 255))

        draw_top_header(
            draw,
            "จำลองการเดินขึ้นวางดอกไม้จันทน์ที่พานพุ่ม หน้าซุ้มถวายดอกไม้จันทน์",
            "แถวตอนเรียง 4 ก้าวขึ้นวางดอกไม้จันทน์พร้อมกันทีละแถว (แถวละ 4 คน) แล้วเลี้ยวซ้ายออกสู่ทางออก",
            "ส่วนที่ 4/4 : การวางที่พานพุ่ม",
            total_progress
        )
        return Image.alpha_composite(frame, overlay).convert("RGB")


def main():
    print("Initializing scenes...")
    s1 = Scene1Plan()
    s2 = Scene2IsoOverview()
    s3 = Scene3AxialView()
    s4 = Scene4CloseUpOffering()

    # Save 4 preview PNG frames first so we can visually inspect every scene!
    s1.render(6.0, 6.0 / 40.0).save(os.path.join(PIC_DIR, "_preview_s1.png"))
    s2.render(5.0, 15.0 / 40.0).save(os.path.join(PIC_DIR, "_preview_s2.png"))
    s3.render(5.0, 23.0 / 40.0).save(os.path.join(PIC_DIR, "_preview_s3.png"))
    s4.render(2.4, 30.4 / 40.0).save(os.path.join(PIC_DIR, "_preview_s4.png"))
    print("Saved preview frames _preview_s1..s4.png")

    # Duration per scene: 10s + 8s + 10s + 12s -> 40 seconds (1200 frames @ 30fps)
    d1, d2, d3, d4 = SCENE_DURATIONS
    total_dur = TOTAL_DURATION
    total_frames = int(total_dur * FPS)

    ffmpeg_exe = r"C:\Users\admin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
    out_mp4_1 = os.path.join(BASE_DIR, "simulation_video.mp4")
    out_mp4_2 = os.path.join(BASE_DIR, "วีดีโอจำลองการเดินถวายดอกไม้จันทน์.mp4")
    pending_mp4 = os.path.join(BASE_DIR, "simulation_video.pending.mp4")
    ffmpeg_log = os.path.join(BASE_DIR, ".devin", "route-review", "render-ffmpeg.log")
    os.makedirs(os.path.dirname(ffmpeg_log), exist_ok=True)

    cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{W}x{H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "20",
        "-movflags", "+faststart",
        pending_mp4
    ]

    print(f"Rendering {total_frames} frames ({total_dur}s @ {FPS}fps) to {pending_mp4}...")
    # stderr goes straight to a log file so the PIPE can't fill up and deadlock the render
    with open(ffmpeg_log, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log_f)
        try:
            for f_idx in range(total_frames):
                t = f_idx / FPS
                prog = t / total_dur
                if t < d1:
                    img = s1.render(t, prog)
                elif t < d1 + d2:
                    img = s2.render(t - d1, prog)
                elif t < d1 + d2 + d3:
                    img = s3.render(t - (d1 + d2), prog)
                else:
                    img = s4.render(t - (d1 + d2 + d3), prog)

                proc.stdin.write(img.tobytes())
                if f_idx % 150 == 0 or f_idx == total_frames - 1:
                    print(f"  Frame {f_idx+1}/{total_frames} ({prog*100:.1f}%)")

            proc.stdin.close()
            proc.wait()
        except BaseException:
            try:
                proc.stdin.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
            raise

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (rc={proc.returncode}), see {ffmpeg_log}")
    if os.path.getsize(pending_mp4) == 0:
        raise RuntimeError("FFmpeg produced an empty output file")

    # Only replace the published outputs after a successful encode
    os.replace(pending_mp4, out_mp4_1)
    shutil.copyfile(out_mp4_1, out_mp4_2)
    print("Successfully created MP4 videos:", os.path.getsize(out_mp4_1), "bytes")

if __name__ == "__main__":
    main()
