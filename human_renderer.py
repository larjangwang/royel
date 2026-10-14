# -*- coding: utf-8 -*-
"""Procedural shaded-3D human sprite renderer (numpy + Pillow only, no external assets).

Analytic orthographic ray/ellipsoid intersection with a per-pixel depth buffer,
Lambert key light + ambient + rim, articulated joints, 4 mourning-attire variants,
and quantized-pose sprite caching.

Public API:
    draw_human(canvas, x, y, height_px, *, heading, phase, state, person_id,
               holding_flower, camera, offer_progress)
        - composites a sprite onto an RGBA PIL image; (x, y) is the foot-center
          ground anchor and stays fixed. height_px is the standing body's
          projected height in pixels.
    render_sprite(...) -> (PIL.Image RGBA, (anchor_x, anchor_y))
        - un-cached render at arbitrary height (tests / contact sheets).
    build_person(...) -> (prims, materials)
        - exposes the primitive list for tests/metadata checks.
"""
import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- materials
MAT_SKIN = 0
MAT_HAIR = 1
MAT_CLOTH = 2
MAT_CLOTH2 = 3
MAT_SHOE = 4
MAT_CHAIR = 5
MAT_FLOWER = 6
MAT_DARK = 7  # eyes / brows
MAT_WOOD = 8  # flower stem

_SKIN_TONES = [(226, 186, 152), (206, 162, 126), (186, 136, 98), (236, 204, 172)]
_HAIR_COLORS = [(30, 25, 22), (58, 48, 42), (95, 93, 91), (34, 30, 28)]
_VARIANTS = [
    # (skin, hair, style, female_skirt, height_scale)
    (_SKIN_TONES[0], _HAIR_COLORS[0], "short", False, 1.00),
    (_SKIN_TONES[1], _HAIR_COLORS[1], "bob", True, 0.97),
    (_SKIN_TONES[2], _HAIR_COLORS[2], "bun", True, 1.03),
    (_SKIN_TONES[3], _HAIR_COLORS[3], "gray", False, 1.01),
]
_CLOTH_RGB = (40, 42, 52)
_CLOTH2_RGB = (33, 35, 44)
_SHOE_RGB = (30, 28, 33)
_CHAIR_RGB = (58, 61, 70)
_FLOWER_RGB = (252, 236, 192)
_DARK_RGB = (40, 34, 32)
_WOOD_RGB = (150, 96, 55)

_CAM_ELEV = {"close": 12.0, "iso": 35.0, "topdown": 85.0}
_HEADING_BINS = 16
_PHASE_BINS = 12
_OFFER_BINS = 8
_BODY_M = 1.75       # canonical adult height for the height_px scale mapping
_BODY_PX_BASE = 120  # base sprite standing-body height in pixels
_MARGIN_BOTTOM_M = 0.65
_MARGIN_TOP_M = 0.20


class _Prim:
    __slots__ = ("c", "r", "rot", "mat", "hair_mode")

    def __init__(self, c, r, rot, mat, hair_mode=0):
        self.c = np.asarray(c, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.rot = np.asarray(rot, dtype=float)  # columns = local axes in world
        self.mat = mat
        self.hair_mode = hair_mode  # 0 none, 1 scalp cap (post-shade), 2 bob shell (pre-depth)


def _basis_z(axis):
    """Orthonormal basis whose z column aligns with `axis` (for limb ellipsoids)."""
    z = np.asarray(axis, dtype=float)
    n = np.linalg.norm(z)
    if n < 1e-9:
        z = np.array([0.0, 0.0, 1.0])
    else:
        z = z / n
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _limb(a, b, rx, ry, mat, taper=1.0):
    """Ellipsoid spanning joint a -> b with perpendicular radii rx/ry."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mid = 0.5 * (a + b)
    axis = b - a
    L = np.linalg.norm(axis)
    rot = _basis_z(axis)
    return _Prim(mid, (rx * taper, ry * taper, max(L * 0.55, 1e-3)), rot, mat)


def _ell(c, r, mat, hair_mode=0, rot=None):
    return _Prim(c, r, rot if rot is not None else np.eye(3), mat, hair_mode)


def _rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _bow(prims, pivot, theta):
    """Rotate primitives about a pivot on the x axis (forward bow)."""
    R = _rot_x(theta)
    out = []
    for p in prims:
        q = _Prim(pivot + R @ (p.c - pivot), p.r, R @ p.rot, p.mat, p.hair_mode)
        out.append(q)
    return out


def _heading_yaw(heading, camera):
    elev = math.radians(_CAM_ELEV[camera])
    dx, dy = heading
    se = max(math.sin(elev), 1e-6)
    wx, wy = float(dx), -float(dy) / se
    n = math.hypot(wx, wy)
    if n < 1e-6:
        wy = -1.0
        n = 1.0
    wx, wy = wx / n, wy / n
    return math.atan2(wx, -wy)  # rotates local forward (0,-1) onto (wx,wy)


def build_person(state="walking", variant=0, yaw=0.0, phase=0.0,
                 offer_progress=0.0, flower=True):
    """Return (prims, materials) in character-local coords (x right, fwd -y, z up),
    with the yaw rotation already applied (still centered on origin, feet z=0)."""
    skin, hairc, style, skirt, hscale = _VARIANTS[variant % len(_VARIANTS)]
    mats = [skin, hairc, _CLOTH_RGB, _CLOTH2_RGB, _SHOE_RGB, _CHAIR_RGB,
            _FLOWER_RGB, _DARK_RGB, _WOOD_RGB]
    prims = []
    hs = hscale
    seated = state == "seated"
    walking = state == "walking"
    # Stationary states never get gait angles; phase is canonicalized upstream anyway
    ph = phase if walking else 0.0
    bob = 0.02 * abs(math.cos(ph)) if walking else 0.0
    bow_p = max(0.0, min(1.0, offer_progress)) if state == "offering" else 0.0

    # ---- legs -------------------------------------------------------------
    hip_z = 0.55 * hs if seated else 0.96 * hs + bob
    for s in (-1, 1):
        hip = np.array([s * 0.085 * hs, 0.0, hip_z])
        if seated:
            knee = hip + np.array([0.0, -0.40 * hs, -0.03])
            ankle = knee + np.array([0.0, -0.02, -0.45 * hs])
            foot_c = ankle + np.array([0.0, -0.055, -0.045])
            pitch = 0.0
        else:
            if walking:
                a = -0.21 * math.sin(ph + (0 if s < 0 else math.pi))
                flex = 0.12 + 0.55 * max(0.0, -math.sin(ph + (0 if s < 0 else math.pi)))
            else:
                a, flex = 0.0, 0.08
            knee = hip + 0.43 * hs * np.array([0.0, math.sin(a), -math.cos(a)])
            a2 = a + min(flex, 0.9)
            ankle = knee + 0.44 * hs * np.array([0.0, math.sin(a2), -math.cos(a2)])
            foot_c = ankle + np.array([0.0, -0.05, -0.05])
            pitch = a2 * 0.4
        leg_mat = MAT_SKIN if skirt else MAT_CLOTH
        prims.append(_limb(hip, knee, 0.072 * hs, 0.066 * hs, MAT_CLOTH if skirt else leg_mat))
        prims.append(_limb(knee, ankle, 0.052 * hs, 0.048 * hs, leg_mat, taper=0.9))
        # Shoe: long axis along local Y (forward), slight pitch following the ankle
        prims.append(_Prim(foot_c, (0.045 * hs, 0.105 * hs, 0.035 * hs),
                           _rot_x(pitch), MAT_SHOE))
    if skirt:
        prims.append(_ell((0, -0.01, hip_z - 0.26 * hs), (0.185 * hs, 0.155 * hs, 0.30 * hs), MAT_CLOTH2))

    # ---- torso / head ------------------------------------------------------
    up_z = hip_z if seated else (0.98 * hs + bob)
    up = []
    up.append(_ell((0, 0.0, up_z + 0.10 * hs), (0.155 * hs, 0.115 * hs, 0.13 * hs), MAT_CLOTH))
    up.append(_ell((0, -0.005, up_z + 0.30 * hs), (0.185 * hs, 0.125 * hs, 0.165 * hs), MAT_CLOTH))
    up.append(_ell((0, 0.0, up_z + 0.50 * hs), (0.05 * hs, 0.045 * hs, 0.06 * hs), MAT_SKIN))  # neck
    head_z = up_z + 0.63 * hs
    head = _ell((0, -0.008, head_z), (0.095 * hs, 0.10 * hs, 0.115 * hs), MAT_SKIN, hair_mode=1)
    up.append(head)
    if style == "bob":
        up.append(_ell((0, 0.012, head_z - 0.01), (0.105 * hs, 0.108 * hs, 0.115 * hs), MAT_HAIR, hair_mode=2))
    elif style == "bun":
        up.append(_ell((0, 0.012, head_z - 0.005), (0.102 * hs, 0.106 * hs, 0.112 * hs), MAT_HAIR, hair_mode=2))
        up.append(_ell((0, 0.105, head_z + 0.075), (0.045 * hs, 0.045 * hs, 0.05 * hs), MAT_HAIR))
    elif style == "gray":
        up.append(_ell((0, 0.008, head_z + 0.005), (0.10 * hs, 0.105 * hs, 0.113 * hs), MAT_HAIR, hair_mode=2))
    for s in (-1, 1):  # ears
        up.append(_ell((s * 0.094 * hs, -0.005, head_z - 0.005), (0.018 * hs, 0.026 * hs, 0.032 * hs), MAT_SKIN))
    up.append(_ell((0, -0.098, head_z - 0.01), (0.014 * hs, 0.018 * hs, 0.02 * hs), MAT_SKIN))  # nose
    for s in (-1, 1):  # subtle brow/eye marks (front only; occluded from behind)
        up.append(_ell((s * 0.034, -0.086, head_z + 0.018), (0.013, 0.009, 0.009), MAT_DARK))

    # ---- arms --------------------------------------------------------------
    shoulder = np.array([0.0, 0.0, up_z + 0.47 * hs])
    for i, s in enumerate((-1, 1)):
        sh = shoulder + np.array([s * 0.20 * hs, 0.0, 0.0])
        holds = flower and s > 0  # right arm cups the sandalwood flower
        if seated:
            elbow = sh + np.array([s * 0.02, -0.10 * hs, -0.24 * hs])
            hand = elbow + np.array([-s * 0.06 * hs, -0.16 * hs, -0.05])
        elif state == "offering":
            reach = 0.20 + 0.15 * bow_p
            hand = sh + np.array([-s * 0.05 * hs, -reach * hs, -(0.10 + 0.06 * bow_p) * hs])
            elbow = sh + np.array([s * 0.06 * hs, -0.14 * hs, -0.20 * hs])
        elif holds:
            # upper arm hangs, forearm brings the hand to cup at lower chest/abdomen
            elbow = sh + np.array([s * 0.03 * hs, -0.02 * hs, -0.26 * hs])
            hand = elbow + np.array([-s * 0.13 * hs, -0.13 * hs, 0.02 * hs])
        else:
            b = -0.18 * math.sin(ph + (0 if s < 0 else math.pi)) if walking else 0.0
            elbow = sh + 0.29 * hs * np.array([0.0, math.sin(b), -math.cos(b)])
            b2 = b + 0.35
            hand = elbow + 0.27 * hs * np.array([0.0, math.sin(b2), -math.cos(b2)])
        up.append(_limb(sh, elbow, 0.048 * hs, 0.044 * hs, MAT_CLOTH))
        up.append(_limb(elbow, hand, 0.040 * hs, 0.036 * hs, MAT_CLOTH, taper=0.9))
        up.append(_ell(hand, (0.040 * hs, 0.045 * hs, 0.045 * hs), MAT_SKIN))
        if holds:
            # Hand-sized cream 5-petal sandalwood flower on a small wooden stem
            fc = hand + np.array([-s * 0.01, -0.035, 0.01])
            up.append(_limb(fc, fc + np.array([0.0, -0.005, 0.06]), 0.006, 0.006, MAT_WOOD))
            head_c = fc + np.array([0.0, -0.008, 0.068])
            up.append(_ell(head_c, (0.010, 0.008, 0.010), MAT_FLOWER))
            for pet in range(5):
                ang = pet * (2 * math.pi / 5)
                pc = head_c + np.array([0.020 * math.cos(ang), 0.0, 0.020 * math.sin(ang)])
                up.append(_ell(pc, (0.013, 0.008, 0.013), MAT_FLOWER))

    if state == "offering":
        up = _bow(up, np.array([0.0, 0.0, up_z]), math.radians(15.0) * bow_p)
    prims.extend(up)

    # ---- chair for seated ---------------------------------------------------
    if seated:
        seat_y = 0.10
        prims.append(_ell((0, seat_y, 0.47 * hs), (0.23, 0.21, 0.05), MAT_CHAIR))
        prims.append(_ell((0, seat_y + 0.22, 0.72 * hs), (0.22, 0.045, 0.26), MAT_CHAIR))
        for s in (-1, 1):
            prims.append(_limb((s * 0.18, seat_y + 0.16, 0.47 * hs),
                               (s * 0.18, seat_y + 0.16, 0.02), 0.02, 0.02, MAT_CHAIR))
    # Ground normalization: lowest grounded primitive (shoe / chair leg) sits at
    # z=0, applied to the whole body so nothing floats or dips below ground
    def bottom_z(p):
        return p.c[2] - float(np.linalg.norm(p.rot[2, :] * p.r))
    grounded = [p for p in prims if p.mat in (MAT_SHOE, MAT_CHAIR)]
    g_min = min(bottom_z(p) for p in grounded)
    if g_min < 0.0 or g_min > 0.005:
        for p in prims:
            p.c[2] -= g_min

    # ---- yaw rotation -------------------------------------------------------
    cy, sy = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    out = []
    for p in prims:
        out.append(_Prim(R @ p.c, p.r, R @ p.rot, p.mat, p.hair_mode))
    return out, mats


# ---------------------------------------------------------------- raycaster
def _camera_vecs(camera):
    e = math.radians(_CAM_ELEV[camera])
    v = np.array([0.0, math.cos(e), -math.sin(e)])   # ray INTO the scene
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, math.sin(e), math.cos(e)])
    return v, right, up


def _render(prims, mats, camera, body_px=_BODY_PX_BASE, supersample=1):
    """Render prims to an RGBA image; returns (img, anchor) in output pixels.
    `body_px` is the standing body's projected height (canonical 1.75m norm)."""
    elev = math.radians(_CAM_ELEV[camera])
    proj_h_m = _BODY_M * (math.cos(elev) if camera != "topdown" else 1.0)
    ss = supersample
    mpp = proj_h_m / (body_px * ss)
    H = int(math.ceil((proj_h_m + _MARGIN_TOP_M + _MARGIN_BOTTOM_M) / mpp))
    W = max(48, int(math.ceil(1.20 / mpp)))  # arm reach / rotated poses need ±0.6m
    cx = W / 2.0
    ay = H - _MARGIN_BOTTOM_M / mpp  # ground (z=0) pixel row; margin covers shadow+feet
    v, right, up = _camera_vecs(camera)

    depth = np.full((H, W), np.inf)
    nrm = np.zeros((H, W, 3))
    mat_id = np.full((H, W), -1, dtype=np.int8)
    loc = np.zeros((H, W, 3))
    prim_sel = np.full((H, W), -1, dtype=np.int16)

    py_i, px_i = np.mgrid[0:H, 0:W]
    sx = (px_i - cx) * mpp
    sy = (ay - py_i) * mpp

    for pi, p in enumerate(prims):
        # projective ROI for this primitive
        c_sx, c_sy = float(p.c @ right), float(p.c @ up)
        rmax = float(p.r.max()) + 0.02
        x0 = int(max(0, (c_sx - rmax) / mpp + cx - 1))
        x1 = int(min(W, (c_sx + rmax) / mpp + cx + 2))
        y0 = int(max(0, ay - (c_sy + rmax) / mpp - 1))
        y1 = int(min(H, ay - (c_sy - rmax) / mpp + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        O = sx[y0:y1, x0:x1, None] * right + sy[y0:y1, x0:x1, None] * up
        o_loc = (O - p.c) @ p.rot
        d_loc = v @ p.rot
        a = ((d_loc / p.r) ** 2).sum()
        b = 2.0 * (o_loc * d_loc / (p.r ** 2)).sum(-1)
        c2 = ((o_loc / p.r) ** 2).sum(-1) - 1.0
        disc = b * b - 4.0 * a * c2
        hit = disc > 0
        if not hit.any():
            continue
        t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a), np.inf)
        if p.hair_mode == 2:
            # hair shell: filter hit pixels to top/back BEFORE they can occlude
            # the face/front of the head behind them
            t_hit = np.where(hit, t, 0.0)
            pl_pre = (O + t_hit[..., None] * v - p.c) @ p.rot
            hair_keep = ~((pl_pre[..., 1] < -0.03) & (pl_pre[..., 2] < 0.04))
            t = np.where(hit & hair_keep, t, np.inf)
        sub_depth = depth[y0:y1, x0:x1]
        closer = t < sub_depth
        if not closer.any():
            continue
        sub_depth[closer] = t[closer]
        t_fin = np.where(closer, t, 0.0)
        P = O + t_fin[..., None] * v
        pl = (P - p.c) @ p.rot
        loc[y0:y1, x0:x1][closer] = pl[closer]
        n_loc = pl / (p.r ** 2)
        n_world = n_loc @ p.rot.T
        n_world /= np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-9
        nrm[y0:y1, x0:x1][closer] = n_world[closer]
        mat_id[y0:y1, x0:x1][closer] = p.mat
        prim_sel[y0:y1, x0:x1][closer] = pi

    # ---- shading ------------------------------------------------------------
    view = -v
    L = np.array([-0.42, -0.38, 0.82])
    L = L / np.linalg.norm(L)
    Hvec = L + view
    Hvec = Hvec / np.linalg.norm(Hvec)
    diff = np.clip(nrm @ L, 0.0, 1.0)
    fres = np.clip(nrm @ view, 0.0, 1.0) ** 3
    spec = np.clip(nrm @ Hvec, 0.0, 1.0) ** 18
    shade = 0.34 + 0.62 * diff + 0.16 * fres * (1.0 - diff) + 0.10 * spec

    rgb = np.zeros((H, W, 3))
    alpha = np.zeros((H, W), dtype=np.uint8)
    body = mat_id >= 0
    for mi, m in enumerate(mats):
        sel = mat_id == mi
        if not sel.any():
            continue
        col = np.asarray(m, dtype=float)
        if mi in (MAT_CLOTH, MAT_CLOTH2):
            # subtle center-seam / crease variation so cloth is not flat black
            seam = (np.abs(loc[..., 0]) < 0.012) & sel
            col_p = np.where(seam[..., None], col - 8.0, col)
            rgb[sel] = col_p[sel] * shade[sel, None]
        else:
            rgb[sel] = col * shade[sel, None]

    # scalp-cap hair mask (mode 1 prims are the skin head itself)
    for pi, p in enumerate(prims):
        if p.hair_mode != 1:
            continue
        sel = prim_sel == pi
        if not sel.any():
            continue
        pl = loc[sel]
        hair = (pl[:, 2] > 0.0) | (pl[:, 1] > 0.0)
        idx = np.where(sel)
        sel2 = np.zeros(sel.shape, bool)
        sel2[idx[0][hair], idx[1][hair]] = True
        rgb[sel2] = np.asarray(mats[MAT_HAIR], float) * shade[sel2, None]

    alpha[body] = 255

    # soft elliptical contact shadow behind/around the feet (never over body)
    sin_e = math.sin(elev)
    shadow_rx = 0.30 / mpp
    shadow_ry = (0.30 * (sin_e if camera == "topdown" else max(sin_e, 0.2))) / mpp
    dsx = (px_i - cx) / shadow_rx
    dsy = (py_i - ay) / shadow_ry
    sd = dsx ** 2 + dsy ** 2
    s_alpha = np.clip(1.0 - sd, 0.0, 1.0) ** 2 * (90 if camera == "topdown" else 110)
    m = (s_alpha > 1) & (~body)
    rgb[m] = 20.0
    alpha[m] = np.maximum(alpha[m], s_alpha[m]).astype(np.uint8)

    img = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha])
    im = Image.fromarray(img, "RGBA")
    if ss > 1:
        im = im.resize((W // ss, H // ss), Image.LANCZOS)
    return im, (cx / ss, ay / ss)


# ---------------------------------------------------------------- public API
def _quantized_key(heading, phase, state, person_id, holding_flower, camera, offer_progress):
    ang = math.atan2(float(heading[1]), float(heading[0]))
    hbin = int(round(ang / (2 * math.pi) * _HEADING_BINS)) % _HEADING_BINS
    pbin = (int((phase % (2 * math.pi)) / (2 * math.pi) * _PHASE_BINS) % _PHASE_BINS
            if state == "walking" else 0)
    obin = (max(0, min(_OFFER_BINS - 1, int(offer_progress * _OFFER_BINS)))
            if state == "offering" else 0)
    return hbin, pbin, state, person_id % 4, obin, camera, bool(holding_flower)


@lru_cache(maxsize=2048)
def _cached_sprite(hbin, pbin, state, variant, obin, camera, flower):
    ang = hbin * (2 * math.pi / _HEADING_BINS)
    heading = (math.cos(ang), math.sin(ang))
    yaw = _heading_yaw(heading, camera)
    phase = pbin * (2 * math.pi / _PHASE_BINS)
    prims, mats = build_person(state=state, variant=variant, yaw=yaw, phase=phase,
                             offer_progress=obin / _OFFER_BINS, flower=flower)
    return _render(prims, mats, camera, body_px=_BODY_PX_BASE, supersample=2)


@lru_cache(maxsize=8192)
def _scaled_sprite(hbin, pbin, state, variant, obin, camera, flower, height_int):
    img, (ax, ay) = _cached_sprite(hbin, pbin, state, variant, obin, camera, flower)
    s = height_int / _BODY_PX_BASE
    if abs(s - 1.0) > 1e-3:
        w2 = max(1, int(round(img.width * s)))
        h2 = max(1, int(round(img.height * s)))
        img = img.resize((w2, h2), Image.LANCZOS)
        ax, ay = ax * s, ay * s
    return img, (ax, ay)


def draw_human(canvas, x, y, height_px, *, heading=(0.0, -1.0), phase=0.0,
               state="walking", person_id=0, holding_flower=True, camera="close",
               offer_progress=0.0):
    """Composite a shaded 3D person onto `canvas` (RGBA PIL image).
    (x, y) is the foot-center ground anchor and is preserved; height_px is the
    standing body's projected height in pixels."""
    key = _quantized_key(heading, phase, state, person_id, holding_flower, camera, offer_progress)
    height_int = max(1, int(round(height_px)))
    img, (ax, ay) = _scaled_sprite(*key, height_int)
    canvas.alpha_composite(img, (int(round(x - ax)), int(round(y - ay))))


def render_sprite(*, heading=(0.0, -1.0), phase=0.0, state="walking", person_id=0,
                  holding_flower=True, camera="close", offer_progress=0.0,
                  height_px=300):
    """Un-cached render at an arbitrary body height (for tests / contact sheets).
    Returns (PIL.Image RGBA, (anchor_x, anchor_y))."""
    yaw = _heading_yaw(heading, camera)
    prims, mats = build_person(state=state, variant=person_id % 4, yaw=yaw, phase=phase,
                             offer_progress=offer_progress, flower=holding_flower)
    return _render(prims, mats, camera, body_px=int(height_px), supersample=1)


def cache_info():
    return _cached_sprite.cache_info()


def scaled_cache_info():
    return _scaled_sprite.cache_info()


def clear_caches():
    _cached_sprite.cache_clear()
    _scaled_sprite.cache_clear()


def make_contact_sheet(path, cell=300):
    """Labeled contact sheet: close camera, 4 variants x poses, on neutral gray."""
    poses = [("walk ph0.7 back", "walking", 0.7, (0.0, -1.0), 0.0, True),
             ("walk ph+pi back", "walking", 0.7 + math.pi, (0.0, -1.0), 0.0, True),
             ("walk left profile", "walking", 0.7, (-1.0, 0.0), 0.0, False),
             ("walk front", "walking", 0.7, (0.0, 1.0), 0.0, True),
             ("seated", "seated", 0.0, (0.0, -1.0), 0.0, True),
             ("offering .7", "offering", 0.0, (0.0, -1.0), 0.7, True),
             ("idle", "idle", 0.0, (0.0, -1.0), 0.0, True)]
    cols = len(poses)
    rows = len(_VARIANTS)
    sheet = Image.new("RGBA", (cols * cell, rows * (cell + 26)), (128, 132, 138, 255))
    d = ImageDraw.Draw(sheet)
    for vi in range(len(_VARIANTS)):
        for cj, (label, st, ph, hd, op, fl) in enumerate(poses):
            sp, (ax, ay) = render_sprite(heading=hd, phase=ph, state=st,
                                       person_id=vi, holding_flower=fl,
                                       camera="close", offer_progress=op,
                                       height_px=int(cell * 0.82))
            ox = int(cj * cell + cell / 2 - ax)
            oy = int(vi * (cell + 26) + cell * 0.94 - ay)
            sheet.alpha_composite(sp, (ox, oy))
        d.text((4, vi * (cell + 26) + cell + 6), f"variant {vi}", fill=(255, 255, 0, 255))
    for cj, (label, *_rest) in enumerate(poses):
        d.text((cj * cell + 4, 2), label, fill=(255, 255, 0, 255))
    sheet.save(path)
    return path
