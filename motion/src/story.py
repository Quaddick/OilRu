"""Godzilla story: animates the two hand-drawn sheets as a short cartoon.

Usage: python3 story.py out.mp4            # full video with sound
       python3 story.py out.mp4 120 480    # only preview frames -> prev_XXXX.jpg
"""
import cv2, numpy as np, subprocess, sys, math, os, wave
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from objs import load

W, H, FPS = 1920, 1080, 30
R = 2.4                       # sprite raster oversampling (for close-ups)
RED = np.array([38, 32, 205], np.float32) / 255
GRAY = np.array([78, 76, 74], np.float32) / 255
rng = np.random.default_rng(7)
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_CAP = os.path.join(HERE, 'Caveat.ttf')
FONT_FX = os.path.join(HERE, 'RubikMonoOne.ttf')

def clamp(x, a=0.0, b=1.0): return min(max(x, a), b)
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)
def win(t, a, b, fi=0.3, fo=0.3):   # 0..1 visibility window
    return clamp((t - a) / fi) * clamp((b - t) / fo)

AUDIO = []                    # (time, kind, params)

# ---------------------------------------------------------------- sprites
def geo_order(mask):
    h, w = mask.shape; order = np.full((h, w), -1.0, np.float32)
    nl, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, nl):
        ys, xs = np.nonzero(lab == i); comps.append((xs.min(), i, ys, xs))
    comps.sort(); base = 0.0
    for _, i, ys, xs in comps:
        k = np.argmin(xs * 1.0 + ys * 0.3); sy, sx = ys[k], xs[k]
        dist = np.full((h, w), -1, np.int32); dist[sy, sx] = 0; q = deque([(sy, sx)]); m = lab == i
        while q:
            y, x = q.popleft(); d = dist[y, x] + 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and dist[yy, xx] < 0:
                        dist[yy, xx] = d; q.append((yy, xx))
        sel = dist >= 0; order[sel] = base + dist[sel]; base += dist.max() + 8
    order[order >= 0] /= max(base, 1)
    return order

def make_raster(red, dark, bbox, S, off):
    x, y, w, h = [int(v) for v in bbox]; p = 6; x -= p; y -= p; w += 2 * p; h += 2 * p
    r = red[y:y + h, x:x + w]; d = dark[y:y + h, x:x + w]
    o = geo_order(r | d)
    k = S * R; sw, sh = int(round(w * k)), int(round(h * k))
    def up(a):
        a = cv2.resize(a.astype(np.float32), (sw, sh), interpolation=cv2.INTER_CUBIC)
        a = cv2.GaussianBlur(a, (0, 0), 0.55 * k)
        return np.clip((a - 0.32) / 0.3, 0, 1)
    ra, da = up(r), up(d)
    a = np.maximum(ra, da)
    col = (RED[None, None] * ra[..., None] + GRAY[None, None] * da[..., None]) / np.maximum(ra + da, 1e-4)[..., None]
    tex = 1 + 0.07 * cv2.GaussianBlur(rng.standard_normal((sh, sw)).astype(np.float32), (0, 0), 1.6)
    col = np.clip(col * tex[..., None], 0, 1)
    oo = cv2.resize(o, (sw, sh), interpolation=cv2.INTER_NEAREST)
    oo = cv2.dilate(oo, np.ones((5, 5), np.uint8)); oo[oo < 0] = 0
    pos = np.array([off[0] + x * S, off[1] + y * S], np.float32)
    return col, a, oo, pos

def rotm(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
def trm(x, y): return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float32)
def scm(a, b): return np.array([[a, 0, 0], [0, b, 0], [0, 0, 1]], np.float32)

class Sprite:
    def __init__(s, col, a, order, pos, tag, group):
        s.col, s.a, s.order, s.pos = col, a, order, pos
        s.tag, s.group = tag, group
        s.h, s.w = a.shape
        s.piv = np.array([s.w / 2, s.h * 0.92], np.float32)   # pivot in raster px
        s.seed = int(rng.integers(1e6)); s.t0 = 0; s.dur = 1; s.phase = 0.0
        s.anim = None; s.boil = 1.3
        ys, xs = np.nonzero(a > 0.5)
        s.center = pos + np.array([xs.mean(), ys.mean()], np.float32) / R if len(xs) else pos

    def matrix(s, t, cam):
        dx = dy = rot = 0.0; sx = sy = 1.0; alpha = 1.0
        if s.anim:
            dx, dy, rot, sx, sy, alpha = s.anim(s, t)
        f = int(t * FPS) // 4; r = np.random.default_rng(s.seed + f)
        b = s.boil
        dx += r.uniform(-b, b); dy += r.uniform(-b, b); rot += r.uniform(-0.45, 0.45)
        px, py = s.piv / R
        A = trm(s.pos[0] + dx + px, s.pos[1] + dy + py) @ rotm(rot) @ scm(sx, sy) @ trm(-px, -py) @ scm(1 / R, 1 / R)
        return cam @ A, alpha

    def draw(s, img, t, cam):
        rv = (t - s.t0) / s.dur
        if rv <= 0: return None
        A, alpha = s.matrix(t, cam)
        if alpha <= 0.003: return None
        a = s.a * alpha
        if rv < 1:
            a = a * np.clip((rv - s.order) / 0.015 + 1, 0, 1)
        A2 = A[:2]
        cs = np.array([[0, 0, 1], [s.w, 0, 1], [0, s.h, 1], [s.w, s.h, 1]], np.float32) @ A2.T
        x0 = int(max(0, cs[:, 0].min() - 2)); x1 = int(min(W, cs[:, 0].max() + 3))
        y0 = int(max(0, cs[:, 1].min() - 2)); y1 = int(min(H, cs[:, 1].max() + 3))
        if x1 <= x0 or y1 <= y0: return None
        A3 = A2.copy(); A3[0, 2] -= x0; A3[1, 2] -= y0
        wp = cv2.warpAffine(np.dstack([s.col, a]), A3, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0)
        al = wp[..., 3:4]; reg = img[y0:y1, x0:x1]
        reg[:] = reg * (1 - al) + (wp[..., :3] * reg) * al * 0.15 + wp[..., :3] * al * 0.85
        if rv < 1:
            band = (np.abs(s.order - rv) < 0.012) & (s.a > 0.3)
            if band.any():
                ys, xs = np.nonzero(band)
                return np.array([xs.mean(), ys.mean(), 1], np.float32) @ A2.T
        return None

    def glow_mask(s, t, cam):
        """screen-space alpha of this sprite (for glow effects)"""
        A, alpha = s.matrix(t, cam)
        return cv2.warpAffine(s.a * alpha, A[:2], (W, H), flags=cv2.INTER_LINEAR)

def shatter(col, a, order, pos, tag, group, n):
    """split a raster into irregular Voronoi chunks, each its own sprite"""
    h, w = a.shape
    ys, xs = np.nonzero(a > 0.2)
    pts = np.c_[xs, ys][rng.choice(len(xs), size=min(n, len(xs)), replace=False)].astype(np.float32)
    gy, gx = np.mgrid[0:h, 0:w]
    d = (gx[..., None] - pts[:, 0]) ** 2 + (gy[..., None] - pts[:, 1]) ** 2
    cell = d.argmin(-1)
    out = []
    for i in range(len(pts)):
        m = (cell == i) & (a > 0.01)
        if m.sum() < 30: continue
        yy, xx = np.nonzero(m); y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
        ca = (a * m)[y0:y1, x0:x1]
        sp = Sprite(col[y0:y1, x0:x1].copy(), ca, order[y0:y1, x0:x1].copy(),
                    pos + np.array([x0, y0], np.float32) / R, tag, group)
        sp.piv = np.array([ca.shape[1] / 2, ca.shape[0] / 2], np.float32)
        out.append(sp)
    return out

# ---------------------------------------------------------------- scenes
def build(name, box, tagger):
    red, dark, objs = load(name)
    x0, y0, x1, y1 = box
    S = min((W - 220) / (x1 - x0), (H - 200) / (y1 - y0))
    off = (W / 2 - (x0 + x1) / 2 * S, H / 2 - (y0 + y1) / 2 * S + 20)
    sprites = []
    for k, o in enumerate(objs):
        m = o['mask']
        for src, lbl in ((red, 'r'), (dark, 'd')):
            pm0 = src & m
            nl, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(pm0.astype(np.uint8), np.ones((9, 9), np.uint8)))
            for i in range(1, nl):
                pm = (lab == i) & pm0
                if pm.sum() < 40: continue
                bb = st[i, :4]
                tag, extra = tagger(k, lbl, bb, int(pm.sum()))
                col, a, order, pos = make_raster(red & pm, dark & pm, bb, S, off)
                if extra.get('shatter'):
                    sprites += shatter(col, a, order, pos, tag, k, extra['shatter'])
                else:
                    sp = Sprite(col, a, order, pos, tag, k)
                    if 'piv' in extra: sp.piv = np.array([sp.w * extra['piv'][0], sp.h * extra['piv'][1]], np.float32)
                    sprites.append(sp)
    return sprites, S, off, red, dark

def tag1(k, lbl, bb, area):
    x = bb[0]
    if k in (0, 2): return 'walker', {}
    if k in (1, 3): return 'arrow', {}
    if k == 4: return 'idle', {}
    if k in (5, 6):
        if lbl == 'd': return f'bld{k}', {'shatter': 7}
        return 'smasher', {}
    if k == 7:
        if lbl == 'r' and area > 9000: return 'breather', {}
        return 'house', {'shatter': 9}
    return 'idle', {}

def tag2(k, lbl, bb, area):
    if k in (0, 1): return 'cheer', {}
    if k == 2: return 'godz', {'piv': (0.35, 0.95)}
    if k == 3: return 'kong', {'piv': (0.8, 0.97)}
    if k == 4 and lbl == 'r' and bb[0] > 1120: return 'stripes', {}
    if k in (4, 5, 6): return 'fighter', {}
    if k == 7: return 'scylla', {'piv': (0.55, 0.9)}
    return 'idle', {}

print('building sprites...', flush=True)
SC1, S1, OFF1, RED1, DARK1 = build('s1', (430, 575, 1740, 1115), tag1)
SC2, S2, OFF2, RED2, DARK2 = build('s2', (275, 590, 1645, 1120), tag2)
def c1(x, y): return np.array([OFF1[0] + x * S1, OFF1[1] + y * S1], np.float32)
def c2(x, y): return np.array([OFF2[0] + x * S2, OFF2[1] + y * S2], np.float32)

# breath path on sheet 1 = the long red line
nl, lab, st, _ = cv2.connectedComponentsWithStats(RED1.astype(np.uint8))
line = lab == max((((lab == i)[560:800, 1450:1700]).sum(), i) for i in range(1, nl))[1]
pts = []
for xv in range(1395, 1716, 3):
    col = np.nonzero(line[:, xv - 1:xv + 2].any(1))[0]; col = col[col < 930]
    if len(col): pts.append((xv, float(np.median(col))))
pts = np.array(pts, np.float32)
pts[:, 1] = np.convolve(np.pad(pts[:, 1], 3, mode='edge'), np.ones(7) / 7, 'valid')
BEAM1 = np.array([c1(*p) for p in pts], np.float32)

# ---------------------------------------------------------------- timeline
# scene 1
D1 = (0.5, 4.6)
WALK = (5.6, 10.6)
SMASH = (13.4, 14.3)               # building shatter times (bld5, bld6)
BREATH1 = (20.6, 23.2); BOOM1 = 21.3
END1 = 24.6; TURN = 1.0
T2 = END1 + TURN                   # scene 2 starts
D2 = (T2 + 0.3, T2 + 4.6)
HITS = [T2 + 7.0, T2 + 7.8, T2 + 8.6]; KO = T2 + 9.3
CLASH = [T2 + 16.4, T2 + 17.1, T2 + 17.8, T2 + 18.5]
BREATH2 = (T2 + 19.6, T2 + 22.0); BOOM2 = T2 + 20.2
END2 = T2 + 27.0; FADE = 1.4
TOTAL = END2 + FADE

CAM1 = [(0, W / 2, H / 2, 1.0), (D1[1], W / 2, H / 2, 1.0),
        (D1[1] + 1.0, *c1(650, 1000), 2.0), (10.6, *c1(700, 1000), 2.0),
        (11.6, *c1(1150, 880), 1.8), (17.4, *c1(1150, 880), 1.8),
        (18.4, *c1(1490, 775), 1.55), (23.5, *c1(1490, 775), 1.55),
        (24.4, W / 2, H / 2, 1.0)]
CAM2 = [(T2, W / 2, H / 2, 1.0), (D2[1], W / 2, H / 2, 1.0),
        (D2[1] + 1.0, *c2(590, 880), 1.6), (T2 + 12.6, *c2(590, 880), 1.6),
        (T2 + 13.6, *c2(1310, 790), 1.65), (T2 + 23.4, *c2(1310, 790), 1.65),
        (T2 + 24.4, W / 2, H / 2, 1.0)]

SHAKES = []                        # (time, strength, duration)
def shake_at(t, s, d): SHAKES.append((t, s, d))

def cam_at(keys, t):
    k = keys[0]
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if t <= a[0]: k = a; break
        if a[0] <= t <= b[0]:
            u = ease((t - a[0]) / max(b[0] - a[0], 1e-6))
            k = tuple(a[j] + (b[j] - a[j]) * u for j in range(4)); break
        k = b
    _, cx, cy, z = k
    sx = sy = 0.0
    for (ts, st_, d) in SHAKES:
        if ts <= t < ts + d:
            e = (1 - (t - ts) / d) ** 2 * st_
            sx += e * math.sin(t * 97 + ts); sy += e * math.cos(t * 83 + ts * 3)
    return np.array([[z, 0, W / 2 - z * cx + sx], [0, z, H / 2 - z * cy + sy], [0, 0, 1]], np.float32)

# ---------------------------------------------------------------- animations
def a_walker(s, t):
    if t < WALK[0]: return 0, 0, 0, 1, 1, 1
    tm = t - WALK[0]; k = clamp(tm / 0.5) * clamp((WALK[1] + 0.6 - t) / 0.6)
    w = 2 * math.pi * 0.9 * tm + s.phase
    return min(tm, WALK[1] - WALK[0]) * 12, -abs(math.sin(w)) * 22 * k, math.sin(w) * 8 * k, 1, 1, 1
def a_arrow(s, t):
    if t < WALK[0] or t > WALK[1] + 0.5: return 0, 0, 0, 1, 1, 1
    c = ((t - WALK[0]) * 0.9 + s.phase) % 1
    return ease(c) * 30 - 6, 0, 0, 1 + 0.1 * math.sin(c * math.pi), 1, 1
def a_smasher(s, t):
    if t < 11.8 or t > 17.2: return 0, 0, 0, 1, 1, 1
    tm = t - 11.8; k = clamp(tm / 0.6)
    # stomping / thrashing on the city
    w = 2 * math.pi * 1.6 * tm + s.phase
    return math.sin(w * 0.5) * 6 * k, -abs(math.sin(w)) * 14 * k, math.sin(w) * 5 * k, 1 + 0.05 * k * math.sin(w), 1, 1

def flyaway(s, t, tb, center, power=1.0):
    """chunk physics after a break at time tb"""
    if t < tb - 0.9: return 0, 0, 0, 1, 1, 1
    if t < tb:   # tremble before breaking
        e = (t - tb + 0.9) / 0.9
        r = np.random.default_rng(s.seed + int(t * FPS))
        return r.uniform(-3, 3) * e, r.uniform(-2, 2) * e, r.uniform(-3, 3) * e, 1, 1, 1
    r = np.random.default_rng(s.seed)
    d = s.center - center; n = np.linalg.norm(d) + 1e-3; d = d / n
    vx = (d[0] * 260 + r.uniform(-120, 120)) * power
    vy = (-r.uniform(250, 520) + d[1] * 80) * power
    spin = r.uniform(-400, 400) * power
    tt = t - tb
    return vx * tt, vy * tt + 900 * tt * tt, spin * tt, 1, 1, clamp(1 - (tt - 0.9) / 0.6)

def a_bld(tb, center):
    return lambda s, t: flyaway(s, t, tb, center, 0.8)

def a_breather(s, t):
    if BREATH1[0] - 1.0 < t < BREATH1[1]:
        e = clamp((t - BREATH1[0] + 1.0) / 1.0)
        r = np.random.default_rng(s.seed + int(t * FPS))
        return r.uniform(-2, 2) * e, r.uniform(-2, 2) * e, 0, 1, 1, 1
    return 0, 0, 0, 1, 1, 1

def a_cheer(s, t):
    tm = t - (D2[1] + 1.0)
    if tm < 0: return 0, 0, 0, 1, 1, 1
    w = 2 * math.pi * 1.4 * tm + s.phase
    k = 1.5 if t > KO else 0.8
    return 0, -abs(math.sin(w)) * 14 * k, math.sin(w) * 6 * k, 1, 1, 1

def a_godz(s, t):
    dx = rot = 0.0
    for h in HITS + [KO]:
        u = (t - (h - 0.35)) / 0.7
        if 0 <= u <= 1:
            p = math.sin(u * math.pi)
            dx += p * 55; rot += p * 14
    if T2 + 6.0 < t < HITS[0] - 0.35:   # stalking in
        dx += math.sin((t - T2) * 8) * 4
    if t > KO + 0.5:                    # victory hop
        w = (t - KO) * 2 * math.pi * 1.5
        return 0, -abs(math.sin(w)) * 24, math.sin(w) * 6, 1, 1, 1
    return dx, 0, rot, 1, 1, 1

def a_kong(s, t):
    dx = rot = 0.0
    for h in HITS:
        u = (t - h) / 0.45
        if 0 <= u <= 1: p = math.sin(u * math.pi); dx += p * 26; rot += p * 10
    if t >= KO:
        u = ease((t - KO) / 0.7)
        bounce = abs(math.sin(clamp((t - KO - 0.7) / 0.5) * math.pi)) * 10 * (1 - clamp((t - KO - 0.7) / 0.5))
        return u * 40, -bounce, u * 82, 1, 1, 1
    return dx, 0, rot, 1, 1, 1

def a_fighter(s, t):
    if t < T2 + 14.0: return 0, 0, 0, 1, 1, 1
    dx = dy = rot = 0.0
    for c in CLASH:
        u = (t - (c - 0.3)) / 0.6
        if 0 <= u <= 1: p = math.sin(u * math.pi); dx += p * 30; rot += p * 6
    if BREATH2[0] - 0.8 < t < BREATH2[1]:
        r = np.random.default_rng(s.seed + int(t * FPS)); dx += r.uniform(-3, 3); dy += r.uniform(-3, 3)
    if t > BREATH2[1] + 0.2:
        w = (t - BREATH2[1]) * 2 * math.pi * 1.5
        dy -= abs(math.sin(w)) * 20; rot += math.sin(w) * 5
    return dx, dy, rot, 1, 1, 1

def a_scylla(s, t):
    if t < T2 + 14.0: return 0, 0, 0, 1, 1, 1
    tm = t - (T2 + 14.0)
    dx = dy = rot = 0.0; sx = sy = 1.0
    if t < CLASH[0] - 0.3:   # menacing approach
        w = tm * 2 * math.pi * 1.2
        dx = -ease(tm / 2.0) * 40; rot = math.sin(w) * 7; sy = 1 + 0.06 * math.sin(w * 2); sx = 2 - sy
    elif t < BOOM2:
        dx = -40
        for c in CLASH:
            u = (t - (c - 0.3)) / 0.6
            if 0 <= u <= 1: p = math.sin(u * math.pi); dx -= p * 35; rot -= p * 8
        if t > CLASH[-1] + 0.4: dx += math.sin(t * 40) * 3
    else:
        u = t - BOOM2
        dx = -40 + ease(u / 0.6) * 140; rot = ease(u / 0.8) * 70; dy = ease(u / 0.8) * 40
        if u > 1.0:
            r = np.random.default_rng(s.seed + int(t * FPS) // 3); dx += r.uniform(-1.5, 1.5)
        return dx, dy, rot, 1, 1, 1
    return dx, dy, rot, sx, sy, 1

def assign():
    for i, s in enumerate(SC1):
        s.phase = (s.center[0] * 0.013) % (2 * math.pi)
        if s.tag == 'walker': s.anim = a_walker
        elif s.tag == 'arrow': s.anim = a_arrow; s.phase = (s.center[0] * 0.002) % 1
        elif s.tag == 'smasher': s.anim = a_smasher
        elif s.tag == 'bld5': s.anim = a_bld(SMASH[1], c1(930, 1000))
        elif s.tag == 'bld6': s.anim = a_bld(SMASH[0], c1(1380, 1000))
        elif s.tag == 'house': s.anim = lambda s_, t: flyaway(s_, t, BOOM1, c1(1700, 900), 1.15)
        elif s.tag == 'breather': s.anim = a_breather
    for s in SC2:
        s.phase = (s.center[0] * 0.017) % (2 * math.pi)
        s.anim = {'cheer': a_cheer, 'godz': a_godz, 'kong': a_kong, 'fighter': a_fighter,
                  'stripes': a_fighter, 'scylla': a_scylla}.get(s.tag)
assign()

def schedule(sprites, t0, t1):
    groups = sorted({s.group for s in sprites}, key=lambda g: min(s.center[0] for s in sprites if s.group == g))
    n = len(groups); span = t1 - t0
    for gi, g in enumerate(groups):
        for s in sprites:
            if s.group == g: s.t0 = t0 + span * gi / n * 0.9; s.dur = max(0.5, span / n * 1.3)
schedule(SC1, *D1); schedule(SC2, *D2)

# sound + shake events
for k in range(40):
    ts = WALK[0] + k / (2 * 0.9)
    if ts > WALK[1]: break
    AUDIO.append((ts, 'stomp', 0.8)); shake_at(ts, 7, 0.25)
for k in range(14):
    ts = 11.8 + k / (2 * 1.6) * 2
    if ts > 17: break
    AUDIO.append((ts, 'stomp', 0.6)); shake_at(ts, 5, 0.2)
for tb in SMASH:
    AUDIO.append((tb, 'crash', 1.0)); shake_at(tb, 22, 0.9)
AUDIO.append((BREATH1[0] - 1.0, 'charge', 1.0)); AUDIO.append((BREATH1[0], 'beam', BREATH1[1] - BREATH1[0]))
AUDIO.append((BOOM1, 'boom', 1.0)); shake_at(BOOM1, 30, 1.2)
for h in HITS: AUDIO.append((h, 'punch', 1.0)); shake_at(h, 14, 0.35)
AUDIO.append((KO, 'punch', 1.4)); shake_at(KO, 22, 0.5); AUDIO.append((KO + 0.75, 'crash', 0.6)); shake_at(KO + 0.7, 14, 0.4)
for c in CLASH: AUDIO.append((c, 'punch', 0.9)); shake_at(c, 14, 0.35)
AUDIO.append((BREATH2[0] - 0.8, 'charge', 0.8)); AUDIO.append((BREATH2[0], 'beam', BREATH2[1] - BREATH2[0]))
AUDIO.append((BOOM2, 'boom', 1.2)); shake_at(BOOM2, 32, 1.3)
AUDIO.append((END2 - 2.4, 'ding', 1.0))

# ---------------------------------------------------------------- effects
def paper():
    base = np.ones((H, W, 3), np.float32) * np.array([0.93, 0.95, 0.965], np.float32)
    n = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 1.0) * 0.012
    n += cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 25) * 0.25
    base += n[..., None]
    yy, xx = np.mgrid[0:H, 0:W]; r = ((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.75)) ** 2
    base *= (1 - 0.22 * r)[..., None]
    return np.clip(base, 0, 1)
PAPER = paper()

def pt(cam, p): q = cam @ np.array([p[0], p[1], 1], np.float32); return q[:2]

def glow_layer(mask, colors=((1.0, 0.55, 0.12), (1.0, 0.9, 0.35), (1.0, 1.0, 1.0)), sig=(22, 8, 2)):
    out = np.zeros((H, W, 3), np.float32)
    for c, sg in zip(colors, sig):
        g = cv2.GaussianBlur(mask, (0, 0), sg)
        g = g / max(g.max(), 1e-4) if sg > 3 else g
        out += g[..., None] * np.array(c, np.float32) * (1.3 if sg > 3 else 1.0)
    return out

def beam(img, cam, path, t, ta, tb, width=1.0):
    """atomic breath along a polyline (canvas coords)"""
    if not (ta <= t <= tb + 0.3): return
    p = clamp((t - ta) / 0.6)
    fade = clamp((tb + 0.3 - t) / 0.3)
    n = max(2, int(len(path) * p))
    sp = np.array([pt(cam, q) for q in path[:n]], np.float32)
    m = np.zeros((H, W), np.float32)
    z = cam[0, 0]
    flick = 1 + 0.25 * math.sin(t * 60) + 0.15 * math.sin(t * 23)
    th = int(max(4, 9 * z * width * flick))
    cv2.polylines(m, [sp.astype(np.int32)], False, 1.0, th, cv2.LINE_AA)
    # crackle sparks along the beam
    r = np.random.default_rng(int(t * FPS))
    for _ in range(int(14 * p)):
        q = sp[r.integers(len(sp))] + r.normal(0, 10 * z, 2)
        cv2.circle(m, tuple(int(v) for v in q), int(r.uniform(2, 5) * z), 1.0, -1, cv2.LINE_AA)
    img *= 1 - 0.28 * fade
    img += glow_layer(m) * 0.9 * fade
    # muzzle flare at the mouth
    cv2.circle(img, tuple(int(v) for v in sp[0]), int(16 * z * flick), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)

def explosion(img, cam, center, t, tb, scale=1.0):
    u = t - tb
    if u < 0 or u > 2.6: return
    c = pt(cam, center); z = cam[0, 0] * scale
    # flash
    if u < 0.25: img += (1 - u / 0.25) * 0.6
    # fireball
    lay = np.zeros((H, W), np.float32)
    r0 = (40 + 160 * ease(u / 0.45)) * z
    rr = np.random.default_rng(int(tb * 100))
    for i in range(14):
        ang = rr.uniform(0, 2 * math.pi); dist = rr.uniform(0, 0.6) * r0
        rad = r0 * rr.uniform(0.35, 0.7)
        q = c + np.array([math.cos(ang), math.sin(ang)]) * dist - np.array([0, u * 60 * z])
        cv2.circle(lay, tuple(int(v) for v in q), int(rad), 1.0, -1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 6)
    heat = clamp(1 - u / 1.1)
    smoke = clamp((u - 0.3) / 0.6) * clamp((2.6 - u) / 1.2)
    fire_col = np.array([0.05, 0.3 + 0.45 * heat, 1.0], np.float32)       # BGR orange
    a = lay[..., None] * heat
    img[:] = img * (1 - a) + fire_col * a
    core = cv2.GaussianBlur(lay, (0, 0), 14)[..., None] * heat ** 2
    img[:] = img * (1 - core) + np.array([0.75, 0.97, 1.0], np.float32) * core
    # smoke puffs
    if smoke > 0:
        sl = np.zeros((H, W), np.float32)
        for i in range(10):
            ang = rr.uniform(0, 2 * math.pi)
            q = c + np.array([math.cos(ang), math.sin(ang) * 0.5]) * r0 * 0.9 - np.array([0, u * 90 * z])
            cv2.circle(sl, tuple(int(v) for v in q), int(r0 * rr.uniform(0.18, 0.32)), 1.0, -1, cv2.LINE_AA)
        sl = cv2.GaussianBlur(sl, (0, 0), 14)[..., None] * smoke * 0.5
        img[:] = img * (1 - sl) + np.array([0.33, 0.33, 0.35], np.float32) * sl
    # shock ring
    if u < 0.6:
        ring = np.zeros((H, W), np.float32)
        cv2.circle(ring, tuple(int(v) for v in c), int((60 + 500 * u) * z), 1.0, int(8 * z), cv2.LINE_AA)
        ring = cv2.GaussianBlur(ring, (0, 0), 3)[..., None] * (1 - u / 0.6) * 0.8
        img[:] = img * (1 - ring) + ring

def dust(img, cam, center, t, tb, width=200):
    u = t - tb
    if u < 0 or u > 2.0: return
    c = pt(cam, center); z = cam[0, 0]
    rr = np.random.default_rng(int(tb * 77))
    sl = np.zeros((H, W), np.float32)
    for i in range(16):
        x = rr.uniform(-1, 1) * width * (0.6 + u)
        q = c + np.array([x, -rr.uniform(0, 40) - u * 40]) * z
        cv2.circle(sl, tuple(int(v) for v in q), int(z * (25 + 50 * ease(u / 0.8)) * rr.uniform(0.6, 1.2)), 1.0, -1, cv2.LINE_AA)
    sl = cv2.GaussianBlur(sl, (0, 0), 12)[..., None] * clamp(1 - u / 2.0) * 0.6
    img[:] = img * (1 - sl) + np.array([0.6, 0.62, 0.64], np.float32) * sl

def impact(img, cam, center, t, tb, text=None):
    u = t - tb
    if u < 0 or u > 0.45: return
    c = pt(cam, center); z = cam[0, 0]
    sc = (0.5 + 0.7 * ease(u / 0.12)) * z * 70
    n = 12; poly = []
    rr = np.random.default_rng(int(tb * 50))
    for i in range(2 * n):
        rad = sc * (1.0 if i % 2 == 0 else 0.45) * rr.uniform(0.85, 1.15)
        a = i * math.pi / n
        poly.append(c + np.array([math.cos(a), math.sin(a)]) * rad)
    poly = np.array(poly, np.int32)
    al = clamp((0.45 - u) / 0.15)
    ov = img.copy()
    cv2.fillPoly(ov, [poly], (0.15, 0.85, 1.0), cv2.LINE_AA)
    cv2.polylines(ov, [poly], True, (0.1, 0.1, 0.75), int(4 * z), cv2.LINE_AA)
    img[:] = img * (1 - al) + ov * al
    if text: fx_text(img, text, c, sc * 0.55, al, -12)

_font_cache = {}
def font(path, size, wght=None):
    k = (path, size)
    if k not in _font_cache:
        f = ImageFont.truetype(path, size)
        if wght:
            try: f.set_variation_by_axes([wght])
            except Exception: pass
        _font_cache[k] = f
    return _font_cache[k]

_txt_cache = {}
def text_sprite(text, path, size, fill, stroke, sw, wght=None):
    k = (text, path, size, fill, stroke, sw)
    if k in _txt_cache: return _txt_cache[k]
    f = font(path, size, wght)
    l, t_, r, b = f.getbbox(text, stroke_width=sw)
    im = Image.new('RGBA', (r - l + 20, b - t_ + 20), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10 - l, 10 - t_), text, font=f, fill=fill, stroke_width=sw, stroke_fill=stroke)
    a = np.asarray(im).astype(np.float32) / 255
    rgba = np.dstack([a[..., 2], a[..., 1], a[..., 0], a[..., 3]])   # -> BGRA
    _txt_cache[k] = rgba
    return rgba

def blit(img, rgba, cx, cy, alpha=1.0, angle=0.0, scale=1.0):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += cx - w / 2; M[1, 2] += cy - h / 2
    wp = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    a = wp[..., 3:4] * alpha
    img[:] = img * (1 - a) + wp[..., :3] * a

def fx_text(img, text, c, size, alpha, angle):
    spr = text_sprite(text, FONT_FX, 64, (255, 230, 40, 255), (200, 20, 20, 255), 6)
    blit(img, spr, c[0], c[1] - size * 0.2, alpha, angle, size / 50)

CAPTIONS = [  # (start, end, text)
    (WALK[0] - 0.4, WALK[1], 'Годзилла идёт...'),
    (11.7, 13.3, '...и натыкается на город!'),
    (13.4, 17.3, 'Годзилла разрушает город!'),
    (18.4, 20.5, 'Идёт дальше — и снова дома!'),
    (20.6, 23.6, 'АТОМНОЕ ДЫХАНИЕ!'),
    (D2[1] + 1.0, HITS[-1] + 0.4, 'Годзилла нападает на Конга!'),
    (KO + 0.2, T2 + 12.6, 'Годзилла побеждает Конга!'),
    (T2 + 13.7, CLASH[0] - 0.2, 'Идёт дальше... а тут Сцилла!'),
    (CLASH[0] - 0.1, BREATH2[0] - 0.1, 'Они борются!'),
    (BREATH2[0], T2 + 23.4, 'Годзилла побеждает атомным дыханием!'),
    (T2 + 24.4, END2 + FADE, 'КОНЕЦ'),
]
def captions(img, t):
    for a, b, txt in CAPTIONS:
        v = win(t, a, b, 0.3, 0.3)
        if v <= 0: continue
        big = txt.isupper()
        size = 112 if big else 84
        spr = text_sprite(txt, FONT_CAP, size, (40, 30, 30, 255), (250, 248, 240, 255), 7, 700)
        pop = 1 + 0.12 * (1 - ease((t - a) / 0.25))
        if txt == 'КОНЕЦ':
            img[:] = img * (1 - 0.6 * v) + np.array([0.94, 0.955, 0.965], np.float32) * 0.6 * v
        y = H / 2 if txt == 'КОНЕЦ' else H - 110
        sc = (1.6 if txt == 'КОНЕЦ' else 1.0) * pop
        blit(img, spr, W / 2, y, v, 0, sc)

def draw_pen(img, p):
    x, y = int(p[0]), int(p[1]); tipx, tipy = x + 70, y - 110
    cv2.line(img, (x + 14, y - 22), (tipx, tipy), (40, 40, 40), 24, cv2.LINE_AA)
    cv2.line(img, (x + 14, y - 22), (tipx, tipy), (60, 60, 215), 18, cv2.LINE_AA)
    cv2.line(img, (x + 40, y - 63), (tipx, tipy), (235, 235, 235), 18, cv2.LINE_AA)
    cv2.line(img, (x + 3, y - 5), (x + 14, y - 22), (40, 40, 190), 10, cv2.LINE_AA)

# ---------------------------------------------------------------- frame rendering
def bg(cam):
    return cv2.warpAffine(PAPER, cam[:2], (W, H), borderMode=cv2.BORDER_REFLECT)

def render1(t):
    cam = cam_at(CAM1, t); img = bg(cam); pen = None
    for s in SC1:
        p = s.draw(img, t, cam)
        if p is not None: pen = p
    dust(img, cam, c1(1390, 960), t, SMASH[0], 90)
    dust(img, cam, c1(930, 960), t, SMASH[1], 70)
    if SMASH[0] - 0.05 < t < SMASH[0] + 0.4: impact(img, cam, c1(1380, 880), t, SMASH[0], 'БАХ!')
    if SMASH[1] - 0.05 < t < SMASH[1] + 0.4: impact(img, cam, c1(930, 900), t, SMASH[1], 'БУМ!')
    # charge glow on Godzilla before breath
    if BREATH1[0] - 1.0 < t < BREATH1[1] + 0.3:
        e = clamp((t - BREATH1[0] + 1.0) / 1.0) * clamp((BREATH1[1] + 0.3 - t) / 0.3)
        g = [s for s in SC1 if s.tag == 'breather'][0].glow_mask(t, cam)
        g[:int(pt(cam, c1(0, 700))[1])] *= 1; img += glow_layer(g * e, sig=(16, 6, 1)) * 0.35 * e * (1 + 0.3 * math.sin(t * 30))
    beam(img, cam, BEAM1, t, *BREATH1)
    explosion(img, cam, c1(1660, 880), t, BOOM1, 0.8)
    out = np.clip(img, 0, 1)
    captions(out, t)
    out = (out * 255).astype(np.uint8)
    if pen is not None: draw_pen(out, pen)
    return out

KONG_HIT = c2(735, 840)
def render2(t):
    cam = cam_at(CAM2, t); img = bg(cam); pen = None
    for s in SC2:
        p = s.draw(img, t, cam)
        if p is not None: pen = p
    for h in HITS: impact(img, cam, KONG_HIT, t, h, ['БАЦ!', 'ХРЯСЬ!', 'БАМ!'][HITS.index(h)])
    impact(img, cam, KONG_HIT + np.array([10, -30]), t, KO, 'БАБАХ!')
    dust(img, cam, c2(860, 990), t, KO + 0.7, 80)
    for i, c in enumerate(CLASH): impact(img, cam, c2(1320, 760) + np.array([0, (i % 2) * 40 - 20]), t, c, ['ТЫЩ!', 'БАЦ!', 'ГРР!', 'БАМ!'][i])
    # stripes glow = Godzilla charging
    if BREATH2[0] - 0.8 < t < BREATH2[1] + 0.3:
        e = clamp((t - BREATH2[0] + 0.8) / 0.8) * clamp((BREATH2[1] + 0.3 - t) / 0.3)
        g = sum(s.glow_mask(t, cam) for s in SC2 if s.tag in ('stripes',))
        img += glow_layer(g * e, sig=(18, 7, 1)) * 0.5 * e * (1 + 0.3 * math.sin(t * 30))
    head = [s for s in SC2 if s.tag == 'fighter' and s.center[1] < c2(0, 690)[1]]
    o = head[0].center if head else c2(1260, 650)
    sc = [s for s in SC2 if s.tag == 'scylla'][0]
    target = c2(1395, 720)
    path = np.array([o + (target - o) * u + np.array([0, -math.sin(u * math.pi) * 25]) for u in np.linspace(0, 1, 40)], np.float32)
    beam(img, cam, path, t, *BREATH2, width=1.3)
    explosion(img, cam, target + np.array([25, 10], np.float32), t, BOOM2, 0.55)
    out = np.clip(img, 0, 1)
    captions(out, t)
    out = (out * 255).astype(np.uint8)
    if pen is not None: draw_pen(out, pen)
    return out

def frame(fi):
    t = fi / FPS
    if t < END1:
        fr = render1(t)
        if t < 0.6: fr = (fr * (t / 0.6) + 238 * (1 - t / 0.6)).astype(np.uint8)
        return fr
    if t < T2:
        u = ease((t - END1) / TURN)
        a = render1(END1 - 0.001); b = render2(t)
        sh = int(u * W); fr = np.empty_like(a)
        fr[:, :W - sh] = a[:, sh:]; fr[:, W - sh:] = b[:, :sh]
        if 0 < sh < W:
            x = W - sh; lo = max(0, x - 40)
            fr[:, lo:x] = (fr[:, lo:x] * np.linspace(1, 0.7, x - lo)[None, :, None]).astype(np.uint8)
        return fr
    fr = render2(t)
    if t > END2:
        u = ease((t - END2) / FADE); fr = (fr * (1 - u) + 240 * u).astype(np.uint8)
    return fr

# ---------------------------------------------------------------- sound
SR = 44100
def synth(path):
    n = int(TOTAL * SR) + SR; out = np.zeros(n, np.float32); r = np.random.default_rng(3)
    def add(t, sig):
        i = int(t * SR); j = min(n, i + len(sig)); out[i:j] += sig[:j - i]
    def env(d, a=0.005, rel=None):
        tt = np.arange(int(d * SR)) / SR
        return np.minimum(tt / a, 1) * np.exp(-tt / (rel or d / 4)), tt
    def lowpass(x, k): return np.convolve(x, np.ones(k) / k, 'same')
    for t, kind, p in AUDIO:
        if kind == 'stomp':
            e, tt = env(0.5, 0.003, 0.11)
            f = 55 * np.exp(-tt * 3)
            sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * e + lowpass(r.standard_normal(len(tt)), 30) * e * 0.5
            add(t, sig * 0.7 * p)
        elif kind == 'crash':
            e, tt = env(1.6, 0.003, 0.35)
            noise = lowpass(r.standard_normal(len(tt)), 6) * e
            rumble = np.sin(2 * np.pi * 42 * tt) * e
            # rubble clicks
            cl = np.zeros(len(tt), np.float32)
            for _ in range(40):
                i = int(r.uniform(0, 1.2) * SR); k = int(0.02 * SR)
                if i + k < len(cl): cl[i:i + k] += r.standard_normal(k) * np.exp(-np.arange(k) / 200) * r.uniform(0.2, 0.6)
            add(t, (noise * 0.6 + rumble * 0.6 + cl * 0.5) * p)
        elif kind == 'punch':
            e, tt = env(0.35, 0.002, 0.06)
            sig = np.sin(2 * np.pi * np.cumsum(120 * np.exp(-tt * 12) + 50) / SR) * e + r.standard_normal(len(tt)) * e * 0.35
            add(t, sig * 0.8 * p)
        elif kind == 'charge':
            d = 1.0 * p; tt = np.arange(int(d * SR)) / SR
            f = 120 + 600 * (tt / d) ** 2
            sig = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.12 * (tt / d)
            add(t, sig)
        elif kind == 'beam':
            d = p + 0.3; tt = np.arange(int(d * SR)) / SR
            e = np.minimum(tt / 0.05, 1) * np.minimum((d - tt) / 0.3, 1)
            saw = ((tt * 95) % 1 - 0.5) + 0.6 * ((tt * 190.7) % 1 - 0.5) + 0.3 * ((tt * 287) % 1 - 0.5)
            trem = 1 + 0.4 * np.sin(2 * np.pi * 17 * tt)
            hiss = lowpass(r.standard_normal(len(tt)), 3) * 0.4
            add(t, (saw * trem * 0.5 + hiss) * e * 0.55)
        elif kind == 'boom':
            e, tt = env(2.8, 0.002, 0.55)
            sig = np.sin(2 * np.pi * np.cumsum(70 * np.exp(-tt * 2) + 28) / SR) * e * 1.1
            sig += lowpass(r.standard_normal(len(tt)), 8) * e * 0.8
            add(t, sig * p)
        elif kind == 'ding':
            tt = np.arange(int(2.5 * SR)) / SR; e = np.exp(-tt * 1.6) * np.minimum(tt / 0.01, 1)
            for f, a in ((523.25, 1), (659.25, 0.8), (783.99, 0.7), (1046.5, 0.4)):
                add(t, np.sin(2 * np.pi * f * tt) * e * a * 0.15)
    # soft background drum loop for drive (scene parts only)
    beat = 60 / 100
    for k in range(int(TOTAL / beat)):
        tb = k * beat
        if tb > END2: break
        e, tt = env(0.18, 0.002, 0.04)
        add(tb, np.sin(2 * np.pi * 80 * tt) * e * 0.12)
    out = np.tanh(out * 1.2) * 0.85
    pcm = (out * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm.tobytes())

# ---------------------------------------------------------------- main
if __name__ == '__main__':
    outp = sys.argv[1] if len(sys.argv) > 1 else 'godzilla.mp4'
    only = [int(x) for x in sys.argv[2:]]
    if only:
        for fi in only: cv2.imwrite(f'prev_{fi:04d}.jpg', frame(fi))
        sys.exit()
    wav = outp + '.wav'; synth(wav)
    NF = int(TOTAL * FPS)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                           '-r', str(FPS), '-i', '-', '-i', wav, '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
                           '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', outp],
                          stdin=subprocess.PIPE)
    for fi in range(NF):
        ff.stdin.write(frame(fi).tobytes())
        if fi % 150 == 0: print(fi, '/', NF, flush=True)
    ff.stdin.close(); ff.wait(); os.remove(wav)
    print('done', TOTAL, 's')
