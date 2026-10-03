"""
Frame renderer (CPU, numba).

Layers, back to front
  background : procedural star catalogue + faint galactic glow, both seen
               through the black hole's gravitational lens (point-lens
               deflection, two images per source, magnification).
  gas        : simulated plasma particles + analytic accretion-disk particles,
               splatted as soft volumetric kernels into a mip pyramid (size =
               physical smoothing length and depth-of-field blur), with
               multi-sample motion blur along the true orbital arc,
               gravitational lensing of everything behind the hole, Doppler
               beaming and black-body colour.
  star       : ray-traced tidal ellipsoid with granulation, limb darkening and
               prominences; fades into the particle gas as layers are stripped.
  shadow     : nothing behind the hole is seen inside the critical impact
               parameter b_c = 3*sqrt(3)/2 R_s.
  post       : bloom, exposure, filmic tone map, grade, vignette, grain.
"""
import os
import sys
import numpy as np
import cv2
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as P
import timeline as TL
import camera as CAM
import disk_trace as DT
from colour import BB_TABLE, LOG_T_MIN, LOG_T_STEP, blackbody

NLEV = 7
SHOCK_K = 6.7e5
DISK_L = 0.10       # volumetric particle part of the disk (haze above/below the plane)
DISK_K = 7.0        # ray-traced thin disk surface brightness
DISK_T0 = 21.9
DISK_ROUT = 55.0
STAR_I = 0.80       # photosphere surface brightness (linear, pre-exposure)
S0 = 0.9            # effective sigma (px) of a splat at pyramid level 0


# ==========================================================================
#  Numba primitives
# ==========================================================================
@njit(inline='always')
def _smooth(a, b, x):
    t = (x - a) / (b - a)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(inline='always')
def _deposit(levels, slab, px, py, sig, r, g, b):
    lf = np.log2(max(sig, S0) / S0)
    if lf > NLEV - 1.001:
        lf = NLEV - 1.001
    L0 = int(lf)
    fr = lf - L0
    for q in range(2):
        L = L0 + q
        w = (1.0 - fr) if q == 0 else fr
        if w <= 1e-4:
            continue
        buf = levels[L]
        sc = 1.0 / (1 << L)
        x = px * sc - 0.5
        y = py * sc - 0.5
        ix = int(np.floor(x))
        iy = int(np.floor(y))
        fx = x - ix
        fy = y - iy
        Hh = buf.shape[1]
        Ww = buf.shape[2]
        for dy in range(2):
            yy = iy + dy
            if yy < 0 or yy >= Hh:
                continue
            wy = fy if dy == 1 else 1.0 - fy
            for dx in range(2):
                xx = ix + dx
                if xx < 0 or xx >= Ww:
                    continue
                ww = w * wy * (fx if dx == 1 else 1.0 - fx)
                buf[slab, yy, xx, 0] += ww * r
                buf[slab, yy, xx, 1] += ww * g
                buf[slab, yy, xx, 2] += ww * b


@njit(inline='always')
def _project(cp, dx, dy, dz, out):
    """Direction -> pixel.  cp layout documented in pack_camera()."""
    xc = dx * cp[3] + dy * cp[4] + dz * cp[5]
    yc = dx * cp[6] + dy * cp[7] + dz * cp[8]
    zc = dx * cp[9] + dy * cp[10] + dz * cp[11]
    if zc < 1e-3:
        return False
    out[0] = cp[13] * 0.5 + cp[12] * xc / zc
    out[1] = cp[14] * 0.5 - cp[12] * yc / zc
    return True


@njit(inline='always')
def _lens(cp, X, Y, Z, d_src_inf, img):
    """Point-lens images of a world point (or of a direction if d_src_inf).

    img rows: (dir_x, dir_y, dir_z, magnification); returns number of images
    and whether the point lies behind the lens plane.
    """
    if d_src_inf:
        px, py, pz = X, Y, Z
        dist = 1e30
    else:
        px, py, pz = X - cp[0], Y - cp[1], Z - cp[2]
        dist = np.sqrt(px * px + py * py + pz * pz) + 1e-12
    nx, ny, nz = cp[18], cp[19], cp[20]
    Dl = cp[17]
    if d_src_inf:
        nrm = np.sqrt(px * px + py * py + pz * pz)
        px /= nrm
        py /= nrm
        pz /= nrm
        zl = px * nx + py * ny + pz * nz
        ex, ey, ez = px - zl * nx, py - zl * ny, pz - zl * nz
        th2 = 2.0 / Dl
        behind = zl > 0.0
    else:
        zl = px * nx + py * ny + pz * nz
        ex, ey, ez = px - zl * nx, py - zl * ny, pz - zl * nz
        dls = zl - Dl
        behind = dls > 0.0
        th2 = 2.0 * dls / (Dl * dist) if behind else 0.0
        zl /= dist
        ex /= dist
        ey /= dist
        ez /= dist
    en = np.sqrt(ex * ex + ey * ey + ez * ez)
    if not behind or th2 <= 0.0:
        img[0, 0], img[0, 1], img[0, 2], img[0, 3] = ex + zl * nx, ey + zl * ny, ez + zl * nz, 1.0
        return 1, False
    if en < 1e-9:
        ex, ey, ez = cp[6], cp[7], cp[8]
        en = 1.0
    ex /= en
    ey /= en
    ez /= en
    psi = np.arctan2(en, zl)
    thE = np.sqrt(th2)
    root = np.sqrt(psi * psi + 4.0 * th2)
    t1 = 0.5 * (psi + root)
    t2 = 0.5 * (root - psi)          # secondary image, opposite side
    u = max(psi / thE, 1e-3)
    mu1 = (u * u + 2.0) / (2.0 * u * np.sqrt(u * u + 4.0)) + 0.5
    mu2 = mu1 - 1.0
    mu1 = min(mu1, 20.0)
    mu2 = min(mu2, 19.0)
    th_sh = cp[21]
    n = 0
    if t1 > th_sh:
        c, s = np.cos(t1), np.sin(t1)
        img[n, 0], img[n, 1], img[n, 2], img[n, 3] = c * nx + s * ex, c * ny + s * ey, c * nz + s * ez, mu1
        n += 1
    if t2 > th_sh and mu2 > 1e-3:
        c, s = np.cos(t2), np.sin(t2)
        img[n, 0], img[n, 1], img[n, 2], img[n, 3] = c * nx - s * ex, c * ny - s * ey, c * nz - s * ez, mu2
        n += 1
    return n, True


@njit(inline='always')
def _bb(T, out):
    if T < 1200.0:
        T = 1200.0
    if T > 59000.0:
        T = 59000.0
    f = (np.log(T) - LOG_T_MIN) / LOG_T_STEP
    i = int(f)
    u = f - i
    for j in range(3):
        out[j] = BB_TABLE[i, j] * (1 - u) + BB_TABLE[i + 1, j] * u


@njit(inline='always')
def _emit(levels, cp, X, Y, Z, h, r, g, b, star_depth, wgt, img, pix):
    """Lens, project, size and deposit one emitting point."""
    px_, py_, pz_ = X - cp[0], Y - cp[1], Z - cp[2]
    dist2 = px_ * px_ + py_ * py_ + pz_ * pz_
    depth = px_ * cp[9] + py_ * cp[10] + pz_ * cp[11]
    if depth < 0.3:
        return
    n, behind = _lens(cp, X, Y, Z, False, img)
    behind_star = depth > star_depth
    slab = 2 - (1 if behind else 0) - (1 if behind_star else 0)
    F = cp[12]
    sig_size = h * F / depth
    coc = cp[16] * abs(1.0 - cp[15] / depth)
    sig = np.sqrt(sig_size * sig_size + 0.25 * coc * coc + 0.35)
    flux = wgt * F * F / dist2
    for k in range(n):
        if not _project(cp, img[k, 0], img[k, 1], img[k, 2], pix):
            continue
        if pix[0] < -60 or pix[1] < -60 or pix[0] > cp[13] + 60 or pix[1] > cp[14] + 60:
            continue
        f = flux * img[k, 3]
        _deposit(levels, slab, pix[0], pix[1], sig, r * f, g * f, b * f)


@njit(fastmath=True)
def splat_gas(levels, cps, dts, pos, vel, col, hs, lab, turb, rnd, par_len, perp_len, star_depth,
              t_now, max_arc):
    """Simulated plasma: children per particle x motion-blur samples x lens images."""
    N = pos.shape[0]
    M = rnd.shape[1]
    K = cps.shape[0]
    img = np.empty((2, 4))
    pix = np.empty(2)
    wk = 1.0 / (K * M)
    for i in range(N):
        cr, cg, cb = col[i, 0], col[i, 1], col[i, 2]
        if cr + cg + cb <= 0.0:
            continue
        x0, y0, z0 = pos[i, 0], pos[i, 1], pos[i, 2]
        vx, vy, vz = vel[i, 0], vel[i, 1], vel[i, 2]
        vm = np.sqrt(vx * vx + vy * vy + vz * vz) + 1e-12
        ux, uy, uz = vx / vm, vy / vm, vz / vm
        # two perpendicular axes
        ax, ay, az = -uy, ux, 0.0
        an = np.sqrt(ax * ax + ay * ay) + 1e-12
        ax /= an
        ay /= an
        bx, by, bz = uy * az - uz * ay, uz * ax - ux * az, ux * ay - uy * ax
        r2 = x0 * x0 + y0 * y0
        rr = np.sqrt(r2) + 1e-9
        om = (x0 * vy - y0 * vx) / (r2 + 1e-9)
        vr = (x0 * vx + y0 * vy) / rr
        # turbulence: slowly evolving displacement keyed to the gas element
        tA = turb[i]
        l0, l1, l2 = lab[i, 0], lab[i, 1], lab[i, 2]
        tx = tA * (np.sin(5.1 * l0 + 3.3 * l2 + 0.0021 * t_now) + 0.5 * np.sin(11.7 * l1 - 0.0037 * t_now))
        ty = tA * (np.sin(4.7 * l1 + 2.9 * l0 + 0.0017 * t_now + 1.3) + 0.5 * np.sin(12.3 * l2 + 0.0041 * t_now))
        tz = 0.5 * tA * np.sin(6.1 * l2 + 3.7 * l1 - 0.0023 * t_now + 2.1)
        for j in range(M):
            oa = rnd[i, j, 0] * par_len[i]
            ob = rnd[i, j, 1] * perp_len[i]
            oc = rnd[i, j, 2] * perp_len[i]
            for k in range(K):
                dt = dts[k]
                ang = om * dt
                if ang > max_arc:
                    ang = max_arc
                if ang < -max_arc:
                    ang = -max_arc
                dr = vr * dt
                lim = 0.3 * rr
                if dr > lim:
                    dr = lim
                if dr < -lim:
                    dr = -lim
                ca, sa = np.cos(ang), np.sin(ang)
                sc = (rr + dr) / rr
                X = (x0 * ca - y0 * sa) * sc
                Y = (x0 * sa + y0 * ca) * sc
                Z = z0 + vz * dt
                X += ux * oa + ax * ob + bx * oc + tx
                Y += uy * oa + ay * ob + by * oc + ty
                Z += uz * oa + az * ob + bz * oc + tz
                _emit(levels, cps[k], X, Y, Z, hs[i], cr, cg, cb, star_depth, wk, img, pix)


@njit(fastmath=True)
def splat_disk(levels, cps, dts, r, phi0, z, om, lum, temp, vis, t_now, h_scale, star_depth, max_arc):
    """Analytic accretion disk on circular orbits (Doppler beamed)."""
    N = r.shape[0]
    K = cps.shape[0]
    img = np.empty((2, 4))
    pix = np.empty(2)
    rgb = np.empty(3)
    wk = 1.0 / K
    for i in range(N):
        if vis[i] <= 0.0:
            continue
        ri = r[i]
        vmag = om[i] * ri                       # orbital speed (c = 1)
        gam = 1.0 / np.sqrt(max(1.0 - vmag * vmag, 0.05))
        grav = np.sqrt(max(1.0 - 1.0 / ri, 0.05))
        for k in range(K):
            ang = om[i] * dts[k]
            if ang > max_arc:
                ang = max_arc
            ph = phi0[i] + om[i] * t_now + ang
            c, s = np.cos(ph), np.sin(ph)
            X, Y, Z = ri * c, ri * s, z[i]
            cp = cps[k]
            # unit vector towards the camera
            qx, qy, qz = cp[0] - X, cp[1] - Y, cp[2] - Z
            qn = np.sqrt(qx * qx + qy * qy + qz * qz)
            bn = vmag * (-s * qx + c * qy) / qn       # beta . n
            D = grav / (gam * (1.0 - bn))
            _bb(temp[i] * D, rgb)
            w = lum[i] * vis[i] * D ** 3 * wk
            hh = h_scale * (0.15 + 0.03 * ri)
            _emit(levels, cp, X, Y, Z, hh, rgb[0] * w, rgb[1] * w, rgb[2] * w, star_depth, 1.0, img, pix)


@njit(fastmath=True)
def splat_stars(levels, cps, dirs, flux, cols, dist, slab, sig):
    """Background stars (directions, or finite distance for parallax)."""
    N = dirs.shape[0]
    K = cps.shape[0]
    img = np.empty((2, 4))
    pix = np.empty(2)
    for i in range(N):
        for k in range(K):
            cp = cps[k]
            if dist[i] > 0:
                n, beh = _lens(cp, dirs[i, 0] * dist[i], dirs[i, 1] * dist[i], dirs[i, 2] * dist[i], False, img)
            else:
                n, beh = _lens(cp, dirs[i, 0], dirs[i, 1], dirs[i, 2], True, img)
            for m in range(n):
                if not _project(cp, img[m, 0], img[m, 1], img[m, 2], pix):
                    continue
                if pix[0] < -8 or pix[1] < -8 or pix[0] > cp[13] + 8 or pix[1] > cp[14] + 8:
                    continue
                f = flux[i] * img[m, 3] / K
                _deposit(levels, slab, pix[0], pix[1], sig[i], cols[i, 0] * f, cols[i, 1] * f, cols[i, 2] * f)


@njit(inline='always')
def _hash3(ix, iy, iz):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(inline='always')
def _worley(x, y, z):
    ix, iy, iz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    f1, f2 = 9.0, 9.0
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                cx, cy, cz = ix + dx, iy + dy, iz + dz
                px = cx + _hash3(cx, cy, cz)
                py = cy + _hash3(cy + 17, cz + 3, cx + 11)
                pz = cz + _hash3(cz + 29, cx + 7, cy + 5)
                d = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
                if d < f1:
                    f2 = f1
                    f1 = d
                elif d < f2:
                    f2 = d
    return np.sqrt(f1), np.sqrt(f2)


@njit(inline='always')
def _vnoise(x, y, z):
    ix, iy, iz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    fx, fy, fz = x - ix, y - iy, z - iz
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    fz = fz * fz * (3 - 2 * fz)
    acc = 0.0
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy) * (fz if dz else 1 - fz)
                acc += w * _hash3(ix + dx, iy + dy, iz + dz)
    return acc


@njit(fastmath=True)
def shade_star(rgb, alpha, depth, cp, x0, x1, y0, y1, c, Ainv, AinvT_n, star_lensed, th2, t_gran,
               T_eff, bright):
    """Ray trace the (tidally deformed) stellar photosphere."""
    F = cp[12]
    W, H = cp[13], cp[14]
    oq = np.empty(3)
    for k in range(3):
        oq[k] = Ainv[k, 0] * (cp[0] - c[0]) + Ainv[k, 1] * (cp[1] - c[1]) + Ainv[k, 2] * (cp[2] - c[2])
    col = np.empty(3)
    nx, ny, nz = cp[18], cp[19], cp[20]
    for py in range(y0, y1):
        for px in range(x0, x1):
            xc = (px + 0.5 - W * 0.5) / F
            yc = -(py + 0.5 - H * 0.5) / F
            dx = cp[3] * xc + cp[6] * yc + cp[9]
            dy = cp[4] * xc + cp[7] * yc + cp[10]
            dz = cp[5] * xc + cp[8] * yc + cp[11]
            dn = np.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dn
            dy /= dn
            dz /= dn
            if star_lensed:
                zl = dx * nx + dy * ny + dz * nz
                ex, ey, ez = dx - zl * nx, dy - zl * ny, dz - zl * nz
                en = np.sqrt(ex * ex + ey * ey + ez * ez) + 1e-12
                th = np.arctan2(en, zl)
                if th < cp[21]:
                    continue
                psi = th - th2 / th
                ex /= en
                ey /= en
                ez /= en
                cs, sn = np.cos(psi), np.sin(psi)
                dx, dy, dz = cs * nx + sn * ex, cs * ny + sn * ey, cs * nz + sn * ez
            dq0 = Ainv[0, 0] * dx + Ainv[0, 1] * dy + Ainv[0, 2] * dz
            dq1 = Ainv[1, 0] * dx + Ainv[1, 1] * dy + Ainv[1, 2] * dz
            dq2 = Ainv[2, 0] * dx + Ainv[2, 1] * dy + Ainv[2, 2] * dz
            a = dq0 * dq0 + dq1 * dq1 + dq2 * dq2
            b = oq[0] * dq0 + oq[1] * dq1 + oq[2] * dq2
            cc = oq[0] * oq[0] + oq[1] * oq[1] + oq[2] * oq[2] - 1.0
            disc = b * b - a * cc
            # closest approach (in units of the stellar radius) for an AA edge
            dmin2 = cc + 1.0 - b * b / a
            dmin = np.sqrt(max(dmin2, 0.0))
            pixq = np.sqrt(cc + 1.0) / (F * np.sqrt(a))
            cov = min(max((1.0 - dmin) / (1.2 * pixq) + 0.5, 0.0), 1.0)
            if cov <= 0.0:
                continue
            s = (-b - np.sqrt(max(disc, 0.0))) / a
            q0, q1, q2 = oq[0] + s * dq0, oq[1] + s * dq1, oq[2] + s * dq2
            qn = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2) + 1e-12
            q0 /= qn
            q1 /= qn
            q2 /= qn
            # world normal = A^-T q
            wnx = AinvT_n[0, 0] * q0 + AinvT_n[0, 1] * q1 + AinvT_n[0, 2] * q2
            wny = AinvT_n[1, 0] * q0 + AinvT_n[1, 1] * q1 + AinvT_n[1, 2] * q2
            wnz = AinvT_n[2, 0] * q0 + AinvT_n[2, 1] * q1 + AinvT_n[2, 2] * q2
            wn = np.sqrt(wnx * wnx + wny * wny + wnz * wnz) + 1e-12
            mu = -(wnx * dx + wny * dy + wnz * dz) / wn
            mu = min(max(mu, 0.0), 1.0)
            # granulation: cellular convection cells + mesogranular mottling
            G = 26.0
            gx, gy, gz = q0 * G, q1 * G, q2 * G
            ta = t_gran
            f1, f2 = _worley(gx + 0.31 * ta, gy - 0.17 * ta, gz + 0.23 * ta)
            cell = min((f2 - f1) * 1.6, 1.0)                 # bright centres, dark lanes
            f1b, f2b = _worley(gx * 0.5 - 0.11 * ta + 7.0, gy * 0.5 + 0.13 * ta, gz * 0.5 + 3.0)
            cell = 0.65 * cell + 0.35 * min((f2b - f1b) * 1.4, 1.0)
            mott = _vnoise(q0 * 6.0 + 0.05 * ta, q1 * 6.0, q2 * 6.0 - 0.04 * ta)
            spot = 1.0
            lim = 1.0 - 0.62 * (1.0 - mu) - 0.18 * (1.0 - mu) ** 2
            I = bright * lim * (0.62 + 0.55 * cell) * (0.88 + 0.24 * mott) * spot
            # faculae: brighter network near the limb
            I *= 1.0 + 0.25 * (1.0 - mu) ** 2 * _smooth(0.55, 0.8, mott)
            T = T_eff * (0.92 + 0.12 * cell) * (0.78 + 0.22 * mu ** 0.5)
            _bb(T, col)
            for j in range(3):
                rgb[py, px, j] = I * col[j]
            alpha[py, px] = cov
            depth[py, px] = s


@njit(fastmath=True)
def nebula(out, cp, step, t_gal):
    """Very faint galactic glow, lensed (rendered at reduced resolution)."""
    H, W = out.shape[0], out.shape[1]
    F = cp[12] / step
    nx, ny, nz = cp[18], cp[19], cp[20]
    th2 = 2.0 / cp[17]
    # galactic plane normal (tilted relative to the orbital plane)
    gnx, gny, gnz = 0.32, -0.55, 0.77
    gn = np.sqrt(gnx * gnx + gny * gny + gnz * gnz)
    gnx /= gn
    gny /= gn
    gnz /= gn
    for py in range(H):
        for px in range(W):
            xc = (px + 0.5 - W * 0.5) / F
            yc = -(py + 0.5 - H * 0.5) / F
            dx = cp[3] * xc + cp[6] * yc + cp[9]
            dy = cp[4] * xc + cp[7] * yc + cp[10]
            dz = cp[5] * xc + cp[8] * yc + cp[11]
            dn = np.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dn
            dy /= dn
            dz /= dn
            zl = dx * nx + dy * ny + dz * nz
            ex, ey, ez = dx - zl * nx, dy - zl * ny, dz - zl * nz
            en = np.sqrt(ex * ex + ey * ey + ez * ez) + 1e-12
            th = np.arctan2(en, zl)
            if th < cp[21]:
                out[py, px, 0] = 0.0
                out[py, px, 1] = 0.0
                out[py, px, 2] = 0.0
                continue
            if zl > 0:
                psi = th - th2 / th
                ex /= en
                ey /= en
                ez /= en
                cs, sn = np.cos(psi), np.sin(psi)
                dx, dy, dz = cs * nx + sn * ex, cs * ny + sn * ey, cs * nz + sn * ez
            lat = dx * gnx + dy * gny + dz * gnz
            band = np.exp(-(lat / 0.16) ** 2)
            n1 = _vnoise(dx * 2.5 + 1.0, dy * 2.5 + 2.0, dz * 2.5) * 0.65 + _vnoise(dx * 7.0, dy * 7.0, dz * 7.0) * 0.35
            dust = _smooth(0.5, 0.8, _vnoise(dx * 10.0 + 5.0, dy * 10.0, dz * 10.0 + 9.0))
            v = band * (0.3 + 0.8 * n1) * (1.0 - 0.6 * dust * band)
            haze = 0.0
            out[py, px, 0] = (v * 1.0 + haze * 0.7)
            out[py, px, 1] = (v * 0.93 + haze * 0.8)
            out[py, px, 2] = (v * 0.86 + haze * 1.0)


# ==========================================================================
#  Scene data
# ==========================================================================
class Scene:
    def __init__(self, snap_dir, seed=11):
        self.dir = snap_dir
        self.pos = np.load(os.path.join(snap_dir, 'pos.npy'), mmap_mode='r')
        self.vel = np.load(os.path.join(snap_dir, 'vel.npy'), mmap_mode='r')
        self.glow = np.load(os.path.join(snap_dir, 'glow.npy'), mmap_mode='r')
        self.bond = np.load(os.path.join(snap_dir, 'bond.npy'), mmap_mode='r')
        self.alive = np.load(os.path.join(snap_dir, 'alive.npy'), mmap_mode='r')
        self.mass = np.load(os.path.join(snap_dir, 'mass.npy')).astype(np.float64)
        # thermal emission is radiated from surfaces, not proportional to the mass of
        # the dense core: a flatter weight keeps the stretched star from looking like a hot point
        wv = self.mass ** 0.35
        self.emis = wv / wv.sum()
        self.rest = np.load(os.path.join(snap_dir, 'rest.npy')).astype(np.float64)
        self.trel = np.load(os.path.join(snap_dir, 'trel.npy'))
        self.centre = np.load(os.path.join(snap_dir, 'centre.npy'))
        self.times = np.load(os.path.join(snap_dir, 'times.npy'))
        self.F = self.pos.shape[0]
        self.N = self.pos.shape[1]
        rng = np.random.default_rng(seed)
        M = 3
        self.rnd = np.empty((self.N, M, 3), np.float64)
        self.rnd[:, :, 0] = rng.uniform(-1, 1, (self.N, M))
        self.rnd[:, :, 1:] = rng.normal(0, 0.6, (self.N, M, 2))
        self.probe = np.sort(rng.choice(self.N, 3000, replace=False))
        self._build_stars(rng)
        self._build_disk(rng)
        self._build_prominences(rng)
        self._calibrate_star()

    # ---------------- background stars
    def _build_stars(self, rng):
        n = 150000
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        g = np.array([0.32, -0.55, 0.77])
        g /= np.linalg.norm(g)
        lat = d @ g
        keep = rng.random(n) < 0.35 + 0.65 * np.exp(-(lat / 0.25) ** 2)
        d = d[keep]
        n = len(d)
        mag = rng.pareto(1.35, n) + 1.0
        flux = np.minimum(mag, 60.0) * 0.045
        T = np.exp(rng.normal(np.log(5600), 0.38, n)).clip(2800, 26000)
        cols = blackbody(T) * np.array([1.0, 1.0, 1.0])
        # a few foreground stars at finite distance: gentle parallax
        m = 1500
        dn = rng.normal(size=(m, 3))
        dn /= np.linalg.norm(dn, axis=1)[:, None]
        dist = np.concatenate([np.zeros(n), rng.uniform(2500, 9000, m)])
        flux = np.concatenate([flux, (rng.pareto(1.4, m) + 1.0).clip(max=40) * 0.06])
        T2 = np.exp(rng.normal(np.log(5800), 0.3, m)).clip(3000, 20000)
        self.st_dir = np.concatenate([d, dn])
        self.st_dist = dist
        self.st_flux = flux
        self.st_col = np.concatenate([cols, blackbody(T2)]).astype(np.float64)

    # ---------------- analytic disk population
    def _build_disk(self, rng):
        n = 350_000
        r_in, r_out = P.R_ISCO, 52.0
        # surface density ~ r^-0.5 with soft outer taper
        rs = []
        while sum(len(x) for x in rs) < n:
            r = rng.uniform(r_in, r_out, 2 * n)
            p = (r / r_in) ** 0.5 * np.exp(-(r / 34.0) ** 4)
            keep = rng.random(2 * n) < p / p.max()
            rs.append(r[keep])
        r = np.concatenate(rs)[:n]
        phi0 = rng.uniform(0, 2 * np.pi, n)
        # comoving structure: sheared by differential rotation into spirals
        x, y = r * np.cos(phi0), r * np.sin(phi0)
        s1 = np.sin(0.35 * x + 1.7 * np.sin(0.21 * y)) * np.cos(0.29 * y + 1.3 * np.sin(0.17 * x))
        s2 = np.sin(1.1 * x + 0.7 * y) * np.sin(0.9 * y - 0.6 * x)
        s3 = np.sin(2.3 * x - 1.1 * y + 2.0 * np.sin(0.5 * x)) * np.sin(1.9 * y + 0.8 * x)
        struct = np.clip(1.0 + 0.5 * s1 + 0.3 * s2 + 0.2 * s3 + 0.25 * rng.normal(size=n), 0.1, None) ** 1.3
        hr = 0.05 + 0.12 * np.exp(-(r - r_in) / 5.0)
        z = rng.normal(size=n) * hr * r * 0.6
        flux = r ** -3 * (1 - np.sqrt(r_in / r)).clip(1e-3)
        lum = (flux / (r ** -0.5)) ** 0.55 * struct
        lum /= lum.sum()
        Tn = (r / 6.0) ** -0.75 * ((1 - np.sqrt(r_in / r)).clip(1e-3) / (1 - np.sqrt(0.5))) ** 0.25
        temp = 11500.0 * Tn
        om = np.sqrt(0.5 / r ** 3)
        r_c = 21.0
        birth = 22.2 + 3.2 * (np.abs(np.log(r / r_c)) / np.log(5.0)) ** 0.8 * rng.uniform(0.5, 1.5, n)
        self.dk = dict(r=r, phi0=phi0, z=z, om=om, lum=lum, temp=temp, birth=birth)

    # ---------------- prominences (render-only arcs on the stellar limb)
    def _build_prominences(self, rng):
        pts = []
        for _ in range(9):
            a = rng.normal(size=3)
            a /= np.linalg.norm(a)
            t = np.cross(a, rng.normal(size=3))
            t /= np.linalg.norm(t)
            span = rng.uniform(0.04, 0.12)
            height = rng.uniform(0.04, 0.13)
            k = 1800
            u = rng.uniform(0, 1, k)
            ang = (u - 0.5) * span * np.pi
            base = np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * t
            lift = 1.0 + height * np.sin(np.pi * u) ** 0.8
            jitter = rng.normal(0, 0.004, (k, 3))
            pts.append(base * lift[:, None] + jitter)
        self.prom = np.concatenate(pts)
        d = rng.normal(size=(6000, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        self.corona = d * (1.0 + rng.exponential(0.07, 6000))[:, None]

    def _calibrate_star(self):
        x = np.asarray(self.pos[0], np.float64)
        w = self.mass
        c = (x * w[:, None]).sum(0) / w.sum()
        d = x - c
        C = (d * w[:, None]).T @ d / w.sum()
        self.cov_scale = P.R_STAR ** 2 / np.mean(np.linalg.eigvalsh(C))

    # ---------------- per-frame helpers
    def star_centre(self, s):
        f = np.clip(s * TL.FPS, 0, self.F - 1)
        i = int(np.floor(f))
        j = min(i + 1, self.F - 1)
        u = f - i
        return self.centre[i, :3] * (1 - u) + self.centre[j, :3] * u

    def star_shape(self, f):
        b = np.asarray(self.bond[f], np.float64)
        w = self.mass * b
        frac = w.sum() / self.mass.sum()
        if frac < 0.02:
            return None, frac
        x = np.asarray(self.pos[f], np.float64)
        sel = b > 0.05
        ws = w[sel]
        c = (x[sel] * ws[:, None]).sum(0) / ws.sum()
        d = x[sel] - c
        C = (d * ws[:, None]).T @ d / ws.sum()
        lam, V = np.linalg.eigh(C * self.cov_scale)
        A = V @ np.diag(np.sqrt(np.maximum(lam, 1e-4))) @ V.T
        return (c, A), frac


# ==========================================================================
#  Frame rendering
# ==========================================================================
def pack_camera(cam, W, H, coc_scale):
    """cp: [0:3] pos, [3:6] right, [6:9] up, [9:12] fwd, 12 F(px), 13 W, 14 H,
    15 focus depth, 16 CoC at infinity (px), 17 D_l, [18:21] dir to BH, 21 shadow angle."""
    F = (H * 0.5) / np.tan(cam.fov * 0.5)
    to_bh = -cam.pos
    Dl = np.linalg.norm(to_bh)
    n = to_bh / Dl
    th_sh = np.arcsin(min(P.R_SHADOW / Dl, 1.0))
    return np.array([*cam.pos, *cam.right, *cam.up, *cam.fwd, F, W, H, cam.focus,
                     cam.aperture * H * coc_scale, Dl, *n, th_sh], np.float64)


def exposure_at(s):
    keys = np.array([
        (0.0, 0.95), (6.0, 0.95), (9.5, 0.85), (11.4, 0.55), (12.8, 0.62), (14.5, 1.05),
        (17.0, 1.45), (18.8, 1.35), (20.0, 1.0), (21.2, 0.82), (22.2, 0.62), (23.2, 0.62),
        (24.5, 0.68), (27.0, 0.8), (30.0, 0.82)])
    return float(np.exp(TL._pchip(keys[:, 0], np.log(keys[:, 1]), s)))


class Renderer:
    def __init__(self, scene, W, H):
        self.sc = scene
        self.W, self.H = W, H
        self.res = H / 2160.0
        self.bvals = DT.b_grid()
        self.levels = tuple(np.zeros((4, (H + (1 << L) - 1) >> L, (W + (1 << L) - 1) >> L, 3), np.float32)
                            for L in range(NLEV))

    # ------------------------------------------------------------------
    def cameras(self, s, K, shutter_s):
        offs = (np.arange(K) + 0.5) / K - 0.5
        sc = self.sc
        cps = np.stack([pack_camera(CAM.camera_at(s + o * shutter_s, sc.star_centre),
                                    self.W, self.H, 1.0) for o in offs])
        return cps, offs

    def render(self, f, stats=None):
        sc = self.sc
        W, H = self.W, self.H
        s = f / TL.FPS
        t = float(sc.times[f])
        rate = float(TL.rate(s))
        shutter_s = 0.5 / TL.FPS
        # longer shutter during the time-lapse: motion trails read as "time passing"
        lapse = float(TL.smoothstep(22.0, 23.5, s) * (1 - TL.smoothstep(27.0, 29.5, s)))
        shutter_s *= 1.0 + 1.6 * lapse
        shutter_sim = shutter_s * rate

        # ---- adaptive number of motion-blur samples (no dotted trails)
        cpe, _ = self.cameras(s, 2, shutter_s * 2.0)
        d_cam = self._cam_motion_px(cpe)
        d_gas = self._gas_motion_px(f, cpe[0], shutter_sim)
        px_step = 1.2 * max(self.res, 0.25) * 2.0
        K = int(np.clip(np.ceil(max(d_cam, d_gas) / px_step), 4, 18))
        Ks = int(np.clip(np.ceil(d_cam / (0.7 * max(self.res, 0.25) * 2.0)), 2, 28))
        cps, offs = self.cameras(s, K, shutter_s)
        dts = offs * shutter_sim
        cp = cps[K // 2]
        for lv in self.levels:
            lv.fill(0.0)
        cam_pos = cp[0:3]
        fwd = cp[9:12]

        # ---------------- star photosphere
        shape, bond_frac = sc.star_shape(f)
        star_rgb = None
        star_depth = 1e30
        star_alpha_k = 0.0
        if shape is not None:
            c, A = shape
            star_depth = float(np.dot(c - cam_pos, fwd))
            star_alpha_k = float(TL.smoothstep(0.04, 0.5, bond_frac))
            if star_alpha_k > 0.002 and star_depth > 1.0:
                star_rgb, star_a = self._shade_star(cp, c, A, s, t, star_alpha_k)

        # ---------------- background stars
        st_sig = np.where(sc.st_flux > 0.5, 1.0, 0.8) * max(self.res, 0.5)
        cps_st, _ = self.cameras(s, Ks, shutter_s)
        splat_stars(self.levels, cps_st, sc.st_dir, sc.st_flux, sc.st_col, sc.st_dist,
                    3, st_sig.astype(np.float64))

        # ---------------- gas
        self._star_hide = (shape, star_alpha_k)
        self._splat_gas(f, s, t, cps, dts, star_depth, lapse)
        self._splat_disk(s, t, cps, dts, star_depth, lapse)
        if star_alpha_k > 0.0 and shape is not None:
            self._splat_prominences(cps, shape, s, star_alpha_k)

        # ---------------- collapse pyramids
        layers = [self._collapse(k) for k in range(4)]
        neb = np.zeros((H // 4 + 1, W // 4 + 1, 3), np.float32)
        nebula(neb, cp, W / neb.shape[1], s)
        neb = cv2.resize(neb, (W, H), interpolation=cv2.INTER_CUBIC)
        bg = layers[3] + neb * 0.012

        # ---------------- shadow, ray-traced disk
        shadow = self._shadow(cp)
        disk_rgb, disk_T = self._trace_disk(cp, s, t, shutter_sim)
        bh_depth = float(np.dot(-cam_pos, fwd))
        ops = [('bh', bh_depth), ('star', star_depth)]
        ops.sort(key=lambda o: -o[1])
        if disk_T is not None:
            bg = bg * disk_T[..., None]      # the opaque disk hides the sky behind it
        img = bg + layers[0]
        slabs = [layers[0], layers[1], layers[2]]
        for k, (name, _) in enumerate(ops):
            if name == 'bh':
                img = img * (1.0 - shadow[..., None])
                if disk_rgb is not None:
                    img = img + disk_rgb
            elif star_rgb is not None:
                a = star_a[..., None]
                img = img * (1.0 - a) + star_rgb
            img = img + slabs[k + 1]

        if stats is not None:
            stats['lum'] = float(np.percentile(img[..., 1], 99.7))
        return self._post(img, s)

    def _cam_motion_px(self, cpe):
        a, b = cpe[0], cpe[1]
        pts = [a[9:12], a[9:12] + 0.35 * a[3:6], a[9:12] + 0.2 * a[6:9]]
        best = 0.0
        for d in pts:
            pa, pb = np.empty(2), np.empty(2)
            if _project(a, d[0], d[1], d[2], pa) and _project(b, d[0], d[1], d[2], pb):
                best = max(best, float(np.hypot(*(pa - pb))))
        return best

    def _gas_motion_px(self, f, cp, shutter_sim):
        sc = self.sc
        idx = sc.probe
        p = np.asarray(sc.pos[f][idx], np.float64)
        v = np.asarray(sc.vel[f][idx], np.float64)
        keep = np.asarray(sc.bond[f][idx]) < 0.5
        if keep.sum() < 10:
            return 0.0
        p, v = p[keep], v[keep]
        d = p - cp[0:3]
        z = d @ cp[9:12]
        ok = z > 1.0
        if ok.sum() < 10:
            return 0.0
        vp = v[ok] - np.outer(v[ok] @ cp[9:12], cp[9:12])
        disp = cp[12] * np.linalg.norm(vp, axis=1) * shutter_sim / z[ok]
        r = np.linalg.norm(p[ok], axis=1)
        arc = np.minimum(disp, cp[12] * r * 0.9 / z[ok])
        return float(np.percentile(arc, 90))

    # ------------------------------------------------------------------
    def _shade_star(self, cp, c, A, s, t, alpha_k):
        W, H = self.W, self.H
        Ainv = np.linalg.inv(A)
        AinvT = Ainv.T
        # bounding box from the ellipsoid extent (+ lensing margin)
        ext = np.sqrt(np.sum(A * A, axis=1)) * 1.08
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * ext + c
        F = cp[12]
        pts = []
        for q in corners:
            d = q - cp[0:3]
            zc = d @ cp[9:12]
            if zc <= 0.5:
                return None, None
            pts.append((W / 2 + F * (d @ cp[3:6]) / zc, H / 2 - F * (d @ cp[6:9]) / zc))
        pts = np.array(pts)
        x0, y0 = np.floor(pts.min(0)).astype(int) - 4
        x1, y1 = np.ceil(pts.max(0)).astype(int) + 4
        to_c = c - cp[0:3]
        Dl = cp[17]
        n = cp[18:21]
        zl = to_c @ n
        lensed = zl > Dl
        th2 = 2.0 * (zl - Dl) / (Dl * np.linalg.norm(to_c)) if lensed else 0.0
        if lensed:
            # include the Einstein-ring region (secondary image) in the box
            bxy = np.array([W / 2 + F * (n @ cp[3:6]) / (n @ cp[9:12]), H / 2 - F * (n @ cp[6:9]) / (n @ cp[9:12])])
            rpx = F * np.sqrt(th2) * 2.5
            x0, y0 = min(x0, int(bxy[0] - rpx)), min(y0, int(bxy[1] - rpx))
            x1, y1 = max(x1, int(bxy[0] + rpx)), max(y1, int(bxy[1] + rpx))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W), min(y1, H)
        rgb = np.zeros((H, W, 3), np.float32)
        alpha = np.zeros((H, W), np.float32)
        depth = np.zeros((H, W), np.float32)
        if x1 <= x0 or y1 <= y0:
            return None, None
        shade_star(rgb, alpha, depth, cp, x0, x1, y0, y1, c, Ainv, AinvT, lensed, th2,
                   s * 0.6, 4750.0, STAR_I)
        alpha *= alpha_k
        rgb *= alpha_k * np.array([1.07, 0.98, 0.84], np.float32)
        # gentle directional blur along the star's screen motion would go here;
        # the camera tracks the star so its screen motion is small.
        return rgb, alpha

    def _splat_gas(self, f, s, t, cps, dts, star_depth, lapse):
        sc = self.sc
        pos = np.asarray(sc.pos[f], np.float64)
        vel = np.asarray(sc.vel[f], np.float64)
        glow = np.asarray(sc.glow[f], np.float64)
        bond = np.asarray(sc.bond[f], np.float64)
        alive = np.asarray(sc.alive[f])
        m = sc.mass
        age = np.where(np.isfinite(sc.trel), t - sc.trel, -1.0)
        age = np.where(age < 0, 0.0, age)
        free = (1.0 - bond)
        # base thermal emission of the stripped stellar gas, cooling as it expands
        Ib = 0.22 * (1.0 + age / 300.0) ** -0.5
        Ib = np.maximum(Ib, 0.18)
        Ib = Ib * (1.0 + 0.8 * float(TL.smoothstep(23.0, 26.0, s)) * np.clip((np.linalg.norm(pos, axis=1) - 60.0) / 60.0, 0, 1))
        Tb = np.maximum(5300.0 * (1.0 + age / 400.0) ** -0.25, 2900.0)
        # shock heating where streams collide
        gref = 4e-3
        # luminosity ~ dissipation; the mild super-linear weighting keeps diffuse
        # pericentre compression heating subtle and makes the stream collision read
        kt = 1.0 - 0.97 * float(TL.smoothstep(21.3, 23.4, s))
        Is = SHOCK_K * kt * glow * np.clip(np.sqrt(glow / 2e-3), 0.02, 1.6)
        Is = np.minimum(Is, 6000.0)
        Ts = 4000.0 + 4300.0 * np.clip(glow / gref, 0, 3) ** 0.5
        r = np.linalg.norm(pos, axis=1)
        fade = np.clip((r - P.R_ABSORB) / 1.5, 0, 1) * alive
        # in the time-lapse the circularised gas is represented by the disk layer
        inner = 1.0 - 0.95 * float(TL.smoothstep(22.0, 24.6, s)) * (1.0 - TL.smoothstep(38.0, 70.0, r))
        # the escaping tail thins out as it expands; keep it readable far out
        inner = inner * (1.0 + np.clip(r / 150.0 - 1.0, 0, None)) ** 0.7
        L = sc.emis * free * fade * inner * np.pi * P.R_STAR ** 2
        shp, ak = getattr(self, '_star_hide', (None, 0.0))
        if shp is not None and ak > 0:
            cc, AA = shp
            qn = np.linalg.norm((pos - cc) @ np.linalg.inv(AA).T, axis=1)
            L = L * (1.0 - ak * np.clip((1.12 - qn) / 0.22, 0, 1))
        Ls = m * free * fade * inner * np.pi * P.R_STAR ** 2
        col = (blackbody(Tb) * (Ib * L)[:, None] + blackbody(Ts) * (Is * Ls)[:, None])
        # Doppler beaming (gas speeds reach ~0.3 c near pericentre)
        n = cps[len(cps) // 2][0:3] - pos
        n /= np.linalg.norm(n, axis=1)[:, None]
        v2 = np.sum(vel * vel, axis=1).clip(max=0.8)
        D = np.sqrt(1 - v2) / np.maximum(1 - np.sum(vel * n, axis=1), 0.05)
        col *= np.nan_to_num(D ** 1.5, nan=1.0).clip(0.5, 2.0)[:, None]
        col = np.nan_to_num(col)
        col = col.astype(np.float64)
        hot = np.clip(glow / gref, 0, 1) ** 0.5
        hs = (0.09 + 0.05 * np.sqrt(age / 100.0)).clip(max=0.6) + 0.35 * hot
        speed = np.linalg.norm(vel, axis=1)
        par = np.clip(speed * 4.0, 0.04, 1.4) * free
        perp = (0.06 + 0.02 * np.sqrt(age / 50.0)).clip(max=0.5)
        turb = 0.22 * free * np.clip(age / 300.0, 0, 1) ** 0.5
        # cull particles that do not emit
        keep = np.flatnonzero(col.sum(1) > 1e-9)
        splat_gas(self.levels, cps, dts, pos[keep], vel[keep], col[keep], hs[keep], sc.rest[keep],
                  turb[keep], sc.rnd[keep], par[keep], perp[keep],
                  star_depth, t, 0.35 + 0.9 * lapse)

    def _splat_disk(self, s, t, cps, dts, star_depth, lapse):
        if s < 21.5:
            return
        dk = self.sc.dk
        vis = TL.smoothstep(dk['birth'], dk['birth'] + 1.4, s)
        total = DISK_L * float(TL.smoothstep(21.8, 26.0, s))
        if total <= 0:
            return
        keep = np.flatnonzero(vis > 0)
        lum = dk['lum'][keep] * total * np.pi * P.R_STAR ** 2
        t_rel = t - TL.sim_time(22.0)
        sel = np.linspace(0, len(cps) - 1, min(len(cps), 8)).round().astype(int)
        cps, dts = cps[sel], dts[sel]
        splat_disk(self.levels, cps, dts, dk['r'][keep], dk['phi0'][keep], dk['z'][keep], dk['om'][keep],
                   lum, dk['temp'][keep], vis[keep], t_rel, 1.0, star_depth, 0.35 + 1.1 * lapse)

    def _splat_prominences(self, cps, shape, s, alpha_k):
        """Prominence arches and a faint corona.  They are put in the slab behind
        the photosphere, so only what rises beyond the limb is seen."""
        c, A = shape
        k = float(1.0 - TL.smoothstep(7.5, 9.5, s)) * alpha_k
        if k <= 0:
            return
        q = self.sc.prom
        wob = 1.0 + 0.02 * np.sin(1.3 * s + 7 * q[:, 0])[:, None]
        pts = c + (q * wob) @ A.T
        n = len(pts)
        lum = 0.010 * k * np.pi * P.R_STAR ** 2 / n
        col = np.tile(blackbody(np.array([3600.0]))[0] * lum, (n, 1)).astype(np.float64)
        zero = np.zeros((n, 3))
        splat_gas(self.levels, cps[:1], np.zeros(1), pts, zero + 1e-6, col, np.full(n, 0.02), zero,
                  np.zeros(n), np.zeros((n, 1, 3)), np.zeros(n), np.zeros(n), -1.0, 0.0, 0.0)
        # no particle corona: the photosphere's own bloom provides the halo

    def _collapse(self, slab):
        acc = None
        for L in range(NLEV - 1, -1, -1):
            lv = self.levels[L][slab]
            b = cv2.GaussianBlur(lv, (0, 0), 0.75)
            if acc is None:
                acc = b
            else:
                acc = cv2.resize(acc, (lv.shape[1], lv.shape[0]), interpolation=cv2.INTER_LINEAR) + b
        return acc[:self.H, :self.W]

    def _shadow(self, cp):
        W, H = self.W, self.H
        F = cp[12]
        n = cp[18:21]
        zc = n @ cp[9:12]
        shadow = np.zeros((H, W), np.float32)
        if zc <= 0.05:
            return shadow
        bx = W / 2 + F * (n @ cp[3:6]) / zc
        by = H / 2 - F * (n @ cp[6:9]) / zc
        rpx = F * np.tan(cp[21])
        R = int(rpx * 1.3 + 8)
        x0, x1 = max(int(bx - R), 0), min(int(bx + R), W)
        y0, y1 = max(int(by - R), 0), min(int(by + R), H)
        if x1 <= x0 or y1 <= y0:
            return shadow
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx + 0.5 - bx) ** 2 + (yy + 0.5 - by) ** 2)
        aa = max(0.8, 0.004 * rpx)
        shadow[y0:y1, x0:x1] = np.clip((rpx - d) / aa + 0.5, 0, 1)
        return shadow

    def _trace_disk(self, cp, s, t, shutter_sim):
        if s < DISK_T0:
            return None, None
        W, H = self.W, self.H
        D = float(np.linalg.norm(cp[0:3]))
        tab = DT.geodesic_table(D, self.bvals, DT.DPHI, DT.NPHI)
        # screen bounding box of the disk (generous: lensed images stay inside)
        R = DISK_ROUT
        ang = np.linspace(0, 2 * np.pi, 64)
        pts = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros_like(ang)], 1)
        pts = np.concatenate([pts, [[0, 0, R * 0.4], [0, 0, -R * 0.4]]])
        xy = []
        for p in pts:
            d = p - cp[0:3]
            z = d @ cp[9:12]
            if z <= 0.5:
                continue
            xy.append((W / 2 + cp[12] * (d @ cp[3:6]) / z, H / 2 - cp[12] * (d @ cp[6:9]) / z))
        if not xy:
            return None, None
        xy = np.array(xy)
        x0, y0 = np.floor(xy.min(0)).astype(int) - 10
        x1, y1 = np.ceil(xy.max(0)).astype(int) + 10
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W), min(y1, H)
        rgb = np.zeros((H, W, 3), np.float32)
        T = np.ones((H, W), np.float32)
        if x1 <= x0 or y1 <= y0:
            return None, None
        kd = DISK_K * float(TL.smoothstep(DISK_T0, 25.0, s))
        t0 = TL.sim_time(22.0)
        ptau, pref, pgen = DT.pattern_clock(TL.sim_time, TL.rate, s, t0)
        DT.trace_disk(rgb, T, cp, x0, x1, y0, y1, tab, self.bvals, t - t0, s,
                      shutter_sim, kd, ptau, pref, pgen, BB_TABLE, LOG_T_MIN, LOG_T_STEP, 22.2, 3.2, 21.0,
                      P.R_ISCO, DISK_ROUT)
        # slight softening so the thin disk sits with the volumetric gas
        rgb = cv2.GaussianBlur(rgb, (0, 0), 1.1 * max(self.res, 0.5))
        return rgb, T

    def _post(self, img, s):
        H, W = self.H, self.W
        # bloom: wide, soft and restrained
        small = img
        blooms = []
        for i in range(6):
            small = cv2.pyrDown(small)
            blooms.append(small)
        acc = None
        for i in range(5, -1, -1):
            b = cv2.GaussianBlur(blooms[i], (0, 0), 1.2)
            if acc is None:
                acc = b
            else:
                acc = cv2.resize(acc, (blooms[i].shape[1], blooms[i].shape[0]), interpolation=cv2.INTER_LINEAR) + b * (1.0 + 0.15 * i)
        acc = cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR) / 6.0
        hdr = img + 0.40 * acc
        e = exposure_at(s)
        x = hdr * (0.85 * e)
        # warm the highlights slightly, keep shadows neutral-cool
        x = x * np.array([1.03, 1.0, 0.95], np.float32)
        # ACES filmic approximation
        a, b, c, d, e2 = 2.51, 0.03, 2.43, 0.59, 0.14
        y = np.clip((x * (a * x + b)) / (x * (c * x + d) + e2), 0, 1)
        # grade: tiny cool lift in the deep shadows, saturation
        lum = (0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2])[..., None]
        y = lum + (y - lum) * 1.12
        y = y + np.array([0.0, 0.0004, 0.0012], np.float32) * (1 - lum) ** 8
        # vignette
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rr = ((xx - W / 2) / (W / 2)) ** 2 * 0.78 + ((yy - H / 2) / (H / 2)) ** 2 * 0.62
        y *= (1.0 - 0.22 * rr ** 1.3)[..., None]
        y = np.clip(y, 0, 1)
        # display encoding (sRGB) + film grain + dither
        y = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
        rng = np.random.default_rng(int(s * 1000))
        grain = rng.normal(0, 1, (H, W)).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 0.6 * max(self.res, 0.5))
        y = y + grain[..., None] * 0.006
        out = np.clip(y * 255 + rng.random((H, W, 1)).astype(np.float32) - 0.5, 0, 255).astype(np.uint8)
        return out[..., ::-1]   # BGR for OpenCV
