"""
Ray-traced thin accretion disk in the Schwarzschild metric.

Photon orbits obey  d^2u/dphi^2 = -u + 3/2 u^2   (u = R_s / r).
Per frame, u(phi) is tabulated for a grid of impact parameters b for the
current camera distance.  For each pixel the photon plane is found, the
angles at which the photon crosses the equatorial (disk) plane are computed,
and the table gives the radius of every crossing - first image, the image of
the far side lifted over the shadow, and the thin photon ring of higher-order
images all come out of the same calculation.  Emission is a black body whose
temperature and intensity are shifted by g = 1/(1+z) with
   1 + z = (1 - Omega L_z) / sqrt(1 - 3M/r)      (circular orbits, M = 1/2).
"""
import numpy as np
from numba import njit

B_CRIT = 1.5 * np.sqrt(3.0)
NB = 1400
DPHI = 0.004
NPHI = 3300


def b_grid():
    return B_CRIT + np.exp(np.linspace(np.log(2e-5), np.log(600.0), NB))


@njit(fastmath=True)
def geodesic_table(D, bvals, dphi, nphi):
    tab = np.full((bvals.shape[0], nphi), -1.0)
    u0 = 1.0 / D
    for ib in range(bvals.shape[0]):
        b = bvals[ib]
        w = 1.0 / (b * b) - u0 * u0 * (1.0 - u0)
        if w < 0:
            continue
        u = u0
        du = np.sqrt(w)
        tab[ib, 0] = u
        for k in range(1, nphi):
            h = dphi
            k1u, k1v = du, -u + 1.5 * u * u
            u2, v2 = u + 0.5 * h * k1u, du + 0.5 * h * k1v
            k2u, k2v = v2, -u2 + 1.5 * u2 * u2
            u3, v3 = u + 0.5 * h * k2u, du + 0.5 * h * k2v
            k3u, k3v = v3, -u3 + 1.5 * u3 * u3
            u4, v4 = u + h * k3u, du + h * k3v
            k4u, k4v = v4, -u4 + 1.5 * u4 * u4
            u += h / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
            du += h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
            if u >= 1.0:
                tab[ib, k] = 9.0      # captured by the horizon
                break
            if u < 0.5 * u0:
                break                 # escaped
            tab[ib, k] = u
    return tab


@njit(inline='always')
def _hash2(ix, iy, s):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (s * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(inline='always')
def _vn2(x, y, s):
    ix, iy = int(np.floor(x)), int(np.floor(y))
    fx, fy = x - ix, y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a = _hash2(ix, iy, s)
    b = _hash2(ix + 1, iy, s)
    c = _hash2(ix, iy + 1, s)
    d = _hash2(ix + 1, iy + 1, s)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@njit(inline='always')
def _smooth(a, b, x):
    t = (x - a) / (b - a)
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(inline='always')
def disk_structure(r, phid, t, shutter, ptau, pref, pgen):
    """Turbulent, differentially rotating pattern.

    Two pattern generations cross-fade on a global clock (ptau, pref, pgen are
    per-generation phase, birth time and seed).  Within a generation the pattern
    is carried by Keplerian rotation, so it shears into trailing spirals; where
    the accumulated shear would wind it into sub-resolution rings, and where the
    shutter spans many radians (= azimuthal motion blur), contrast is reduced."""
    om = np.sqrt(0.5 / (r * r * r))
    acc = 0.0
    wsum = 0.0
    for j in range(2):
        w = np.sin(np.pi * ptau[j]) ** 2
        dtj = t - pref[j]
        phic = phid - om * dtj
        X = r * np.cos(phic)
        Y = r * np.sin(phic)
        seed = int(pgen[j])
        n = 0.62 * _vn2(0.16 * X, 0.16 * Y, seed) + 0.28 * _vn2(0.45 * X + 3.1, 0.45 * Y, seed + 7) \
            + 0.10 * _vn2(1.1 * X, 1.1 * Y + 1.3, seed + 13)
        n = 0.62 + 1.6 * (n - 0.5)
        n += 0.10 * np.sin(5.0 * np.log(r) + 2.0 * phic + 6.0 * _vn2(0.3 * X, 0.3 * Y, seed + 3))
        shear = 1.5 * om * abs(dtj) / 8.0
        n = 0.62 + (n - 0.62) / (1.0 + shear * shear)
        acc += w * n
        wsum += w
    s = acc / max(wsum, 1e-6)
    blur = om * shutter
    con = 1.0 / (1.0 + 0.6 * blur)
    return max(0.62 + con * (s - 0.62), 0.05)


@njit(fastmath=True)
def trace_disk(rgb, trans, cp, x0, x1, y0, y1, tab, bvals, t, s_now, shutter, kdisk, ptau, pref, pgen,
               bb_table, log_tmin, log_tstep, birth0, birth_span, r_c, r_in, r_out):
    W, H = cp[13], cp[14]
    F = cp[12]
    Cx, Cy, Cz = cp[0], cp[1], cp[2]
    D = np.sqrt(Cx * Cx + Cy * Cy + Cz * Cz)
    e1x, e1y, e1z = Cx / D, Cy / D, Cz / D
    lb0 = np.log(bvals[0] - 1.5 * np.sqrt(3.0))
    lb1 = np.log(bvals[-1] - 1.5 * np.sqrt(3.0))
    nb = bvals.shape[0]
    nphi = tab.shape[1]
    dphi = 0.004
    sq = np.sqrt(1.0 - 1.0 / D)
    col = np.empty(3)
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
            cth = -(dx * e1x + dy * e1y + dz * e1z)
            sth = np.sqrt(max(1.0 - cth * cth, 1e-12))
            b = D * sth / sq
            if b > bvals[-1]:
                continue
            # transverse direction in the photon plane
            tx, ty, tz = dx + cth * e1x, dy + cth * e1y, dz + cth * e1z
            tn = np.sqrt(tx * tx + ty * ty + tz * tz)
            if tn < 1e-9:
                continue
            e2x, e2y, e2z = tx / tn, ty / tn, tz / tn
            # photon angular momentum (disk -> camera), z component
            nzc = e1x * e2y - e1y * e2x
            Lz = -b * nzc
            # fractional b index
            lb = np.log(max(b - 1.5 * np.sqrt(3.0), 1e-12))
            fb = (lb - lb0) / (lb1 - lb0) * (nb - 1)
            if fb < 0:
                fb = 0.0
            ib = int(fb)
            if ib >= nb - 1:
                ib = nb - 2
            ub = fb - ib
            phi0 = np.arctan2(-e1z, e2z)
            if phi0 < 0:
                phi0 += np.pi
            T = 1.0
            ar, ag, ab = 0.0, 0.0, 0.0
            for k in range(4):
                phi = phi0 + k * np.pi
                fk = phi / dphi
                ik = int(fk)
                if ik >= nphi - 1:
                    break
                uk = fk - ik
                a00 = tab[ib, ik]
                a01 = tab[ib, ik + 1]
                a10 = tab[ib + 1, ik]
                a11 = tab[ib + 1, ik + 1]
                if a00 < 0 or a01 < 0 or a10 < 0 or a11 < 0:
                    break
                if a00 > 2 or a01 > 2 or a10 > 2 or a11 > 2:
                    break
                u = (a00 * (1 - uk) + a01 * uk) * (1 - ub) + (a10 * (1 - uk) + a11 * uk) * ub
                r = 1.0 / u
                if r < r_in or r > r_out:
                    continue
                cphi, sphi = np.cos(phi), np.sin(phi)
                Px = r * (cphi * e1x + sphi * e2x)
                Py = r * (cphi * e1y + sphi * e2y)
                phid = np.arctan2(Py, Px)
                # growth of the disk: from the circularisation radius outwards and inwards
                lr = abs(np.log(r / r_c)) / np.log(5.0)
                nb_ = _vn2(0.15 * Px + 17.0, 0.15 * Py, 99)
                birth = birth0 + birth_span * lr ** 0.8 * (0.6 + 0.8 * nb_)
                vis = _smooth(birth, birth + 1.4, s_now)
                if vis <= 0:
                    continue
                st = disk_structure(r, phid, t, shutter, ptau, pref, pgen)
                edge = _smooth(r_in, r_in + 1.2, r) * np.exp(-(r / 34.0) ** 4)
                alpha = min(0.97, vis * edge * (0.55 + 0.6 * st))
                if alpha <= 0:
                    continue
                om = np.sqrt(0.5 / (r * r * r))
                zf = (1.0 - om * Lz) / np.sqrt(max(1.0 - 1.5 / r, 0.02))
                g = 1.0 / zf
                fl = (r ** -3.0 * max(1.0 - np.sqrt(r_in / r), 1e-3)) ** 0.3
                Iem = kdisk * fl * st ** 1.6
                Tn = (r / 6.0) ** -0.75 * (max(1.0 - np.sqrt(r_in / r), 1e-3) / (1.0 - np.sqrt(0.5))) ** 0.25
                Tobs = 11500.0 * Tn * (0.9 + 0.2 * st) * g
                Tobs = min(max(Tobs, 1250.0), 59000.0)
                fT = (np.log(Tobs) - log_tmin) / log_tstep
                it = int(fT)
                ut = fT - it
                for j in range(3):
                    col[j] = bb_table[it, j] * (1 - ut) + bb_table[it + 1, j] * ut
                I = Iem * g ** 3 * alpha * T
                ar += I * col[0]
                ag += I * col[1]
                ab += I * col[2]
                T *= (1.0 - alpha)
                if T < 0.01:
                    break
            rgb[py, px, 0] = ar
            rgb[py, px, 1] = ag
            rgb[py, px, 2] = ab
            trans[py, px] = T


def pattern_clock(t_of_s, rate_of_s, s_now, t_rel0):
    """Global pattern clock: one generation per max(300, 0.6*rate) sim units."""
    s = np.linspace(0.0, 30.5, 30501)
    rate = rate_of_s(s)
    pg = np.maximum(300.0, 0.6 * rate)
    clock = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] / pg[1:] + rate[:-1] / pg[:-1]) * np.diff(s))])
    c_now = np.interp(s_now, s, clock)
    tau, ref, gen = np.zeros(2), np.zeros(2), np.zeros(2)
    for j in range(2):
        ph = c_now + 0.5 * j
        g = np.floor(ph)
        tau[j] = ph - g
        s_birth = np.interp(g - 0.5 * j, clock, s)
        ref[j] = t_of_s(s_birth) - t_rel0
        gen[j] = 2 * g + j
    return tau, ref, gen
