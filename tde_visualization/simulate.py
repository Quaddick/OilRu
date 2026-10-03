"""
Particle simulation of the tidal disruption.

Model (cheap, but physically motivated):
  * Each gas element moves in the Paczynski-Wiita potential  Phi = -GM/(r - R_s),
    which reproduces the relativistic apsidal precession that makes the
    returning debris stream cross itself.
  * Before disruption the star is held together by a "self-gravity + pressure"
    restoring force towards its unperturbed position in the star frame.  The
    stiffness of each layer is proportional to its local self-gravity, so the
    loose outer layers deform first and the dense core last (overlapping action
    comes from the physics, not from keyframes).
  * A layer is released once the tidal acceleration exceeds its self-binding
    (chi > 1).  Stripping starts at the L1/L2 points (along the BH direction),
    which produces the two tidal tails.  After release elements are ballistic,
    so the debris energy spread is frozen near the tidal radius, as in the
    standard picture: half of the gas is bound, half escapes.
  * Stream-stream collisions are treated as inelastic: on a 3D grid, where the
    velocity dispersion inside a cell is large (crossing streams), velocities are
    relaxed to the cell mean (momentum conserving) and the lost kinetic energy is
    converted into heat -> luminosity.
  * During the time-lapse disk phase an extra drag acts on radial and vertical
    motion only (angular momentum is conserved), standing in for many orbits of
    shocks: gas settles at the circularisation radius ~2 r_p.
"""
import os
import sys
import time
import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import params as P


# --------------------------------------------------------------------------
#  Stellar structure
# --------------------------------------------------------------------------
def lane_emden(n, h=1e-4):
    xi = h
    th = 1.0 - xi * xi / 6.0
    dth = -xi / 3.0
    xs, ms = [0.0], [0.0]
    while th > 0:
        def f(x, y):
            return np.array([y[1], -max(y[0], 0.0) ** n - 2.0 * y[1] / x])
        y = np.array([th, dth])
        k1 = f(xi, y)
        k2 = f(xi + h / 2, y + h / 2 * k1)
        k3 = f(xi + h / 2, y + h / 2 * k2)
        k4 = f(xi + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xi += h
        th, dth = y
        xs.append(xi)
        ms.append(-xi * xi * dth)
        h = min(h * 1.002, 2e-3)
    xs = np.array(xs)
    ms = np.array(ms)
    return xs / xs[-1], ms / ms[-1]


def make_star(n, rng):
    s_tab, m_tab = lane_emden(P.POLY_N)
    # Sample more particles in the outer layers than strict equal-mass would
    # (better resolution of the stripped envelope) and carry a mass weight.
    u = rng.random(n)
    mix = 0.45
    m_equal = np.interp(u, m_tab, s_tab)          # equal-mass radii
    vol = u ** (1.0 / 3.0)                         # uniform-volume radii
    s = np.where(rng.random(n) < mix, vol, m_equal)
    s = np.clip(s, 1e-3, 0.999)
    # density at s (from dm/ds / s^2) for the weights
    ds = 1e-3
    dm = (np.interp(s + ds, s_tab, m_tab) - np.interp(s - ds, s_tab, m_tab)) / (2 * ds)
    pdf_equal = dm
    pdf_vol = 3.0 * s * s
    w = dm / ((1 - mix) * pdf_equal + mix * pdf_vol + 1e-9)
    w = w / w.sum()
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    o = d * (s * P.R_STAR)[:, None]
    # local self-gravity stiffness  w^2 = G m(<s) / s^3  (compressed dynamic range)
    menc = np.interp(s, s_tab, m_tab)
    S = (menc / s ** 3)
    S = (S / S.min()) ** 0.3
    w2 = P.GM_STAR / P.R_STAR ** 3 * S
    return o.astype(np.float64), w2, w, s


# --------------------------------------------------------------------------
#  Centre-of-mass orbit (parabolic in the PW potential)
# --------------------------------------------------------------------------
def pw_acc(x):
    r = np.linalg.norm(x)
    return -P.GM / (r - 1.0) ** 2 * x / r


def centre_orbit(t_start, t_end, dt=0.05):
    ang = P.PERI_ANGLE
    rp = P.R_PERI
    xp = np.array([rp * np.cos(ang), rp * np.sin(ang), 0.0])
    vp_mag = np.sqrt(2 * P.GM / (rp - 1.0))
    vp = vp_mag * np.array([-np.sin(ang), np.cos(ang), 0.0])

    def integ(t_to, h):
        n = int(np.ceil(abs(t_to) / dt))
        xs = np.zeros((n + 1, 3))
        vs = np.zeros((n + 1, 3))
        x, v = xp.copy(), vp.copy()
        xs[0], vs[0] = x, v
        for k in range(n):
            def f(y):
                return np.concatenate([y[3:], pw_acc(y[:3])])
            y = np.concatenate([x, v])
            k1 = f(y)
            k2 = f(y + h / 2 * k1)
            k3 = f(y + h / 2 * k2)
            k4 = f(y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x, v = y[:3], y[3:]
            xs[k + 1], vs[k + 1] = x, v
        return xs, vs

    xb, vb = integ(t_start, -dt)
    xf, vf = integ(t_end, dt)
    xs = np.concatenate([xb[::-1], xf[1:]])
    vs = np.concatenate([vb[::-1], vf[1:]])
    t0 = -dt * (len(xb) - 1)
    return t0, dt, xs, vs


# --------------------------------------------------------------------------
#  Numba kernels
# --------------------------------------------------------------------------
@njit(inline='always')
def centre_at(t, ct0, cdt, cx, cv, out_x, out_v):
    f = (t - ct0) / cdt
    k = int(f)
    if k < 0:
        k = 0
        f = 0.0
    if k >= cx.shape[0] - 1:
        k = cx.shape[0] - 2
        f = float(k + 1)
    u = f - k
    for j in range(3):
        out_x[j] = cx[k, j] * (1 - u) + cx[k + 1, j] * u
        out_v[j] = cv[k, j] * (1 - u) + cv[k + 1, j] * u


@njit(inline='always')
def smoothstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return t * t * (3 - 2 * t)


@njit(parallel=True, fastmath=True)
def advance(x, v, o, w2, jit, chimax, alive, heat,
            t0, t1, ct0, cdt, cx, cv, GM, eta, drag_k, r_abs, bind):
    N = x.shape[0]
    for i in prange(N):
        if not alive[i]:
            continue
        xc = np.empty(3)
        vc = np.empty(3)
        a = np.empty(3)
        t = t0
        xi0, xi1, xi2 = x[i, 0], x[i, 1], x[i, 2]
        vi0, vi1, vi2 = v[i, 0], v[i, 1], v[i, 2]
        cm = chimax[i]
        hacc = 0.0
        steps = 0
        while t < t1 and steps < 200000:
            steps += 1
            r = np.sqrt(xi0 * xi0 + xi1 * xi1 + xi2 * xi2)
            if r < r_abs:
                alive[i] = False
                break
            b = 1.0 - smoothstep(0.85, 1.35, cm * jit[i])
            dt = eta * np.sqrt((r - 1.0) ** 3 / GM)
            w2i = w2[i] * bind
            if b > 0.0:
                dts = 0.12 / np.sqrt(w2[i])
                if dts < dt:
                    dt = dts
            if dt > t1 - t:
                dt = t1 - t
            # ---- kick-drift-kick, force evaluated at two points
            for half in range(2):
                tt = t if half == 0 else t + dt
                r = np.sqrt(xi0 * xi0 + xi1 * xi1 + xi2 * xi2)
                g = -GM / ((r - 1.0) ** 2 * r)
                a[0] = g * xi0
                a[1] = g * xi1
                a[2] = g * xi2
                if b > 0.0:
                    centre_at(tt, ct0, cdt, cx, cv, xc, vc)
                    d0 = xi0 - xc[0] - o[i, 0]
                    d1 = xi1 - xc[1] - o[i, 1]
                    d2 = xi2 - xc[2] - o[i, 2]
                    k = b * w2i
                    gam = b * 0.6 * np.sqrt(w2i)
                    a[0] += -k * d0 - gam * (vi0 - vc[0])
                    a[1] += -k * d1 - gam * (vi1 - vc[1])
                    a[2] += -k * d2 - gam * (vi2 - vc[2])
                    # tidal breaking criterion, anisotropic (L1/L2 first)
                    rc = np.sqrt(xc[0] ** 2 + xc[1] ** 2 + xc[2] ** 2)
                    so = np.sqrt(o[i, 0] ** 2 + o[i, 1] ** 2 + o[i, 2] ** 2) + 1e-9
                    cth = (o[i, 0] * xc[0] + o[i, 1] * xc[1] + o[i, 2] * xc[2]) / (so * rc)
                    ang = 0.3 + 0.7 * max(0.0, (3 * cth * cth - 1) * 0.5)
                    chi = 2 * GM / ((rc - 1.0) ** 3 * w2i) * ang
                    if chi > cm:
                        cm = chi
                eps = 0.5 * (vi0 * vi0 + vi1 * vi1 + vi2 * vi2) - GM / (r - 1.0)
                if drag_k > 0.0 and eps < 0.0 and r < 160.0:
                    # dissipate radial + vertical motion of bound gas (keeps angular momentum)
                    rr = np.sqrt(xi0 * xi0 + xi1 * xi1) + 1e-9
                    vr = (vi0 * xi0 + vi1 * xi1) / rr
                    a[0] += -drag_k * vr * xi0 / rr
                    a[1] += -drag_k * vr * xi1 / rr
                    a[2] += -drag_k * vi2 * 2.0
                    hacc += 0.08 * drag_k * (vr * vr + 2.0 * vi2 * vi2) * dt * 0.5
                if half == 0:
                    vi0 += 0.5 * dt * a[0]
                    vi1 += 0.5 * dt * a[1]
                    vi2 += 0.5 * dt * a[2]
                    xi0 += dt * vi0
                    xi1 += dt * vi1
                    xi2 += dt * vi2
                else:
                    vi0 += 0.5 * dt * a[0]
                    vi1 += 0.5 * dt * a[1]
                    vi2 += 0.5 * dt * a[2]
            t += dt
        x[i, 0], x[i, 1], x[i, 2] = xi0, xi1, xi2
        v[i, 0], v[i, 1], v[i, 2] = vi0, vi1, vi2
        chimax[i] = cm
        heat[i] += hacc


@njit(fastmath=True)
def collide(x, v, m, alive, chimax, jit, heat, cell, half_xy, half_z, GM, strength,
            sm, sv, sv2, cnt, idx, f_perp, ox, oy, oz):
    """Inelastic stream-stream collisions on a 3D grid (buffers are reused)."""
    nx = int(2 * half_xy / cell)
    nz = int(2 * half_z / cell)
    N = x.shape[0]
    for i in range(N):
        c = idx[i]
        if c >= 0:
            sm[c] = 0.0
            sv[c, 0] = 0.0
            sv[c, 1] = 0.0
            sv[c, 2] = 0.0
            sv2[c] = 0.0
            cnt[c] = 0
        idx[i] = -1
    for i in range(N):
        if not alive[i]:
            continue
        # only free (released) gas collides
        if 1.0 - smoothstep(0.85, 1.35, chimax[i] * jit[i]) > 0.5:
            continue
        ix = int((x[i, 0] + half_xy + ox) / cell)
        iy = int((x[i, 1] + half_xy + oy) / cell)
        iz = int((x[i, 2] + half_z + oz) / cell)
        if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= nx or iz >= nz:
            continue
        c = (iz * nx + iy) * nx + ix
        idx[i] = c
        mi = m[i]
        sm[c] += mi
        cnt[c] += 1
        for j in range(3):
            sv[c, j] += mi * v[i, j]
        sv2[c] += mi * (v[i, 0] ** 2 + v[i, 1] ** 2 + v[i, 2] ** 2)
    for i in range(N):
        c = idx[i]
        if c < 0 or cnt[c] < 3:
            continue
        M = sm[c]
        m0 = sv[c, 0] / M
        m1 = sv[c, 1] / M
        m2 = sv[c, 2] / M
        sig2 = sv2[c] / M - (m0 * m0 + m1 * m1 + m2 * m2)
        if sig2 <= 0:
            continue
        d0 = v[i, 0] - m0
        d1 = v[i, 1] - m1
        d2 = v[i, 2] - m2
        e0 = d0 * d0 + d1 * d1 + d2 * d2
        r = np.sqrt(x[i, 0] ** 2 + x[i, 1] ** 2 + x[i, 2] ** 2)
        vesc = np.sqrt(2 * GM / max(r - 1.0, 0.5))
        rel = np.sqrt(sig2) / vesc
        # (1) shocks where streams cross: relax to the cell mean velocity
        f = strength * smoothstep(0.08, 0.3, rel) * min(1.0, cnt[c] / 12.0)
        # (2) weak transverse confinement (proxy for the stream's self-gravity):
        #     damp only the velocity component perpendicular to the local flow
        vm = np.sqrt(m0 * m0 + m1 * m1 + m2 * m2) + 1e-12
        u0, u1, u2 = m0 / vm, m1 / vm, m2 / vm
        dpar = d0 * u0 + d1 * u1 + d2 * u2
        p0, p1, p2 = d0 - dpar * u0, d1 - dpar * u1, d2 - dpar * u2
        v[i, 0] -= f * d0 + (1 - f) * f_perp * p0
        v[i, 1] -= f * d1 + (1 - f) * f_perp * p1
        v[i, 2] -= f * d2 + (1 - f) * f_perp * p2
        ep = p0 * p0 + p1 * p1 + p2 * p2
        heat[i] += 0.5 * e0 * (1 - (1 - f) ** 2) + 0.02 * ep * f_perp * (1 - f)


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------
def run(out_times, out_screen, out_dir, n=P.N_STREAM, drag_of_t=None,
        glow_tau=0.7, max_sync=12.0, verbose=True):
    """Integrate and write one snapshot for every entry of out_times.

    out_screen: matching screen-time (seconds) of each output, used so that the
    visual cooling of shock-heated gas is defined in screen time.
    """
    rng = np.random.default_rng(P.SEED)
    o, w2, mw, s = make_star(n, rng)
    jit = rng.normal(1.0, 0.12, n).clip(0.7, 1.4)
    t_start = float(out_times[0])
    t_end = float(out_times[-1])
    ct0, cdt, cx, cv = centre_orbit(t_start - 10.0, 3000.0)
    xc = np.zeros(3)
    vc = np.zeros(3)
    k = int((t_start - ct0) / cdt)
    x = cx[k] + o
    v = np.repeat(cv[k][None, :], n, axis=0)
    alive = np.ones(n, np.bool_)
    chimax = np.zeros(n)
    heat = np.zeros(n)
    glow = np.zeros(n)
    trel = np.full(n, np.inf)
    m = mw.copy()

    os.makedirs(out_dir, exist_ok=True)
    F = len(out_times)
    pos_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'pos.npy'), 'w+', np.float32, (F, n, 3))
    vel_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'vel.npy'), 'w+', np.float16, (F, n, 3))
    glow_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'glow.npy'), 'w+', np.float16, (F, n))
    bond_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'bond.npy'), 'w+', np.float16, (F, n))
    alive_mm = np.lib.format.open_memmap(os.path.join(out_dir, 'alive.npy'), 'w+', np.bool_, (F, n))
    centre = np.zeros((F, 6))
    np.save(os.path.join(out_dir, 'mass.npy'), m.astype(np.float32))
    np.save(os.path.join(out_dir, 'rest.npy'), (o / P.R_STAR).astype(np.float32))

    cell, hxy, hz = 1.6, 260.0, 24.0
    ncell = int(2 * hxy / cell) ** 2 * int(2 * hz / cell)
    g_sm = np.zeros(ncell)
    g_sv = np.zeros((ncell, 3))
    g_sv2 = np.zeros(ncell)
    g_cnt = np.zeros(ncell, np.int32)
    g_idx = np.full(n, -1, np.int64)
    t = t_start
    s_prev = out_screen[0]
    tic = time.time()
    for f in range(F):
        t_target = float(out_times[f])
        while t < t_target - 1e-9:
            tn = min(t_target, t + max_sync)
            dk = drag_of_t(0.5 * (t + tn)) if drag_of_t else 0.0
            bnow = 1.0 - np.clip((chimax * jit - 0.85) / 0.5, 0, 1)
            # mass-loss runaway: the remaining core is bound only by what is left
            bind = max(float((m * (bnow * bnow * (3 - 2 * bnow))).sum()), 0.03)
            advance(x, v, o, w2, jit, chimax, alive, heat, t, tn, ct0, cdt, cx, cv,
                    P.GM, 0.03, dk, P.R_ABSORB, bind)
            if t > 0.0 and bind < 0.75:
                # a core stripped of its envelope is over-pressured and unbinds
                chimax *= np.exp((tn - t) / 45.0)
            collide(x, v, m, alive, chimax, jit, heat, cell, hxy, hz, P.GM, 0.6,
                    g_sm, g_sv, g_sv2, g_cnt, g_idx, 1.0 - np.exp(-(tn - t) / 80.0),
                    *(rng.random(3) * cell))
            t = tn
        ds = out_screen[f] - s_prev
        s_prev = out_screen[f]
        glow = glow * np.exp(-ds / glow_tau) + heat
        heat[:] = 0
        b = 1.0 - np.clip((chimax * jit - 0.85) / 0.5, 0, 1)
        b = b * b * (3 - 2 * b)
        trel[(b < 0.5) & ~np.isfinite(trel)] = t
        pos_mm[f] = x
        vel_mm[f] = v
        glow_mm[f] = np.minimum(glow, 6e4)
        bond_mm[f] = b
        alive_mm[f] = alive
        kk = int((t - ct0) / cdt)
        kk = min(max(kk, 0), len(cx) - 1)
        centre[f, :3] = cx[kk]
        centre[f, 3:] = cv[kk]
        if verbose and (f % 25 == 0 or f == F - 1):
            print(f'frame {f:4d}  t={t:10.1f}  alive={alive.mean():.3f}  bonded={(b>0.5).mean():.3f}'
                  f'  glow_max={glow.max():.3g}  {time.time()-tic:6.1f}s', flush=True)
    np.save(os.path.join(out_dir, 'centre.npy'), centre)
    np.save(os.path.join(out_dir, 'trel.npy'), trel)
    np.save(os.path.join(out_dir, 'times.npy'), np.asarray(out_times))
    for mm in (pos_mm, vel_mm, glow_mm, bond_mm, alive_mm):
        mm.flush()


if __name__ == '__main__':
    import timeline as TL
    out = sys.argv[1] if len(sys.argv) > 1 else 'snap'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else P.N_STREAM
    s = TL.frame_screen_times()
    t = TL.sim_time(s)
    run(t, s, out, n=n, drag_of_t=TL.drag_of_t)
