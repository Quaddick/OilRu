"""
Screen time (seconds) -> simulation time mapping, and the global beat sheet.

The film is 30 s.  Simulation time is warped smoothly: the playback rate
(sim-time units per screen second) is keyed in log space and integrated, so
there is never a jump in time, only easing in and out of slow motion / time-lapse.

Beat sheet (screen seconds):
   0 -  6   approach            rate ~ steady, the star accelerates by itself
   6 - 13   tidal stretching    gentle slow motion around pericentre
  13 - 22   disruption          rate ramps up, the stream flies out and the
                                bound part falls back; stream collision ~19.5 s
  22 - 30   disk formation      time-lapse (rate x20) then settles to a calm end
"""
import numpy as np

FPS = 30
DURATION = 30.0
N_FRAMES = int(FPS * DURATION)

S_PERI = 11.4          # screen time of the stellar pericentre passage (t_sim = 0)
S_COLLIDE = 19.6       # intended peak of the stream-stream collision

# (screen second, sim units per screen second)
RATE_KEYS = np.array([
    (0.0, 150.0),
    (4.5, 165.0),
    (7.0, 60.0),
    (9.5, 22.0),
    (11.4, 18.0),
    (13.0, 40.0),
    (14.0, 120.0),
    (15.0, 300.0),
    (16.5, 380.0),
    (18.0, 480.0),
    (19.6, 420.0),
    (21.0, 520.0),
    (22.4, 1200.0),
    (24.0, 3600.0),
    (26.0, 4200.0),
    (27.6, 900.0),
    (29.0, 260.0),
    (30.0, 200.0),
])


def _pchip(xk, yk, x):
    """Monotone cubic Hermite interpolation (no overshoot in the rate curve)."""
    xk = np.asarray(xk, float)
    yk = np.asarray(yk, float)
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.zeros_like(yk)
    for i in range(1, len(yk) - 1):
        if d[i - 1] * d[i] > 0:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0] = 0.0
    m[-1] = 0.0
    x = np.clip(np.asarray(x, float), xk[0], xk[-1])
    i = np.clip(np.searchsorted(xk, x) - 1, 0, len(h) - 1)
    t = (x - xk[i]) / h[i]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2
    return h00 * yk[i] + h10 * h[i] * m[i] + h01 * yk[i + 1] + h11 * h[i] * m[i + 1]


def rate(s):
    return np.exp(_pchip(RATE_KEYS[:, 0], np.log(RATE_KEYS[:, 1]), s))


_S_FINE = np.linspace(0.0, DURATION + 1.0, 200001)
_R_FINE = rate(_S_FINE)
_T_FINE = np.concatenate([[0.0], np.cumsum(0.5 * (_R_FINE[1:] + _R_FINE[:-1]) * np.diff(_S_FINE))])
_T_FINE -= np.interp(S_PERI, _S_FINE, _T_FINE)


def sim_time(s):
    return np.interp(s, _S_FINE, _T_FINE)


def screen_time(t):
    return np.interp(t, _T_FINE, _S_FINE)


def frame_screen_times():
    return np.arange(N_FRAMES) / FPS


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def drag_of_t(t):
    """Circularisation drag (1/sim-time) standing in for repeated shocks in the time-lapse."""
    s = screen_time(t)
    return 1.0 / 600.0 * float(smoothstep(21.5, 25.0, s))


if __name__ == '__main__':
    for s in [0, 3, 6, 9, 11.4, 13, 15, 17, 19.6, 22, 24, 26, 28, 30]:
        print(f's={s:5.1f}  t={sim_time(s):10.1f}  rate={rate(s):8.1f}')
