"""
Camera choreography.  One continuous move, no cuts.

All channels are keyed against screen time and interpolated with monotone
cubic splines (continuous velocity, no overshoot), so every move eases in and
out.  The aim point is a blend of a keyed world point and the star's centre of
mass, so the camera follows the star smoothly while it is the subject and then
hands over to the black hole / collision region.

World frame: black hole at the origin, orbital plane z = 0, lengths in R_s.
"""
import numpy as np
from timeline import _pchip

# s, aim point (x, y, z), star-follow weight, distance, azimuth, elevation, fov, aperture,
# focus weight (1 = focus on the star, 0 = focus on the aim point)
#   azimuth: direction from the aim point towards the camera, degrees in the xy-plane
#   aperture: circle of confusion at infinity, as a fraction of frame height
KEYS = np.array([
    #  s      ax      ay    az  wstar   dist    azim   elev   fov    aper  wfocus
    (0.0,  -61.1,   91.3, 0.0, 0.00, 100.0,  147.4,  9.0, 34.0, 0.0030, 1.0),
    (3.0,  -46.8,   51.6, 0.0, 0.00,  90.7,  156.7, 10.5, 34.0, 0.0032, 1.0),
    (6.0,  -23.7,    8.8, 0.0, 0.00,  71.9,  191.8, 14.6, 34.0, 0.0034, 1.0),
    (8.5,  -11.2,   -3.6, 0.0, 0.00,  54.4,  212.3, 16.9, 34.0, 0.0036, 1.0),
    (10.2,  -4.5,   -7.3, 0.0, 0.00,  54.7,  221.0, 18.6, 34.0, 0.0036, 1.0),
    (11.4,   0.0,   -6.1, 0.0, 0.00,  54.2,  227.9, 18.8, 34.0, 0.0036, 1.0),
    (13.0,   3.7,   -4.8, 0.0, 0.00,  70.3,  238.0, 21.2, 34.0, 0.0032, 0.6),
    (15.5,  15.0,   30.0, 0.0, 0.00, 150.0,  244.0, 25.0, 34.5, 0.0026, 0.0),
    (17.5,  10.0,   45.0, 0.0, 0.00, 205.0,  247.0, 27.0, 35.0, 0.0022, 0.0),
    (19.0,  -8.0,   28.0, 0.0, 0.00, 165.0,  250.0, 27.0, 35.0, 0.0026, 0.0),
    (20.5, -12.0,   20.0, 0.0, 0.00, 138.0,  253.0, 26.0, 34.5, 0.0028, 0.0),
    (22.0,  -8.0,   14.0, 0.0, 0.00, 135.0,  258.0, 26.0, 34.5, 0.0028, 0.0),
    (24.0,   0.0,   12.0, 0.0, 0.00, 150.0,  268.0, 26.0, 34.5, 0.0026, 0.0),
    (27.0,   8.0,   16.0, 0.0, 0.00, 180.0,  283.0, 25.0, 34.5, 0.0022, 0.0),
    (30.0,  10.0,   18.0, 0.0, 0.00, 198.0,  291.0, 24.5, 34.5, 0.0020, 0.0),
])


def _ch(col, s):
    return _pchip(KEYS[:, 0], KEYS[:, col], s)


class Cam:
    pass


def camera_at(s, star_centre_fn):
    """Return a camera for screen time s.  star_centre_fn(s) -> world position."""
    aim_key = np.array([_ch(1, s), _ch(2, s), _ch(3, s)])
    w = float(np.clip(_ch(4, s), 0.0, 1.0))
    star = np.asarray(star_centre_fn(s), float)
    aim = (1 - w) * aim_key + w * star
    dist = float(_ch(5, s))
    az = np.deg2rad(float(_ch(6, s)))
    el = np.deg2rad(float(_ch(7, s)))
    # very slow, low-amplitude drift so the frame breathes (no handheld shake)
    drift = 0.004 * dist * np.array([np.sin(0.21 * s + 1.0), np.sin(0.17 * s + 2.3), 0.5 * np.sin(0.13 * s)])
    pos = aim + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]) + drift
    c = Cam()
    c.pos = pos
    c.aim = aim
    fwd = aim - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c.fwd, c.right, c.up = fwd, right, up
    c.fov = np.deg2rad(float(_ch(8, s)))
    c.aperture = float(_ch(9, s))
    wf = float(np.clip(_ch(10, s), 0.0, 1.0))
    fpt = (1 - wf) * aim + wf * star
    c.focus = float(np.dot(fpt - pos, fwd))
    return c
