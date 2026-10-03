"""
Physical scenario and simulation parameters for the tidal disruption event (TDE).

Code units:  length = Schwarzschild radius R_s,  c = 1,  hence  G*M_bh = 0.5.

Physical scenario (what the picture represents)
------------------------------------------------
  Black hole        M_bh = 5e5 M_sun          (intermediate / light SMBH)
                    R_s  = 2GM/c^2 ~ 1.48e6 km ~ 2.1 R_sun
                    shadow radius  ~ 2.6 R_s ~ 5.5 R_sun
  Star              subgiant, M* = 1.3 M_sun,  R* = 6 R_sun ~ 2.85 R_s
                    -> stellar DIAMETER (12 R_sun) is ~2.9x the horizon diameter,
                       but the star is ~4e5 times less massive than the hole.
  Tidal radius      r_t = R* (M_bh/M*)^(1/3) ~ 440 R_sun ~ 200 R_s   >>  R_s
  Pericentre        r_p ~ 7 R_s (deep encounter)  > r_mb = 2 R_s  > R_s
  -> the star is torn apart far outside the event horizon; only a tiny fraction
     of debris ever gets close enough to be swallowed directly.

Artistic compression (stated honestly)
--------------------------------------
  * Time is compressed non-uniformly (see timeline.py): seconds of screen time
    span minutes (pericentre passage) up to months (disk formation).
  * The SPATIAL scale of the debris orbits is compressed: the debris energy
    spread is computed with an effective mass ratio q_eff = 1e3 instead of 4e5,
    so the most-bound debris turns around at ~300 R_s instead of ~1e4 R_s and the
    returning streams fit in one frame. Star size, horizon size, shadow size and
    pericentre distance keep their true ratios.
"""
import numpy as np

GM = 0.5                    # G*M_bh in code units (R_s = 1, c = 1)
R_HORIZON = 1.0
R_SHADOW = 1.5 * np.sqrt(3.0)   # critical impact parameter 3*sqrt(3)/2 R_s ~ 2.598
R_ISCO = 3.0

R_STAR = 2.85               # stellar radius in R_s
Q_EFF = 118.0              # effective mass ratio for the debris dynamics
GM_STAR = GM / Q_EFF
R_TIDAL = R_STAR * Q_EFF ** (1.0 / 3.0)
R_PERI = 12.0               # pericentre of the stellar orbit
R_ABSORB = 1.15             # particles inside this radius are swallowed

POLY_N = 3.0                # polytropic index for the stellar density profile

# Orbital plane is z = 0, star moves counter-clockwise seen from +z.
PERI_ANGLE = np.deg2rad(-90.0)  # direction of pericentre in the orbital plane

N_STREAM = 220_000          # simulated gas particles
SEED = 7
