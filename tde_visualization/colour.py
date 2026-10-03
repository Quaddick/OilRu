"""Black-body colours (Planck spectrum x CIE 1931 observer -> linear sRGB)."""
import numpy as np


def _cie_xyz(lam_nm):
    # Wyman, Sloan & Shirley (2013) multi-lobe fit of the CIE 1931 2-deg observer
    l = lam_nm

    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    x = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2)
    y = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1)
    z = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8)
    return x, y, z


def _planck(lam_nm, T):
    lam = lam_nm * 1e-9
    h, c, k = 6.62607e-34, 2.99792e8, 1.380649e-23
    return 1.0 / (lam ** 5 * (np.exp(h * c / (lam * k * T)) - 1.0))


_M = np.array([[3.2406, -1.5372, -0.4986],
               [-0.9689, 1.8758, 0.0415],
               [0.0557, -0.2040, 1.0570]])

T_MIN, T_MAX, T_N = 1200.0, 60000.0, 1024
_T_TAB = np.geomspace(T_MIN, T_MAX, T_N)


def _build():
    lam = np.linspace(380, 780, 401)
    xb, yb, zb = _cie_xyz(lam)
    out = np.zeros((T_N, 3))
    for i, T in enumerate(_T_TAB):
        sp = _planck(lam, T)
        X, Y, Z = (sp * xb).sum(), (sp * yb).sum(), (sp * zb).sum()
        rgb = _M @ np.array([X, Y, Z]) / Y
        rgb = np.clip(rgb, 0.0, None)
        # luminance-normalised so that colour and brightness are independent
        out[i] = rgb / (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
    return out.astype(np.float32)


BB_TABLE = _build()
LOG_T_MIN = np.log(T_MIN)
LOG_T_STEP = (np.log(T_MAX) - np.log(T_MIN)) / (T_N - 1)


def blackbody(T):
    """Luminance-normalised linear-sRGB colour of a black body at temperature T (array)."""
    T = np.clip(np.asarray(T, np.float32), T_MIN, T_MAX)
    f = (np.log(T) - LOG_T_MIN) / LOG_T_STEP
    i = np.clip(f.astype(np.int32), 0, T_N - 2)
    u = (f - i)[..., None]
    return BB_TABLE[i] * (1 - u) + BB_TABLE[i + 1] * u
