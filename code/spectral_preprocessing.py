"""Segment-wise smoothing and continuum removal from the frozen spectral rebuild."""
import numpy as np

BAD_INTERVALS = [(400.0, 430.0, '400-430'), (900.0, 1050.0, '900-1050'), (1350.0, 1450.0, '1350-1450'), (1800.0, 1950.0, '1800-1950'), (2450.0, 2500.0, '2450-2500')]


SG_WINDOW = 11


SG_POLYORDER = 2


def sg_per_segment(y, seg, n_segments):
    from scipy.signal import savgol_filter
    out = np.full_like(y, np.nan)
    for s in range(1, n_segments + 1):
        m = seg == s
        seg_len = int(m.sum())
        wl = SG_WINDOW
        if seg_len < wl:
            # 段长不足窗口时退化为该段可用的最大奇数窗口（记录但不改变数学含义）
            wl = seg_len if seg_len % 2 == 1 else seg_len - 1
            if wl < 3:
                out[:, m] = y[:, m]
                continue
        out[:, m] = savgol_filter(y[:, m], window_length=wl, polyorder=SG_POLYORDER,
                                  axis=1, mode="interp")
    return out


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def upper_convex_hull(x, y):
    """上凸包（天花板），返回 (ux, uy)，x 升序。"""
    pts = sorted(zip(x, y))
    upper = []
    for p in pts:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) >= 0:
            upper.pop()
        upper.append(p)
    return np.array([p[0] for p in upper]), np.array([p[1] for p in upper])


def continuum_remove_one(x, y):
    """单条光谱的分段连续统去除。返回 CR。x 升序、y 为 SG 反射率。"""
    if len(x) < 2:
        return np.full_like(y, np.nan)
    ux, uy = upper_convex_hull(x, y)
    cont = np.interp(x, ux, uy)
    if np.any(cont <= 0):
        raise ValueError("continuum 非正")
    cr = y / cont
    # 仅裁剪浮点误差造成的极小超界到 [0,1]
    cr = np.clip(cr, 0.0, 1.0)
    return cr
