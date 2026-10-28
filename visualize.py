"""
visualize.py — Koyu temalı (siyah zemin, neon çizgiler) FuncAnimation.

Üstte  : robotun 2D yan görünümü (gövde, uyluk, baldır, tekerlek + dönen kollar, zemin profili,
         temas noktası ve normal kuvvet vektörü).  Kamera robotu takip eder.
Altta  : zamanla ilerleyen iki grafik —  tekerlek torku [N·m]  ve  tekerlek yükü (normal kuvvet) [N].

Kullanım:
    python visualize.py out_lqr.npz --terrain course                # pencere
    python visualize.py out_lqr.npz --terrain course --save anim.mp4  # mp4 (ffmpeg) veya .gif (pillow)
    python visualize.py out_lqr.npz out_mpc.npz --terrain course --compare compare.png
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np

from dynamics import Params, WheeledBiped
from simulate import SimResult
from terrain import Terrain

# ---- neon palet ------------------------------------------------------------------------
BG = "#000000"
CYAN, MAGENTA, LIME, ORANGE, YELLOW, PINK, BLUE = ("#00F0FF", "#FF2BD6", "#39FF14", "#FF9A00", "#FFF01F",
                                                    "#FF3B6B", "#4D7CFF")
GRID, TXT = "#1f1f1f", "#b8c4cc"


def _style_axis(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#3a3a3a")
    ax.tick_params(colors=TXT, labelsize=9)
    ax.grid(color=GRID, lw=0.6)


class Glow:
    """Neon parlama efekti: aynı çizgiyi 3 kalınlıkta, azalan opaklıkla çizer."""

    def __init__(self, ax, color, lw=2.0, **kw):
        self.lines = [ax.plot([], [], color=color, lw=lw * m, alpha=a, solid_capstyle="round", **kw)[0]
                      for m, a in ((5.0, 0.07), (2.4, 0.20), (1.0, 1.0))]

    def set_data(self, x, y):
        for l in self.lines:
            l.set_data(x, y)


def _frame_data(res: SimResult, model: WheeledBiped, terrain: Terrain, fps: int, speed: float):
    """Animasyon karelerinin zamanlarını seçip q, tork, N değerlerini ve ileri kinematiği önceden hesaplar."""
    tf = np.arange(0.0, res.t[-1], speed / fps)
    interp = lambda a: np.stack([np.interp(tf, res.t, a[:, i]) for i in range(a.shape[1])], axis=1)
    q = interp(res.q)
    fr = [model.fk(qi) for qi in q]
    pts = {k: np.array([f[k] for f in fr]) for k in ("hip", "knee", "wheel", "c_body", "com")}
    pts["psi"] = np.array([f["psi"] for f in fr])
    return tf, q, pts


def animate(res: SimResult, terrain: Terrain, model: WheeledBiped | None = None, fps=30, speed=1.0,
            save: str | None = None, show: bool = True, window_w=4.0, hist=None):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, Polygon

    model = model or WheeledBiped(Params())
    p = model.p
    tf, q, pts = _frame_data(res, model, terrain, fps, speed)
    wheel_xy = pts["wheel"]
    N_f = np.interp(tf, res.t, res.N)
    tau_w_f = np.interp(tf, res.t, res.tau[:, 2])
    v_f = np.interp(tf, res.t, res.v)

    if save is None and not show:
        matplotlib.use("Agg")
    fig = plt.figure(figsize=(12, 8), facecolor=BG)
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.35, 1, 1], hspace=0.28, left=0.07, right=0.97, top=0.93, bottom=0.07)
    ax = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2])
    for a in (ax, ax1, ax2):
        _style_axis(a)
    fig.text(0.07, 0.955, f"TEKERLEKLİ İKİ-BACAKLI ROBOT  ·  {res.controller}  ·  {res.terrain}", color=CYAN,
             fontsize=12, fontweight="bold", family="monospace")

    # ---- zemin (statik) ---------------------------------------------------------------------
    x_lo, x_hi = min(wheel_xy[:, 0].min() - 4, -3), wheel_xy[:, 0].max() + 6
    xs, gs_ = terrain.sample(x_lo, x_hi, 2500)
    ax.fill_between(xs, gs_, gs_.min() - 3, color="#0a2a0a", lw=0)
    ground_glow = Glow(ax, LIME, lw=1.8); ground_glow.set_data(xs, gs_)
    for xm in np.arange(np.floor(x_lo / 5) * 5, x_hi, 5.0):
        gm = float(terrain.g(xm))
        ax.plot([xm, xm], [gm - 0.05, gm], color=LIME, lw=1, alpha=0.6)
        ax.text(xm, gm - 0.09, f"{xm:.0f} m", color=LIME, alpha=0.7, fontsize=8, ha="center", va="top")
    for xl, lab in terrain.labels():
        ax.text(xl, float(terrain.g(xl)) + 0.62, lab, color=YELLOW, alpha=0.75, fontsize=9, ha="center", family="monospace")
        ax.plot([xl, xl], [float(terrain.g(xl)) + 0.12, float(terrain.g(xl)) + 0.56], color=YELLOW, lw=0.6, alpha=0.25, ls=":")

    # ---- robot (dinamik) ----------------------------------------------------------------------
    body = Polygon(np.zeros((4, 2)), closed=True, fc=MAGENTA + "22", ec=MAGENTA, lw=2.0, joinstyle="round")
    ax.add_patch(body)
    body_glow = Polygon(np.zeros((4, 2)), closed=True, fc="none", ec=MAGENTA, lw=7, alpha=0.12)
    ax.add_patch(body_glow)
    thigh = Glow(ax, CYAN, lw=3.2); shank = Glow(ax, BLUE, lw=3.2)
    wheel = Circle((0, 0), p.R, fc="#ffffff08", ec=ORANGE, lw=2.2); ax.add_patch(wheel)
    wheel_glow = Circle((0, 0), p.R, fc="none", ec=ORANGE, lw=8, alpha=0.12); ax.add_patch(wheel_glow)
    spokes = [ax.plot([], [], color=ORANGE, lw=1.6, alpha=0.9)[0] for _ in range(3)]
    joints = ax.scatter([], [], s=38, c="white", zorder=6)
    com_dot = ax.scatter([], [], s=55, c=YELLOW, marker="x", zorder=6, lw=1.6)
    contact_dot = ax.scatter([], [], s=34, c=PINK, zorder=7)
    force_vec = Glow(ax, PINK, lw=1.8)
    hud = ax.text(0.012, 0.93, "", transform=ax.transAxes, color=TXT, fontsize=10, family="monospace", va="top")
    ax.set_yticks([]); ax.set_xticks([])

    # ---- alt grafikler ---------------------------------------------------------------------------
    tmax = res.t[-1]
    ax1.set_xlim(0, tmax); ax2.set_xlim(0, tmax)
    lim = p.tau_max_wheel
    ax1.axhline(0, color="#444", lw=0.8)
    for s_ in (-1, 1):
        ax1.axhline(s_ * lim, color=PINK, lw=0.9, ls="--", alpha=0.6)
    pad = 0.12 * (res.tau[:, 2].max() - res.tau[:, 2].min() + 1e-6) + 0.3
    ax1.set_ylim(min(res.tau[:, 2].min(), -lim * 0.0) - pad, res.tau[:, 2].max() + pad)
    ax1.set_ylabel("Tekerlek torku [N·m]", color=TXT, fontsize=10)
    mg = model.m_total * p.g
    ax2.axhline(mg, color="#888", lw=0.9, ls="--"); ax2.text(tmax * 0.995, mg, " m·g", color="#888", fontsize=8, va="bottom", ha="right")
    pad2 = 0.1 * (res.N.max() - res.N.min() + 1e-6) + 2
    ax2.set_ylim(min(res.N.min(), mg) - pad2, max(res.N.max(), mg) + pad2)
    ax2.set_ylabel("Tekerlek yükü N [N]", color=TXT, fontsize=10)
    ax2.set_xlabel("zaman [s]", color=TXT, fontsize=10)
    tau_line = Glow(ax1, CYAN, lw=1.6); N_line = Glow(ax2, LIME, lw=1.6)
    tau_dot = ax1.scatter([], [], s=40, c=CYAN, zorder=5); N_dot = ax2.scatter([], [], s=40, c=LIME, zorder=5)
    cur1 = ax1.axvline(0, color="white", lw=0.7, alpha=0.4); cur2 = ax2.axvline(0, color="white", lw=0.7, alpha=0.4)

    # kamera oranı: eksen kutusunun en-boy oranından pencere yüksekliği
    pos = ax.get_position()
    ratio = (pos.height * fig.get_figheight()) / (pos.width * fig.get_figwidth())
    win_h = window_w * ratio
    cam_y = [float(wheel_xy[0, 1]) + 0.25]

    def update(i):
        w, hip, knee, cb, com = wheel_xy[i], pts["hip"][i], pts["knee"][i], pts["c_body"][i], pts["com"][i]
        th = q[i, 2]
        eb = np.array([np.sin(th), np.cos(th)]); nb = np.array([np.cos(th), -np.sin(th)])
        s0, s1, hw = -0.04, p.body_h, p.body_w / 2
        poly = np.array([hip + s0 * eb - hw * nb, hip + s0 * eb + hw * nb, hip + s1 * eb + hw * nb, hip + s1 * eb - hw * nb])
        body.set_xy(poly); body_glow.set_xy(poly)
        thigh.set_data([hip[0], knee[0]], [hip[1], knee[1]]); shank.set_data([knee[0], w[0]], [knee[1], w[1]])
        wheel.center = tuple(w); wheel_glow.center = tuple(w)
        for k, sp_ in enumerate(spokes):
            ang = pts["psi"][i] + k * 2 * np.pi / 3 + np.pi / 2
            sp_.set_data([w[0], w[0] + p.R * np.cos(ang)], [w[1], w[1] + p.R * np.sin(ang)])
        joints.set_offsets(np.array([hip, knee, w])); com_dot.set_offsets(com[None, :])
        c = terrain.contact(w, p.R, w[0])
        contact_dot.set_offsets(c.p[None, :])
        L = 0.0022 * N_f[i]
        force_vec.set_data([c.p[0], c.p[0] + c.n[0] * L], [c.p[1], c.p[1] + c.n[1] * L])
        # kamera (yatay: robotu takip, dikey: yumuşatılmış)
        cam_y[0] += (1.0 if i == 0 else 0.10) * ((w[1] + 0.40) - cam_y[0])
        ax.set_xlim(w[0] - 0.38 * window_w, w[0] + 0.62 * window_w)
        ax.set_ylim(cam_y[0] - 0.5 * win_h, cam_y[0] + 0.5 * win_h)
        hud.set_text(f"t = {tf[i]:5.1f} s    v = {v_f[i]:4.2f} m/s    θ = {np.degrees(th):+5.1f}°    "
                     f"x = {c.xc:5.1f} m    N = {N_f[i]:5.1f} N")
        j = i + 1
        tau_line.set_data(tf[:j], tau_w_f[:j]); N_line.set_data(tf[:j], N_f[:j])
        tau_dot.set_offsets([[tf[i], tau_w_f[i]]]); N_dot.set_offsets([[tf[i], N_f[i]]])
        cur1.set_xdata([tf[i], tf[i]]); cur2.set_xdata([tf[i], tf[i]])
        return []

    anim = FuncAnimation(fig, update, frames=len(tf), interval=1000 / fps, blit=False, repeat=False)
    if save:
        if save.lower().endswith(".gif"):
            anim.save(save, writer="pillow", fps=fps, dpi=80, savefig_kwargs=dict(facecolor=BG))
        else:
            anim.save(save, writer="ffmpeg", fps=fps, dpi=110, savefig_kwargs=dict(facecolor=BG),
                      extra_args=["-pix_fmt", "yuv420p"])
        print("kaydedildi:", save)
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return anim, fig, update


def compare_plot(results: list[SimResult], terrain: Terrain, save: str | None = None, xaxis: str = "x"):
    """LQR ve MPC sonuçlarını koyu temada karşılaştıran statik şekil."""
    import matplotlib.pyplot as plt
    cols = [CYAN, MAGENTA, LIME, ORANGE]
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), facecolor=BG, sharex=True,
                            gridspec_kw=dict(height_ratios=[0.8, 1, 1, 1, 1], hspace=0.25, left=0.08, right=0.97, top=0.95, bottom=0.07))
    for a in axs:
        _style_axis(a)
    ref = results[0]
    X = (lambda r: r.xc) if xaxis == "x" else (lambda r: r.t)
    if xaxis == "x":
        xs, g = terrain.sample(ref.xc.min(), ref.xc.max(), 1500)
        axs[0].fill_between(xs, g, g.min() - 0.1, color="#0a2a0a"); axs[0].plot(xs, g, color=LIME, lw=1.6)
        axs[0].set_ylabel("zemin [m]", color=TXT)
        for xl, lab in terrain.labels():
            axs[0].text(xl, max(float(terrain.g(xl)), 0.0) + 0.07, lab, color=YELLOW, fontsize=8, ha="center", family="monospace")
    else:
        axs[0].plot(ref.t, np.degrees(ref.q[:, 2]) * 0, color=LIME); axs[0].set_ylabel("zemin: düz", color=TXT)
    axs[1].plot(X(ref), ref.v_ref, color="#888", ls="--", lw=1, label="referans")
    styles = [dict(ls="-", lw=3.2, alpha=0.85), dict(ls=(0, (4, 2)), lw=1.5, alpha=1.0)]   # çakışan eğriler görünür kalsın
    for k, (r, c) in enumerate(zip(results, cols)):
        st = styles[min(k, 1)]
        axs[1].plot(X(r), r.v, color=c, label=r.controller, **st)
        axs[2].plot(X(r), np.degrees(r.q[:, 2] - r.theta_ref), color=c, label=r.controller, **st)
        axs[3].plot(X(r), r.tau[:, 2], color=c, label=r.controller, **st)
        axs[4].plot(X(r), r.N, color=c, label=r.controller, **st)
    axs[1].set_ylabel("ileri hız [m/s]", color=TXT); axs[2].set_ylabel("pitch hatası [°]", color=TXT)
    axs[3].set_ylabel("tekerlek torku [N·m]", color=TXT); axs[4].set_ylabel("tekerlek yükü [N]", color=TXT)
    axs[4].set_xlabel("konum x [m]" if xaxis == "x" else "zaman [s]", color=TXT)
    for a in axs[1:]:
        leg = a.legend(facecolor=BG, edgecolor="#333", labelcolor=TXT, fontsize=9, loc="upper right")
    fig.suptitle("LQR  vs  MPC — " + ref.terrain, color=CYAN, fontsize=13, fontweight="bold", family="monospace")
    if save:
        fig.savefig(save, dpi=110, facecolor=BG)
        print("kaydedildi:", save)
    return fig


def _terrain_from_name(name: str) -> Terrain:
    return Terrain.course() if name == "course" else Terrain.flat()


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")   # Windows konsolunda Türkçe karakterler için
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="simulate.py --save ile üretilen .npz dosyaları")
    ap.add_argument("--terrain", choices=["flat", "course"], default="course")
    ap.add_argument("--save", default=None, help="animasyonu .mp4 / .gif olarak kaydet")
    ap.add_argument("--compare", default=None, help="LQR/MPC karşılaştırma şeklini kaydet (png)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    terrain = _terrain_from_name(a.terrain)
    results = [SimResult.load(f) for f in a.files]
    if a.compare:
        compare_plot(results, terrain, a.compare, "x" if a.terrain == "course" else "t")
    else:
        animate(results[0], terrain, fps=a.fps, speed=a.speed, save=a.save, show=not a.no_show and a.save is None)


if __name__ == "__main__":
    main()
