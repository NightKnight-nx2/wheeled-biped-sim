"""
terrain.py — Smooth (C∞) zemin profili ve tekerlek–zemin temas geometrisi.

Zemin  y = g(x)  fonksiyonudur.  Birleşik "engel parkuru" şu analitik bileşenlerin toplamıdır:

    tümsek (bump)   h · exp(−(x−x0)² / 2σ²)                      (h>0 tümsek, h<0 çukur)
    çukur  (pit)    aynı Gauss, h<0
    rampa  (ramp)   eğim m = tan(açı) olan, kenarları tanh ile yumuşatılmış eğik düzlem
                    g = m·[ (w/2)(ln cosh((x−a)/w) − ln cosh((x−b)/w)) + (b−a)/2 ]
    S-viraj (scurve) H · ½(1 + tanh((x−x0)/w))                    (sigmoid, S biçimli yükseliş)

Türevler (g', g'', g''') sympy ile analitik alınır, lambdify ile hızlı sayısal fonksiyona çevrilir.

Temas geometrisi
----------------
Tekerlek merkezi c=(x_w, y_w), yarıçap R.  Temas noktası P = (x_c, g(x_c)), c'ye en yakın zemin noktasıdır:

        (x_c − x_w) + g'(x_c)·(g(x_c) − y_w) = 0        (Newton ile çözülür)

Zemin çerçevesi:  t = (1, g')/W ,  n = (−g', 1)/W ,  W = √(1+g'²) ,  eğrilik  κ = g''/W³
(κ>0 çukurda/vadide, κ<0 tümsekte).  Yuvarlanma için R·κ < 1 olmalıdır.
Normal kuvvet N ve teğet kuvvet F_t, dynamics.py'deki KKT çözümünden (Lagrange çarpanları) gelir.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp


@dataclass
class ContactFrame:
    xc: float                 # temas noktasının x'i
    p: np.ndarray             # temas noktası (xc, g(xc))
    t: np.ndarray             # birim teğet (ileri)
    n: np.ndarray             # birim normal (yukarı)
    kappa: float              # işaretli eğrilik  dt/ds = κ n
    d: float                  # tekerlek merkezinin zemine uzaklığı (= R beklenir)
    wheel_center: np.ndarray  # tekerlek merkezi
    slope: float              # yerel eğim açısı atan(g') [rad]
    gp: float = 0.0           # g'(xc)


class Terrain:
    """y = g(x) zemini.  `features`: [(tip, {parametreler}), ...]"""

    def __init__(self, features: list[tuple[str, dict]] | None = None, name: str = "custom"):
        self.features = list(features or [])
        self.name = name
        x = sp.Symbol("x", real=True)
        g = sp.Integer(0)
        for kind, a in self.features:
            if kind in ("bump", "pit"):
                h = a["h"] if kind == "bump" else -abs(a["h"])
                g += h * sp.exp(-((x - a["x0"]) ** 2) / (2 * a["sigma"] ** 2))
            elif kind == "ramp":
                m = np.tan(np.deg2rad(a["angle_deg"]))
                lc = lambda z: sp.log(sp.cosh(z))
                A, Bb, w = a["x_start"], a["x_end"], a["w"]
                g += m * (w / 2 * (lc((x - A) / w) - lc((x - Bb) / w)) + (Bb - A) / 2)
            elif kind == "scurve":
                g += a["H"] * sp.Rational(1, 2) * (1 + sp.tanh((x - a["x0"]) / a["w"]))
            else:
                raise ValueError(kind)
        self.g_expr = g
        derivs = [g, sp.diff(g, x), sp.diff(g, x, 2), sp.diff(g, x, 3)]
        self._f = sp.lambdify(x, derivs, modules="numpy", cse=True)

    # ---- fabrika fonksiyonları ----------------------------------------------------------
    @classmethod
    def flat(cls) -> "Terrain":
        return cls([], name="düz zemin")

    @classmethod
    def course(cls) -> "Terrain":
        """Birleşik engel parkuru: tümsek → çukur → rampa çıkış/iniş → S-viraj (yükseliş + iniş)."""
        return cls([
            ("bump", dict(x0=7.0, h=0.05, sigma=0.30)),
            ("pit", dict(x0=10.0, h=0.05, sigma=0.35)),
            ("ramp", dict(x_start=14.0, x_end=18.0, angle_deg=6.0, w=0.6)),     # yukarı
            ("ramp", dict(x_start=21.0, x_end=25.0, angle_deg=-6.0, w=0.6)),    # aşağı
            ("scurve", dict(x0=30.0, H=0.20, w=1.0)),
            ("scurve", dict(x0=36.0, H=-0.20, w=1.0)),
        ], name="engel parkuru")

    def labels(self) -> list[tuple[float, str]]:
        """Engel etiketleri (animasyonda gösterilir): [(x, metin)]"""
        out = []
        for kind, a in self.features:
            if kind == "bump":
                out.append((a["x0"], "tümsek"))
            elif kind == "pit":
                out.append((a["x0"], "çukur"))
            elif kind == "ramp":
                out.append(((a["x_start"] + a["x_end"]) / 2, "rampa ↗" if a["angle_deg"] > 0 else "rampa ↘"))
            elif kind == "scurve":
                out.append((a["x0"], "S-viraj ↗" if a["H"] > 0 else "S-viraj ↘"))
        return out

    # ---- profil -------------------------------------------------------------------------------
    def g(self, x):
        return self._f(np.asarray(x, dtype=float))[0] * np.ones_like(np.asarray(x, dtype=float))

    def derivs(self, x: float):
        """(g, g', g'', g''') — skaler x için."""
        r = self._f(float(x))
        return tuple(float(v) for v in r)

    def sample(self, x0: float, x1: float, n: int = 800):
        xs = np.linspace(x0, x1, n)
        return xs, np.array([self.derivs(v)[0] for v in xs])

    def max_curvature(self, x0=-5.0, x1=45.0, n=4000):
        xs = np.linspace(x0, x1, n)
        k = []
        for v in xs:
            _, g1, g2, _ = self.derivs(v)
            k.append(g2 / (1 + g1**2) ** 1.5)
        return float(np.max(k)), float(np.min(k))

    # ---- temas ----------------------------------------------------------------------------------
    def _frame(self, xc: float, wheel_center: np.ndarray, R: float) -> ContactFrame:
        g, g1, g2, _ = self.derivs(xc)
        W = np.sqrt(1.0 + g1**2)
        t = np.array([1.0, g1]) / W
        n = np.array([-g1, 1.0]) / W
        p = np.array([xc, g])
        return ContactFrame(xc=xc, p=p, t=t, n=n, kappa=g2 / W**3, d=float((wheel_center - p) @ n),
                            wheel_center=np.asarray(wheel_center, float), slope=float(np.arctan(g1)), gp=g1)

    def contact(self, wheel_center, R: float, xc_guess: float | None = None) -> ContactFrame:
        """Tekerlek merkezine en yakın zemin noktasını (Newton) bulup temas çerçevesini döndürür."""
        xw, yw = float(wheel_center[0]), float(wheel_center[1])
        xc = xw if xc_guess is None else xc_guess
        for _ in range(30):
            g, g1, g2, _ = self.derivs(xc)
            F = (xc - xw) + g1 * (g - yw)
            dF = 1.0 + g2 * (g - yw) + g1**2
            step = F / dF
            xc -= step
            if abs(step) < 1e-12:
                break
        return self._frame(xc, np.array([xw, yw]), R)

    def contact_at(self, xc: float, R: float) -> ContactFrame:
        """Temas x'i verilen durumda tekerlek merkezi ofset eğri üzerinde:  c = P + R n."""
        g, g1, *_ = self.derivs(xc)
        W = np.sqrt(1.0 + g1**2)
        c = np.array([xc, g]) + R * np.array([-g1, 1.0]) / W
        return self._frame(xc, c, R)


if __name__ == "__main__":
    T = Terrain.course()
    kmax, kmin = T.max_curvature()
    print(f"κ_max (vadi) = {kmax:.3f} 1/m,  κ_min (tümsek) = {kmin:.3f} 1/m  → R·κ_max = {0.1 * kmax:.3f} (<1 olmalı)")
    for xq in [0, 7, 10, 16, 19.5, 23, 30, 33, 36, 40]:
        g, g1, g2, g3 = T.derivs(xq)
        print(f"x={xq:5.1f}  g={g:+.3f}  eğim={np.degrees(np.arctan(g1)):+5.1f}°")
    c = T.contact_at(16.0, 0.1)
    c2 = T.contact(c.wheel_center, 0.1, xc_guess=15.7)
    print("temas çözümü doğru mu:", abs(c2.xc - 16.0) < 1e-9, " d =", c2.d)
