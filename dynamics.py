"""
dynamics.py — Tekerlekli-iki-linkli-bacaklı (wheeled-biped) robotun sagittal düzlem dinamiği.

Model (2D, sagittal düzlem)
---------------------------
  * Gövde (rijit) — kalça eklemine bağlı.
  * Bacak = iki link (uyluk + baldır); sağ/sol bacak sagittal düzlemde çakıştığı için tek eşdeğer
    bacak olarak modellenir (kütle/atalet iki bacağın toplamıdır).  Kalça ve diz motorlu.
  * Baldırın ucunda motorlu tekerlek (yarıçap R).

Genelleştirilmiş koordinatlar (6 adet, floating-base):

    q = [ x , y , θ , q1 , q2 , φ ]

    x, y : kalça ekleminin (gövde referans noktası) dünya konumu  → y = gövde yüksekliği [m]
    θ    : gövde pitch açısı, düşeyden ölçülür, İLERİ eğilme (+)   [rad]
    q1   : kalça açısı — uyluğun gövdeye göre açısı (ayak ileri = +) [rad]
    q2   : diz açısı   — baldırın uyluğa göre açısı                  [rad]
    φ    : tekerlek dönüşü (baldıra göre), İLERİ yuvarlanma (+)      [rad]

  Mutlak açılar (CCW, "aşağı" yönünden ölçülür):  a1 = q1 − θ ,  a2 = a1 + q2 ,  ψ_tekerlek = a2 − φ
  Birim vektör  e(a) = (sin a, −cos a)   (a=0 → düz aşağı, a>0 → ayak ileri)

Hareket denklemleri (kısıtsız, Euler-Lagrange'dan):

    M(q) q̈ + h(q, q̇) + G(q) = B τ            τ = [τ_kalça, τ_diz, τ_tekerlek]

Tekerlek–zemin teması (yuvarlanma kısıtı, kayma yok) bilateral kısıt olarak eklenir:

    M q̈ + h + G = B τ + Jcᵀ λ ,        Jc q̈ = −J̇c q̇  (+ Baumgarte)       λ = [N, F_t]

  N   : temas noktasındaki normal kuvvet ("tekerlek yükü")
  F_t : teğet (çekiş) kuvveti
  Kısıt satırları (zemin çerçevesi n = normal, t = teğet, κ = eğrilik):
      1) n·v_w = 0                (tekerlek merkezi ofset eğri üzerinde, d = R)
      2) t·v_w + R ω_w = 0        (temas noktasında kayma yok)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import fsolve

# --------------------------------------------------------------------------------------
#  Parametreler
# --------------------------------------------------------------------------------------


@dataclass
class Params:
    # gövde
    mb: float = 6.0        # kütle [kg]
    Ib: float = 0.065      # kütle merkezi etrafında atalet [kg m²]
    db: float = 0.12       # kalçadan gövde KM'sine mesafe (gövde ekseni boyunca yukarı) [m]
    body_h: float = 0.30   # çizim için gövde boyu [m]
    body_w: float = 0.16   # çizim için gövde eni [m]
    # uyluk
    m1: float = 0.9
    l1: float = 0.25
    lc1: float = 0.125
    I1: float = 0.0047
    # baldır
    m2: float = 0.7
    l2: float = 0.25
    lc2: float = 0.125
    I2: float = 0.0036
    # tekerlek
    mw: float = 1.0
    R: float = 0.10
    Iw: float = 0.005
    # genel
    g: float = 9.81
    b_joint: float = 0.02   # kalça/diz viskoz sürtünme [N m s/rad]
    b_wheel: float = 0.005  # tekerlek rulman sürtünmesi [N m s/rad]
    # motor sınırları (kontrolör ve simülasyon bunları kullanır)
    tau_max_hip: float = 40.0
    tau_max_knee: float = 40.0
    tau_max_wheel: float = 10.0

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, s) for s in _PARAM_NAMES], dtype=float)

    @property
    def tau_max(self) -> np.ndarray:
        return np.array([self.tau_max_hip, self.tau_max_knee, self.tau_max_wheel])


# türetmede sembolik olarak kullanılan parametreler (sıra önemli)
_PARAM_NAMES = ["mb", "Ib", "db", "m1", "l1", "lc1", "I1", "m2", "l2", "lc2", "I2",
                "mw", "R", "Iw", "g", "b_joint", "b_wheel"]


# --------------------------------------------------------------------------------------
#  Sembolik türetme (sympy)
# --------------------------------------------------------------------------------------


def derive_symbolic() -> dict:
    """Lagrangian'ı kurar; M, h, G, B ve temas kinematiğinin sembolik ifadelerini döndürür."""
    x, y, th, q1, q2, ph = sp.symbols("x y theta q1 q2 phi", real=True)
    xd, yd, thd, q1d, q2d, phd = sp.symbols("xd yd thetad q1d q2d phid", real=True)
    q = sp.Matrix([x, y, th, q1, q2, ph])
    qd = sp.Matrix([xd, yd, thd, q1d, q2d, phd])
    P = {n: sp.Symbol(n, real=True) for n in _PARAM_NAMES}

    def e(a):  # "aşağı"dan ileri doğru a kadar dönmüş birim vektör
        return sp.Matrix([sp.sin(a), -sp.cos(a)])

    # ---- kinematik --------------------------------------------------------------
    a1 = q1 - th
    a2 = a1 + q2
    hip = sp.Matrix([x, y])
    knee = hip + P["l1"] * e(a1)
    wheel = knee + P["l2"] * e(a2)                         # tekerlek merkezi
    c_body = hip + P["db"] * sp.Matrix([sp.sin(th), sp.cos(th)])
    c_thigh = hip + P["lc1"] * e(a1)
    c_shank = knee + P["lc2"] * e(a2)

    w_body = -thd
    w_thigh = -thd + q1d
    w_shank = -thd + q1d + q2d
    w_wheel = w_shank - phd                                # tekerlek mutlak açısal hızı (CCW +)

    # ---- enerjiler -----------------------------------------------------------------
    def vel(r):
        return r.jacobian(q) * qd

    bodies = [  # (kütle, KM konumu, atalet, açısal hız)
        (P["mb"], c_body, P["Ib"], w_body),
        (P["m1"], c_thigh, P["I1"], w_thigh),
        (P["m2"], c_shank, P["I2"], w_shank),
        (P["mw"], wheel, P["Iw"], w_wheel),
    ]
    T = sum(sp.Rational(1, 2) * m * (vel(r).T * vel(r))[0] + sp.Rational(1, 2) * I * w**2
            for m, r, I, w in bodies)
    V = sum(m * P["g"] * r[1] for m, r, _, _ in bodies)
    L = T - V                                              # Lagrangian

    # ---- Euler-Lagrange:  d/dt(∂L/∂q̇) − ∂L/∂q = Q  →  M q̈ + h + G = Q ---------------
    M = sp.hessian(L, tuple(qd))                           # ∂²L/∂q̇²
    M = M.applyfunc(sp.trigsimp)
    Mqd = M * qd
    dL_dq_kin = sp.Matrix([sp.diff(T, qi) for qi in q])    # ∂T/∂q
    h_cor = Mqd.jacobian(q) * qd - dL_dq_kin               # Ṁq̇ − ∂T/∂q  (Coriolis + merkezkaç)
    G = sp.Matrix([sp.diff(V, qi) for qi in q])            # ∂V/∂q
    D = sp.diag(0, 0, 0, P["b_joint"], P["b_joint"], P["b_wheel"])
    h = h_cor + D * qd                                     # viskoz sürtünme h içinde

    B = sp.zeros(6, 3)
    B[3, 0] = B[4, 1] = B[5, 2] = 1                        # τ → (q1, q2, φ)

    # ---- temas kinematiği ------------------------------------------------------------
    Jw = wheel.jacobian(q)                                 # tekerlek merkezi Jacobian'ı (2×6)
    Jw_qd = Jw * qd
    Jdot_qd = Jw_qd.jacobian(q) * qd                       # J̇_w q̇
    e_omega = sp.Matrix([[sp.diff(w_wheel, v) for v in qd]])  # ω_w = e_omega · q̇  (1×6)

    com_m = sum(m for m, *_ in bodies)
    com_x = sum(m * r[0] for m, r, _, _ in bodies) / com_m
    com_y = sum(m * r[1] for m, r, _, _ in bodies) / com_m

    return dict(q=q, qd=qd, P=[P[n] for n in _PARAM_NAMES], L=L, T=T, V=V, M=M, h=h, G=G, B=B,
                wheel=wheel, hip=hip, knee=knee, c_body=c_body, Jw=Jw, Jdot_qd=Jdot_qd,
                e_omega=e_omega, com_x=com_x, com_y=com_y, com_m=com_m,
                psi=a2 - ph, a1=a1, a2=a2)


_SYM_CACHE: dict | None = None
_NUM_CACHE: dict | None = None


def _load_or_derive() -> dict:
    """Sembolik türetme ~10 s sürer; sonucu pickle ile önbelleğe alır (dynamics.py değişirse yenilenir)."""
    cache = Path(__file__).with_name(".dynamics_cache.pkl")
    src = Path(__file__)
    try:
        if cache.exists() and cache.stat().st_mtime > src.stat().st_mtime:
            with open(cache, "rb") as fh:
                return pickle.load(fh)
    except Exception:
        pass
    S = derive_symbolic()
    try:
        with open(cache, "wb") as fh:
            pickle.dump(S, fh)
    except Exception:
        pass
    return S


def _build_numeric():
    """sympy ifadelerini lambdify ile sayısal fonksiyonlara çevirir (tek seferlik)."""
    global _SYM_CACHE, _NUM_CACHE
    if _NUM_CACHE is not None:
        return _NUM_CACHE
    S = _load_or_derive()
    args = (list(S["q"]), list(S["qd"]), S["P"])
    # tek bir lambdify çağrısı → ortak alt ifadeler (cse) paylaşılır, çağrı başına overhead düşer
    f_dyn = sp.lambdify(args, [S["M"], S["h"], S["G"], S["Jw"], S["Jdot_qd"]],
                        modules="numpy", cse=True)
    qargs = (list(S["q"]), S["P"])
    f_fk = sp.lambdify(qargs, [S["hip"], S["knee"], S["wheel"], S["c_body"], S["psi"],
                               S["com_x"], S["com_y"], S["V"]], modules="numpy", cse=True)
    f_omega = sp.lambdify([S["P"]], S["e_omega"], modules="numpy")
    _SYM_CACHE = S
    _NUM_CACHE = dict(f_dyn=f_dyn, f_fk=f_fk, f_omega=f_omega)
    return _NUM_CACHE


# --------------------------------------------------------------------------------------
#  Sayısal model
# --------------------------------------------------------------------------------------


class WheeledBiped:
    """Lambdify edilmiş dinamik + kısıtlı (temaslı) dinamik çözücüsü."""

    nq = 6
    nu = 3
    idx = dict(x=0, y=1, th=2, q1=3, q2=4, ph=5)

    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self._pv = self.p.vector()
        self._num = _build_numeric()
        self.e_omega = np.asarray(self._num["f_omega"](self._pv), dtype=float).reshape(6)
        self.B = np.zeros((6, 3))
        self.B[3, 0] = self.B[4, 1] = self.B[5, 2] = 1.0
        self.m_total = self.p.mb + self.p.m1 + self.p.m2 + self.p.mw

    # ---- M, h, G ------------------------------------------------------------------------
    def terms(self, q, qd):
        """M(q), h(q,q̇), G(q), J_w(q), J̇_w q̇  —  tek lambdify çağrısıyla."""
        M, h, G, Jw, Jdq = self._num["f_dyn"](q, qd, self._pv)
        return (np.asarray(M, float), np.asarray(h, float).reshape(6), np.asarray(G, float).reshape(6),
                np.asarray(Jw, float), np.asarray(Jdq, float).reshape(2))

    def energy(self, q, qd):
        M, *_ = self.terms(q, qd)
        V = self._num["f_fk"](q, self._pv)[-1]
        return 0.5 * qd @ M @ qd + float(V)

    # ---- ileri kinematik (çizim ve ölçüm) -------------------------------------------------
    def fk(self, q):
        hip, knee, wheel, c_body, psi, cx, cy, _ = self._num["f_fk"](q, self._pv)
        return dict(hip=np.asarray(hip, float).ravel(), knee=np.asarray(knee, float).ravel(),
                    wheel=np.asarray(wheel, float).ravel(), c_body=np.asarray(c_body, float).ravel(),
                    psi=float(psi), com=np.array([float(cx), float(cy)]))

    def wheel_center(self, q):
        """Tekerlek merkezi (hızlı, saf numpy) — temas aramasında kullanılır."""
        p = self.p
        a1 = q[3] - q[2]
        a2 = a1 + q[4]
        return np.array([q[0] + p.l1 * np.sin(a1) + p.l2 * np.sin(a2),
                         q[1] - p.l1 * np.cos(a1) - p.l2 * np.cos(a2)])

    # ---- temas kısıtlı dinamik --------------------------------------------------------------
    def constraint_rows(self, q, qd, c, Jw, Jdq, alpha=30.0, gamma=30.0):
        """Kısıt Jacobian'ı Jc (2×6) ve sağ taraf (Jc q̈ = rhs).  Baumgarte: α (konum/hız), γ (kayma)."""
        R = self.p.R
        n, t, kap, d = c.n, c.t, c.kappa, c.d
        vw = Jw @ qd
        tv, nv = t @ vw, n @ vw
        denom = max(1.0 - d * kap, 0.2)           # ṡ = t·v_w / (1 − dκ)
        sdot = tv / denom
        om = self.e_omega @ qd
        J = np.vstack([n @ Jw, t @ Jw + R * self.e_omega])
        rhs = np.array([
            kap * tv * sdot - n @ Jdq - 2 * alpha * nv - alpha**2 * (d - R),   # d̈ = n·a_w − κ ṡ (t·v_w)
            -t @ Jdq - kap * sdot * nv - gamma * (tv + R * om),                # d/dt(t·v_w + Rω) = −γ(...)
        ])
        return J, rhs

    def constrained_accel(self, q, qd, tau, c, alpha=30.0, gamma=30.0):
        """KKT sistemini çözer → (q̈, λ=[N, F_t])."""
        M, h, G, Jw, Jdq = self.terms(q, qd)
        J, rhs = self.constraint_rows(q, qd, c, Jw, Jdq, alpha, gamma)
        K = np.zeros((8, 8))
        K[:6, :6] = M
        K[:6, 6:] = -J.T
        K[6:, :6] = J
        b = np.concatenate([self.B @ tau - h - G, rhs])
        sol = np.linalg.solve(K, b)
        return sol[:6], sol[6:]

    # ---- konfigürasyon / hız kurulumu --------------------------------------------------------
    def config_on_contact(self, c, theta, q1, q2, phi=0.0):
        """Tekerlek merkezi c.wheel_center'da olacak şekilde kalça konumunu hesaplar → q."""
        p = self.p
        a1 = q1 - theta
        a2 = a1 + q2
        off = p.l1 * np.array([np.sin(a1), -np.cos(a1)]) + p.l2 * np.array([np.sin(a2), -np.cos(a2)])
        hip = np.asarray(c.wheel_center) - off
        return np.array([hip[0], hip[1], theta, q1, q2, phi])

    def consistent_velocity(self, q, c, v_ground, thd=0.0, q1d=0.0, q2d=0.0):
        """Kısıtları sağlayan q̇: (θ̇,q̇1,q̇2) verilir, x-hızı ayarlanıp ẏ ve φ̇ kısıttan çözülür.

        v_ground: temas noktasının zemin boyunca (teğet) hızı ṡ.
        """
        M, h, G, Jw, Jdq = self.terms(q, np.zeros(6))
        R = self.p.R
        kap = c.kappa
        vw_t = (1.0 - c.d * kap) * v_ground            # tekerlek merkezinin teğet hızı
        # bilinmeyenler: ẋ, ẏ, φ̇ ;  bilinenler: θ̇, q̇1, q̇2
        known = np.zeros(6)
        known[2:5] = [thd, q1d, q2d]
        A = np.vstack([c.n @ Jw, c.t @ Jw + R * self.e_omega])        # 2×6
        # satır 1: n·v_w = 0 ; satır 2: t·v_w = vw_t ve t·v_w + Rω = 0  → ω = −vw_t / R
        # üç bilinmeyen (ẋ, ẏ, φ̇) için üç denklem:
        rows = [c.n @ Jw, c.t @ Jw, self.e_omega]
        rhs = [0.0, vw_t, -vw_t / R]
        free = [0, 1, 5]
        Amat = np.array([[r[j] for j in free] for r in rows])
        bvec = np.array(rhs) - np.array([r @ known for r in rows])
        sol = np.linalg.solve(Amat, bvec)
        qd = known.copy()
        qd[free] = sol
        return qd

    # ---- statik denge (trim) ----------------------------------------------------------------
    def static_trim(self, q1, q2, slope=0.0):
        """Yerel eğimi `slope` (rad) olan düzlemde, bacak (q1,q2) sabitken statik dengeyi çözer.

        Bilinmeyenler: θ, τ (3), λ=[N,F_t]  →  G − Bτ − Jcᵀλ = 0  (6 denklem).
        Dönüş: dict(theta, tau, N, Ft)
        """
        sa, ca = np.sin(slope), np.cos(slope)

        class _C:  # düz eğimli zemin çerçevesi
            n = np.array([-sa, ca]); t = np.array([ca, sa]); kappa = 0.0; d = self.p.R
            wheel_center = np.zeros(2)

        def resid(z):
            th, tau, lam = z[0], z[1:4], z[4:6]
            q = self.config_on_contact(_C, th, q1, q2)
            _, _, G, Jw, _ = self.terms(q, np.zeros(6))
            J = np.vstack([_C.n @ Jw, _C.t @ Jw + self.p.R * self.e_omega])
            return G - self.B @ tau - J.T @ lam

        z0 = np.array([0.0, 0, 0, 0, self.m_total * self.p.g, 0])
        z, info, ier, msg = fsolve(resid, z0, full_output=True, xtol=1e-12)
        if ier != 1 or np.linalg.norm(resid(z)) > 1e-8:
            raise RuntimeError(f"trim bulunamadı (slope={slope:.3f}): {msg}")
        return dict(theta=z[0], tau=z[1:4].copy(), N=z[4], Ft=z[5])


if __name__ == "__main__":  # hızlı duman testi
    import time
    t0 = time.time()
    m = WheeledBiped()
    print(f"sympy türetme + lambdify: {time.time() - t0:.1f} s")
    q = np.array([0, 0.4, 0.05, 0.6, -1.2, 0.0]); qd = np.zeros(6)
    M, h, G, *_ = m.terms(q, qd)
    print("M simetrik:", np.allclose(M, M.T), " min özdeğer:", np.linalg.eigvalsh(M).min())
    print("trim:", m.static_trim(0.6, -1.2, 0.0))
