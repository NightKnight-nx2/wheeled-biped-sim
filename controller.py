"""
controller.py — Gövde dikliği + ileri hız kontrolü: LQR (python-control) ve kısıtlı MPC (do-mpc).

Yöntem
------
1) Düz zemin + yuvarlanma kısıtı altında sistem 4 bağımsız koordinata iner:  ξ = [x_c, θ, q1, q2]
   (y ve φ kısıttan çıkar).  P(ξ) = ∂Φ/∂ξ  ile  q̇ = P ξ̇ ,  q̈ = P ξ̈ + Ṗ ξ̇  yazılır ve
   Pᵀ(M q̈ + h + G − Bτ) = 0  (kısıt kuvvetleri P'ye dik → elenir) ile indirgenmiş dinamik elde edilir:

        (Pᵀ M P) ξ̈ = Pᵀ (Bτ − h − G − M Ṗ ξ̇)

2) Bu 8-durumlu doğrusal olmayan sistem `control.nlsys` ile sarılır, trim (denge) noktasında
   `control.linearize` ile lineerize edilir.  x_c (konum) diğer durumları etkilemediği için atılır
   → 7 durum  z = [θ, q1, q2, ẋ_c, θ̇, q̇1, q̇2],  3 giriş  τ = [kalça, diz, tekerlek].

3) Kontrol yasası (hata koordinatlarında):

        τ = τ_ff(α) − K (z − z_ref(α, v_ref))

   α = tekerleğin altındaki yerel zemin eğimi.  Eğimli zeminde denge (KM temas noktasının düşey
   hizasında olmalı) ve gereken tutma torkları değişir; bunlar `static_trim` ile bir tablo olarak
   (θ_ref(α), τ_ff(α)) önceden hesaplanır.  K: LQR (ayrık, kontrol periyoduna uygun) veya kısıtlı MPC.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import control as ct
import numpy as np
import sympy as sp

from dynamics import WheeledBiped

# ---------------------------------------------------------------------------------------
#  İndirgenmiş düz-zemin modeli ve lineerizasyon
# ---------------------------------------------------------------------------------------


class ReducedFlatModel:
    """Düz zeminde yuvarlanan robotun 4 bağımsız koordinatlı (8 durumlu) modeli."""

    def __init__(self, model: WheeledBiped):
        self.m = model
        p = model.p
        xc, th, q1, q2 = sp.symbols("xc th q1 q2", real=True)
        xcd, thd, q1d, q2d = sp.symbols("xcd thd q1d q2d", real=True)
        xi, xid = sp.Matrix([xc, th, q1, q2]), sp.Matrix([xcd, thd, q1d, q2d])
        e = lambda a: sp.Matrix([sp.sin(a), -sp.cos(a)])
        a1 = q1 - th
        a2 = a1 + q2
        wc = sp.Matrix([xc, p.R])                                    # tekerlek merkezi (düz zemin)
        hip = wc - p.l1 * e(a1) - p.l2 * e(a2)
        phi = a2 + xc / p.R                                          # yuvarlanma: ω_w = −ẋ_c/R
        Phi = sp.Matrix([hip[0], hip[1], th, q1, q2, phi])           # ξ → q
        P = Phi.jacobian(xi)
        Pd = (P * xid).jacobian(xi) * xid                            # Ṗ ξ̇
        self._Phi = sp.lambdify(list(xi), Phi, "numpy", cse=True)
        self._P = sp.lambdify(list(xi), P, "numpy", cse=True)
        self._Pd = sp.lambdify(list(xi) + list(xid), Pd, "numpy", cse=True)

    def full_state(self, xi, xid):
        q = np.asarray(self._Phi(*xi), float).ravel()
        P = np.asarray(self._P(*xi), float)
        return q, P @ xid, P

    def f(self, x, tau):
        """ẋ = f(x, τ),  x = [ξ, ξ̇] (8)."""
        xi, xid = x[:4], x[4:]
        q, qd, P = self.full_state(xi, xid)
        Pd = np.asarray(self._Pd(*xi, *xid), float).ravel()
        M, h, G, _, _ = self.m.terms(q, qd)
        rhs = P.T @ (self.m.B @ tau - h - G - M @ Pd)
        xidd = np.linalg.solve(P.T @ M @ P, rhs)
        return np.concatenate([xid, xidd])

    def as_nlsys(self):
        return ct.nlsys(lambda t, x, u, params: self.f(x, u), None, inputs=3, states=8,
                        outputs=8, name="wheeled_biped_flat")


@dataclass
class Linearization:
    A: np.ndarray       # 7×7
    B: np.ndarray       # 7×3
    z_eq: np.ndarray
    u_eq: np.ndarray
    sys: ct.StateSpace


def linearize_flat(model: WheeledBiped, q1_ref: float, q2_ref: float) -> Linearization:
    """python-control ile trim noktasında lineerleştir, x_c durumunu at."""
    red = ReducedFlatModel(model)
    trim = model.static_trim(q1_ref, q2_ref, 0.0)
    x_eq = np.array([0.0, trim["theta"], q1_ref, q2_ref, 0, 0, 0, 0])
    u_eq = trim["tau"]
    resid = np.linalg.norm(red.f(x_eq, u_eq))
    assert resid < 1e-7, f"trim noktası denge değil: |f|={resid:.2e}"
    sys = red.as_nlsys()
    lin = ct.linearize(sys, x_eq, u_eq, eps=1e-6)
    A, B = np.asarray(lin.A), np.asarray(lin.B)
    assert np.abs(A[:, 0]).max() < 1e-6, "x_c diğer durumları etkilememeliydi"
    keep = slice(1, 8)
    A7, B7 = A[keep, keep], B[keep, :]
    return Linearization(A7, B7, x_eq[keep], u_eq, ct.ss(A7, B7, np.eye(7), np.zeros((7, 3))))


# ---------------------------------------------------------------------------------------
#  Trim tablosu (eğime bağlı denge pitch'i ve tutma torkları)
# ---------------------------------------------------------------------------------------


class TrimTable:
    def __init__(self, model: WheeledBiped, q1_ref: float, q2_ref: float, amax=0.35, n=71):
        self.alpha = np.linspace(-amax, amax, n)
        res = [model.static_trim(q1_ref, q2_ref, a) for a in self.alpha]
        self.theta = np.array([r["theta"] for r in res])
        self.tau = np.array([r["tau"] for r in res])

    def __call__(self, alpha):
        a = np.clip(alpha, self.alpha[0], self.alpha[-1])
        th = np.interp(a, self.alpha, self.theta)
        tau = np.array([np.interp(a, self.alpha, self.tau[:, i]) for i in range(3)])
        return th, tau


# ---------------------------------------------------------------------------------------
#  Ölçüm / referans
# ---------------------------------------------------------------------------------------


def measure(model: WheeledBiped, q, qd, c) -> np.ndarray:
    """z = [θ, q1, q2, ẋ_c, θ̇, q̇1, q̇2]  (ẋ_c: temas noktasının yatay hızı)."""
    _, _, _, Jw, _ = model.terms(q, qd)
    vw = Jw @ qd
    sdot = (c.t @ vw) / max(1.0 - c.d * c.kappa, 0.2)
    return np.array([q[2], q[3], q[4], sdot * c.t[0], qd[2], qd[3], qd[4]])


def smooth_step(t, t0, T):
    s = np.clip((t - t0) / T, 0.0, 1.0)
    return s * s * (3 - 2 * s)


class SpeedProfile:
    """v_ref(t): t0'da başlayıp T sürede v_cmd'ye çıkan yumuşak (smoothstep) hız komutu."""

    def __init__(self, v_cmd=1.0, t0=1.0, T=1.5, v_start=0.0):
        self.v_cmd, self.t0, self.T, self.v_start = v_cmd, t0, T, v_start

    def __call__(self, t):
        return self.v_start + (self.v_cmd - self.v_start) * smooth_step(t, self.t0, self.T)


# ---------------------------------------------------------------------------------------
#  Ortak taban sınıf
# ---------------------------------------------------------------------------------------


class BaseController:
    name = "base"

    def __init__(self, model: WheeledBiped, q_ref=(0.6, -1.2), v_ref=None, dt=0.01,
                 Q=None, R=None):
        self.model = model
        self.dt = dt
        self.q1_ref, self.q2_ref = q_ref
        self.v_ref = v_ref if v_ref is not None else SpeedProfile(0.0)
        self.tau_max = model.p.tau_max
        self.lin = linearize_flat(model, *q_ref)
        self.trim = TrimTable(model, *q_ref)
        self.Q = np.diag([300., 30., 30., 60., 8., 1., 1.]) if Q is None else Q
        self.R = np.diag([0.02, 0.02, 0.2]) if R is None else R
        # ZOH ayrıklaştırma
        sd = ct.c2d(self.lin.sys, dt, method="zoh")
        self.Ad, self.Bd = np.asarray(sd.A), np.asarray(sd.B)
        self.Qd, self.Rd = self.Q * dt, self.R * dt
        self.last_solve_time = 0.0
        # ayrık LQR çözümü (LQR kontrolörü K'yı doğrudan kullanır; MPC terminal maliyet S ve yedek K olarak)
        K, S, _ = ct.dlqr(self.Ad, self.Bd, self.Qd, self.Rd)
        self.K, self.S = np.asarray(K), np.asarray(S)

    # hata ve referans
    def reference(self, t, c):
        th_ref, tau_ff = self.trim(c.slope)
        z_ref = np.array([th_ref, self.q1_ref, self.q2_ref, self.v_ref(t), 0.0, 0.0, 0.0])
        return z_ref, tau_ff

    def __call__(self, t, q, qd, c) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------------------
#  LQR
# ---------------------------------------------------------------------------------------


class LQRController(BaseController):
    """Ayrık-zamanlı LQR (kontrol periyoduna göre ayrıklaştırılmış lineer model, ct.dlqr)."""
    name = "LQR"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        ctrb_rank = np.linalg.matrix_rank(ct.ctrb(self.lin.A, self.lin.B))
        assert ctrb_rank == 7, f"kontrol edilebilir değil (rank={ctrb_rank})"
        self.open_loop_poles = np.linalg.eigvals(self.lin.A)
        self.closed_loop_poles = np.linalg.eigvals(self.Ad - self.Bd @ self.K)

    def __call__(self, t, q, qd, c):
        t0 = time.perf_counter()
        z = measure(self.model, q, qd, c)
        z_ref, tau_ff = self.reference(t, c)
        tau = tau_ff - self.K @ (z - z_ref)
        self.last_solve_time = time.perf_counter() - t0
        return np.clip(tau, -self.tau_max, self.tau_max)


# ---------------------------------------------------------------------------------------
#  do-mpc ile kısıtlı MPC
# ---------------------------------------------------------------------------------------


class MPCController(BaseController):
    """Doğrusal model + kısıtlar üzerinde MPC (do-mpc, IPOPT).

    Hata koordinatında  e = z − z_ref ,  δu = τ − τ_ff:   e⁺ = A_d e + B_d δu
    Maliyet : Σ eᵀQe + δuᵀRδu  + terminal eᵀ S e  (S: ayrık Riccati → kısıtsız MPC ≡ LQR)
    Kısıtlar: |τ_ff + δu| ≤ τ_max (motor sınırı, τ_ff eğime göre değişir → zamanla değişen parametre),
              eklem açısı sınırları (kalça/diz), pitch hatası sınırı |θ−θ_ref| ≤ pitch_limit (yumuşak).
    Çözüm başarısız olursa LQR geri-beslemesine düşülür (`n_fail` sayar).
    """
    name = "MPC"

    def __init__(self, *a, horizon=25, joint_dev=0.6, pitch_limit=0.35, pitch_hard=0.9, tau_scale=1.0, **kw):
        super().__init__(*a, **kw)
        import casadi as cs
        try:                                   # casadi 3.7+: eski numpy davranışını sessizce koru
            cs.GlobalOptions.setNumpyMode(-1)
        except AttributeError:
            pass
        import do_mpc

        self.N = horizon
        self.pitch_limit = pitch_limit
        model = do_mpc.model.Model("discrete")
        e = model.set_variable("_x", "e", shape=(7, 1))
        du = model.set_variable("_u", "du", shape=(3, 1))
        uff = model.set_variable("_tvp", "uff", shape=(3, 1))
        model.set_rhs("e", cs.DM(self.Ad) @ e + cs.DM(self.Bd) @ du)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.settings.n_horizon = horizon
        mpc.settings.t_step = self.dt
        mpc.settings.n_robust = 0
        mpc.settings.store_full_solution = False
        mpc.settings.nlpsol_opts = {"ipopt.max_iter": 100, "ipopt.tol": 1e-8, "ipopt.print_level": 0,
                                    "ipopt.sb": "yes", "print_time": 0}
        Qd, Rd, S = cs.DM(self.Qd), cs.DM(self.Rd), cs.DM(self.S)
        mpc.set_objective(mterm=e.T @ S @ e, lterm=e.T @ Qd @ e + du.T @ Rd @ du)
        mpc.set_rterm(du=1e-6 * np.ones((3, 1)))
        # motor limitleri: τ_ff + δu ∈ [−τ_max, τ_max]  (τ_ff zamanla değişen parametre)
        tm = cs.DM(self.tau_max * tau_scale)
        mpc.set_nl_cons("tau_ub", uff + du - tm, ub=0.0, soft_constraint=False)
        mpc.set_nl_cons("tau_lb", -tm - (uff + du), ub=0.0, soft_constraint=False)
        # eklem ve pitch sınırları (hata koordinatında, referans etrafında)
        if pitch_limit is not None:      # pitch hatası sınırı: yumuşak kısıt (çözümsüzlüğü önler)
            mpc.set_nl_cons("pitch_ub", e[0] - pitch_limit, ub=0.0, soft_constraint=True, penalty_term_cons=1e5)
            mpc.set_nl_cons("pitch_lb", -pitch_limit - e[0], ub=0.0, soft_constraint=True, penalty_term_cons=1e5)
        lo = np.array([-pitch_hard, -joint_dev, -joint_dev, -50., -50., -50., -50.])
        hi = -lo
        mpc.bounds["lower", "_x", "e"] = lo
        mpc.bounds["upper", "_x", "e"] = hi
        self._uff = np.zeros(3)
        tvp = mpc.get_tvp_template()

        def tvp_fun(t_now):
            tvp["_tvp", :, "uff"] = self._uff
            return tvp

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        mpc.x0 = np.zeros((7, 1))
        mpc.u0 = np.zeros((3, 1))
        mpc.set_initial_guess()
        self.mpc = mpc
        self.n_fail = 0

    def __call__(self, t, q, qd, c):
        z = measure(self.model, q, qd, c)
        z_ref, tau_ff = self.reference(t, c)
        e0 = (z - z_ref).reshape(7, 1)
        self._uff = tau_ff
        t0 = time.perf_counter()
        try:
            du = np.asarray(self.mpc.make_step(e0)).ravel()
            ok = bool(self.mpc.data["success"][-1]) and np.all(np.isfinite(du))
        except Exception:
            ok = False
        if not ok:                       # çözüm başarısızsa LQR geri-besleme yedeği
            self.n_fail += 1
            du = -self.K @ e0.ravel()
        self.last_solve_time = time.perf_counter() - t0
        return np.clip(tau_ff + du, -self.tau_max, self.tau_max)


if __name__ == "__main__":
    m = WheeledBiped()
    lqr = LQRController(m)
    print("açık çevrim kutupları :", np.round(lqr.open_loop_poles, 2))
    print("kapalı çevrim (ayrık) |λ|:", np.round(np.abs(lqr.closed_loop_poles), 3))
    print("K =\n", np.round(lqr.K, 2))
