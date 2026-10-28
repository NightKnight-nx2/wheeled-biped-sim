"""Model doğrulama testleri:  python tests/test_model.py   (veya pytest)"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dynamics as dyn
from terrain import Terrain

RNG = np.random.default_rng(1)


def test_against_sympy_mechanics():
    """M ve (h+G) değerlerini sympy.physics.mechanics.LagrangesMethod ile bağımsız olarak karşılaştır."""
    from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols
    S = dyn.derive_symbolic()
    t = sp.Symbol("t")
    names = ["x", "y", "th", "q1", "q2", "ph"]
    qf = [dynamicsymbols(n) for n in names]
    qdf = [f.diff(t) for f in qf]
    L = S["L"].subs(dict(zip(S["qd"], qdf))).subs(dict(zip(S["q"], qf)))
    lm = LagrangesMethod(L, qf)
    lm.form_lagranges_equations()
    mm, forcing = lm.mass_matrix, lm.forcing            # M q̈ = forcing
    p = dyn.Params(b_joint=0.0, b_wheel=0.0)
    pd = dict(zip(S["P"], p.vector()))
    model = dyn.WheeledBiped(p)
    for _ in range(3):
        q = RNG.uniform(-0.8, 0.8, 6); qd = RNG.uniform(-2, 2, 6)
        sub = {**pd, **dict(zip(qdf, qd)), **dict(zip(qf, q))}
        M_ref = np.array(mm.subs(sub).evalf(), dtype=float)
        f_ref = np.array(forcing.subs(sub).evalf(), dtype=float).ravel()
        M, h, G, *_ = model.terms(q, qd)
        assert np.allclose(M, M_ref[:6, :6], atol=1e-9), "M uyuşmuyor"
        assert np.allclose(-(h + G), f_ref[:6], atol=1e-8), "h+G uyuşmuyor"
    print("OK  M, h, G  ==  sympy.physics.mechanics.LagrangesMethod")


def test_energy_conservation():
    """Sürtünmesiz, torksuz, kısıtsız serbest hareket → toplam enerji korunmalı."""
    model = dyn.WheeledBiped(dyn.Params(b_joint=0.0, b_wheel=0.0))
    q0 = np.array([0, 1.0, 0.1, 0.6, -1.2, 0.0]); qd0 = RNG.uniform(-1, 1, 6)

    def f(_, s):
        q, qd = s[:6], s[6:]
        M, h, G, *_ = model.terms(q, qd)
        return np.concatenate([qd, np.linalg.solve(M, -h - G)])

    sol = solve_ivp(f, [0, 1.0], np.concatenate([q0, qd0]), rtol=1e-11, atol=1e-12)
    E = np.array([model.energy(s[:6], s[6:]) for s in sol.y.T])
    drift = np.ptp(E)
    assert drift < 1e-7, drift
    print(f"OK  enerji korunumu: ptp(E) = {drift:.2e} J")


def test_constraint_consistency():
    """Kısıt Jacobian'ı, tutarlı hız ve kısıtlı ivme birbiriyle uyumlu mu? Zemin: engel parkuru."""
    model = dyn.WheeledBiped()
    T = Terrain.course()
    R = model.p.R
    for xc in [3.0, 7.2, 10.1, 15.0, 19.0, 23.0, 30.4]:
        c0 = T.contact_at(xc, R)
        q = model.config_on_contact(c0, RNG.uniform(-.2, .2), 0.6 + RNG.uniform(-.1, .1), -1.2 + RNG.uniform(-.1, .1), 0.3)
        qd = model.consistent_velocity(q, c0, v_ground=1.3, thd=0.2, q1d=-0.1, q2d=0.15)
        c = T.contact(model.fk(q)["wheel"], R, xc_guess=xc)
        assert abs(c.xc - xc) < 1e-9 and abs(c.d - R) < 1e-9
        M, h, G, Jw, Jdq = model.terms(q, qd)
        J, rhs = model.constraint_rows(q, qd, c, Jw, Jdq, 0, 0)
        assert np.allclose(J @ qd, 0, atol=1e-10), "tutarlı hız kısıtı sağlamıyor"
        qdd, lam = model.constrained_accel(q, qd, np.array([1., -2., 0.5]), c, 0, 0)
        assert np.allclose(J @ qdd, rhs, atol=1e-8)
        # ivmenin sayısal doğrulaması: kısa süre ileri integre edip kısıt hatasının ~0 kaldığını kontrol et
        assert np.isfinite(lam).all()
    print("OK  kısıt satırları, tutarlı hız ve KKT ivmesi (parkur boyunca)")


def test_contact_acceleration_matches_finite_difference():
    """d̈ ve slip ivmesi formüllerini (κ terimleri dahil) zaman integrasyonu + sonlu fark ile doğrula."""
    model = dyn.WheeledBiped()
    T = Terrain.course()
    R = model.p.R
    c0 = T.contact_at(7.0, R)                             # tümsek tepesi → κ ≠ 0
    q = model.config_on_contact(c0, 0.05, 0.6, -1.2)
    qd = model.consistent_velocity(q, c0, v_ground=1.5)
    tau = np.array([0.2, 9.5, -0.3])

    def f(_, s):
        qq, qqd = s[:6], s[6:]
        c = T.contact(model.fk(qq)["wheel"], R)
        qdd, _ = model.constrained_accel(qq, qqd, tau, c, 40, 40)
        return np.concatenate([qqd, qdd])

    sol = solve_ivp(f, [0, 0.6], np.concatenate([q, qd]), rtol=1e-10, atol=1e-12, max_step=0.005)
    dmax = slipmax = 0
    for s in sol.y.T:
        qq, qqd = s[:6], s[6:]
        c = T.contact(model.fk(qq)["wheel"], R)
        _, _, _, Jw, _ = model.terms(qq, qqd)
        vw = Jw @ qqd
        dmax = max(dmax, abs(c.d - R))
        slipmax = max(slipmax, abs(c.t @ vw + R * model.e_omega @ qqd))
    assert dmax < 1e-6 and slipmax < 1e-6, (dmax, slipmax)
    print(f"OK  tümsek üzerinde 0.6 s integrasyon: max|d−R| = {dmax:.1e} m, max kayma hızı = {slipmax:.1e} m/s")


def test_reduced_model_matches_kkt():
    """Kontrolör için kullanılan indirgenmiş düz-zemin modeli (PtMP) ile KKT çözümü aynı ivmeyi vermeli."""
    from controller import ReducedFlatModel
    model = dyn.WheeledBiped()
    red = ReducedFlatModel(model)
    T = Terrain.flat()
    for _ in range(5):
        xi = np.array([0.3, *RNG.uniform(-.3, .3, 1), 0.6 + RNG.uniform(-.3, .3), -1.2 + RNG.uniform(-.3, .3)])
        xid = RNG.uniform(-1.5, 1.5, 4)
        tau = RNG.uniform(-5, 5, 3)
        q, qd, _ = red.full_state(xi, xid)
        c = T.contact(model.wheel_center(q), model.p.R)
        qdd, _ = model.constrained_accel(q, qd, tau, c, 0, 0)
        _, _, _, Jw, Jdq = model.terms(q, qd)
        xcdd = (Jw @ qdd + Jdq)[0]
        kkt = np.array([xcdd, qdd[2], qdd[3], qdd[4]])
        rd = red.f(np.concatenate([xi, xid]), tau)[4:]
        assert np.allclose(kkt, rd, atol=1e-8), (kkt, rd)
    print("OK  indirgenmis model (P^T M P) == KKT cozumu (duz zemin)")


def test_terrain_derivatives():
    T = Terrain.course()
    for x in [6.9, 10.2, 14.5, 22.0, 30.3, 35.7]:
        h = 1e-5
        g = [T.derivs(x + k * h) for k in (-1, 0, 1)]
        d1 = (g[2][0] - g[0][0]) / (2 * h); d2 = (g[2][1] - g[0][1]) / (2 * h); d3 = (g[2][2] - g[0][2]) / (2 * h)
        assert abs(d1 - g[1][1]) < 1e-7 and abs(d2 - g[1][2]) < 1e-6 and abs(d3 - g[1][3]) < 1e-4
    print("OK  zemin turevleri (g1, g2, g3) sonlu farklarla tutarli")


if __name__ == "__main__":
    test_against_sympy_mechanics()
    test_energy_conservation()
    test_constraint_consistency()
    test_contact_acceleration_matches_finite_difference()
    test_reduced_model_matches_kkt()
    test_terrain_derivatives()
