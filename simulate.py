"""
simulate.py — Kapalı çevrim simülasyon: solve_ivp (RK45) + sıfırıncı derece tutucu (ZOH) kontrol döngüsü.

Her kontrol periyodunda (dt):  ölçüm → kontrolör (LQR / MPC) → τ sabit tutulur → solve_ivp(RK45) ile
[t, t+dt] aralığı çözülür.  İntegre edilen durum 12 boyutludur:  s = [q (6), q̇ (6)];  tekerlek–zemin
kısıtı her RHS çağrısında KKT sistemi (dynamics.constrained_accel) ile uygulanır; normal kuvvet N ve
çekiş kuvveti F_t Lagrange çarpanlarından gelir.

Kullanım:
    python simulate.py --scenario flat   --controller lqr            # düz zeminde denge
    python simulate.py --scenario course --controller both --save out.npz
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from controller import LQRController, MPCController, SpeedProfile, measure
from dynamics import Params, WheeledBiped
from terrain import Terrain


@dataclass
class SimResult:
    t: np.ndarray
    q: np.ndarray            # (n,6)
    qd: np.ndarray           # (n,6)
    tau: np.ndarray          # (n,3)  uygulanan (doymuş) torklar
    N: np.ndarray            # normal kuvvet [N]
    Ft: np.ndarray           # teğet kuvvet [N]
    xc: np.ndarray           # temas noktası x
    slope: np.ndarray        # yerel eğim [rad]
    v: np.ndarray            # ölçülen ileri hız ẋ_c
    v_ref: np.ndarray
    theta_ref: np.ndarray
    d_err: np.ndarray        # |d − R|  (konum kısıtı hatası)
    slip: np.ndarray         # kayma hızı
    solve_time: np.ndarray   # kontrolör hesap süresi [s]
    controller: str = ""
    terrain: str = ""
    status: str = "ok"
    meta: dict = field(default_factory=dict)

    def metrics(self, t_from: float = 0.0) -> dict:
        m = self.t >= t_from
        return dict(
            controller=self.controller, status=self.status,
            rms_speed_err=float(np.sqrt(np.mean((self.v[m] - self.v_ref[m]) ** 2))),
            rms_pitch_err=float(np.sqrt(np.mean((self.q[m, 2] - self.theta_ref[m]) ** 2))),
            max_pitch_err_deg=float(np.degrees(np.max(np.abs(self.q[m, 2] - self.theta_ref[m])))),
            rms_tau_wheel=float(np.sqrt(np.mean(self.tau[m, 2] ** 2))),
            max_abs_tau=[float(x) for x in np.max(np.abs(self.tau[m]), axis=0)],
            N_min=float(self.N[m].min()), N_max=float(self.N[m].max()),
            max_friction_ratio=float(np.max(np.abs(self.Ft[m]) / np.maximum(self.N[m], 1e-6))),
            max_constraint_err=float(self.d_err.max()), max_slip=float(np.abs(self.slip).max()),
            mean_ctrl_ms=float(1e3 * self.solve_time.mean()), max_ctrl_ms=float(1e3 * self.solve_time.max()),
        )

    def save(self, path: str):
        d = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path, **d, controller=self.controller, terrain=self.terrain,
                            status=self.status, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: str) -> "SimResult":
        z = np.load(path, allow_pickle=False)
        arrs = {k: z[k] for k in z.files if k not in ("controller", "terrain", "status", "meta")}
        return cls(**arrs, controller=str(z["controller"]), terrain=str(z["terrain"]),
                   status=str(z["status"]), meta=json.loads(str(z["meta"])))


def initial_state(model: WheeledBiped, terrain: Terrain, xc0=0.0, theta0=0.0, q_ref=(0.6, -1.2), v0=0.0):
    c = terrain.contact_at(xc0, model.p.R)
    q = model.config_on_contact(c, theta0, *q_ref)
    qd = model.consistent_velocity(q, c, v0 / c.t[0])
    return q, qd


def simulate(model: WheeledBiped, terrain: Terrain, controller, t_end: float, q0, qd0,
             x_end: float | None = None, kicks: list[tuple[float, float]] | None = None,
             rtol=1e-7, atol=1e-9, verbose=True) -> SimResult:
    """Kapalı çevrim simülasyon.  kicks: [(t, Δθ̇)] — gövdeye anlık itme (pitch hız darbesi)."""
    R, dt = model.p.R, controller.dt
    n = int(round(t_end / dt))
    kicks = sorted(kicks or [])
    q, qd = q0.copy(), qd0.copy()
    logs = {k: [] for k in ["t", "q", "qd", "tau", "N", "Ft", "xc", "slope", "v", "v_ref", "theta_ref",
                            "d_err", "slip", "solve_time"]}
    guess = [terrain.contact(model.wheel_center(q), R).xc]
    status = "ok"
    t0_wall = time.perf_counter()
    t = 0.0

    for k in range(n):
        c = terrain.contact(model.wheel_center(q), R, guess[0])
        guess[0] = c.xc
        tau = np.asarray(controller(t, q, qd, c), float)
        z_ref, _ = controller.reference(t, c)
        qdd, lam = model.constrained_accel(q, qd, tau, c)
        _, _, _, Jw, _ = model.terms(q, qd)
        slip = c.t @ (Jw @ qd) + R * (model.e_omega @ qd)
        for key, val in zip(logs, [t, q, qd, tau, lam[0], lam[1], c.xc, c.slope, measure(model, q, qd, c)[3],
                                   z_ref[3], z_ref[0], abs(c.d - R), slip, controller.last_solve_time]):
            logs[key].append(np.copy(val))

        # --- ZOH: tau sabit, [t, t+dt] RK45 ---
        def rhs(_, s):
            cc = terrain.contact(model.wheel_center(s[:6]), R, guess[0])
            a, _ = model.constrained_accel(s[:6], s[6:], tau, cc)
            return np.concatenate([s[6:], a])

        sol = solve_ivp(rhs, [t, t + dt], np.concatenate([q, qd]), method="RK45", rtol=rtol, atol=atol)
        if not sol.success:
            status = f"integrasyon hatası t={t:.2f}"
            break
        q, qd = sol.y[:6, -1], sol.y[6:, -1]
        t += dt

        for kt, dth in [kk for kk in kicks if abs(kk[0] - t) < dt / 2]:   # itme: θ̇ += Δ (tutarlı hızla)
            c2 = terrain.contact(model.wheel_center(q), R, guess[0])
            ds = measure(model, q, qd, c2)[3] / c2.t[0]
            qd = model.consistent_velocity(q, c2, ds, qd[2] + dth, qd[3], qd[4])

        if abs(q[2] - z_ref[0]) > 1.0:
            status = f"devrildi (t={t:.2f} s, θ={np.degrees(q[2]):.0f}°)"
            break
        if lam[0] < 0.0:
            status = f"temas kaybı (N<0, t={t - dt:.2f} s)"
            if verbose:
                print("  uyarı:", status)
        if x_end is not None and guess[0] > x_end:
            status = status if status != "ok" else "ok"
            break

    arr = {k: np.array(v) for k, v in logs.items()}
    if verbose:
        print(f"  [{controller.name}] {len(arr['t'])} adım, {time.perf_counter() - t0_wall:.1f} s duvar süresi, durum: {status}")
    return SimResult(**arr, controller=controller.name, terrain=terrain.name, status=status,
                     meta=dict(dt=dt))


# ---------------------------------------------------------------------------------------
#  Senaryolar
# ---------------------------------------------------------------------------------------

def scenario_flat(model, ctrl_cls, dt=0.01, **ckw):
    """Düz zemin: 0.15 rad ilk pitch hatası + t=2 s'de 0.8 m/s hız komutu + t=6 s'de itme."""
    terrain = Terrain.flat()
    vref = SpeedProfile(v_cmd=0.8, t0=2.0, T=1.5)
    ctrl = ctrl_cls(model, v_ref=vref, dt=dt, **ckw)
    q0, qd0 = initial_state(model, terrain, 0.0, theta0=ctrl.trim(0.0)[0] + 0.15)
    res = simulate(model, terrain, ctrl, 9.0, q0, qd0, kicks=[(6.0, 1.0)])
    return terrain, ctrl, res


def scenario_course(model, ctrl_cls, dt=0.01, v_cmd=1.5, **ckw):
    terrain = Terrain.course()
    vref = SpeedProfile(v_cmd=v_cmd, t0=0.5, T=2.0)
    ctrl = ctrl_cls(model, v_ref=vref, dt=dt, **ckw)
    q0, qd0 = initial_state(model, terrain, 0.0, theta0=ctrl.trim(0.0)[0])
    res = simulate(model, terrain, ctrl, 45.0 / v_cmd + 4.0, q0, qd0, x_end=44.0)
    return terrain, ctrl, res


def scenario_step(model, ctrl_cls, dt=0.01, v_cmd=2.5, **ckw):
    """Düz zeminde t=1 s'de ani hız komutu (0 → v_cmd).  Kısıt etkisini göstermek için kullanılır."""
    terrain = Terrain.flat()
    ctrl = ctrl_cls(model, v_ref=SpeedProfile(v_cmd=v_cmd, t0=1.0, T=0.05), dt=dt, **ckw)
    q0, qd0 = initial_state(model, terrain, 0.0, theta0=ctrl.trim(0.0)[0])
    res = simulate(model, terrain, ctrl, 6.0, q0, qd0)
    return terrain, ctrl, res


def print_metrics(results: list[SimResult], t_from=3.0):
    keys = ["status", "rms_speed_err", "rms_pitch_err", "max_pitch_err_deg", "rms_tau_wheel", "max_abs_tau",
            "N_min", "N_max", "max_friction_ratio", "max_constraint_err", "max_slip", "mean_ctrl_ms", "max_ctrl_ms"]
    for r in results:
        m = r.metrics(t_from)
        print(f"--- {m['controller']} ---")
        for k in keys:
            v = m[k]
            print(f"  {k:20s}: {np.round(v, 4) if not isinstance(v, str) else v}")


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")   # Windows konsolunda Türkçe karakterler için
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["flat", "course", "step"], default="flat")
    ap.add_argument("--controller", choices=["lqr", "mpc", "both"], default="lqr")
    ap.add_argument("--dt", type=float, default=0.01, help="kontrol periyodu [s]")
    ap.add_argument("--vcmd", type=float, default=1.5)
    ap.add_argument("--pitch_limit", type=float, default=0.35, help="MPC pitch hatası sınırı [rad] (step senaryosu)")
    ap.add_argument("--save", type=str, default=None, help="sonucu <prefix>_<ctrl>.npz olarak kaydet")
    a = ap.parse_args()

    model = WheeledBiped(Params())
    classes = {"lqr": [LQRController], "mpc": [MPCController], "both": [LQRController, MPCController]}[a.controller]
    results = []
    for cls in classes:
        if a.scenario == "flat":
            terrain, ctrl, res = scenario_flat(model, cls, a.dt)
        elif a.scenario == "step":
            kw = dict(pitch_limit=a.pitch_limit) if cls is MPCController else {}
            terrain, ctrl, res = scenario_step(model, cls, a.dt, a.vcmd, **kw)
        else:
            terrain, ctrl, res = scenario_course(model, cls, a.dt, a.vcmd)
        results.append(res)
        if a.save:
            res.save(f"{a.save}_{cls.name.lower()}.npz")
    print_metrics(results, t_from=3.0 if a.scenario == "course" else 0.0)


if __name__ == "__main__":
    main()
