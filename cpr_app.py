# cpr_app.py
# Streamlit UI for CPR dynamic with two tabs:
#   Shared parameters: n, m
#
#   Tab 1: Single specification (full w vector)
#          - Full-width drift plot
#          - Black zero line
#          - Direction arrows (phase-line style)
#          - Equilibria marked on the zero line:
#              * stable = solid red dot
#              * unstable = hollow red dot
#          - Markers are drawn in front of arrows (zorder)
#
#   Tab 2: Parameterized comparison with affine weights:
#          w(ω) = intercept + ω*slope
#          Default ω-range is automatic:
#             ω_min = 0
#             ω_max = min(1, first ω where any component would turn negative)
#          Optional manual restriction of ω-domain via checkbox:
#             user can set ω_min and ω_max, both capped into [0, ω_max_auto]
#
#          Plot of equilibrium branches vs ω (continuation):
#             stable solid, unstable dashed
#             continuation includes Newton fallback + jump filter + reseed each ω step
#
# Run:
#   streamlit run cpr_app.py

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import brentq


# ===================== Core model =====================

def q_zr(n, m, z, r, x):
    outsiders = n - z
    k = m - r
    if k <= 0:
        return 1.0
    if k > outsiders:
        return 0.0
    return float(binom.sf(k - 1, outsiders, x))


def P_zr_all(n, m, z, x):
    q = np.array([q_zr(n, m, z, r, x) for r in range(z + 1)])
    one_minus_q = 1.0 - q

    pref = np.ones(z + 1)
    for r in range(1, z + 1):
        pref[r] = pref[r - 1] * one_minus_q[r - 1]

    P = np.zeros(z + 1)
    P[0] = q[0] + np.prod(one_minus_q)
    for r in range(1, z + 1):
        P[r] = q[r] * pref[r]

    return P / P.sum()


def Delta_z(n, m, z, x):
    P = P_zr_all(n, m, z, x)
    r = np.arange(z + 1)
    return (r @ P) / z - x


def Delta_w(n, m, w, x):
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return sum(w[z - 1] * Delta_z(n, m, z, x) for z in range(1, n + 1))


# ===================== Equilibria =====================

def find_equilibria(n, m, w, grid=4001, endpoint_tol=1e-6):
    xs = np.linspace(0.0, 1.0, grid)
    fs = np.array([Delta_w(n, m, w, x) for x in xs])

    roots = []

    for i in range(len(xs) - 1):
        if fs[i] == 0 or fs[i] * fs[i + 1] < 0:
            try:
                r = brentq(lambda x: Delta_w(n, m, w, x), xs[i], xs[i + 1])
                roots.append(float(r))
            except ValueError:
                pass

    if abs(float(fs[0])) <= endpoint_tol:
        roots.append(0.0)
    if abs(float(fs[-1])) <= endpoint_tol:
        roots.append(1.0)

    return sorted(set(round(r, 6) for r in roots))


def stability_at_root(n, m, w, x, h=1e-5):
    f1 = Delta_w(n, m, w, max(0.0, x - h))
    f2 = Delta_w(n, m, w, min(1.0, x + h))
    d = (f2 - f1) / (2 * h)
    return float(d), ("stable" if d < 0 else "unstable")


# ===================== Helpers for Tab 2 ω-range =====================

def omega_max_from_affine(intercept, slope, cap=1.0, eps=1e-12):
    intercept = np.asarray(intercept, dtype=float)
    slope = np.asarray(slope, dtype=float)

    if np.any(intercept < -eps):
        return 0.0

    candidates = [cap]
    for a, b in zip(intercept, slope):
        if b < -eps:
            if a <= eps:
                candidates.append(0.0)
            else:
                candidates.append(a / (-b))

    wmax = float(min(candidates))
    return max(0.0, wmax)


# ===================== Plot: drift with arrows + equilibrium markers =====================

def plot_drift(n, m, w, roots, xmin=0.0, xmax=1.0):
    xs = np.linspace(xmin, xmax, 2001)
    ys = np.array([Delta_w(n, m, w, x) for x in xs])

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(xs, ys, linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1.8)

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Delta_w(x)$")
    ax.set_title(f"CPR drift (n={n}, m={m})")
    ax.set_xlim(xmin, xmax)

    # Direction arrows
    k = 20
    span = (xmax - xmin)
    if span <= 0:
        span = 1.0

    xA = np.linspace(xmin + 0.04 * span, xmax - 0.04 * span, k)
    yA = np.array([Delta_w(n, m, w, x) for x in xA])
    sgn = np.sign(yA)

    eps = 1e-6
    dx = 0.045 * span

    for x, d, val in zip(xA, sgn, yA):
        if abs(val) > eps:
            ax.annotate(
                "",
                xy=(x + d * dx, 0.0),
                xytext=(x, 0.0),
                arrowprops=dict(arrowstyle="->", lw=2.2, mutation_scale=22),
                clip_on=True,
                zorder=2
            )

    # Equilibria markers (red; above arrows)
    for r in roots:
        if r < xmin - 1e-12 or r > xmax + 1e-12:
            continue
        _, cls = stability_at_root(n, m, w, r)

        if cls == "stable":
            ax.plot(
                r, 0.0,
                marker="o",
                markersize=10,
                markerfacecolor="red",
                markeredgecolor="red",
                linestyle="None",
                zorder=5
            )
        else:
            ax.plot(
                r, 0.0,
                marker="o",
                markersize=10,
                markerfacecolor="white",
                markeredgecolor="red",
                markeredgewidth=1.8,
                linestyle="None",
                zorder=5
            )

    return fig


# ===================== Tab 2: affine weights and continuation =====================

def affine_weights(intercept, slope, omega):
    w = intercept + omega * slope
    w = np.maximum(w, 0.0)  # safety; ω-range chosen to avoid clipping
    s = float(w.sum())
    if s <= 0:
        raise ValueError("Weights sum to 0 after clipping; adjust intercept/slope or ω-range.")
    return w / s


def _track_one_root(f, x_prev, max_expand=0.35):
    deltas = [0.02, 0.05, 0.10, 0.20, max_expand]
    for d in deltas:
        a = max(0.0, x_prev - d)
        b = min(1.0, x_prev + d)
        fa, fb = f(a), f(b)
        if np.isnan(fa) or np.isnan(fb):
            continue
        if fa == 0.0:
            return a, True
        if fb == 0.0:
            return b, True
        if fa * fb < 0:
            try:
                r = brentq(f, a, b)
                return float(r), True
            except ValueError:
                continue
    return np.nan, False


def _newton_refine(f, x0, max_iter=12, tol=1e-10, h=1e-6):
    x = float(np.clip(x0, 0.0, 1.0))
    for _ in range(max_iter):
        fx = f(x)
        if not np.isfinite(fx):
            return np.nan, False
        if abs(fx) < tol:
            return x, True

        xh1 = min(1.0, x + h)
        xh0 = max(0.0, x - h)
        denom = (xh1 - xh0) if (xh1 != xh0) else (2 * h)
        d = (f(xh1) - f(xh0)) / denom

        if (not np.isfinite(d)) or abs(d) < 1e-12:
            return np.nan, False

        x_new = x - fx / d
        if not np.isfinite(x_new):
            return np.nan, False

        x_new = float(np.clip(x_new, 0.0, 1.0))
        if abs(x_new - x) < 1e-10:
            return x_new, True
        x = x_new

    return np.nan, False


JUMP_TOL = 0.03


@st.cache_data(show_spinner=False)
def branches_over_omega(n, m, intercept, slope, w_min, w_max, n_w,
                       seed_every=1, scan_grid=1201, tol_match=0.015):
    omegas = np.linspace(float(w_min), float(w_max), int(n_w))

    def w_of(omega):
        return affine_weights(intercept, slope, omega)

    w0 = w_of(omegas[0])
    roots0 = find_equilibria(n, m, w0, grid=scan_grid)

    branches = []
    for r in roots0:
        _, cls = stability_at_root(n, m, w0, r)
        branches.append({"x": [r], "stab": [cls]})

    for j in range(1, len(omegas)):
        omega = float(omegas[j])
        wj = w_of(omega)

        def f(x):
            return Delta_w(n, m, wj, x)

        tracked_roots = []

        for br in branches:
            x_prev = br["x"][-1]
            if np.isnan(x_prev):
                br["x"].append(np.nan)
                br["stab"].append("unstable")
                continue

            x_new, ok = _track_one_root(f, x_prev)
            if not ok:
                x_new, ok = _newton_refine(f, x_prev)

            if ok and abs(x_new - x_prev) <= JUMP_TOL:
                _, cls = stability_at_root(n, m, wj, x_new)
                br["x"].append(x_new)
                br["stab"].append(cls)
                tracked_roots.append(x_new)
            else:
                br["x"].append(np.nan)
                br["stab"].append("unstable")

        if (j % seed_every) == 0:
            roots_scan = find_equilibria(n, m, wj, grid=scan_grid)
            for r in roots_scan:
                if any((not np.isnan(t)) and abs(r - t) <= tol_match for t in tracked_roots):
                    continue
                _, cls = stability_at_root(n, m, wj, r)
                new_br = {"x": [np.nan] * j + [r], "stab": ["unstable"] * j + [cls]}
                branches.append(new_br)
                tracked_roots.append(r)

        for br in branches:
            if len(br["x"]) < j + 1:
                br["x"].append(np.nan)
                br["stab"].append("unstable")

    for br in branches:
        br["x"] = np.array(br["x"], dtype=float)

    return omegas, branches


def plot_branches(omegas, branches):
    fig, ax = plt.subplots(figsize=(9, 5))

    for br in branches:
        x = br["x"]
        stab = br["stab"]

        start = 0
        while start < len(omegas) - 1:
            while start < len(omegas) and np.isnan(x[start]):
                start += 1
            if start >= len(omegas) - 1:
                break

            cls = stab[start]
            ls = "-" if cls == "stable" else "--"

            end = start + 1
            while end < len(omegas) and (not np.isnan(x[end])) and (stab[end] == cls):
                end += 1

            ax.plot(omegas[start:end], x[start:end], linestyle=ls, linewidth=2)
            start = end

    ax.set_xlabel("ω")
    ax.set_ylabel("x*")
    ax.set_title("Equilibrium branches vs ω (solid=stable, dashed=unstable)")
    ax.set_xlim(float(omegas[0]), float(omegas[-1]))
    ax.set_ylim(0, 1)
    return fig


# ===================== Streamlit UI =====================

st.set_page_config(layout="wide")
st.title("Coalitional Procedural Rationality dynamic")

with st.sidebar:
    st.header("Core parameters")
    n = st.number_input("n (number of players)", 1, 200, 4, step=1)
    m = st.number_input("m (threshold)", 0, int(n), min(3, int(n)), step=1)

    st.divider()

    st.header("Tab 1: Single specification")
    st.caption("Enter n weights (one per coalition size). They will be renormalized to sum to 1.")
    w_str = st.text_input("w₁ … wₙ", "0.75 0.25 0 0")

    st.subheader("Restrict x-domain (optional)")
    use_x = st.checkbox("Restrict x-domain", False)
    if use_x:
        xmin = st.number_input("x min", 0.0, 1.0, 0.0, format="%.3f")
        xmax = st.number_input("x max", 0.0, 1.0, 1.0, format="%.3f")
    else:
        xmin, xmax = 0.0, 1.0

    st.divider()

    st.header("Tab 2: Parameterized comparison")
    st.caption("Affine family w(ω) = intercept + ω*slope, with multiplication componentwise. \n Negative w(ω) are not permitted. Positive w(ω) are normalized to sum to 1.")
    intercept_str = st.text_input("Intercept vector (length n)", "1 0 0 0")
    slope_str = st.text_input("Slope vector (length n)", "-2 1 1 0")

    restrict_omega = st.checkbox("Restrict ω-domain (manual)", False)
    if restrict_omega:
        omega_min_user = st.number_input("Manual ω min (≥ 0)", 0.0, 1.0, 0.0, format="%.6f")
        omega_max_user = st.number_input("Manual ω max (≤ automatic ω max)", 0.0, 1.0, 1.0, format="%.6f")
    else:
        omega_min_user = 0.0
        omega_max_user = 1.0  # ignored unless restrict_omega=True

    n_w = st.number_input("Number of ω points used for plot", 5, 301, 100, step=1)


tab1, tab2 = st.tabs(
    ["Single specification", "Parameterized comparison, w = intercept vector + ω * slope vector"]
)


# ---------- Tab 1 ----------
with tab1:
    st.subheader("Drift and equilibria")

    ok1 = True
    try:
        w = np.array([float(x) for x in w_str.split()], dtype=float)
        if len(w) != int(n):
            st.error(f"Need exactly {int(n)} weights.")
            ok1 = False
        elif np.any(w < 0) or w.sum() <= 0:
            st.error("Weights must be nonnegative and sum to > 0.")
            ok1 = False
        else:
            w = w / w.sum()
    except Exception as e:
        st.error(f"Could not parse w: {e}")
        ok1 = False

    if ok1:
        roots = find_equilibria(int(n), int(m), w, grid=4001)
        st.pyplot(plot_drift(int(n), int(m), w, roots, xmin, xmax), use_container_width=True)

        st.subheader("Equilibria (table)")
        if roots:
            rows = []
            for r in roots:
                d, s = stability_at_root(int(n), int(m), w, r)
                rows.append({"x*": r, "Δ'(x*)": d, "type": s})
            st.dataframe(rows, use_container_width=True)
        else:
            st.write("No equilibria found.")
            st.caption(
                f"Diagnostics: Δ_w(0)={Delta_w(int(n), int(m), w, 0.0):.6g}, "
                f"Δ_w(1)={Delta_w(int(n), int(m), w, 1.0):.6g}"
            )
    else:
        st.info("Fix inputs in the sidebar to display results.")


# ---------- Tab 2 ----------
with tab2:
    st.subheader("Parameterized comparison: equilibrium branches vs ω")

    ok2 = True

    try:
        intercept = np.array([float(x) for x in intercept_str.split()], dtype=float)
        slope = np.array([float(x) for x in slope_str.split()], dtype=float)
        if len(intercept) != int(n) or len(slope) != int(n):
            st.error(f"intercept and slope must both have length {int(n)}.")
            ok2 = False
    except Exception as e:
        st.error(f"Could not parse intercept/slope: {e}")
        ok2 = False

    if ok2:
        w_max_auto = omega_max_from_affine(intercept, slope, cap=1.0)
        if w_max_auto <= 0:
            st.error("Automatic ω_max is 0. Your affine family becomes negative immediately (or intercept has negatives).")
            ok2 = False
        else:
            if restrict_omega:
                w_min = float(max(0.0, omega_min_user))
                w_max = float(min(w_max_auto, omega_max_user))
                if w_max <= w_min:
                    st.error("Manual ω max must be strictly greater than manual ω min.")
                    ok2 = False
                else:
                    st.info(
                        f"Using ω range: [{w_min:.6g}, {w_max:.6g}] "
                        f"(manual, capped by automatic ω_max={w_max_auto:.6g})"
                    )
            else:
                w_min = 0.0
                w_max = float(w_max_auto)
                st.info(f"Using ω range: [0, {w_max:.6g}] (automatic)")

    if ok2:
        try:
            _ = affine_weights(intercept, slope, float(w_min))
            _ = affine_weights(intercept, slope, float(w_max))
        except Exception as e:
            st.error(f"Invalid affine family over ω-range: {e}")
            ok2 = False

    if ok2:
        with st.spinner("Tracking equilibrium branches across ω…"):
            omegas, branches = branches_over_omega(
                int(n), int(m),
                intercept, slope,
                float(w_min), float(w_max),
                int(n_w),
                seed_every=1,
                scan_grid=1201,
                tol_match=0.015
            )
        st.pyplot(plot_branches(omegas, branches), use_container_width=True)
    else:
        st.info("Fix parameters in the sidebar to see results.")
