# cpr_app.py
# Streamlit UI for CPR / CBR dynamic with two toggles:
#   Toggle 1: CPR vs CBR
#   Toggle 2: Single specification vs Parameterized comparison
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


def Delta_z_cpr(n, m, z, x):
    P = P_zr_all(n, m, z, x)
    r = np.arange(z + 1)
    return (r @ P) / z - x


def Delta_w_cpr(n, m, w, x):
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return sum(w[z - 1] * Delta_z_cpr(n, m, z, x) for z in range(1, n + 1))


# ===================== Coalitional best response (CBR) =====================

def r_star_cbr(n, m, z, x, b_over_c):
    """
    Coalitional best response for coalition size z.
    Objective (scaled by c=1): μ_{z,r}(x) = z*(b/c)*q_{z,r}(x) - r.

    Tie-break: smallest r among maximizers.
    """
    rs = np.arange(z + 1, dtype=int)
    qs = np.array([q_zr(n, m, z, int(r), x) for r in rs], dtype=float)
    mu = z * float(b_over_c) * qs - rs.astype(float)

    max_mu = np.max(mu)
    r_candidates = rs[np.isclose(mu, max_mu, rtol=0.0, atol=1e-12)]
    return int(r_candidates.min())


def Delta_z_cbr(n, m, z, x, b_over_c):
    r_star = r_star_cbr(n, m, z, x, b_over_c)
    return (r_star / z) - x


def Delta_w_cbr(n, m, w, x, b_over_c):
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return sum(w[z - 1] * Delta_z_cbr(n, m, z, x, b_over_c) for z in range(1, n + 1))


# ===================== Generic equilibria + stability =====================

def find_equilibria_from_delta(delta_func, grid=2001, endpoint_tol=1e-6):
    xs = np.linspace(0.0, 1.0, grid)
    fs = np.array([float(delta_func(x)) for x in xs], dtype=float)

    roots = []

    for i in range(len(xs) - 1):
        if fs[i] == 0 or fs[i] * fs[i + 1] < 0:
            try:
                r = brentq(lambda x: float(delta_func(x)), xs[i], xs[i + 1])
                roots.append(float(r))
            except ValueError:
                pass

    if abs(float(fs[0])) <= endpoint_tol:
        roots.append(0.0)
    if abs(float(fs[-1])) <= endpoint_tol:
        roots.append(1.0)

    return sorted(set(round(r, 6) for r in roots))


def stability_at_root_from_delta(delta_func, x, h=1e-5):
    f1 = float(delta_func(max(0.0, x - h)))
    f2 = float(delta_func(min(1.0, x + h)))
    d = (f2 - f1) / (2 * h)
    return float(d), ("stable" if d < 0 else "unstable")


# ===================== Helpers for ω-range =====================

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

def plot_drift_generic(delta_func, roots, title, xmin=0.0, xmax=1.0):
    xs = np.linspace(xmin, xmax, 2001)
    ys = np.array([float(delta_func(x)) for x in xs], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(xs, ys, linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1.8)

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Delta(x)$")
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)

    # Direction arrows
    k = 20
    span = (xmax - xmin)
    if span <= 0:
        span = 1.0

    xA = np.linspace(xmin + 0.04 * span, xmax - 0.04 * span, k)
    yA = np.array([float(delta_func(x)) for x in xA], dtype=float)
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
        _, cls = stability_at_root_from_delta(delta_func, r)

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


# ===================== Affine weights and continuation =====================

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
def branches_over_omega_generic(mode, n, m, intercept, slope, w_min, w_max, n_w,
                               b_over_c=2.0,
                               seed_every=1, scan_grid=401, tol_match=0.015):
    omegas = np.linspace(float(w_min), float(w_max), int(n_w))

    def w_of(omega):
        return affine_weights(intercept, slope, omega)

    def delta_of_w(w_vec):
        if mode == "CPR":
            return lambda x: Delta_w_cpr(n, m, w_vec, x)
        else:
            return lambda x: Delta_w_cbr(n, m, w_vec, x, b_over_c)

    w0 = w_of(omegas[0])
    roots0 = find_equilibria_from_delta(delta_of_w(w0), grid=scan_grid)

    branches = []
    for r in roots0:
        _, cls = stability_at_root_from_delta(delta_of_w(w0), r)
        branches.append({"x": [r], "stab": [cls]})

    for j in range(1, len(omegas)):
        omega = float(omegas[j])
        wj = w_of(omega)
        delta_j = delta_of_w(wj)

        def f(x):
            return float(delta_j(x))

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
                _, cls = stability_at_root_from_delta(delta_j, x_new)
                br["x"].append(x_new)
                br["stab"].append(cls)
                tracked_roots.append(x_new)
            else:
                br["x"].append(np.nan)
                br["stab"].append("unstable")

        if (j % seed_every) == 0:
            roots_scan = find_equilibria_from_delta(delta_j, grid=scan_grid)
            for r in roots_scan:
                if any((not np.isnan(t)) and abs(r - t) <= tol_match for t in tracked_roots):
                    continue
                _, cls = stability_at_root_from_delta(delta_j, r)
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


def plot_branches(omegas, branches, title):
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
    ax.set_title(title)
    ax.set_xlim(float(omegas[0]), float(omegas[-1]))
    ax.set_ylim(0, 1)
    return fig


# ===================== Streamlit UI =====================

st.set_page_config(layout="wide")

st.markdown(
    """
    <div style="margin-top: -0.5rem;">
      <div style="font-size: 1.4rem; font-weight: 700;">Little platoons</div>
      <div style="font-size: 0.85rem; line-height: 1.2;">
        Little Platoons: Coalitional Procedural Rationality and the Provision of Public Goods in Large Populations
        <br/>
        <a href="https://dx.doi.org/10.2139/ssrn.5710262" target="_blank">https://dx.doi.org/10.2139/ssrn.5710262</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Core parameters")
    n = st.number_input("n (number of players)", 1, 200, 4, step=1)
    m = st.number_input("m (threshold)", 0, int(n), min(3, int(n)), step=1)

    st.divider()

    st.header("View")
    mode = st.radio("CPR vs. CBR", ["CPR", "CBR"], index=0)
    view = st.radio("Single specification vs. Parameterized comparison",
                    ["Single specification", "Parameterized comparison"], index=0)

    st.divider()

    st.header("Single specification inputs")
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

    st.header("Parameterized comparison inputs")
    st.caption("Affine family w(ω) = intercept + ω*slope, with multiplication componentwise. \nNegative w(ω) are not permitted. Positive w(ω) are normalized to sum to 1.")
    intercept_str = st.text_input("Intercept vector (length n)", "1 0 0 0")
    slope_str = st.text_input("Slope vector (length n)", "-2 1 1 0")

    restrict_omega = st.checkbox("Restrict ω-domain (manual)", False)
    if restrict_omega:
        omega_min_user = st.number_input("Manual ω min (≥ 0)", 0.0, 1.0, 0.0, format="%.6f")
        omega_max_user = st.number_input("Manual ω max (≤ automatic ω max)", 0.0, 1.0, 1.0, format="%.6f")
    else:
        omega_min_user = 0.0
        omega_max_user = 1.0

    n_w = st.number_input("Number of ω points used for plot", 5, 201, 80, step=1)

    st.divider()

    st.header("Coalitional best response (CBR)")
    b_over_c = st.number_input("b/c (must be > 1)", 1.000001, 1e6, 2.0, step=0.1, format="%.6f")


# ===================== Content =====================

# Parse w for single spec use (both CPR and CBR)
ok_w = True
try:
    w = np.array([float(x) for x in w_str.split()], dtype=float)
    if len(w) != int(n):
        st.error(f"Need exactly {int(n)} weights.")
        ok_w = False
    elif np.any(w < 0) or w.sum() <= 0:
        st.error("Weights must be nonnegative and sum to > 0.")
        ok_w = False
    else:
        w = w / w.sum()
except Exception as e:
    st.error(f"Could not parse w: {e}")
    ok_w = False

# Parse intercept/slope for parameterized view
ok_aff = True
try:
    intercept = np.array([float(x) for x in intercept_str.split()], dtype=float)
    slope = np.array([float(x) for x in slope_str.split()], dtype=float)
    if len(intercept) != int(n) or len(slope) != int(n):
        st.error(f"intercept and slope must both have length {int(n)}.")
        ok_aff = False
except Exception as e:
    st.error(f"Could not parse intercept/slope: {e}")
    ok_aff = False


if mode == "CBR" and float(b_over_c) <= 1.0:
    st.error("Need b/c > 1 for coalitional best response.")
    st.stop()


if view == "Single specification":
    st.subheader(f"{mode}: Single specification")

    if not ok_w:
        st.info("Fix inputs in the sidebar to display results.")
        st.stop()

    if mode == "CPR":
        delta = lambda x: Delta_w_cpr(int(n), int(m), w, x)
        title = f"CPR drift (n={int(n)}, m={int(m)})"
    else:
        delta = lambda x: Delta_w_cbr(int(n), int(m), w, x, float(b_over_c))
        title = f"CBR drift (n={int(n)}, m={int(m)}, b/c={float(b_over_c):g})"

    roots = find_equilibria_from_delta(delta, grid=2001)

    st.pyplot(
        plot_drift_generic(delta, roots, title=title, xmin=xmin, xmax=xmax),
        use_container_width=True
    )

    st.subheader("Equilibria (table)")
    if roots:
        rows = []
        for r in roots:
            _, s = stability_at_root_from_delta(delta, r)
            rows.append({"x*": r, "type": s})
        st.dataframe(rows, use_container_width=True)
    else:
        st.write("No equilibria found.")
        st.caption(f"Diagnostics: Δ(0)={delta(0.0):.6g}, Δ(1)={delta(1.0):.6g}")

else:
    st.subheader(f"{mode}: Parameterized comparison")

    if not ok_aff:
        st.info("Fix parameters in the sidebar to see results.")
        st.stop()

    w_max_auto = omega_max_from_affine(intercept, slope, cap=1.0)
    if w_max_auto <= 0:
        st.error("Automatic ω_max is 0. Your affine family becomes negative immediately (or intercept has negatives).")
        st.stop()

    if restrict_omega:
        w_min = float(max(0.0, omega_min_user))
        w_max = float(min(w_max_auto, omega_max_user))
        if w_max <= w_min:
            st.error("Manual ω max must be strictly greater than manual ω min.")
            st.stop()
        st.info(
            f"Using ω range: [{w_min:.6g}, {w_max:.6g}] "
            f"(manual, capped by automatic ω_max={w_max_auto:.6g})"
        )
    else:
        w_min = 0.0
        w_max = float(w_max_auto)
        st.info(f"Using ω range: [0, {w_max:.6g}] (automatic)")

    try:
        _ = affine_weights(intercept, slope, float(w_min))
        _ = affine_weights(intercept, slope, float(w_max))
    except Exception as e:
        st.error(f"Invalid affine family over ω-range: {e}")
        st.stop()

    with st.spinner("Tracking equilibrium branches across ω…"):
        omegas, branches = branches_over_omega_generic(
            mode,
            int(n), int(m),
            intercept, slope,
            float(w_min), float(w_max),
            int(n_w),
            b_over_c=float(b_over_c),
            seed_every=1,
            scan_grid=401,
            tol_match=0.015
        )

    title = "Equilibrium branches vs ω (solid=stable, dashed=unstable)"
    if mode == "CBR":
        title = "CBR equilibrium branches vs ω (solid=stable, dashed=unstable)"

    st.pyplot(
        plot_branches(omegas, branches, title=title),
        use_container_width=True
    )
