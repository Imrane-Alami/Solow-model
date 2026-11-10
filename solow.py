"""
The Solow Growth Model Applied to Morocco
=========================================
Simulates the Solow (1956) model and confronts it with Moroccan and US data.

Cobb-Douglas production: y = A k^alpha.
Central equation:        k(t+1) = s A f(k(t)) + (1 - delta) k(t).

Run:  python solow.py   -> figures are saved in figures/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Textbook style: LaTeX font, arrow axes, no grid
# ------------------------------------------------------------------
from matplotlib import font_manager

for font in Path("/usr/share/texmf/fonts/opentype/public/lm").glob("lmroman10-*.otf"):
    font_manager.fontManager.addfont(str(font))   # Latin Modern if installed (TeX Live)

BLACK, GREEN, ORANGE, BLUE, GREY = "#000000", "#0b6e4f", "#d9731f", "#1f4e9c", "#6b6b6b"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11, "axes.labelsize": 11, "axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.dpi": 110,
    "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def textbook_axes(ax, xlabel, ylabel, xmax, ymax):
    """Textbook axes: two arrows from the origin, no tick marks."""
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, xmax * 1.06); ax.set_ylim(0, ymax * 1.08)
    arrow = dict(arrowstyle="-|>", lw=0.8, color=BLACK, mutation_scale=9)
    ax.annotate("", (xmax * 1.06, 0), (0, 0), arrowprops=arrow, annotation_clip=False)
    ax.annotate("", (0, ymax * 1.08), (0, 0), arrowprops=arrow, annotation_clip=False)
    ax.text(xmax * 1.06, -ymax * 0.035, xlabel, ha="center", va="top")
    ax.text(-xmax * 0.02, ymax * 1.08, ylabel, ha="right", va="center")


def project(ax, x, y, label_x=None, label_y=None, color=BLACK):
    """Dashed lines from (x, y) to both axes, with labels on the axes."""
    ax.plot([x, x], [0, y], ls=(0, (3, 3)), lw=0.8, color=color)
    ax.plot([0, x], [y, y], ls=(0, (3, 3)), lw=0.8, color=color)
    lim_y = ax.get_ylim()[1]
    lim_x = ax.get_xlim()[1]
    if label_x:
        ax.text(x, -lim_y * 0.035, label_x, ha="center", va="top")
    if label_y:
        ax.text(-lim_x * 0.015, y, label_y, ha="right", va="center")


def label(ax, x, y, text, color=BLACK, **kw):
    """Curve name written at its end (no legend box)."""
    ax.text(x, y, text, color=color, va="center", **kw)


def caption(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha="center", va="top", fontsize=10.5)


def style_series(ax):
    """Time series / data charts: plain, thin graduated axes."""
    ax.spines["left"].set_linewidth(0.8); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=10)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf / .png")


# ------------------------------------------------------------------
# 1. The model
# ------------------------------------------------------------------
ALPHA = 1 / 3  # capital share of income (usually around 1/3)


def f(k, alpha=ALPHA):
    """Output per worker without A: f(k) = k^alpha."""
    return k ** alpha


def central_equation(k, s, A, delta, alpha=ALPHA):
    """k(t+1) = s A f(k) + (1 - delta) k   (central equation)."""
    return s * A * f(k, alpha) + (1 - delta) * k


def simulate(k0, T, s, A, delta, alpha=ALPHA):
    """Path of k over T periods. s and A may be arrays (shocks)."""
    s = np.broadcast_to(s, T)
    A = np.broadcast_to(A, T)
    k = np.empty(T + 1)
    k[0] = k0
    for t in range(T):
        k[t + 1] = central_equation(k[t], s[t], A[t], delta, alpha)
    return k


def steady_state(s, A, delta, alpha=ALPHA):
    """Closed-form steady state: k*, y*, c*."""
    k = (s * A / delta) ** (1 / (1 - alpha))
    y = A * k ** alpha
    return k, y, (1 - s) * y


# Check: the simulation converges to the closed-form solution
_k = simulate(1.0, 3000, 0.3, 1.0, 0.05)[-1]
assert abs(_k / steady_state(0.3, 1.0, 0.05)[0] - 1) < 1e-6, "simulation != formula"


# ------------------------------------------------------------------
# 2. Data
# ------------------------------------------------------------------
data = pd.read_csv(HERE / "data" / "morocco_usa_data.csv")
s_country = data.groupby("code")["gcf_pct_gdp"].mean() / 100     # average investment rate 2018-2024
y_country = data.dropna(subset=["gdp_pc_ppp_2024"]).set_index("code")["gdp_pc_ppp_2024"]
S_MAR, S_USA = s_country["MAR"], s_country["USA"]
DELTA = 0.05  # annual depreciation rate (standard value)

# Illustrative parameters for the diagrams (visible steps)
S_ILL, DELTA_ILL = 0.30, 0.20




# ==================================================================
# FIGURES
# ==================================================================

# ------------------------------------------------------------------
# Fig. 1 — The steady state
# ------------------------------------------------------------------
def fig_steady_state():
    k_star = steady_state(S_ILL, 1, DELTA_ILL)[0]
    xmax = 1.75 * k_star
    kk = np.linspace(0, xmax, 400)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    textbook_axes(ax, r"$k_t$", r"$k_{t+1}$", xmax, xmax)
    ax.plot(kk, kk, color=BLACK, lw=0.9)
    ax.plot(kk, central_equation(kk, S_ILL, 1, DELTA_ILL), color=BLACK, lw=1.6)
    label(ax, xmax * 1.0, xmax * 1.03, r"$k_{t+1}=k_t$", ha="left")
    label(ax, xmax, central_equation(xmax, S_ILL, 1, DELTA_ILL), r"$sAf(k_t)+(1-\delta)k_t$", ha="left")
    ax.plot(k_star, k_star, "o", color=BLACK, ms=3.5)
    project(ax, k_star, k_star, r"$k^*$", r"$k^*$")
    caption(fig, "Fig. 1: The steady state, where the central equation crosses the 45° line.")
    save(fig, "01_steady_state")


# ------------------------------------------------------------------
# Fig. 2 — Dynamics: under- and over-accumulation
# ------------------------------------------------------------------
def fig_staircases():
    S_ESC, D_ESC = 0.5, 0.5          # wider steps for readability
    k_star = steady_state(S_ESC, 1, D_ESC)[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, k0, title, color in [(axes[0], 0.08 * k_star, "(a) Under-accumulation: $k_t < k^*$", BLUE),
                                   (axes[1], 2.6 * k_star, "(b) Over-accumulation: $k_t > k^*$", ORANGE)]:
        xmax = max(1.6 * k_star, 1.1 * k0)
        kk = np.linspace(0, xmax, 400)
        textbook_axes(ax, r"$k_t$", r"$k_{t+1}$", xmax, xmax)
        ax.plot(kk, kk, color=BLACK, lw=0.9)
        ax.plot(kk, central_equation(kk, S_ESC, 1, D_ESC), color=BLACK, lw=1.6)
        label(ax, xmax, xmax * 1.03, r"$45^\circ$", ha="left")
        k = simulate(k0, 12, S_ESC, 1, D_ESC)
        xs, ys = [k[0]], [0]
        for t in range(12):
            xs += [k[t], k[t + 1]]
            ys += [k[t + 1], k[t + 1]]
        ax.plot(xs, ys, color=color, lw=1.1)
        names = [r"$k_t$", r"$k_{t+1}$", r"$k_{t+2}$"]
        for i, nm in enumerate(names):
            ax.plot([k[i], k[i]], [0, k[i]], ls=(0, (2, 2)), lw=0.7, color=GREY)
            ax.text(k[i], -xmax * 0.035, nm, ha="center", va="top", fontsize=10)
        ax.plot([k_star, k_star], [0, k_star], ls=(0, (3, 3)), lw=0.8, color=BLACK)
        ax.text(k_star, -xmax * 0.035, r"$k^*$", ha="center", va="top")
        ax.set_title(title, fontsize=11, pad=12)
    caption(fig, "Fig. 2: Wherever it starts, capital per worker converges to $k^*$: the equilibrium is stable.")
    fig.tight_layout()
    save(fig, "02_dynamics_staircases")


# ------------------------------------------------------------------
# Fig. 3 — Investment vs depreciation
# ------------------------------------------------------------------
def fig_investment_depreciation():
    k_star = steady_state(S_ILL, 1, DELTA_ILL)[0]
    xmax = 2.2 * k_star
    kk = np.linspace(1e-4, xmax, 400)
    inv, dep = S_ILL * f(kk), DELTA_ILL * kk
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    textbook_axes(ax, r"$k$", "", xmax, dep[-1])
    ax.plot(kk, inv, color=BLACK, lw=1.6)
    ax.plot(kk, dep, color=BLACK, lw=0.9)
    label(ax, xmax * 1.01, inv[-1], r"$sAf(k)$", ha="left")
    label(ax, xmax * 1.01, dep[-1], r"$\delta k$", ha="left")
    project(ax, k_star, DELTA_ILL * k_star, r"$k^*$")
    ax.annotate("", (0.75 * k_star, 0.05 * dep[-1]), (0.25 * k_star, 0.05 * dep[-1]),
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate("", (1.25 * k_star, 0.05 * dep[-1]), (1.75 * k_star, 0.05 * dep[-1]),
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.text(0.5 * k_star, 0.1 * dep[-1], r"$k$ rises", ha="center", color=BLUE, fontsize=10)
    ax.text(1.5 * k_star, 0.1 * dep[-1], r"$k$ falls", ha="center", color=ORANGE, fontsize=10)
    caption(fig, "Fig. 3: In the steady state, investment per worker exactly offsets depreciation.")
    save(fig, "03_investment_vs_depreciation")


# ------------------------------------------------------------------
# Fig. 4 — A higher saving rate: a level effect
# ------------------------------------------------------------------
def fig_saving_increase():
    s0, s1, T, t_shock = 0.20, S_MAR, 150, 10
    k0_star, k1_star = steady_state(s0, 1, DELTA)[0], steady_state(s1, 1, DELTA)[0]
    s_path = np.where(np.arange(T) < t_shock, s0, s1)
    k = simulate(k0_star, T, s_path, 1, DELTA)
    y = f(k)
    growth = 100 * np.diff(np.log(y))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    xmax = 1.45 * k1_star
    kk = np.linspace(1e-4, xmax, 300)
    textbook_axes(ax, r"$k$", "", xmax, DELTA * xmax)
    ax.plot(kk, s0 * f(kk), color=BLACK, lw=1.4)
    ax.plot(kk, s1 * f(kk), color=BLUE, lw=1.6)
    ax.plot(kk, DELTA * kk, color=BLACK, lw=0.9)
    label(ax, xmax * 1.01, s0 * f(xmax), rf"$s_0Af(k)$, $s_0={s0:.0%}$".replace("%", r"\%"), ha="left", fontsize=10)
    label(ax, xmax * 1.01, s1 * f(xmax), rf"$s_1Af(k)$, $s_1={s1:.0%}$".replace("%", r"\%"), BLUE, ha="left", fontsize=10)
    label(ax, xmax * 1.01, DELTA * xmax * 1.03, r"$\delta k$", ha="left")
    project(ax, k0_star, DELTA * k0_star, r"$k_0^*$")
    project(ax, k1_star, DELTA * k1_star, r"$k_1^*$", color=BLUE)
    ax.set_title("(a) The investment curve shifts up", fontsize=11, pad=12)

    ax = axes[1]
    style_series(ax)
    ax.plot(y / y[0], color=BLACK)
    ax.axhline(steady_state(s1, 1, DELTA)[1] / y[0], color=GREY, ls=(0, (3, 3)), lw=0.8)
    ax.axvline(t_shock, color=GREY, ls=":", lw=0.8)
    ax.set(xlabel="years", ylabel=r"$y_t / y_0$")
    ax.set_title("(b) Income per worker reaches a new plateau", fontsize=11)

    ax = axes[2]
    style_series(ax)
    ax.plot(growth, color=BLACK)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axvline(t_shock, color=GREY, ls=":", lw=0.8)
    ax.set(xlabel="years", ylabel="growth of $y$ (% per year)")
    ax.set_title("(c) …but its growth falls back to zero", fontsize=11)
    fig.tight_layout(w_pad=3)
    caption(fig, f"Fig. 4: Saving rate raised from {s0:.0%} to {s1:.0%} (Morocco's level): "
                     "a level effect, not permanent growth.")
    save(fig, "04_saving_increase")


# ------------------------------------------------------------------
# Fig. 5 — Raising s or raising A
# ------------------------------------------------------------------
def fig_s_vs_A():
    T, t_shock = 150, 10
    k0 = steady_state(S_MAR, 1, DELTA)[0]
    y0 = f(k0)
    time = np.arange(T + 1)
    s_path = np.where(np.arange(T) < t_shock, S_MAR, S_MAR * 1.10)
    A_path = np.where(np.arange(T) < t_shock, 1.0, 1.10)
    y_s = f(simulate(k0, T, s_path, 1, DELTA))
    y_A = np.where(time <= t_shock, 1.0, 1.10) * f(simulate(k0, T, S_MAR, A_path, DELTA))
    gain_s = 100 * (1.10 ** (ALPHA / (1 - ALPHA)) - 1)
    gain_A = 100 * (1.10 ** (1 / (1 - ALPHA)) - 1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    style_series(ax)
    ax.plot(time, 100 * (y_A / y0 - 1), color=GREEN, lw=1.6)
    ax.plot(time, 100 * (y_s / y0 - 1), color=BLACK, lw=1.6)
    for g, c in [(gain_s, BLACK), (gain_A, GREEN)]:
        ax.axhline(g, color=c, ls=(0, (3, 3)), lw=0.7)
    label(ax, T + 2, gain_A, rf"$A$ +10%  $\rightarrow$  $y^*$ +{gain_A:.1f}%", GREEN, ha="left")
    label(ax, T + 2, gain_s, rf"$s$ +10%  $\rightarrow$  $y^*$ +{gain_s:.1f}%", BLACK, ha="left")
    ax.annotate("immediate jump:\n$A$ acts directly on $y=Af(k)$", (t_shock + 1, 10.1), xytext=(30, 6.5),
                fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set(xlabel="years", ylabel="income per worker (% deviation)", xlim=(0, T))
    caption(fig, "Fig. 5: The same 10% shock: productivity yields three times more than saving.")
    save(fig, "05_shock_s_vs_shock_A")


# ------------------------------------------------------------------
# Fig. 6 — The golden rule
# ------------------------------------------------------------------
def fig_golden_rule():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # (a) bell-shaped c*(s)
    ax = axes[0]
    s = np.linspace(1e-3, 0.999, 500)
    c = steady_state(s, 1, DELTA)[2]
    textbook_axes(ax, r"$s$", r"$c^*$", 1, c.max())
    ax.plot(s, c, color=BLACK, lw=1.6)
    c_gr = steady_state(ALPHA, 1, DELTA)[2]
    ax.plot(ALPHA, c_gr, "o", color=BLACK, ms=3.5)
    project(ax, ALPHA, c_gr, r"$s_{gr}=\alpha$")
    ax.text(1.0, -c.max() * 0.035, "1", ha="center", va="top")
    for s_, nm, col, dx, dy in [(S_USA, "United States", BLUE, 0.16, 0.62), (S_MAR, "Morocco", ORANGE, 0.49, 0.5)]:
        c_ = steady_state(s_, 1, DELTA)[2]
        ax.plot(s_, c_, "s", color=col, ms=4.5)
        ax.annotate(f"{nm} ({s_:.1%})", (s_, c_), xytext=(dx, dy * c.max()),
                    color=col, fontsize=10, ha="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=col))
    ax.text(0.14, 0.12 * c.max(), "$s<s_{gr}$:\nsaving more\nraises $c^*$", ha="center", fontsize=9.5)
    ax.text(0.72, 0.12 * c.max(), "$s>s_{gr}$:\nsaving more\nlowers $c^*$", ha="center", fontsize=9.5)
    ax.set_title("(a) The cake-eating trade-off", fontsize=11, pad=12)

    # (b) the tangency condition
    ax = axes[1]
    k_gr = steady_state(ALPHA, 1, DELTA)[0]
    xmax = 2.1 * k_gr
    kk = np.linspace(1e-4, xmax, 400)
    y = f(kk)
    textbook_axes(ax, r"$k_t$", r"$y_t,\ i_t,\ \delta k_t$", xmax, y[-1])
    ax.plot(kk, y, color=GREEN, lw=1.6)
    ax.plot(kk, ALPHA * y, color=BLACK, lw=1.6)
    ax.plot(kk, DELTA * kk, color=BLACK, lw=0.9)
    zone = (kk > 0.45 * k_gr) & (kk < 1.65 * k_gr)
    ax.plot(kk[zone], f(k_gr) + DELTA * (kk[zone] - k_gr), color=GREEN, ls=(0, (4, 3)), lw=0.9)
    label(ax, xmax * 1.01, y[-1], r"$y_t=Af(k_t)$", GREEN, ha="left")
    label(ax, xmax * 1.01, ALPHA * y[-1], r"$i_t=sAf(k_t)$", ha="left")
    label(ax, xmax * 1.01, DELTA * xmax, r"$\delta k_t$", ha="left")
    ax.text(0.5 * k_gr, f(k_gr) - DELTA * 0.5 * k_gr + 0.04 * y[-1], r"slope $\delta$", color=GREEN, fontsize=10, ha="right")
    project(ax, k_gr, f(k_gr), r"$k_{gr}$", r"$y^*$")
    project(ax, k_gr, ALPHA * f(k_gr), None, r"$i^*$")
    ax.plot([k_gr, k_gr], [ALPHA * f(k_gr), f(k_gr)], color=ORANGE, lw=2.2)
    ax.text(k_gr * 1.04, (1 + ALPHA) / 2 * f(k_gr), r"$c^*$", color=ORANGE)
    for yy in (f(k_gr), ALPHA * f(k_gr)):
        ax.plot(k_gr, yy, "o", color=BLACK, ms=3.5)
    ax.set_title(r"(b) At the golden rule: $Af'(k_{gr})=\delta$", fontsize=11, pad=12)
    fig.tight_layout(w_pad=6)
    caption(fig, r"Fig. 6: The golden rule: $c^*$ is maximized when $s=\alpha$, "
                     r"i.e. when the tangent to $y=Af(k)$ has slope $\delta$.")
    save(fig, "06_golden_rule")


# ------------------------------------------------------------------
# Fig. 7 — Augmented Solow model
# ------------------------------------------------------------------
def fig_augmented_solow():
    n, z, T = 0.01, 0.02, 120
    factor = (1 + n) * (1 + z)
    k_hat_star = (S_MAR / (DELTA + n + z + n * z)) ** (1 / (1 - ALPHA))
    k_hat = np.empty(T + 1)
    k_hat[0] = 0.3 * k_hat_star
    for t in range(T):
        k_hat[t + 1] = (S_MAR * f(k_hat[t]) + (1 - DELTA) * k_hat[t]) / factor
    time = np.arange(T + 1)
    y_hat = f(k_hat)
    y = y_hat * (1 + z) ** time
    Y = y * (1 + n) ** time
    series = [(Y, r"aggregate $Y$", ORANGE, n + z + n * z), (y, r"$y=Y/L$ per worker", BLUE, z),
              (y_hat, r"$\hat y=Y/(ZL)$", GREEN, 0)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    style_series(ax)
    for ser, nm, col, _ in series:
        ax.plot(time, np.log(ser / ser[0]), color=col)
        label(ax, T + 2, np.log(ser[-1] / ser[0]), nm, col, ha="left", fontsize=10)
    ax.set(xlabel="years", ylabel="log (0 at start)", xlim=(0, T))
    ax.set_title("(a) Levels", fontsize=11)
    ax = axes[1]
    style_series(ax)
    for ser, nm, col, target in series:
        ax.plot(time[1:], 100 * (ser[1:] / ser[:-1] - 1), color=col)
        ax.axhline(100 * target, color=col, ls=(0, (3, 3)), lw=0.7)
        label(ax, T + 2, 100 * target, rf"$\rightarrow$ {100 * target:.1f}%", col, ha="left", fontsize=10)
    ax.set(xlabel="years", ylabel="growth (% per year)", xlim=(0, T))
    ax.set_title("(b) Growth rates", fontsize=11)
    fig.tight_layout(w_pad=6)
    caption(fig, f"Fig. 7: Augmented Solow ($n$ = {n:.0%}, $z$ = {z:.0%}): in the long run, "
                     r"$\hat y$ is constant and $y$ grows at the rate $z$ of technical progress.")
    save(fig, "07_augmented_solow")


# ------------------------------------------------------------------
# Fig. 8 — Confronting the data: Morocco vs United States
# ------------------------------------------------------------------
def fig_data():
    expo = ALPHA / (1 - ALPHA)                      # y* ∝ A^(1/(1-α)) · s^(α/(1-α))
    ratio_pred = (S_MAR / S_USA) ** expo
    ratio_obs = y_country["MAR"] / y_country["USA"]
    ratio_A = (ratio_obs / ratio_pred) ** (1 - ALPHA)
    s_ratio_needed = ratio_obs ** (-1 / expo)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    style_series(ax)
    values = [100 * ratio_pred, 100 * ratio_obs, 100 * ratio_A]
    bars = ax.bar([0, 1, 2], values, width=0.55, color=["white", BLACK, GREEN], edgecolor=BLACK, lw=0.9,
                    hatch=["///", "", ""])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}%", ha="center")
    ax.axhline(100, color=GREY, ls=(0, (3, 3)), lw=0.8)
    ax.text(2.42, 101, "United States = 100", ha="right", va="bottom", fontsize=9, color=GREY)
    ax.set_xticks([0, 1, 2], ["predicted income\n(only $s$ differs)", "observed income\n(GDP p.c. PPP, 2024)",
                              "implied TFP\n$A_{MAR}/A_{USA}$"])
    ax.set(ylabel="Morocco, % of United States", ylim=(0, 135))
    ax.set_title(f"(a) Morocco invests more ({S_MAR:.1%} vs {S_USA:.1%} of GDP)", fontsize=11)

    ax = axes[1]
    style_series(ax)
    r = np.logspace(0, 2.2, 300)
    ax.plot(r, r ** expo, color=BLACK, lw=1.6)
    label(ax, r[-1] * 1.05, r[-1] ** expo, r"$\dfrac{y_1^*}{y_2^*}=\left(\dfrac{s_1}{s_2}\right)^{\frac{\alpha}{1-\alpha}}$", ha="left")
    x0, y0 = s_ratio_needed, 1 / ratio_obs
    ax.plot([x0, x0], [0, y0], ls=(0, (3, 3)), lw=0.8, color=ORANGE)
    ax.plot([1, x0], [y0, y0], ls=(0, (3, 3)), lw=0.8, color=ORANGE)
    ax.plot(x0, y0, "o", color=ORANGE, ms=4)
    ax.text(1.15, y0 + 0.35, f"observed income gap: ×{y0:.1f}", color=ORANGE, fontsize=10)
    ax.text(x0 * 0.93, 0.6, f"saving gap\nrequired: ×{x0:.0f}", color=ORANGE, fontsize=10, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set(xlabel=r"ratio of saving rates $s_1/s_2$ (log scale)", ylabel="ratio of long-run incomes",
           ylim=(0, 13))
    ax.set_title("(b) Saving cannot explain the gap", fontsize=11)
    fig.tight_layout(w_pad=4)
    caption(fig, "Fig. 8: Confronting Solow with the data: through the lens of the model, "
                     "$A$ accounts for most of the income gap.")
    save(fig, "08_morocco_vs_usa_data")
    return ratio_pred, ratio_obs, ratio_A, s_ratio_needed


# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures…")
    fig_steady_state()
    fig_staircases()
    fig_investment_depreciation()
    fig_saving_increase()
    fig_s_vs_A()
    fig_golden_rule()
    fig_augmented_solow()
    pred, obs, A_rel, s_nec = fig_data()

    print("\nKey results (alpha = 1/3, delta = 5%):")
    print(f"  Average investment rate 2018-2024: Morocco {S_MAR:.1%}, United States {S_USA:.1%}")
    for code, s_ in [("Morocco", S_MAR), ("United States", S_USA)]:
        k, y, c = steady_state(s_, 1, DELTA)
        print(f"  {code:<14} k* = {k:5.2f}  y* = {y:4.2f}  c* = {c:4.2f}  (A normalized to 1)")
    print(f"  Morocco/US income predicted by s alone: {pred:.0%}  |  observed: {obs:.0%}")
    print(f"  Implied TFP A_Morocco / A_USA          : {A_rel:.0%}")
    print(f"  Required saving gap                     : x{s_nec:.0f}")
