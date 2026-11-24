"""Figure 6: the local time of a bridge drawn by Algorithm 2 against its distribution function.

Output fig_lt_check.pdf, and the numbers for the caption.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import sticky_bridge as sb

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": True,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}",
    "font.family": "serif", "font.size": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
})
EXACT, EMP = dict(color="black", lw=1.0), dict(color="0.45", lw=0.9, ls=(0, (3, 2)))
r, T, mu, s = 0.8, 2.0, 0.5, 0.9
n, seed = 100_000, 20261117
ZOOM = 0.008

t = T - mu * r
lo, hi = max(0.0, (s - t) / mu), min(r, s / mu)


def F(a):
    """P{l_s <= a} from Proposition 4.4."""
    if a <= lo:
        return 0.0
    if a >= hi:
        return 1.0
    return 1.0 - sb.passage_bridge_cdf(s - mu * a, a, r - a, t)


rng = np.random.default_rng(seed)
L = np.sort([sb.bridge(rng, 0.0, 0.0, r, T, s, mu)[1] for _ in range(n)])

v, k = np.unique(L, return_counts=True)
right, left = np.cumsum(k) / n, (np.cumsum(k) - k) / n
Fv = np.array([F(a) for a in v])
i_up, i_dn = int(np.argmax(right - Fv)), int(np.argmax(Fv - left))
if right[i_up] - Fv[i_up] >= Fv[i_dn] - left[i_dn]:
    i, Fn_i = i_up, right[i_up]
else:
    i, Fn_i = i_dn, left[i_dn]
gap, centre = abs(Fn_i - Fv[i]), v[i]

w = ZOOM * (hi - lo)
xlo = max(lo, min(centre - w / 2, hi - w)); xhi = xlo + w
X, Yn = np.r_[lo, v, hi], np.r_[0.0, right, 1.0]
j0 = max(0, np.searchsorted(X, xlo) - 1); j1 = min(len(X), np.searchsorted(X, xhi, side="right") + 1)
ylo = min(F(xlo), np.searchsorted(L, xlo, side="right") / n, Fv[i], Fn_i)
yhi = max(F(xhi), np.searchsorted(L, xhi, side="right") / n, Fv[i], Fn_i)
pad = 0.14 * max(yhi - ylo, gap)
ylo, yhi = max(0.0, ylo - pad), min(1.0, yhi + pad)

fig = plt.figure(figsize=(6.5, 2.2))
full = fig.add_axes([0.05, 0.20, 0.45, 0.76])
zoom = fig.add_axes([0.62, 0.20, 0.36, 0.66])
g = np.linspace(lo, hi, 800); gz = np.linspace(xlo, xhi, 600)
e_line, = full.step(X, Yn, where="post", label="empirical", **EMP)
x_line, = full.plot(g, [F(a) for a in g], label="exact", **EXACT)
zoom.step(X[j0:j1], Yn[j0:j1], where="post", **EMP)
zoom.plot(gz, [F(a) for a in gz], **EXACT)

full.set_xlim(lo, hi); full.set_ylim(0, 1)
full.set_xticks([lo, hi]); full.set_xticklabels([rf"${lo:g}$", rf"${hi:g}$"])
full.set_yticks([0, 1]); full.set_yticklabels([r"$0$", r"$1$"])
full.legend(handles=[x_line, e_line], frameon=False, fontsize=7.5, loc="lower right", handlelength=2.2)
zoom.set_xlim(xlo, xhi); zoom.set_ylim(ylo, yhi)
for ax in (zoom.xaxis, zoom.yaxis):
    ax.set_major_locator(MaxNLocator(nbins=3))
    fmt = ScalarFormatter(useOffset=False); fmt.set_scientific(False); ax.set_major_formatter(fmt)
for ax in (full, zoom):
    ax.set_xlabel(r"local time $a$ at time $s$", fontsize=8, labelpad=2)
full.add_patch(Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, fill=False, edgecolor="0.55", lw=0.65, zorder=4))
for yy in (ylo, yhi):
    fig.add_artist(ConnectionPatch(xyA=(xhi, yy), coordsA=full.transData, xyB=(xlo, yy), coordsB=zoom.transData,
                                   color="0.65", lw=0.55, ls=":", zorder=3, clip_on=False))
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_lt_check.pdf"))

print(f"r={r:g} T={T:g} mu={mu:g} s={s:g}, n={n}, seed {seed}; local time in ({lo:g}, {hi:g})")
print(f"largest |F_n - F| = {gap:.4f} at a = {centre:.4f} (1/sqrt(n) = {n ** -0.5:.4f}); "
      f"magnified window [{xlo:.4f}, {xhi:.4f}] x [{ylo:.4f}, {yhi:.4f}]")
