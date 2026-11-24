"""Figure 5: a bridge after many queries (Algorithm 3).

Output fig_refined.pdf.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import sticky_bridge as sb

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": True,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}",
    "font.family": "serif", "font.size": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
})
PATH, VAL = "0.35", "black"
x, y, r, T, mu, nq, seed = 0.6, 0.4, 0.8, 2.0, 0.5, 3000, 20261116

rng = np.random.default_rng(seed)
R = sb.Skeleton.bridge(rng, x, y, r, T, mu)
for _ in range(nq):
    gaps = np.diff(R.s)
    i = int(np.argmax(gaps))
    R.query(rng, R.s[i] + gaps[i] / 2.0)
s, Y, L = np.array(R.s), np.array(R.z), np.array(R.a)

fig, (aY, aL) = plt.subplots(1, 2, figsize=(6.5, 1.55), gridspec_kw={"wspace": 0.22})
aY.plot(s, Y, color=PATH, lw=0.45)
aL.plot(s, L, color=PATH, lw=0.9)
for ax, a0, a1 in ((aY, x, y), (aL, 0.0, r)):
    ax.plot([0, T], [a0, a1], "o", color=VAL, ms=4, zorder=4, clip_on=False)
    ax.set_xticks([0, T]); ax.set_xticklabels([r"$0$", r"$T$"]); ax.set_xlim(-0.03, T + 0.05)
aY.set_ylim(0, None); aY.set_yticks([0]); aY.set_yticklabels([r"$0$"])
aY.annotate(r"$x$", (0, x), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=8)
aY.annotate(r"$y$", (T, y), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8, annotation_clip=False)
aL.set_yticks([0, r]); aL.set_yticklabels([r"$0$", r"$r$"]); aL.set_ylim(0, 1.05 * r)
aY.set_title(r"position $\mathfrak Y$", fontsize=8.5, pad=4)
aL.set_title(r"local time $\ell$", fontsize=8.5, pad=4)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_refined.pdf"), bbox_inches="tight")
time_at_0 = float(np.sum(np.diff(s)[(Y[:-1] == 0) & (Y[1:] == 0)]))
print("points", len(s), "zeros recorded", int(np.sum(Y == 0)), "max gap", float(np.max(np.diff(s))))
