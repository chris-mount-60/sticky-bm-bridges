# Numerics

`fig_lt_check.py` makes Figure 6, the numerical check of Section 6. It draws the local time at time 0.9 of the bridge
from (0, 0) to (0, 0.8) over [0, 2] with stickiness 0.5, 10^5 times by Algorithm 2, and compares its empirical
distribution function with the exact one of Proposition 4.4. It writes `fig_lt_check.pdf` and prints the largest
difference between the two functions and where it occurs, the numbers of the caption. It takes about 15 seconds.

`fig_refined.py` makes Figure 5, one bridge after 3000 queries by Algorithm 3 at the midpoint of the longest
interval. It writes `fig_refined.pdf`.

Both scripts import the samplers from `../code/sticky_bridge.py` and write their PDF next to themselves. Run them
from this folder, for example `python fig_lt_check.py`. Their seeds are fixed (20261117 and 20261116), and with
numpy 2.5 and scipy 1.18 they reproduce the figures of the paper. Numpy does not promise the same random numbers
from one version to the next, so another version can give slightly different samples.

## LaTeX

The scripts use matplotlib's pgf backend, which sets the labels with pdfLaTeX in the fonts of the paper. They need a
LaTeX installation, such as TeX Live or MiKTeX, with `pdflatex` on the path and the packages `lmodern`, `amsmath` and
`amssymb`. Without it they stop with an error when they draw the figure.
