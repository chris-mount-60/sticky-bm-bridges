# Exact bridges for sticky reflected Brownian motion

This repository holds the paper

C. Mount and I. Fatkullin, *Exact bridges for sticky reflected Brownian motion* (2026, to appear on arXiv),

with its code and numerics, and the interactive pages and the movies of the talk on it at the AMS Fall Western
Sectional Meeting, Arizona State University, November 8, 2026.

The samplers draw the position of sticky reflected Brownian motion and its local time at 0 exactly, from a few
uniform, normal and exponential variables. They take a forward step, sample bridges given the position and the local
time at the end, and refine a path at any times in any order.

- `paper/exact-bridges-sticky-rbm.pdf` the paper.
- `code/sticky_bridge.py` the samplers, Algorithms 1 to 3 of the paper, in Python 3 with numpy and scipy. See
  `code/README.md` for the functions and a short example.
- `numerics/` the scripts of Figure 5 and of the numerical check of Section 6 (Figure 6). They draw with matplotlib
  and set their labels with pdfLaTeX, so they need a LaTeX installation; see `numerics/README.md`.
- `docs/` the pages, served at https://chris-mount-60.github.io/sticky-bm-bridges/. *One path, refined* lets you query
  a bridge by clicking on it, with the samplers ported to JavaScript. *The zero set of reflected Brownian motion, and
  its local time* shows the approximations of the zero set and of the local time closing in on their limits. The
  movies in `docs/movies` show a heavy particle in a gas of light molecules above a reflecting wall and above a sticky
  wall, and one sticky path refined by the samplers.

To install the Python packages, run `pip install -r requirements.txt`.

## Citing

```
@misc{mountExactBridges2026,
  author = {Mount, Christopher and Fatkullin, Ibrahim},
  title = {Exact bridges for sticky reflected {B}rownian motion},
  year = {2026},
  note = {Preprint, to appear on arXiv}
}
```

## License

The code, including the JavaScript of the pages, is under the MIT License (LICENSE). The movies and the text and
pictures of the pages are under CC BY 4.0 (LICENSE-media). The paper in `paper/` is not covered by either license. Its copyright stays with the authors, and arXiv distributes
it under the arXiv.org perpetual, non-exclusive license.