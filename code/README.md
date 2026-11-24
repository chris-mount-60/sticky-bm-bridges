# Code

`sticky_bridge.py` holds the samplers of the paper, in Python 3 with numpy and scipy and nothing else. Every function
takes a numpy `Generator` and draws fresh elementary random variables from it, and the arguments use the symbols of
the paper.

- `forward` is Algorithm 1, the forward step.
- `bridge` is Algorithm 2, the bridge given the position and the local time at the end.
- `Skeleton` is the skeleton of Algorithm 3. `Skeleton.bridge` and `Skeleton.start` make one, and
  `Skeleton.query` refines it at any time, in any order.

A short example draws a bridge from 0 to 0 over [0, 1] with local time 0.5 at the end, for the stickiness 0.4, and
queries it at three times.

```python
import numpy as np
from sticky_bridge import Skeleton

rng = np.random.default_rng(1)
R = Skeleton.bridge(rng, 0.0, 0.0, 0.5, 1.0, 0.4)
for u in (0.5, 0.25, 0.75):
    y, l = R.query(rng, u)
    print(u, y, l)
```
