"""Exact samplers for sticky reflected Brownian motion and its local time: Algorithms 1-3 of the paper.

The arguments use the symbols of the paper: x and y for positions, r for a local time, s and T for times, and mu for
the stickiness. Every sampler takes a numpy Generator rng.
"""
import math
from scipy.special import ndtr, erfcx

SQRT2 = math.sqrt(2.0)


# densities
def G(a, t):
    return math.exp(-a * a / (2.0 * t)) / math.sqrt(2.0 * math.pi * t)


def h(a, t):
    return a / t * G(a, t) if t > 0 else 0.0


def pA(y, t, x):
    return G(y - x, t) - G(y + x, t)


# Section 2.4
def inverse_gaussian(rng, m, lam):
    """IG(m, lam) from one normal and one uniform (Michael, Schucany and Haas 1976)."""
    z = rng.standard_normal()
    y = m * z * z / (2.0 * lam)
    w = m / (1.0 + y + math.sqrt(y * (2.0 + y)))  # no cancellation
    return w if rng.random() <= m / (m + w) else m * m / w


def passage_bridge(rng, a, b, t):
    """tau_a given tau_{a+b} = t, for a, b, t > 0 (Proposition B.1)."""
    if rng.random() < b / (a + b):
        v = inverse_gaussian(rng, a / b, a * a / t)
        return t * v / (1.0 + v)
    w = inverse_gaussian(rng, b / a, b * b / t)
    return t / (1.0 + w)


def ig_cdf(v, m, lam):
    """Distribution function of IG(m, lam)."""
    q = math.sqrt(lam / v)
    return ndtr(q * (v / m - 1.0)) + 0.5 * erfcx(q * (v / m + 1.0) / SQRT2) * math.exp(-lam * (v - m) ** 2 / (2.0 * m * m * v))


def passage_bridge_cdf(u, a, b, t):
    """P{tau_a <= u | tau_{a+b} = t}, equation (7)."""
    if u <= 0.0:
        return 0.0
    if u >= t:
        return 1.0
    return (b / (a + b) * ig_cdf(u / (t - u), a / b, a * a / t)
            + a / (a + b) * (1.0 - ig_cdf(t / u - 1.0, b / a, b * b / t)))


def bessel_bridge(rng, x, y, t, u):
    """Bessel(3) bridge from x to y over [0, t] at time u, with x + y > 0 (Lemma B.1)."""
    kappa = x * y / t
    c = 1.0
    if kappa > 0.0:
        c = 1.0 + math.log1p(rng.random() * math.expm1(-2.0 * kappa)) / kappa
    p = u / t
    m2 = (1.0 - p) ** 2 * x * x + p * p * y * y + 2.0 * p * (1.0 - p) * x * y * c
    m = math.sqrt(max(m2, 0.0))
    v = u * (t - u) / t
    z = m + math.sqrt(v) * rng.standard_normal()
    return math.sqrt(z * z + 2.0 * v * rng.exponential())


# Algorithm 1
def forward(rng, x, s, mu):
    """Algorithm 1: (Y_s, L_s) under P_x (Theorem 1)."""
    H = abs(rng.standard_normal())
    if H <= x / math.sqrt(s):
        m = math.sqrt(s) * H
        e = rng.exponential()
        return x - m + 2.0 * s * e / (math.sqrt(m * m + 2.0 * s * e) + m), 0.0
    r = 2.0 * (s * H * H - x * x) / (2.0 * x + mu * H * H + H * math.sqrt(mu * mu * H * H + 4.0 * (s + mu * x)))
    c = x + r
    t = (c / H) ** 2  # no cancellation
    if rng.random() <= mu * c / (mu * c + 2.0 * t):
        return 0.0, r
    e = rng.exponential()
    return 2.0 * t * e / (math.sqrt(c * c + 2.0 * t * e) + c), r


def _bisect(above, lo, hi, scale, rtol=1e-14, maxit=2200):
    """Bisection for the root of a test that holds left of the root."""
    for _ in range(maxit):
        a = 0.5 * (lo + hi)
        if a <= lo or a >= hi:
            break
        if above(a):
            lo = a
        else:
            hi = a
        if hi - lo <= rtol * scale(lo, hi):
            break
    return 0.5 * (lo + hi)


# Algorithm 2
def middle_local_time(rng, r, t, s, mu, rtol=1e-14, maxit=2200):
    """Local time at time s of the middle bridge (Proposition 4.4)."""
    U = rng.random()
    return _bisect(lambda a: passage_bridge_cdf(s - mu * a, a, r - a, t) >= U,
                   max(0.0, (s - t) / mu), min(r, s / mu),
                   lambda lo, hi: min(lo, r - hi, (s - mu * hi) / mu, (t - s + mu * lo) / mu), rtol, maxit)


def middle_position(rng, a, r, t, s, mu, counter=None):
    """Position given the local time a in the middle bridge (Proposition 4.5, Lemma C.1)."""
    s1 = s - mu * a
    s2 = t - s1
    v = s1 * s2 / t
    sv = math.sqrt(v)
    k = sv * (a / s1 + (r - a) / s2)
    c = (a * (r - a), r * sv, v)
    gam = [2.0 * (j + 1) / (math.sqrt(k * k + 4.0 * (j + 1)) + k) for j in range(3)]
    lam = [k + gam[j] for j in range(3)]
    mass = [sv * c[j] * math.exp(0.5 * gam[j] ** 2) * math.factorial(j) / lam[j] ** (j + 1) for j in range(3)]
    atom = 0.5 * mu * a * (r - a)
    M = atom + mass[0] + mass[1] + mass[2]
    passes = 0
    while True:
        passes += 1
        w = rng.random() * M
        if w < atom:
            z = 0.0
            break
        w -= atom
        j = 0 if w < mass[0] else (1 if w < mass[0] + mass[1] else 2)
        x = sum(rng.exponential() for _ in range(j + 1)) / lam[j]
        if rng.random() <= math.exp(-0.5 * (x - gam[j]) ** 2):
            z = sv * x
            break
    if counter is not None:
        counter.append(passes)
    return z


def middle(rng, r, t, s, mu, counter=None):
    """(Y_s, L_s) in the middle bridge of local time r and operational length t."""
    a = middle_local_time(rng, r, t, s, mu)
    return middle_position(rng, a, r, t, s, mu, counter), a


def bridge(rng, x, y, r, T, s, mu, counter=None):
    """Algorithm 2: (Y_s, L_s, f, g) under P_x{ . | Y_T = y, L_T = r} (Theorem 2)."""
    if r == 0.0:
        return bessel_bridge(rng, x, y, T, s), 0.0, None, None
    f = passage_bridge(rng, x, r + y, T - mu * r) if x > 0.0 else 0.0
    t = passage_bridge(rng, r, y, T - mu * r - f) if y > 0.0 else T - mu * r - f
    g = f + t + mu * r
    if s < f:
        return bessel_bridge(rng, x, 0.0, f, s), 0.0, f, g
    if s > g:
        return bessel_bridge(rng, 0.0, y, T - g, s - g), r, f, g
    Y, L = middle(rng, r, t, s - f, mu, counter)
    return Y, L, f, g


# Algorithm 3
import bisect


class Skeleton:
    """A skeleton (Section 5.1): points (s, z, a), with operational times th = s - mu a."""

    def __init__(self, mu):
        self.mu = mu
        self.s, self.z, self.a, self.th = [], [], [], []
        self.T = None

    def add(self, s, z, a, th):
        i = bisect.bisect_left(self.s, s)
        self.s.insert(i, s); self.z.insert(i, z); self.a.insert(i, a); self.th.insert(i, th)
        return i

    @classmethod
    def start(cls, x, mu):
        """The skeleton {(0, x, 0)} under P_x."""
        R = cls(mu)
        R.add(0.0, x, 0.0, 0.0)
        return R

    @classmethod
    def bridge(cls, rng, x, y, r, T, mu):
        """The completed skeleton of the bridge from (0, x, 0) to (T, y, r)."""
        R = cls(mu)
        R.T = T
        R.add(0.0, x, 0.0, 0.0)
        R.add(T, y, r, T - mu * r)
        if r > 0.0:
            R.complete(rng, 0)
        return R

    def complete(self, rng, i):
        """Draw the first and last zeros of the open interval (i, i+1)."""
        mu = self.mu
        s1, z1, a1, th1 = self.s[i], self.z[i], self.a[i], self.th[i]
        z2, a2, th2 = self.z[i + 1], self.a[i + 1], self.th[i + 1]
        delta = a2 - a1
        top = th2 - th1
        f = passage_bridge(rng, z1, delta + z2, top) if z1 > 0.0 else 0.0
        d = s1 + f
        if z2 > 0.0:
            t = passage_bridge(rng, delta, z2, top - f)
            self.add(d + mu * delta + t, 0.0, a2, th1 + f + t)
        if z1 > 0.0:
            self.add(d, 0.0, a1, th1 + f)

    def query(self, rng, u, counter=None):
        """The query Q_u (Section 5.1). Returns (Y_u, L_u)."""
        mu = self.mu
        i = bisect.bisect_left(self.s, u)
        if i < len(self.s) and self.s[i] == u:
            return self.z[i], self.a[i]
        if i == len(self.s):
            if self.T is not None:
                raise ValueError('query time beyond the end of the bridge')
            m = i - 1
            z, a = forward(rng, self.z[m], u - self.s[m], mu)
            self.add(u, z, self.a[m] + a, self.th[m] + (u - self.s[m]) - mu * a)
            if a > 0.0 and self.z[m] + z > 0.0:
                self.complete(rng, m)
            return z, self.a[m] + a
        i -= 1
        si, zi, ai, thi = self.s[i], self.z[i], self.a[i], self.th[i]
        sj, zj, aj, thj = self.s[i + 1], self.z[i + 1], self.a[i + 1], self.th[i + 1]
        if aj == ai:
            z = bessel_bridge(rng, zi, zj, sj - si, u - si)
            self.add(u, z, ai, thi + (u - si))
            return z, ai
        if zi > 0.0 or zj > 0.0:
            raise ValueError('open interval (increasing local time, an end above 0): complete it first')
        delta, top = aj - ai, thj - thi
        z, a = middle(rng, delta, top, u - si, mu, counter)
        w1 = (u - si) - mu * a
        thu = thi + w1
        self.add(u, z, ai + a, thu)
        if z > 0.0:
            back = passage_bridge(rng, z, a, w1)
            ahead = passage_bridge(rng, z, delta - a, top - w1)
            self.add(u - back, 0.0, ai + a, thu - back)
            self.add(u + ahead, 0.0, ai + a, thu + ahead)
        return z, ai + a
