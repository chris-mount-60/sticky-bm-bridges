/* Exact samplers for sticky reflected Brownian motion and its local time, for the click-to-refine page.
   A line-by-line port of code/sticky_bridge.py (Algorithms 1-3 of the paper): the same formulas, branches, stopping
   rule and names. Numbers are IEEE doubles, as in the Python code.
   Mount & Fatkullin, "Exact bridges for sticky reflected Brownian motion".

   Elementary random variables come from a seeded generator (xoshiro128**), so a path can be reproduced from its seed.
   In node: const S = require('./sampler.js'). In the browser: the global StickySampler. */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------------------------- random numbers
  function makeRng(seed) {
    // xoshiro128** seeded by splitmix32; uniform() returns a double in (0, 1) with 53 random bits.
    let x = seed >>> 0;
    function splitmix() {
      x = (x + 0x9e3779b9) >>> 0;
      let z = x;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
      return (z ^ (z >>> 16)) >>> 0;
    }
    let s0 = splitmix(), s1 = splitmix(), s2 = splitmix(), s3 = splitmix();
    function next() {
      const r = Math.imul(rotl(Math.imul(s1, 5) >>> 0, 7), 9) >>> 0;
      const t = (s1 << 9) >>> 0;
      s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = rotl(s3, 11);
      s0 >>>= 0; s1 >>>= 0; s2 >>>= 0; s3 >>>= 0;
      return r;
    }
    function rotl(v, k) { return ((v << k) | (v >>> (32 - k))) >>> 0; }
    function uniform() {
      const hi = next() >>> 5, lo = next() >>> 6;                // 27 + 26 = 53 bits
      return (hi * 67108864 + lo + 0.5) / 9007199254740992;       // strictly inside (0, 1)
    }
    let spare = null;
    function normal() {
      if (spare !== null) { const v = spare; spare = null; return v; }
      const u = uniform(), v = uniform();
      const r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * v;
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    }
    function exponential() { return -Math.log(uniform()); }
    return { random: uniform, standard_normal: normal, exponential: exponential };
  }

  // ------------------------------------------------------------------------------------- special functions
  const SQRT2 = Math.SQRT2, SQRTPI = Math.sqrt(Math.PI);

  function erfcxPos(x) {
    // exp(x^2) erfc(x) for x >= 0: the power series of erf for x < 0.5, the continued fraction of erfc otherwise.
    if (x < 0.5) {
      let term = x, sum = x, n = 0;                               // erf(x) = 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1))
      const x2 = x * x;
      for (n = 1; n < 60; n++) {
        term *= -x2 / n;
        const add = term / (2 * n + 1);
        sum += add;
        if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
      }
      return Math.exp(x2) * (1 - 2 / SQRTPI * sum);
    }
    // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))), by the modified Lentz method
    const tiny = 1e-300;
    let f = x, C = x, D = 0;
    for (let n = 1; n < 5000; n++) {
      const an = n / 2;
      D = x + an * D; if (Math.abs(D) < tiny) D = tiny; D = 1 / D;
      C = x + an / C; if (Math.abs(C) < tiny) C = tiny;
      const delta = C * D;
      f *= delta;
      if (Math.abs(delta - 1) < 1e-16) break;
    }
    return 1 / (SQRTPI * f);
  }

  function erfcx(x) {
    if (x >= 0) return erfcxPos(x);
    return 2 * Math.exp(x * x) - erfcxPos(-x);
  }

  function ndtr(z) {
    // standard normal distribution function; 0.5 erfc(-z / sqrt 2) = 0.5 erfcx(-z / sqrt 2) exp(-z^2 / 2)
    const x = -z / SQRT2;
    if (x >= 0) return 0.5 * erfcxPos(x) * Math.exp(-x * x);
    return 1 - 0.5 * erfcxPos(-x) * Math.exp(-x * x);
  }

  // ------------------------------------------------------------------------------------- Section 2.4
  function inverse_gaussian(rng, m, lam) {
    // IG(m, lam) from one normal and one uniform (Michael, Schucany and Haas 1976)
    const z = rng.standard_normal();
    const y = m * z * z / (2 * lam);
    const w = m / (1 + y + Math.sqrt(y * (2 + y)));               // the smaller root, without cancellation
    return rng.random() <= m / (m + w) ? w : m * m / w;
  }

  function passage_bridge(rng, a, b, t) {
    // tau_a given tau_{a+b} = t, for a, b, t > 0 (Proposition B.1)
    if (rng.random() < b / (a + b)) {
      const v = inverse_gaussian(rng, a / b, a * a / t);
      return t * v / (1 + v);
    }
    const w = inverse_gaussian(rng, b / a, b * b / t);
    return t / (1 + w);
  }

  function ig_cdf(v, m, lam) {
    // distribution function of IG(m, lam), with erfcx in place of exp * Phi to avoid overflow
    const q = Math.sqrt(lam / v);
    return ndtr(q * (v / m - 1)) + 0.5 * erfcx(q * (v / m + 1) / SQRT2) * Math.exp(-lam * (v - m) * (v - m) / (2 * m * m * v));
  }

  function passage_bridge_cdf(u, a, b, t) {
    // P{tau_a <= u | tau_{a+b} = t}, equation (7)
    if (u <= 0) return 0;
    if (u >= t) return 1;
    return b / (a + b) * ig_cdf(u / (t - u), a / b, a * a / t) + a / (a + b) * (1 - ig_cdf(t / u - 1, b / a, b * b / t));
  }

  function bessel_bridge(rng, x, y, t, u) {
    // value at time u of the Bessel(3) bridge from x to y over [0, t], with x + y > 0 (Lemma B.1)
    const kappa = x * y / t;
    let c = 1;
    if (kappa > 0) c = 1 + Math.log1p(rng.random() * Math.expm1(-2 * kappa)) / kappa;
    const p = u / t;
    const m2 = (1 - p) * (1 - p) * x * x + p * p * y * y + 2 * p * (1 - p) * x * y * c;
    const m = Math.sqrt(Math.max(m2, 0));
    const v = u * (t - u) / t;
    const z = m + Math.sqrt(v) * rng.standard_normal();
    return Math.sqrt(z * z + 2 * v * rng.exponential());
  }

  // ------------------------------------------------------------------------------------- Algorithm 1
  function forward(rng, x, s, mu) {
    // (Y_s, L_s) under P_x, for x >= 0 and s > 0 (Theorem 1)
    const H = Math.abs(rng.standard_normal());
    if (H <= x / Math.sqrt(s)) {
      const m = Math.sqrt(s) * H, e = rng.exponential();
      return [x - m + 2 * s * e / (Math.sqrt(m * m + 2 * s * e) + m), 0];
    }
    const r = 2 * (s * H * H - x * x) / (2 * x + mu * H * H + H * Math.sqrt(mu * mu * H * H + 4 * (s + mu * x)));
    const c = x + r;
    const t = (c / H) * (c / H);
    if (rng.random() <= mu * c / (mu * c + 2 * t)) return [0, r];
    const e = rng.exponential();
    return [2 * t * e / (Math.sqrt(c * c + 2 * t * e) + c), r];
  }

  // ------------------------------------------------------------------------------------- Algorithm 2
  function bisect_(above, lo, hi, scale, rtol, maxit) {
    rtol = rtol || 1e-14; maxit = maxit || 2200;
    for (let k = 0; k < maxit; k++) {
      const a = 0.5 * (lo + hi);
      if (a <= lo || a >= hi) break;
      if (above(a)) lo = a; else hi = a;
      if (hi - lo <= rtol * scale(lo, hi)) break;
    }
    return 0.5 * (lo + hi);
  }

  function middle_local_time(rng, r, t, s, mu) {
    // local time at the physical time s of the middle bridge of local time r, operational length t (Prop. 4.4)
    const U = rng.random();
    return bisect_(a => passage_bridge_cdf(s - mu * a, a, r - a, t) >= U,
      Math.max(0, (s - t) / mu), Math.min(r, s / mu),
      (lo, hi) => Math.min(lo, r - hi, (s - mu * hi) / mu, (t - s + mu * lo) / mu));
  }

  function factorial(j) { return j === 0 ? 1 : j === 1 ? 1 : 2; }

  function middle_position(rng, a, r, t, s, mu) {
    // position given the local time a in the middle bridge (Prop. 4.5), with the Gamma envelopes of Lemma C.1
    const s1 = s - mu * a, s2 = t - s1;
    const v = s1 * s2 / t, sv = Math.sqrt(v);
    const k = sv * (a / s1 + (r - a) / s2);
    const c = [a * (r - a), r * sv, v];
    const gam = [0, 1, 2].map(j => 2 * (j + 1) / (Math.sqrt(k * k + 4 * (j + 1)) + k));   // lambda_j - k
    const lam = gam.map(g => k + g);
    const mass = [0, 1, 2].map(j => sv * c[j] * Math.exp(0.5 * gam[j] * gam[j]) * factorial(j) / Math.pow(lam[j], j + 1));
    const atom = 0.5 * mu * a * (r - a);
    const M = atom + mass[0] + mass[1] + mass[2];
    for (;;) {
      let w = rng.random() * M;
      if (w < atom) return 0;
      w -= atom;
      const j = w < mass[0] ? 0 : (w < mass[0] + mass[1] ? 1 : 2);
      let x = 0;
      for (let i = 0; i <= j; i++) x += rng.exponential();
      x /= lam[j];
      if (rng.random() <= Math.exp(-0.5 * (x - gam[j]) * (x - gam[j]))) return sv * x;
    }
  }

  function middle(rng, r, t, s, mu) {
    const a = middle_local_time(rng, r, t, s, mu);
    return [middle_position(rng, a, r, t, s, mu), a];
  }

  // ------------------------------------------------------------------------------------- Algorithm 3
  function bisectLeft(arr, v) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; }
    return lo;
  }

  class Skeleton {
    // points (s, z, a) sorted by time, with the operational time th = s - mu a of each point (Section 5.1)
    constructor(mu) { this.mu = mu; this.s = []; this.z = []; this.a = []; this.th = []; this.T = null; }

    add(s, z, a, th) {
      const i = bisectLeft(this.s, s);
      this.s.splice(i, 0, s); this.z.splice(i, 0, z); this.a.splice(i, 0, a); this.th.splice(i, 0, th);
      return i;
    }

    static start(x, mu) { const R = new Skeleton(mu); R.add(0, x, 0, 0); return R; }

    static bridge(rng, x, y, r, T, mu) {
      const R = new Skeleton(mu);
      R.T = T;
      R.add(0, x, 0, 0);
      R.add(T, y, r, T - mu * r);
      if (r > 0) R.complete(rng, 0);
      return R;
    }

    complete(rng, i) {
      // completion of the open pair (i, i+1), a pair with increasing local time and an end above 0: its first and last zeros (Section 5.1)
      const mu = this.mu;
      const s1 = this.s[i], z1 = this.z[i], a1 = this.a[i], th1 = this.th[i];
      const z2 = this.z[i + 1], a2 = this.a[i + 1], th2 = this.th[i + 1];
      const delta = a2 - a1, top = th2 - th1;
      const f = z1 > 0 ? passage_bridge(rng, z1, delta + z2, top) : 0;
      const d = s1 + f;
      if (z2 > 0) {
        const t = passage_bridge(rng, delta, z2, top - f);
        this.add(d + mu * delta + t, 0, a2, th1 + f + t);
      }
      if (z1 > 0) this.add(d, 0, a1, th1 + f);
    }

    query(rng, u) {
      // the query Q_u (Section 5.1); adds the point at u and the recorded zeros it needs; returns [Y_u, L_u]
      const mu = this.mu;
      let i = bisectLeft(this.s, u);
      if (i < this.s.length && this.s[i] === u) return [this.z[i], this.a[i]];
      if (i === this.s.length) {                                   // after the last point: a forward step
        if (this.T !== null) throw new Error('query time beyond the end of the bridge');
        const m = i - 1;
        const [z, a] = forward(rng, this.z[m], u - this.s[m], mu);
        this.add(u, z, this.a[m] + a, this.th[m] + (u - this.s[m]) - mu * a);
        if (a > 0 && this.z[m] + z > 0) this.complete(rng, m);
        return [z, this.a[m] + a];
      }
      i -= 1;                                                      // u lies in (s_i, s_{i+1})
      const si = this.s[i], zi = this.z[i], ai = this.a[i], thi = this.th[i];
      const sj = this.s[i + 1], zj = this.z[i + 1], aj = this.a[i + 1], thj = this.th[i + 1];
      if (aj === ai) {                                             // constant interval: a Bessel(3) bridge
        const z = bessel_bridge(rng, zi, zj, sj - si, u - si);
        this.add(u, z, ai, thi + (u - si));
        return [z, ai];
      }
      if (zi > 0 || zj > 0) throw new Error('open interval (increasing local time, an end above 0): complete it first');
      const delta = aj - ai, top = thj - thi;                      // increasing interval: a middle bridge
      const [z, a] = middle(rng, delta, top, u - si, mu);
      const w1 = (u - si) - mu * a;
      const thu = thi + w1;
      this.add(u, z, ai + a, thu);
      if (z > 0) {
        const back = passage_bridge(rng, z, a, w1);                // u - g_u
        const ahead = passage_bridge(rng, z, delta - a, top - w1); // d_u - u
        this.add(u - back, 0, ai + a, thu - back);
        this.add(u + ahead, 0, ai + a, thu + ahead);
      }
      return [z, ai + a];
    }

    kind(i) {
      // the interval (s_i, s_{i+1}): 'constant' or 'increasing' (a skeleton has no open interval)
      return this.a[i + 1] === this.a[i] ? 'constant' : 'increasing';
    }
  }

  // ------------------------------------------------------------------------------------- the local time given the positions (not in the paper)
  function atom_prob_(v, K) {
    // expm1(v) / (expm1(v) + K) for v >= 0 and K > 0, without overflow for large v
    if (v <= 1) { const e = Math.expm1(v); return e / (e + K); }
    return 1 / (1 + K * Math.exp(-v) / -Math.expm1(-v));
  }

  function local_time_given_positions(rng, x, y, T, mu) {
    // L_T given Y_0 = x and Y_T = y; this sampler is not in the paper
    const has_atom = x > 0 && y > 0;
    const weight = 2 * Math.sqrt(2 * Math.PI * T) / mu;
    const b = x + y + 2 * T / mu;
    const q0 = b / Math.sqrt(T);
    if (has_atom && rng.random() <= atom_prob_(2 * x * y / T, weight * erfcx(q0 / SQRT2))) return 0;
    const gam = 2 / (Math.sqrt(q0 * q0 + 4) + q0);
    const lam = q0 + gam;
    let w;
    for (;;) {
      w = rng.exponential() / lam;
      if (rng.random() <= Math.exp(-0.5 * (w - gam) * (w - gam))) break;
    }
    const Q = q0 + w, V = w * (w + 2 * q0);
    return 2 * T * V / (mu * (Q * Math.sqrt(V + (x + y) * (x + y) / T) + V + (x + y) * b / T));
  }

  function bridge_given_positions(rng, x, y, T, mu) {
    return Skeleton.bridge(rng, x, y, local_time_given_positions(rng, x, y, T, mu), T, mu);
  }

  const api = { makeRng, erfcx, ndtr, ig_cdf, passage_bridge_cdf, inverse_gaussian, passage_bridge, bessel_bridge,
    forward, middle_local_time, middle_position, middle, Skeleton, local_time_given_positions, bridge_given_positions };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.StickySampler = api;
})(typeof self !== 'undefined' ? self : this);
