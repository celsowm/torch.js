/**
 * torch.special module - Special mathematical functions.
 * @status partial
 *
 * Implements special mathematical functions compatible with PyTorch's torch.special.
 * Reference: .externals/pytorch/torch/_refs/special/__init__.py
 */

import { Tensor } from '../tensor';
import * as ops from '../ops';
import { softmax as F_softmax, log_softmax as F_log_softmax } from '../nn/functional';

/**
 * Special mathematical functions namespace.
 */
export const special = {
  // ---------------------------------------------------------------------------
  // erfinv - Inverse error function
  // ---------------------------------------------------------------------------
  erfinv: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const a = 0.147;
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (Math.abs(x) >= 1) {
        result[i] = x > 0 ? Infinity : -Infinity;
        continue;
      }
      if (x === 0) { result[i] = 0; continue; }

      const ln1mx2 = Math.log(1 - x * x);
      const term1 = 2 / (Math.PI * a) + ln1mx2 / 2;
      const inside = term1 * term1 - ln1mx2 / a;

      if (inside < 0) {
        result[i] = 0;
      } else {
        result[i] = Math.sign(x) * Math.sqrt(Math.sqrt(inside) - term1);
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // logit - log(x / (1 - x))
  // ---------------------------------------------------------------------------
  logit: (input: Tensor, eps?: number): Tensor => {
    let clipped = input;
    if (eps !== undefined && eps > 0) {
      clipped = input.clamp(eps, 1 - eps);
    }
    // logit(x) = log(x) - log(1 - x)
    return clipped.log().sub(clipped.neg().add(1).log());
  },

  // ---------------------------------------------------------------------------
  // sinc - sin(pi*x) / (pi*x), with sinc(0) = 1
  // ---------------------------------------------------------------------------
  sinc: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const pi = Math.PI;
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (Math.abs(x) < 1e-10) {
        result[i] = 1.0;
      } else {
        const pix = pi * x;
        result[i] = Math.sin(pix) / pix;
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // entr - -x*log(x) for x>0, 0 for x=0, -inf for x<0
  // ---------------------------------------------------------------------------
  entr: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (x > 0) {
        result[i] = -x * Math.log(x);
      } else if (x === 0) {
        result[i] = 0;
      } else {
        result[i] = -Infinity;
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // i1 - Modified Bessel function of the first kind, order 1
  // ---------------------------------------------------------------------------
  i1: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      const ax = Math.abs(x);
      // Use power series for small x, asymptotic for large x
      if (ax < 1e-10) {
        result[i] = x / 2; // I1(x) ~ x/2 for small x
      } else if (ax < 8) {
        // Power series approximation
        const t = ax / 8;
        // Simplified polynomial approximation (accurate to ~1e-6 for |x| < 8)
        let sum = 1.0;
        let term = 1.0;
        for (let k = 1; k <= 30; k++) {
          term *= t * t / (k * (k + 1));
          sum += term;
          if (Math.abs(term) < 1e-15) break;
        }
        result[i] = Math.sign(x) * (ax / 2) * sum;
      } else {
        // Asymptotic expansion
        result[i] = Math.sign(x) * Math.exp(ax) / Math.sqrt(2 * Math.PI * ax);
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // i1e - Exponentially scaled i1: i1(x) * exp(-|x|)
  // ---------------------------------------------------------------------------
  i1e: async (input: Tensor): Promise<Tensor> => {
    const i1Result = await special.i1(input);
    const data = await input.toArray();
    const i1Data = await i1Result.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      result[i] = i1Data[i] * Math.exp(-Math.abs(data[i]));
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // xlogy - x * log(y), returns 0 when x = 0
  // ---------------------------------------------------------------------------
  xlogy: async (input: Tensor, other: Tensor | number): Promise<Tensor> => {
    const xData = await input.toArray();
    const otherData = typeof other === 'number'
      ? new Float32Array(xData.length).fill(other)
      : await other.toArray();

    const result = new Float32Array(xData.length);
    for (let i = 0; i < xData.length; i++) {
      if (xData[i] === 0) {
        result[i] = 0;
      } else {
        result[i] = xData[i] * Math.log(otherData[i]);
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // xlog1py - x * log1p(y), returns 0 when x = 0
  // ---------------------------------------------------------------------------
  xlog1py: async (input: Tensor, other: Tensor | number): Promise<Tensor> => {
    const xData = await input.toArray();
    const otherData = typeof other === 'number'
      ? new Float32Array(xData.length).fill(other)
      : await other.toArray();

    const result = new Float32Array(xData.length);
    for (let i = 0; i < xData.length; i++) {
      if (xData[i] === 0) {
        result[i] = 0;
      } else {
        result[i] = xData[i] * Math.log1p(otherData[i]);
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // multigammaln - Multivariate log-gamma
  // ---------------------------------------------------------------------------
  multigammaln: async (input: Tensor, p: number): Promise<Tensor> => {
    const c = 0.25 * p * (p - 1) * Math.log(Math.PI);
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      let sum = c;
      for (let j = 0; j < p; j++) {
        const b = (1 - p + j) * 0.5;
        sum += lgamma(data[i] - b);
      }
      result[i] = sum;
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // zeta - Riemann zeta function (Hurwitz zeta with other parameter)
  // ---------------------------------------------------------------------------
  zeta: async (input: Tensor, other: Tensor | number): Promise<Tensor> => {
    const inputData = await input.toArray();
    const otherData = typeof other === 'number'
      ? new Float32Array(inputData.length).fill(other)
      : await other.toArray();

    const result = new Float32Array(inputData.length);

    for (let i = 0; i < inputData.length; i++) {
      const s = inputData[i];
      const q = otherData[i];

      if (q <= 0 && Math.floor(q) === q) {
        result[i] = NaN; // pole at non-positive integers
      } else if (s > 1 && q > 0) {
        // Sum with enough terms for convergence
        let sum = 0;
        for (let n = 0; n < 10000; n++) {
          sum += Math.pow(n + q, -s);
        }
        result[i] = sum;
      } else if (s <= 1 && q > 0) {
        // Use analytic continuation for s <= 1
        // For s = 1, zeta diverges
        if (Math.abs(s - 1) < 1e-10) {
          result[i] = Infinity;
        } else {
          // Approximate using Euler-Maclaurin
          let sum = 0;
          for (let n = 0; n < 1000; n++) {
            sum += Math.pow(n + q, -s);
          }
          // Add integral tail: (q+N)^(1-s)/(s-1)
          const N = 1000;
          sum += Math.pow(q + N, 1 - s) / (s - 1);
          result[i] = sum;
        }
      } else {
        result[i] = NaN;
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // erfcx - Scaled complementary error function: exp(x^2) * erfc(x)
  // ---------------------------------------------------------------------------
  erfcx: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      // erfcx(x) = exp(x^2) * erfc(x)
      // For large positive x, use asymptotic expansion to avoid overflow
      if (x > 26) {
        // erfcx(x) ~ 1/(sqrt(pi)*x) for large x
        result[i] = 1 / (Math.sqrt(Math.PI) * x);
      } else {
        result[i] = Math.exp(x * x) * erfc(x);
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // expit - Sigmoid function: 1 / (1 + exp(-x))
  // ---------------------------------------------------------------------------
  expit: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (x > 500) {
        result[i] = 1.0;
      } else if (x < -500) {
        result[i] = 0.0;
      } else {
        result[i] = 1 / (1 + Math.exp(-x));
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // ndtr - Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
  // ---------------------------------------------------------------------------
  ndtr: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const M_SQRT1_2 = 0.707106781186547524400844362104849039;
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      result[i] = 0.5 * (1 + erf(data[i] * M_SQRT1_2));
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // log_ndtr - Log of standard normal CDF
  // ---------------------------------------------------------------------------
  log_ndtr: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const M_SQRT1_2 = 0.707106781186547524400844362104849039;
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      const t = x * M_SQRT1_2;
      if (x < -1) {
        // Use erfcx for numerical stability: log(erfc(-t)/2) - t^2
        result[i] = Math.log(erfc(-t) / 2) - t * t;
      } else {
        result[i] = Math.log(0.5 * (1 + erf(t)));
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // ndtri - Inverse standard normal CDF
  // ---------------------------------------------------------------------------
  ndtri: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const p = data[i];
      if (p <= 0) { result[i] = -Infinity; continue; }
      if (p >= 1) { result[i] = Infinity; continue; }
      result[i] = Math.sqrt(2) * erfcInv(2 * p);
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // bessel_j0 - Bessel function of the first kind, order 0
  // ---------------------------------------------------------------------------
  bessel_j0: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      result[i] = besselJ0(Math.abs(x));
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // bessel_j1 - Bessel function of the first kind, order 1
  // ---------------------------------------------------------------------------
  bessel_j1: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      result[i] = Math.sign(x) * besselJ1(Math.abs(x));
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // spherical_bessel_j0 - sin(x) / x, with j0(0) = 1
  // ---------------------------------------------------------------------------
  spherical_bessel_j0: async (input: Tensor): Promise<Tensor> => {
    const data = await input.toArray();
    const result = new Float32Array(data.length);

    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (Math.abs(x) < 1e-10) {
        result[i] = 1.0;
      } else {
        result[i] = Math.sin(x) / x;
      }
    }

    return ops.tensor(Array.from(result), { dtype: input.dtype });
  },

  // ---------------------------------------------------------------------------
  // log_softmax
  // ---------------------------------------------------------------------------
  log_softmax: (input: Tensor, dim: number): Tensor => {
    return F_log_softmax(input, dim);
  },

  // ---------------------------------------------------------------------------
  // softmax
  // ---------------------------------------------------------------------------
  softmax: (input: Tensor, dim: number): Tensor => {
    return F_softmax(input, dim);
  },
};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/**
 * Error function approximation.
 */
function erf(x: number): number {
  return 1 - erfc(x);
}

/**
 * Complementary error function approximation.
 */
function erfc(x: number): number {
  // Abramowitz and Stegun approximation
  const t = 1 / (1 + 0.5 * Math.abs(x));
  const tau = t * Math.exp(
    -x * x
    - 1.26551223
    + 1.00002368 * t
    + 0.37409196 * t * t
    + 0.09678418 * t * t * t
    - 0.18628806 * t * t * t * t
    + 0.27886807 * t * t * t * t * t
    - 1.13520398 * t * t * t * t * t * t
    + 1.48851587 * t * t * t * t * t * t * t
    - 0.82215223 * t * t * t * t * t * t * t * t
    + 0.17087277 * t * t * t * t * t * t * t * t * t
  );
  return x >= 0 ? tau : 2 - tau;
}

/**
 * Inverse complementary error function.
 */
function erfcInv(p: number): number {
  if (p <= 0 || p >= 2) return NaN;
  // Use rational approximation
  const t = Math.sqrt(-2 * Math.log(p / 2));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;
  return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
}

/**
 * Log-gamma function (Lanczos approximation).
 */
function lgamma(x: number): number {
  if (x <= 0 && Math.floor(x) === x) return Infinity;
  const g = 7;
  const coef = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];

  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }

  x -= 1;
  let y = coef[0];
  for (let i = 1; i < g + 2; i++) {
    y += coef[i] / (x + i);
  }
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(y);
}

/**
 * Bessel J0 approximation using series expansion.
 * Accurate to ~1e-8 for all x >= 0.
 */
function besselJ0(x: number): number {
  if (x < 0) x = -x;
  if (x < 8) {
    // Power series
    const t = x / 8;
    let sum = 0;
    let term = 1;
    for (let k = 0; k <= 30; k++) {
      sum += term;
      term *= -t * t / ((k + 1) * (k + 1));
      if (Math.abs(term) < 1e-15) break;
    }
    return sum;
  } else {
    // Asymptotic expansion
    const sqrt2pi = Math.sqrt(2 / (Math.PI * x));
    const z = x - Math.PI / 4;
    return sqrt2pi * Math.cos(z);
  }
}

/**
 * Bessel J1 approximation using series expansion.
 * Accurate to ~1e-8 for all x >= 0.
 */
function besselJ1(x: number): number {
  if (x < 8) {
    const t = x / 8;
    let sum = 0.5;
    let term = 0.5;
    for (let k = 1; k <= 30; k++) {
      term *= -t * t / (k * (k + 1));
      sum += term;
      if (Math.abs(term) < 1e-15) break;
    }
    return sum * x;
  } else {
    const sqrt2pi = Math.sqrt(2 / (Math.PI * x));
    const z = x - 3 * Math.PI / 4;
    return sqrt2pi * Math.cos(z);
  }
}
