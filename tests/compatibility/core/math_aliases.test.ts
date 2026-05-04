import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Math Aliases and Less Common Functions', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.absolute() — alias of abs
  // ==========================================================================
  describe('absolute', () => {
    it('computes absolute value (alias of abs)', async () => {
      const t = torch.tensor([-1, -2, 3]);
      const r = torch.absolute(t);
      expect(Array.from(await r.toArray())).toEqual([1, 2, 3]);
    });

    it('absolute of positive is unchanged', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.absolute(t);
      expect(Array.from(await r.toArray())).toEqual([1, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.arccos() — alias of acos
  // ==========================================================================
  describe('arccos', () => {
    it('arccos([1, 0, -1]) → [0, pi/2, pi]', async () => {
      const t = torch.tensor([1, 0, -1]);
      const r = torch.arccos(t);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.PI / 2, 4);
      expect(arr[2]).toBeCloseTo(Math.PI, 4);
    });

    it('arccos(0.5) ≈ pi/3', async () => {
      const t = torch.tensor([0.5]);
      const r = torch.arccos(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(Math.PI / 3, 4);
    });
  });

  // ==========================================================================
  // torch.arccosh() — alias of acosh
  // ==========================================================================
  describe('arccosh', () => {
    it('arccosh([1, 2, 3]) returns values >= 0', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.arccosh(t);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr.every((v: number) => v >= 0)).toBe(true);
    });

    it('arccosh(1) = 0', async () => {
      const t = torch.tensor([1]);
      const r = torch.arccosh(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(0, 4);
    });
  });

  // ==========================================================================
  // torch.arcsin() — alias of asin
  // ==========================================================================
  describe('arcsin', () => {
    it('arcsin([0, 1, -1]) → [0, pi/2, -pi/2]', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.arcsin(t);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.PI / 2, 4);
      expect(arr[2]).toBeCloseTo(-Math.PI / 2, 4);
    });

    it('arcsin(0.5) ≈ pi/6', async () => {
      const t = torch.tensor([0.5]);
      const r = torch.arcsin(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(Math.PI / 6, 4);
    });
  });

  // ==========================================================================
  // torch.arcsinh() — alias of asinh
  // ==========================================================================
  describe('arcsinh', () => {
    it('arcsinh(0) = 0', async () => {
      const t = torch.tensor([0]);
      const r = torch.arcsinh(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(0, 4);
    });

    it('arcsinh(1) ≈ asinh(1)', async () => {
      const t = torch.tensor([1]);
      const r = torch.arcsinh(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(Math.asinh(1), 4);
    });
  });

  // ==========================================================================
  // torch.arctan() — alias of atan
  // ==========================================================================
  describe('arctan', () => {
    it('arctan([0, 1, -1]) → [0, pi/4, -pi/4]', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.arctan(t);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.PI / 4, 4);
      expect(arr[2]).toBeCloseTo(-Math.PI / 4, 4);
    });

    it('arctan is inverse of tan: arctan(tan(x)) ≈ x', async () => {
      const t = torch.tensor([0.5]);
      const r = torch.arctan(torch.tan(t));
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(0.5, 4);
    });
  });

  // ==========================================================================
  // torch.arctan2() — alias of atan2
  // ==========================================================================
  describe('arctan2', () => {
    it('arctan2([1, 0], [0, 1]) → [pi/2, 0]', async () => {
      const y = torch.tensor([1, 0]);
      const x = torch.tensor([0, 1]);
      const r = torch.arctan2(y, x);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(Math.PI / 2, 4);
      expect(arr[1]).toBeCloseTo(0, 4);
    });

    it('arctan2([1, -1], [1, 1]) → [pi/4, -pi/4]', async () => {
      const y = torch.tensor([1, -1]);
      const x = torch.tensor([1, 1]);
      const r = torch.arctan2(y, x);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(Math.PI / 4, 4);
      expect(arr[1]).toBeCloseTo(-Math.PI / 4, 4);
    });
  });

  // ==========================================================================
  // torch.arctanh() — alias of atanh
  // ==========================================================================
  describe('arctanh', () => {
    it('arctanh(0) = 0', async () => {
      const t = torch.tensor([0]);
      const r = torch.arctanh(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(0, 4);
    });

    it('arctanh(0.5) ≈ atanh(0.5)', async () => {
      const t = torch.tensor([0.5]);
      const r = torch.arctanh(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(Math.atanh(0.5), 4);
    });
  });

  // ==========================================================================
  // torch.atan2() — arctan(y/x) with quadrant awareness
  // ==========================================================================
  describe('atan2', () => {
    it('atan2([1, 0], [0, 1]) → [pi/2, 0]', async () => {
      const y = torch.tensor([1, 0]);
      const x = torch.tensor([0, 1]);
      const r = torch.atan2(y, x);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(Math.PI / 2, 4);
      expect(arr[1]).toBeCloseTo(0, 4);
    });

    it('atan2 handles all four quadrants', async () => {
      const y = torch.tensor([1, -1, -1, 1]);
      const x = torch.tensor([1, 1, -1, -1]);
      const r = torch.atan2(y, x);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(Math.PI / 4, 4);
      expect(arr[1]).toBeCloseTo(-Math.PI / 4, 4);
      expect(arr[2]).toBeCloseTo(-3 * Math.PI / 4, 3);
      expect(arr[3]).toBeCloseTo(3 * Math.PI / 4, 3);
    });
  });

  // ==========================================================================
  // torch.deg2rad() — degrees to radians
  // ==========================================================================
  describe('deg2rad', () => {
    it('deg2rad([0, 90, 180]) → [0, pi/2, pi]', async () => {
      const t = torch.tensor([0, 90, 180]);
      const r = torch.deg2rad(t);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.PI / 2, 4);
      expect(arr[2]).toBeCloseTo(Math.PI, 4);
    });

    it('deg2rad(360) = 2*pi', async () => {
      const t = torch.tensor([360]);
      const r = torch.deg2rad(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(2 * Math.PI, 4);
    });
  });

  // ==========================================================================
  // torch.digamma() — derivative of log gamma
  // ==========================================================================
  describe('digamma', () => {
    it('digamma([1, 2]) — negative for x < ~1.46', async () => {
      const t = torch.tensor([1, 2]);
      const r = torch.digamma(t);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeLessThan(0);
      expect(arr[1]).toBeGreaterThan(0);
    });

    it('digamma(1) = -euler_mascheroni ≈ -0.5772', async () => {
      const t = torch.tensor([1]);
      const r = torch.digamma(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(-0.5772, 3);
    });
  });

  // ==========================================================================
  // torch.expm1() — exp(x) - 1, more precise for small x
  // ==========================================================================
  describe('expm1', () => {
    it('expm1(0) = 0', async () => {
      const t = torch.tensor([0]);
      const r = torch.expm1(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(0, 4);
    });

    it('expm1(1) = e - 1 ≈ 1.718', async () => {
      const t = torch.tensor([1]);
      const r = torch.expm1(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(Math.E - 1, 4);
    });

    it('expm1(x) ≈ x for very small x (precision advantage)', async () => {
      const t = torch.tensor([1e-10]);
      const r = torch.expm1(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(1e-10, 8);
    });
  });

  // ==========================================================================
  // torch.fix() — alias of trunc (truncate toward zero)
  // ==========================================================================
  describe('fix', () => {
    it('fix([1.7, -2.3]) → [1, -2]', async () => {
      const t = torch.tensor([1.7, -2.3]);
      const r = torch.fix(t);
      expect(Array.from(await r.toArray())).toEqual([1, -2]);
    });

    it('fix of integers is identity', async () => {
      const t = torch.tensor([1, -3, 0, 5]);
      const r = torch.fix(t);
      expect(Array.from(await r.toArray())).toEqual([1, -3, 0, 5]);
    });
  });

  // ==========================================================================
  // torch.i0() — modified Bessel function of first kind, order 0
  // ==========================================================================
  describe('i0', () => {
    it('i0(0) = 1', async () => {
      const t = torch.tensor([0]);
      const r = torch.i0(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(1, 4);
    });

    it('i0 returns positive for real input', async () => {
      const t = torch.tensor([-2, 0, 2]);
      const r = torch.i0(t);
      const arr = Array.from(await r.toArray());
      expect(arr.every((v: number) => v > 0)).toBe(true);
    });
  });

  // ==========================================================================
  // torch.isreal() — element-wise check for real values
  // ==========================================================================
  describe('isreal', () => {
    it('isreal returns all 1s for real tensors', async () => {
      const t = torch.tensor([1, 2.5, -3, 0]);
      const r = torch.isreal(t);
      expect(Array.from(await r.toArray())).toEqual([1, 1, 1, 1]);
    });

    it('isreal returns 1 for scalar float', async () => {
      const t = torch.tensor([3.14]);
      const r = torch.isreal(t);
      expect(Array.from(await r.toArray())).toEqual([1]);
    });
  });

  // ==========================================================================
  // torch.lgamma() — log of absolute gamma function
  // ==========================================================================
  describe('lgamma', () => {
    it('lgamma(1) = 0 (log(0!) = log(1) = 0)', async () => {
      const t = torch.tensor([1]);
      const r = torch.lgamma(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(0, 4);
    });

    it('lgamma(2) = 0 (log(1!) = log(1) = 0)', async () => {
      const t = torch.tensor([2]);
      const r = torch.lgamma(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(0, 4);
    });

    it('lgamma(5) = log(24) ≈ 3.178', async () => {
      const t = torch.tensor([5]);
      const r = torch.lgamma(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(Math.log(24), 4);
    });
  });

  // ==========================================================================
  // torch.logaddexp() — log(exp(x) + exp(y))
  // ==========================================================================
  describe('logaddexp', () => {
    it('logaddexp(0, 0) = log(2) ≈ 0.693', async () => {
      const a = torch.tensor([0]);
      const b = torch.tensor([0]);
      const r = torch.logaddexp(a, b);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(Math.log(2), 4);
    });

    it('logaddexp(x, y) = log(exp(x) + exp(y))', async () => {
      const a = torch.tensor([1]);
      const b = torch.tensor([2]);
      const r = torch.logaddexp(a, b);
      const expected = Math.log(Math.exp(1) + Math.exp(2));
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(expected, 4);
    });
  });

  // ==========================================================================
  // torch.logical_not() — element-wise NOT
  // ==========================================================================
  describe('logical_not', () => {
    it('logical_not([0, 1, 1]) → [1, 0, 0]', async () => {
      const t = torch.tensor([0, 1, 1]);
      const r = torch.logical_not(t);
      expect(Array.from(await r.toArray())).toEqual([1, 0, 0]);
    });

    it('logical_not of all zeros is all ones', async () => {
      const t = torch.tensor([0, 0, 0]);
      const r = torch.logical_not(t);
      expect(Array.from(await r.toArray())).toEqual([1, 1, 1]);
    });
  });

  // ==========================================================================
  // torch.rad2deg() — radians to degrees
  // ==========================================================================
  describe('rad2deg', () => {
    it('rad2deg([0, pi/2, pi]) → [0, 90, 180]', async () => {
      const t = torch.tensor([0, Math.PI / 2, Math.PI]);
      const r = torch.rad2deg(t);
      const arr = Array.from(await r.toArray());
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(90, 4);
      expect(arr[2]).toBeCloseTo(180, 4);
    });

    it('rad2deg(2*pi) = 360', async () => {
      const t = torch.tensor([2 * Math.PI]);
      const r = torch.rad2deg(t);
      expect(Array.from(await r.toArray())[0]).toBeCloseTo(360, 4);
    });
  });

  // ==========================================================================
  // torch.sgn() — sign of complex number
  // ==========================================================================
  describe('sgn', () => {
    it('sgn of positive real is 1', async () => {
      const t = torch.tensor([5]);
      const r = torch.sgn(t);
      expect(Array.from(await r.toArray())).toEqual([1]);
    });

    it('sgn of negative real is -1', async () => {
      const t = torch.tensor([-3]);
      const r = torch.sgn(t);
      expect(Array.from(await r.toArray())).toEqual([-1]);
    });

    it('sgn(0) = 0', async () => {
      const t = torch.tensor([0]);
      const r = torch.sgn(t);
      expect(Array.from(await r.toArray())).toEqual([0]);
    });
  });
});
