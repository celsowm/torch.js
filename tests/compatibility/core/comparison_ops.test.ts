import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Comparison Operations', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.eq()
  // ==========================================================================
  describe('eq', () => {
    it('element-wise equality', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([1, 5, 3]);
      const r = torch.eq(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0, 1]);
    });

    it('equality with broadcasting', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([1, 2]);
      const r = torch.eq(a, b);
      expect(r.shape).toEqual([2, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 1, 0, 0]);
    });
  });

  // ==========================================================================
  // torch.ne()
  // ==========================================================================
  describe('ne', () => {
    it('element-wise not equal', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([1, 5, 3]);
      const r = torch.ne(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 1, 0]);
    });

    it('not_equal alias', async () => {
      const a = torch.tensor([1, 2]);
      const b = torch.tensor([1, 3]);
      const r = torch.not_equal(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 1]);
    });
  });

  // ==========================================================================
  // torch.lt()
  // ==========================================================================
  describe('lt', () => {
    it('element-wise less than', async () => {
      const a = torch.tensor([1, 5, 3]);
      const b = torch.tensor([2, 5, 1]);
      const r = torch.lt(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0, 0]);
    });

    it('less alias', async () => {
      const a = torch.tensor([1, 5]);
      const b = torch.tensor([2, 3]);
      const r = torch.less(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0]);
    });
  });

  // ==========================================================================
  // torch.le()
  // ==========================================================================
  describe('le', () => {
    it('element-wise less than or equal', async () => {
      const a = torch.tensor([1, 5, 3]);
      const b = torch.tensor([2, 5, 1]);
      const r = torch.le(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 1, 0]);
    });

    it('less_equal alias', async () => {
      const a = torch.tensor([1, 5, 3]);
      const b = torch.tensor([2, 5, 1]);
      const r = torch.less_equal(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 1, 0]);
    });
  });

  // ==========================================================================
  // torch.gt()
  // ==========================================================================
  describe('gt', () => {
    it('element-wise greater than', async () => {
      const a = torch.tensor([3, 5, 1]);
      const b = torch.tensor([2, 5, 1]);
      const r = torch.gt(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0, 0]);
    });

    it('greater alias', async () => {
      const a = torch.tensor([3, 5, 1]);
      const b = torch.tensor([2, 5, 1]);
      const r = torch.greater(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0, 0]);
    });
  });

  // ==========================================================================
  // torch.ge()
  // ==========================================================================
  describe('ge', () => {
    it('element-wise greater than or equal', async () => {
      const a = torch.tensor([3, 5, 1]);
      const b = torch.tensor([2, 5, 1]);
      const r = torch.ge(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 1, 1]);
    });

    it('greater_equal alias', async () => {
      const a = torch.tensor([3, 5, 1]);
      const b = torch.tensor([2, 5, 1]);
      const r = torch.greater_equal(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 1, 1]);
    });
  });

  // ==========================================================================
  // torch.equal()
  // ==========================================================================
  describe('equal', () => {
    it('returns true for identical tensors', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([1, 2, 3]);
      const r = await torch.equal(a, b);
      expect(r).toBe(true);
    });

    it('returns false for different tensors', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([1, 2, 4]);
      const r = await torch.equal(a, b);
      expect(r).toBe(false);
    });

    it('returns false for different shapes', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([[1, 2, 3]]);
      const r = await torch.equal(a, b);
      expect(r).toBe(false);
    });
  });

  // ==========================================================================
  // torch.isclose()
  // ==========================================================================
  describe('isclose', () => {
    it('detects close values with default tolerance', async () => {
      const a = torch.tensor([1.0, 1.0001, 1.1]);
      const b = torch.tensor([1.0, 1.0001, 1.2]);
      const r = torch.isclose(a, b, 1e-4, 1e-8);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
      expect(arr[1]).toBe(1);
      expect(arr[2]).toBe(0);
    });

    it('handles NaN with equal_nan', async () => {
      const a = torch.tensor([1, NaN, 3]);
      const b = torch.tensor([1, NaN, 3]);
      const r = torch.isclose(a, b, 1e-5, 1e-8, true);
      const arr = await r.toArray();
      expect(arr[1]).toBe(1);
    });

    it('handles different tolerances', async () => {
      const a = torch.tensor([1.0, 1.1]);
      const b = torch.tensor([1.0, 1.05]);
      const r = torch.isclose(a, b, 0.1, 0.01);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
      expect(arr[1]).toBe(1);
    });
  });

  // ==========================================================================
  // torch.allclose()
  // ==========================================================================
  describe('allclose', () => {
    it('returns true for close tensors', async () => {
      const a = torch.tensor([1.0, 1.0001, 0.9999]);
      const b = torch.tensor([1.0, 1.0001, 0.9999]);
      const r = await torch.allclose(a, b);
      expect(r).toBe(true);
    });

    it('returns false for different tensors', async () => {
      const a = torch.tensor([1.0, 2.0, 3.0]);
      const b = torch.tensor([1.0, 2.0, 4.0]);
      const r = await torch.allclose(a, b, 1e-5, 1e-8);
      expect(r).toBe(false);
    });

    it('handles NaN with equal_nan', async () => {
      const a = torch.tensor([1, NaN, 3]);
      const b = torch.tensor([1, NaN, 3]);
      const r = await torch.allclose(a, b, 1e-5, 1e-8, true);
      expect(r).toBe(true);
    });
  });

  // ==========================================================================
  // torch.isinf()
  // ==========================================================================
  describe('isinf', () => {
    it('detects infinite values', async () => {
      const t = torch.tensor([1, Infinity, -Infinity, 0, NaN]);
      const r = torch.isinf(t);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 1, 1, 0, 0]);
    });

    it('returns all zeros for finite tensor', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.isinf(t);
      const arr = await r.toArray();
      expect(arr.every((v: number) => v === 0)).toBe(true);
    });
  });

  // ==========================================================================
  // torch.isfinite()
  // ==========================================================================
  describe('isfinite', () => {
    it('detects finite values', async () => {
      const t = torch.tensor([1, Infinity, -Infinity, 0, NaN]);
      const r = torch.isfinite(t);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0, 0, 1, 0]);
    });

    it('returns all ones for finite tensor', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.isfinite(t);
      const arr = await r.toArray();
      expect(arr.every((v: number) => v === 1)).toBe(true);
    });
  });

  // ==========================================================================
  // torch.isposinf()
  // ==========================================================================
  describe('isposinf', () => {
    it('detects positive infinity', async () => {
      const t = torch.tensor([1, Infinity, -Infinity, 0]);
      const r = torch.isposinf(t);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 1, 0, 0]);
    });
  });

  // ==========================================================================
  // torch.isneginf()
  // ==========================================================================
  describe('isneginf', () => {
    it('detects negative infinity', async () => {
      const t = torch.tensor([1, Infinity, -Infinity, 0]);
      const r = torch.isneginf(t);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 0, 1, 0]);
    });
  });

  // ==========================================================================
  // torch.bitwise_and()
  // ==========================================================================
  describe('bitwise_and', () => {
    it('computes bitwise AND', async () => {
      const a = torch.tensor([0b1100, 0b1010, 0b1111], { dtype: 'int32' });
      const b = torch.tensor([0b1010, 0b1100, 0b1001], { dtype: 'int32' });
      const r = torch.bitwise_and(a, b);
      const arr = await r.toArray();
      // 1100 & 1010 = 1000 = 8
      // 1010 & 1100 = 1000 = 8
      // 1111 & 1001 = 1001 = 9
      expect(arr).toEqual([8, 8, 9]);
    });

    it('bitwise AND with boolean tensors', async () => {
      const a = torch.tensor([true, true, false, false], { dtype: 'bool' });
      const b = torch.tensor([true, false, true, false], { dtype: 'bool' });
      const r = torch.bitwise_and(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0, 0, 0]);
    });
  });

  // ==========================================================================
  // torch.bitwise_or()
  // ==========================================================================
  describe('bitwise_or', () => {
    it('computes bitwise OR', async () => {
      const a = torch.tensor([0b1100, 0b1010], { dtype: 'int32' });
      const b = torch.tensor([0b1010, 0b0101], { dtype: 'int32' });
      const r = torch.bitwise_or(a, b);
      const arr = await r.toArray();
      // 1100 | 1010 = 1110 = 14
      // 1010 | 0101 = 1111 = 15
      expect(arr).toEqual([14, 15]);
    });

    it('bitwise OR with boolean tensors', async () => {
      const a = torch.tensor([true, true, false, false], { dtype: 'bool' });
      const b = torch.tensor([true, false, true, false], { dtype: 'bool' });
      const r = torch.bitwise_or(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 1, 1, 0]);
    });
  });

  // ==========================================================================
  // torch.bitwise_xor()
  // ==========================================================================
  describe('bitwise_xor', () => {
    it('computes bitwise XOR', async () => {
      const a = torch.tensor([0b1100, 0b1010], { dtype: 'int32' });
      const b = torch.tensor([0b1010, 0b1010], { dtype: 'int32' });
      const r = torch.bitwise_xor(a, b);
      const arr = await r.toArray();
      // 1100 ^ 1010 = 0110 = 6
      // 1010 ^ 1010 = 0000 = 0
      expect(arr).toEqual([6, 0]);
    });

    it('bitwise XOR with boolean tensors', async () => {
      const a = torch.tensor([true, true, false, false], { dtype: 'bool' });
      const b = torch.tensor([true, false, true, false], { dtype: 'bool' });
      const r = torch.bitwise_xor(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 1, 1, 0]);
    });
  });

  // ==========================================================================
  // torch.fmax()
  // ==========================================================================
  describe('fmax', () => {
    it('element-wise max, ignores NaN', async () => {
      const a = torch.tensor([1, NaN, 3]);
      const b = torch.tensor([2, 5, NaN]);
      const r = torch.fmax(a, b);
      const arr = await r.toArray();
      expect(arr[0]).toBe(2);
      expect(arr[1]).toBe(5);
      expect(arr[2]).toBe(3);
    });

    it('fmax with both NaN', async () => {
      const a = torch.tensor([1, NaN, 3]);
      const b = torch.tensor([2, NaN, 4]);
      const r = torch.fmax(a, b);
      const arr = await r.toArray();
      // NaN vs NaN should be NaN
      expect(arr[0]).toBe(2);
    });
  });

  // ==========================================================================
  // torch.fmin()
  // ==========================================================================
  describe('fmin', () => {
    it('element-wise min, ignores NaN', async () => {
      const a = torch.tensor([1, NaN, 3]);
      const b = torch.tensor([2, 5, NaN]);
      const r = torch.fmin(a, b);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
      expect(arr[1]).toBe(5);
      expect(arr[2]).toBe(3);
    });

    it('fmin with both NaN', async () => {
      const a = torch.tensor([1, NaN, 3]);
      const b = torch.tensor([2, NaN, 4]);
      const r = torch.fmin(a, b);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
    });
  });
});
