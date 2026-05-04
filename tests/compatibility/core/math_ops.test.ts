import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Math Operations', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.add()
  // ==========================================================================
  describe('add', () => {
    it('adds two tensors element-wise', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.add(a, b);
      const arr = await c.toArray();
      expect(arr).toEqual([5, 7, 9]);
    });

    it('adds scalar to tensor', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.add(t, 10);
      const arr = await r.toArray();
      expect(arr).toEqual([11, 12, 13]);
    });

    it('broadcasts during add', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([10, 20]);
      const c = torch.add(a, b);
      expect(c.shape).toEqual([2, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([11, 22, 13, 24]);
    });
  });

  // ==========================================================================
  // torch.sub()
  // ==========================================================================
  describe('sub', () => {
    it('subtracts two tensors', async () => {
      const a = torch.tensor([5, 7, 9]);
      const b = torch.tensor([1, 2, 3]);
      const c = torch.sub(a, b);
      const arr = await c.toArray();
      expect(arr).toEqual([4, 5, 6]);
    });

    it('subtracts scalar from tensor', async () => {
      const t = torch.tensor([10, 20, 30]);
      const r = torch.sub(t, 5);
      const arr = await r.toArray();
      expect(arr).toEqual([5, 15, 25]);
    });
  });

  // ==========================================================================
  // torch.mul()
  // ==========================================================================
  describe('mul', () => {
    it('multiplies two tensors element-wise', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.mul(a, b);
      const arr = await c.toArray();
      expect(arr).toEqual([4, 10, 18]);
    });

    it('multiplies by scalar', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.mul(t, 3);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 6, 9]);
    });
  });

  // ==========================================================================
  // torch.div()
  // ==========================================================================
  describe('div', () => {
    it('divides two tensors element-wise', async () => {
      const a = torch.tensor([4, 10, 18]);
      const b = torch.tensor([2, 5, 9]);
      const c = torch.div(a, b);
      const arr = await c.toArray();
      expect(arr).toEqual([2, 2, 2]);
    });

    it('divides by scalar', async () => {
      const t = torch.tensor([10, 20, 30]);
      const r = torch.div(t, 5);
      const arr = await r.toArray();
      expect(arr).toEqual([2, 4, 6]);
    });
  });

  // ==========================================================================
  // torch.abs()
  // ==========================================================================
  describe('abs', () => {
    it('computes absolute value', async () => {
      const t = torch.tensor([-1, -2, 3, -4]);
      const r = torch.abs(t);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3, 4]);
    });

    it('abs of positive is unchanged', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.abs(t);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.ceil()
  // ==========================================================================
  describe('ceil', () => {
    it('computes ceiling', async () => {
      const t = torch.tensor([1.2, 1.7, -0.5, 3.0]);
      const r = torch.ceil(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(2);
      expect(arr[1]).toBe(2);
      expect(arr[2]).toBe(0);
      expect(arr[3]).toBe(3);
    });
  });

  // ==========================================================================
  // torch.floor()
  // ==========================================================================
  describe('floor', () => {
    it('computes floor', async () => {
      const t = torch.tensor([1.2, 1.7, -0.5, 3.0]);
      const r = torch.floor(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
      expect(arr[1]).toBe(1);
      expect(arr[2]).toBe(-1);
      expect(arr[3]).toBe(3);
    });
  });

  // ==========================================================================
  // torch.round()
  // ==========================================================================
  describe('round', () => {
    it('rounds to nearest integer', async () => {
      const t = torch.tensor([1.2, 1.5, 1.7, 2.5]);
      const r = torch.round(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
      expect(arr[1]).toBe(2);
      expect(arr[2]).toBe(2);
    });
  });

  // ==========================================================================
  // torch.trunc()
  // ==========================================================================
  describe('trunc', () => {
    it('truncates toward zero', async () => {
      const t = torch.tensor([1.7, -1.7, 3.0]);
      const r = torch.trunc(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
      expect(arr[1]).toBe(-1);
      expect(arr[2]).toBe(3);
    });
  });

  // ==========================================================================
  // torch.frac()
  // ==========================================================================
  describe('frac', () => {
    it('computes fractional part', async () => {
      const t = torch.tensor([1.5, -1.5, 3.0]);
      const r = torch.frac(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0.5);
      expect(arr[1]).toBeCloseTo(-0.5);
      expect(arr[2]).toBeCloseTo(0);
    });
  });

  // ==========================================================================
  // torch.exp()
  // ==========================================================================
  describe('exp', () => {
    it('computes e^x', async () => {
      const t = torch.tensor([0, 1, 2]);
      const r = torch.exp(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(1);
      expect(arr[1]).toBeCloseTo(Math.E);
      expect(arr[2]).toBeCloseTo(Math.E * Math.E);
    });
  });

  // ==========================================================================
  // torch.exp2()
  // ==========================================================================
  describe('exp2', () => {
    it('computes 2^x', async () => {
      const t = torch.tensor([0, 1, 2, 3]);
      const r = torch.exp2(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(1);
      expect(arr[1]).toBeCloseTo(2);
      expect(arr[2]).toBeCloseTo(4);
      expect(arr[3]).toBeCloseTo(8);
    });
  });

  // ==========================================================================
  // torch.log()
  // ==========================================================================
  describe('log', () => {
    it('computes natural log', async () => {
      const t = torch.tensor([1, Math.E, Math.E * Math.E]);
      const r = torch.log(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo(1);
      expect(arr[2]).toBeCloseTo(2);
    });
  });

  // ==========================================================================
  // torch.log2()
  // ==========================================================================
  describe('log2', () => {
    it('computes log base 2', async () => {
      const t = torch.tensor([1, 2, 4, 8]);
      const r = torch.log2(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo(1);
      expect(arr[2]).toBeCloseTo(2);
      expect(arr[3]).toBeCloseTo(3);
    });
  });

  // ==========================================================================
  // torch.log10()
  // ==========================================================================
  describe('log10', () => {
    it('computes log base 10', async () => {
      const t = torch.tensor([1, 10, 100]);
      const r = torch.log10(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo(1);
      expect(arr[2]).toBeCloseTo(2);
    });
  });

  // ==========================================================================
  // torch.log1p()
  // ==========================================================================
  describe('log1p', () => {
    it('computes log(1 + x)', async () => {
      const t = torch.tensor([0, Math.E - 1]);
      const r = torch.log1p(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo(1);
    });
  });

  // ==========================================================================
  // torch.sqrt()
  // ==========================================================================
  describe('sqrt', () => {
    it('computes square root', async () => {
      const t = torch.tensor([0, 1, 4, 9]);
      const r = torch.sqrt(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo(1);
      expect(arr[2]).toBeCloseTo(2);
      expect(arr[3]).toBeCloseTo(3);
    });
  });

  // ==========================================================================
  // torch.rsqrt()
  // ==========================================================================
  describe('rsqrt', () => {
    it('computes 1/sqrt(x)', async () => {
      const t = torch.tensor([1, 4]);
      const r = torch.rsqrt(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(1);
      expect(arr[1]).toBeCloseTo(0.5);
    });
  });

  // ==========================================================================
  // torch.square()
  // ==========================================================================
  describe('square', () => {
    it('computes x^2', async () => {
      const t = torch.tensor([1, 2, 3, -4]);
      const r = torch.square(t);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 4, 9, 16]);
    });
  });

  // ==========================================================================
  // torch.reciprocal()
  // ==========================================================================
  describe('reciprocal', () => {
    it('computes 1/x', async () => {
      const t = torch.tensor([1, 2, 4]);
      const r = torch.reciprocal(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(1);
      expect(arr[1]).toBeCloseTo(0.5);
      expect(arr[2]).toBeCloseTo(0.25);
    });
  });

  // ==========================================================================
  // torch.pow()
  // ==========================================================================
  describe('pow', () => {
    it('computes x^exponent', async () => {
      const t = torch.tensor([2, 3, 4]);
      const r = torch.pow(t, 2);
      const arr = await r.toArray();
      expect(arr).toEqual([4, 9, 16]);
    });

    it('works with float exponent', async () => {
      const t = torch.tensor([4, 9, 16]);
      const r = torch.pow(t, 0.5);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(2);
      expect(arr[1]).toBeCloseTo(3);
      expect(arr[2]).toBeCloseTo(4);
    });
  });

  // ==========================================================================
  // torch.sin()
  // ==========================================================================
  describe('sin', () => {
    it('computes sine', async () => {
      const t = torch.tensor([0, Math.PI / 2, Math.PI]);
      const r = torch.sin(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(1, 4);
      expect(arr[2]).toBeCloseTo(0, 4);
    });
  });

  // ==========================================================================
  // torch.cos()
  // ==========================================================================
  describe('cos', () => {
    it('computes cosine', async () => {
      const t = torch.tensor([0, Math.PI / 2, Math.PI]);
      const r = torch.cos(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(1, 4);
      expect(arr[1]).toBeCloseTo(0, 4);
      expect(arr[2]).toBeCloseTo(-1, 4);
    });
  });

  // ==========================================================================
  // torch.tan()
  // ==========================================================================
  describe('tan', () => {
    it('computes tangent', async () => {
      const t = torch.tensor([0, Math.PI / 4]);
      const r = torch.tan(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(1, 4);
    });
  });

  // ==========================================================================
  // torch.asin()
  // ==========================================================================
  describe('asin', () => {
    it('computes arcsine', async () => {
      const t = torch.tensor([0, 0.5, 1]);
      const r = torch.asin(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.PI / 6, 3);
      expect(arr[2]).toBeCloseTo(Math.PI / 2, 4);
    });
  });

  // ==========================================================================
  // torch.acos()
  // ==========================================================================
  describe('acos', () => {
    it('computes arccosine', async () => {
      const t = torch.tensor([0, 0.5, 1]);
      const r = torch.acos(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(Math.PI / 2, 3);
      expect(arr[1]).toBeCloseTo(Math.PI / 3, 3);
      expect(arr[2]).toBeCloseTo(0, 4);
    });
  });

  // ==========================================================================
  // torch.atan()
  // ==========================================================================
  describe('atan', () => {
    it('computes arctangent', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.atan(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.PI / 4, 4);
      expect(arr[2]).toBeCloseTo(-Math.PI / 4, 4);
    });
  });

  // ==========================================================================
  // torch.sinh()
  // ==========================================================================
  describe('sinh', () => {
    it('computes hyperbolic sine', async () => {
      const t = torch.tensor([0, 1]);
      const r = torch.sinh(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo((Math.E - 1 / Math.E) / 2, 4);
    });
  });

  // ==========================================================================
  // torch.cosh()
  // ==========================================================================
  describe('cosh', () => {
    it('computes hyperbolic cosine', async () => {
      const t = torch.tensor([0, 1]);
      const r = torch.cosh(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(1);
      expect(arr[1]).toBeCloseTo((Math.E + 1 / Math.E) / 2, 4);
    });
  });

  // ==========================================================================
  // torch.tanh()
  // ==========================================================================
  describe('tanh', () => {
    it('computes hyperbolic tangent', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.tanh(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo(Math.tanh(1), 4);
      expect(arr[2]).toBeCloseTo(-Math.tanh(1), 4);
    });
  });

  // ==========================================================================
  // torch.asinh()
  // ==========================================================================
  describe('asinh', () => {
    it('computes inverse hyperbolic sine', async () => {
      const t = torch.tensor([0, 1]);
      const r = torch.asinh(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.asinh(1), 4);
    });
  });

  // ==========================================================================
  // torch.acosh()
  // ==========================================================================
  describe('acosh', () => {
    it('computes inverse hyperbolic cosine', async () => {
      const t = torch.tensor([1, 2]);
      const r = torch.acosh(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.acosh(2), 4);
    });
  });

  // ==========================================================================
  // torch.atanh()
  // ==========================================================================
  describe('atanh', () => {
    it('computes inverse hyperbolic tangent', async () => {
      const t = torch.tensor([0, 0.5]);
      const r = torch.atanh(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.atanh(0.5), 4);
    });
  });

  // ==========================================================================
  // torch.sigmoid()
  // ==========================================================================
  describe('sigmoid', () => {
    it('computes sigmoid function', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.sigmoid(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0.5, 4);
      expect(arr[1]).toBeCloseTo(1 / (1 + Math.exp(-1)), 4);
      expect(arr[2]).toBeCloseTo(1 / (1 + Math.exp(1)), 4);
    });

    it('sigmoid of large positive approaches 1', async () => {
      const t = torch.tensor([100]);
      const r = torch.sigmoid(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeGreaterThan(0.99);
    });

    it('sigmoid of large negative approaches 0', async () => {
      const t = torch.tensor([-100]);
      const r = torch.sigmoid(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeLessThan(0.01);
    });
  });

  // ==========================================================================
  // torch.relu()
  // ==========================================================================
  describe('relu', () => {
    it('computes ReLU', async () => {
      const t = torch.tensor([-1, 0, 1, 2]);
      const r = torch.relu(t);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 0, 1, 2]);
    });

    it('ReLU of all positive is identity', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.relu(t);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.gelu()
  // ==========================================================================
  describe('gelu', () => {
    it('computes GELU', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.gelu(t);
      const arr = await r.toArray();
      // GELU(0) should be close to 0
      expect(arr[0]).toBeCloseTo(0, 3);
      // GELU(1) should be positive
      expect(arr[1]).toBeGreaterThan(0);
      // GELU(-1) should be small negative
      expect(arr[2]).toBeLessThan(0);
    });
  });

  // ==========================================================================
  // torch.softplus()
  // ==========================================================================
  describe('softplus', () => {
    it('computes softplus', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.softplus(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(Math.log(2), 3);
      expect(arr[1]).toBeCloseTo(Math.log(1 + Math.exp(1)), 3);
      expect(arr[2]).toBeCloseTo(Math.log(1 + Math.exp(-1)), 3);
    });
  });

  // ==========================================================================
  // torch.silu()
  // ==========================================================================
  describe('silu', () => {
    it('computes SiLU (x * sigmoid(x))', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.silu(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 3);
      expect(arr[1]).toBeCloseTo(1 * (1 / (1 + Math.exp(-1))), 3);
      expect(arr[2]).toBeCloseTo(-1 * (1 / (1 + Math.exp(1))), 3);
    });
  });

  // ==========================================================================
  // torch.neg()
  // ==========================================================================
  describe('neg', () => {
    it('negates tensor', async () => {
      const t = torch.tensor([1, -2, 0, 3]);
      const r = torch.neg(t);
      const arr = await r.toArray();
      expect(arr).toEqual([-1, 2, 0, -3]);
    });
  });

  // ==========================================================================
  // torch.sign()
  // ==========================================================================
  describe('sign', () => {
    it('computes sign', async () => {
      const t = torch.tensor([-5, 0, 3]);
      const r = torch.sign(t);
      const arr = await r.toArray();
      expect(arr).toEqual([-1, 0, 1]);
    });
  });

  // ==========================================================================
  // torch.erf()
  // ==========================================================================
  describe('erf', () => {
    it('computes error function', async () => {
      const t = torch.tensor([0, 1, -1]);
      const r = torch.erf(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(0.8427, 3);
      expect(arr[2]).toBeCloseTo(-0.8427, 3);
    });
  });

  // ==========================================================================
  // torch.erfc()
  // ==========================================================================
  describe('erfc', () => {
    it('computes complementary error function', async () => {
      const t = torch.tensor([0, 1]);
      const r = torch.erfc(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(1, 4);
      expect(arr[1]).toBeCloseTo(1 - 0.8427, 3);
    });
  });

  // ==========================================================================
  // torch.isnan()
  // ==========================================================================
  describe('isnan', () => {
    it('detects NaN values', async () => {
      const t = torch.tensor([1, NaN, 3, NaN]);
      const r = torch.isnan(t);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 1, 0, 1]);
    });

    it('non-NaN values are false', async () => {
      const t = torch.tensor([0, 1, -1, 100]);
      const r = torch.isnan(t);
      const arr = await r.toArray();
      expect(arr.every((v: number) => v === 0)).toBe(true);
    });
  });

  // ==========================================================================
  // torch.clip() / torch.clamp()
  // ==========================================================================
  describe('clip', () => {
    it('clips values to range', async () => {
      const t = torch.tensor([1, 5, 10, 15]);
      const r = torch.clip(t, 3, 12);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 5, 10, 12]);
    });

    it('clips with only min', async () => {
      const t = torch.tensor([-5, 0, 5, 10]);
      const r = torch.clip(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 0, 5, 10]);
    });

    it('clips with only max', async () => {
      const t = torch.tensor([-5, 0, 5, 10]);
      const r = torch.clip(t, undefined, 5);
      const arr = await r.toArray();
      expect(arr).toEqual([-5, 0, 5, 5]);
    });
  });

  describe('clamp', () => {
    it('clamps values to range', async () => {
      const t = torch.tensor([1, 5, 10, 15]);
      const r = torch.clamp(t, 3, 12);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 5, 10, 12]);
    });

    it('clamp with only min', async () => {
      const t = torch.tensor([-5, 0, 5, 10]);
      const r = torch.clamp(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 0, 5, 10]);
    });
  });

  describe('clamp_min', () => {
    it('clamps minimum values', async () => {
      const t = torch.tensor([-5, 0, 5, 10]);
      const r = torch.clamp_min(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([0, 0, 5, 10]);
    });
  });

  describe('clamp_max', () => {
    it('clamps maximum values', async () => {
      const t = torch.tensor([-5, 0, 5, 10]);
      const r = torch.clamp_max(t, 5);
      const arr = await r.toArray();
      expect(arr).toEqual([-5, 0, 5, 5]);
    });
  });

  // ==========================================================================
  // torch.fmod()
  // ==========================================================================
  describe('fmod', () => {
    it('computes remainder with sign of dividend', async () => {
      const t = torch.tensor([5.5, -5.5, 5.5, -5.5]);
      const r = torch.fmod(t, 2);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(1.5);
      expect(arr[1]).toBeCloseTo(-1.5);
    });
  });

  // ==========================================================================
  // torch.remainder()
  // ==========================================================================
  describe('remainder', () => {
    it('computes remainder always positive', async () => {
      const t = torch.tensor([5, -5, 5, -5]);
      const r = torch.remainder(t, 3);
      const arr = await r.toArray();
      expect(arr[0]).toBeCloseTo(2);
      expect(arr[1]).toBeCloseTo(1); // -5 mod 3 = 1
    });
  });

  // ==========================================================================
  // torch.maximum()
  // ==========================================================================
  describe('maximum', () => {
    it('element-wise maximum', async () => {
      const a = torch.tensor([1, 5, 3]);
      const b = torch.tensor([4, 2, 6]);
      const r = torch.maximum(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([4, 5, 6]);
    });

    it('maximum with broadcasting', async () => {
      const a = torch.tensor([1, 5, 3]);
      const b = torch.tensor(4);
      const r = torch.maximum(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([4, 5, 4]);
    });
  });

  // ==========================================================================
  // torch.minimum()
  // ==========================================================================
  describe('minimum', () => {
    it('element-wise minimum', async () => {
      const a = torch.tensor([1, 5, 3]);
      const b = torch.tensor([4, 2, 6]);
      const r = torch.minimum(a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3]);
    });
  });
});
