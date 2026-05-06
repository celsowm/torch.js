import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Tensor Math Methods', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ========== add ==========
  describe('tensor.add()', () => {
    it('adds a scalar to each element', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.add(5);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([6, 7, 8]);
    });

    it('adds two tensors element-wise', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const result = a.add(b);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([5, 7, 9]);
    });

    it('preserves shape and dtype', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.add(1);
      expect(result.shape).toEqual([2, 2]);
      expect(result.dtype).toBe('float32');
    });

    it('works with empty-like single element', async () => {
      const t = torch.tensor([42]);
      const result = t.add(-42);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0]);
    });
  });

  // ========== sub ==========
  describe('tensor.sub()', () => {
    it('subtracts a scalar from each element', async () => {
      const t = torch.tensor([10, 20, 30]);
      const result = t.sub(5);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([5, 15, 25]);
    });

    it('subtracts two tensors element-wise', async () => {
      const a = torch.tensor([5, 7, 9]);
      const b = torch.tensor([1, 2, 3]);
      const result = a.sub(b);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 5, 6]);
    });

    it('handles negative results', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.sub(10);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([-9, -8, -7]);
    });
  });

  // ========== mul ==========
  describe('tensor.mul()', () => {
    it('multiplies each element by a scalar', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.mul(3);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([3, 6, 9]);
    });

    it('multiplies two tensors element-wise', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const result = a.mul(b);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 10, 18]);
    });

    it('multiplying by zero yields zeros', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.mul(0);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0, 0, 0]);
    });
  });

  // ========== div ==========
  describe('tensor.div()', () => {
    it('divides each element by a scalar', async () => {
      const t = torch.tensor([10, 20, 30]);
      const result = t.div(2);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([5, 10, 15]);
    });

    it('divides two tensors element-wise', async () => {
      const a = torch.tensor([10, 20, 30]);
      const b = torch.tensor([2, 4, 5]);
      const result = a.div(b);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([5, 5, 6]);
    });

    it('handles division resulting in fractions', async () => {
      const t = torch.tensor([1, 1, 1]);
      const result = t.div(2);
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0.5, 5);
    });
  });

  // ========== pow ==========
  describe('tensor.pow()', () => {
    it('raises each element to a scalar power', async () => {
      const t = torch.tensor([2, 3, 4]);
      const result = t.pow(2);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 9, 16]);
    });

    it('raises to power 0 yields ones', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.pow(0);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 1, 1]);
    });

    it('raises to power 1 returns same values', async () => {
      const t = torch.tensor([5, 6, 7]);
      const result = t.pow(1);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([5, 6, 7]);
    });

    it('handles fractional powers', async () => {
      const t = torch.tensor([4, 9, 16]);
      const result = t.pow(0.5);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 3, 4]);
    });
  });

  // ========== abs ==========
  describe('tensor.abs()', () => {
    it('computes absolute value of each element', async () => {
      const t = torch.tensor([-3, 0, 5, -1.5]);
      const result = t.abs();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(3, 5);
      expect(arr[1]).toBeCloseTo(0, 5);
      expect(arr[2]).toBeCloseTo(5, 5);
      expect(arr[3]).toBeCloseTo(1.5, 5);
    });

    it('all positive values remain unchanged', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.abs();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3]);
    });
  });

  // ========== ceil ==========
  describe('tensor.ceil()', () => {
    it('rounds up to nearest integer', async () => {
      const t = torch.tensor([1.1, 2.9, -1.5, 3.0]);
      const result = t.ceil();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 3, -1, 3]);
    });
  });

  // ========== floor ==========
  describe('tensor.floor()', () => {
    it('rounds down to nearest integer', async () => {
      const t = torch.tensor([1.9, 2.1, -1.5, 3.0]);
      const result = t.floor();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, -2, 3]);
    });
  });

  // ========== round ==========
  describe('tensor.round()', () => {
    it('rounds to nearest integer', async () => {
      const t = torch.tensor([1.4, 1.5, 2.6, -1.5]);
      const result = t.round();
      const arr = await result.toArray();
      // PyTorch uses round-half-to-even (banker's rounding): -1.5 → -2
      expect(Array.from(arr)).toEqual([1, 2, 3, -2]);
    });
  });

  // ========== trunc ==========
  describe('tensor.trunc()', () => {
    it('truncates toward zero', async () => {
      const t = torch.tensor([1.9, -1.9, 2.1, -2.1]);
      const result = t.trunc();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, -1, 2, -2]);
    });
  });

  // ========== exp ==========
  describe('tensor.exp()', () => {
    it('computes e^x for each element', async () => {
      const t = torch.tensor([0, 1, 2]);
      const result = t.exp();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(1, 4);
      expect(arr[1]).toBeCloseTo(Math.E, 4);
      expect(arr[2]).toBeCloseTo(Math.E ** 2, 4);
    });
  });

  // ========== log ==========
  describe('tensor.log()', () => {
    it('computes natural logarithm', async () => {
      const t = torch.tensor([1, Math.E, Math.E ** 2]);
      const result = t.log();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(1, 4);
      expect(arr[2]).toBeCloseTo(2, 4);
    });

    it('log(1) is 0', async () => {
      const t = torch.tensor([1, 1, 1]);
      const result = await t.log().toArray();
      expect(Array.from(result)).toEqual([0, 0, 0]);
    });
  });

  // ========== sqrt ==========
  describe('tensor.sqrt()', () => {
    it('computes square root', async () => {
      const t = torch.tensor([1, 4, 9, 16]);
      const result = t.sqrt();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3, 4]);
    });

    it('sqrt(0) is 0', async () => {
      const t = torch.tensor([0]);
      const result = t.sqrt();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 5);
    });
  });

  // ========== square ==========
  describe('tensor.square()', () => {
    it('squares each element', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const result = t.square();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 4, 9, 16]);
    });

    it('square of negative is positive', async () => {
      const t = torch.tensor([-3, -2, -1]);
      const result = t.square();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([9, 4, 1]);
    });
  });

  // ========== reciprocal ==========
  describe('tensor.reciprocal()', () => {
    it('computes 1/x for each element', async () => {
      const t = torch.tensor([1, 2, 4]);
      const result = t.reciprocal();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(1, 5);
      expect(arr[1]).toBeCloseTo(0.5, 5);
      expect(arr[2]).toBeCloseTo(0.25, 5);
    });
  });

  // ========== sin ==========
  describe('tensor.sin()', () => {
    it('computes sine of each element', async () => {
      const t = torch.tensor([0, Math.PI / 2, Math.PI]);
      const result = t.sin();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 3);
      expect(arr[1]).toBeCloseTo(1, 3);
      expect(arr[2]).toBeCloseTo(0, 3);
    });
  });

  // ========== cos ==========
  describe('tensor.cos()', () => {
    it('computes cosine of each element', async () => {
      const t = torch.tensor([0, Math.PI / 2, Math.PI]);
      const result = t.cos();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(1, 3);
      expect(arr[1]).toBeCloseTo(0, 3);
      expect(arr[2]).toBeCloseTo(-1, 3);
    });
  });

  // ========== tan ==========
  describe('tensor.tan()', () => {
    it('computes tangent of each element', async () => {
      const t = torch.tensor([0, Math.PI / 4]);
      const result = t.tan();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 3);
      expect(arr[1]).toBeCloseTo(1, 2);
    });
  });

  // ========== sinh ==========
  describe('tensor.sinh()', () => {
    it('computes hyperbolic sine', async () => {
      const t = torch.tensor([0, 1]);
      const result = t.sinh();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.sinh(1), 4);
    });
  });

  // ========== cosh ==========
  describe('tensor.cosh()', () => {
    it('computes hyperbolic cosine', async () => {
      const t = torch.tensor([0, 1]);
      const result = t.cosh();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(1, 4);
      expect(arr[1]).toBeCloseTo(Math.cosh(1), 4);
    });
  });

  // ========== tanh ==========
  describe('tensor.tanh()', () => {
    it('computes hyperbolic tangent', async () => {
      const t = torch.tensor([0, 1, -1]);
      const result = t.tanh();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.tanh(1), 4);
      expect(arr[2]).toBeCloseTo(-Math.tanh(1), 4);
    });
  });

  // ========== sigmoid ==========
  describe('tensor.sigmoid()', () => {
    it('computes sigmoid function', async () => {
      const t = torch.tensor([0, 1, -1]);
      const result = t.sigmoid();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0.5, 4);
      expect(arr[1]).toBeCloseTo(1 / (1 + Math.exp(-1)), 4);
      expect(arr[2]).toBeCloseTo(1 / (1 + Math.exp(1)), 4);
    });

    it('sigmoid output is between 0 and 1', async () => {
      const t = torch.tensor([-100, 0, 100]);
      const result = t.sigmoid();
      const arr = await result.toArray();
      // exp(100) overflows in float32, so sigmoid(-100) ≈ 0
      expect(arr[0]).toBeGreaterThanOrEqual(0);
      expect(arr[0]).toBeLessThan(0.01);
      expect(arr[1]).toBeCloseTo(0.5, 2);
      expect(arr[2]).toBeGreaterThan(0.99);
      expect(arr[2]).toBeLessThanOrEqual(1);
    });
  });

  // ========== relu ==========
  describe('tensor.relu()', () => {
    it('zeros out negative values', async () => {
      const t = torch.tensor([-2, -1, 0, 1, 2]);
      const result = t.relu();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0, 0, 0, 1, 2]);
    });

    it('positive values unchanged', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.relu();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3]);
    });
  });

  // ========== gelu ==========
  describe('tensor.gelu()', () => {
    it('computes GELU activation', async () => {
      const t = torch.tensor([0, 1, -1]);
      const result = t.gelu();
      const arr = await result.toArray();
      // gelu(0) ≈ 0
      expect(arr[0]).toBeCloseTo(0, 2);
      // gelu(1) ≈ 0.84
      expect(arr[1]).toBeGreaterThan(0.5);
      // gelu(-1) ≈ -0.16
      expect(arr[2]).toBeLessThan(0);
    });
  });

  // ========== softmax ==========
  describe('tensor.softmax()', () => {
    it('softmax sums to 1', async () => {
      const t = torch.tensor([[1, 2, 3]]);
      const result = t.softmax(-1);
      const arr = await result.toArray();
      const sum = arr.reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1, 5);
    });

    it('all softmax values are positive', async () => {
      const t = torch.tensor([[1, 2, 3]]);
      const result = t.softmax(-1);
      const arr = await result.toArray();
      for (const v of arr) {
        expect(v).toBeGreaterThan(0);
      }
    });

    it('preserves shape', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.softmax(-1);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  // ========== log_softmax ==========
  describe('tensor.log_softmax()', () => {
    it('log_softmax is log of softmax', async () => {
      const t = torch.tensor([[1, 2, 3]]);
      const logSm = t.log_softmax(-1);
      const sm = t.softmax(-1);
      const logSmArr = await logSm.toArray();
      const smArr = await sm.toArray();
      for (let i = 0; i < logSmArr.length; i++) {
        expect(logSmArr[i]).toBeCloseTo(Math.log(smArr[i]), 4);
      }
    });

    it('preserves shape', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.log_softmax(-1);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  // ========== neg ==========
  describe('tensor.neg()', () => {
    it('negates all elements', async () => {
      const t = torch.tensor([1, -2, 0, 3]);
      const result = t.neg();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([-1, 2, 0, -3]);
    });
  });

  // ========== sign ==========
  describe('tensor.sign()', () => {
    it('returns sign of each element', async () => {
      const t = torch.tensor([5, -3, 0, 1]);
      const result = t.sign();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, -1, 0, 1]);
    });
  });

  // ========== isnan ==========
  describe('tensor.isnan()', () => {
    it('identifies NaN values', async () => {
      // Create tensor with NaN using math operation
      const t = torch.tensor([0]).div(torch.tensor([0]));
      const result = t.isnan();
      const arr = await result.toArray();
      expect(arr[0]).toBe(1);
    });

    it('non-NaN values return false', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.isnan();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0, 0, 0]);
    });
  });

  // ========== isinf ==========
  describe('tensor.isinf()', () => {
    it('identifies infinite values', async () => {
      const t = torch.tensor([1]).div(torch.tensor([0]));
      const result = t.isinf();
      const arr = await result.toArray();
      expect(arr[0]).toBe(1);
    });

    it('finite values return false', async () => {
      const t = torch.tensor([1, -1, 0]);
      const result = t.isinf();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0, 0, 0]);
    });
  });

  // ========== isfinite ==========
  describe('tensor.isfinite()', () => {
    it('identifies finite values', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.isfinite();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 1, 1]);
    });

    it('non-finite values return false', async () => {
      const t = torch.tensor([1]).div(torch.tensor([0]));
      const result = t.isfinite();
      const arr = await result.toArray();
      expect(arr[0]).toBe(0);
    });
  });
});
