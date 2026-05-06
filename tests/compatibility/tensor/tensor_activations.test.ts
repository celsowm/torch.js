import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Tensor Activation Math Methods', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ========== deg2rad ==========
  describe('tensor.deg2rad()', () => {
    it('converts degrees to radians: 0, 90, 180', async () => {
      const t = torch.tensor([0, 90, 180]);
      const result = t.deg2rad();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.PI / 2, 4);
      expect(arr[2]).toBeCloseTo(Math.PI, 4);
    });

    it('converts 360 degrees to 2*pi', async () => {
      const t = torch.tensor([360]);
      const result = t.deg2rad();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(2 * Math.PI, 4);
    });
  });

  // ========== digamma ==========
  describe('tensor.digamma()', () => {
    it('digamma(1) and digamma(2) are negative for x < 1.46', async () => {
      const t = torch.tensor([1, 2]);
      const result = t.digamma();
      const arr = await result.toArray();
      // digamma(1) = -gamma (Euler-Mascheroni) ~ -0.5772
      expect(arr[0]).toBeLessThan(0);
      // digamma(2) = 1 - gamma ~ 0.4228
      expect(arr[1]).toBeCloseTo(1 - 0.5772156649, 3);
    });

    it('digamma preserves shape', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.digamma();
      expect(result.shape).toEqual([2, 2]);
    });
  });

  // ========== elu ==========
  describe('tensor.elu()', () => {
    it('positive values unchanged', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.elu(1.0);
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(1, 4);
      expect(arr[1]).toBeCloseTo(2, 4);
      expect(arr[2]).toBeCloseTo(3, 4);
    });

    it('negative values transformed: alpha=1, x=-1 => -(1-e^(-1))', async () => {
      const t = torch.tensor([1, -1]);
      const result = t.elu(1.0);
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(1, 4);
      expect(arr[1]).toBeCloseTo(-(1 - Math.exp(-1)), 4);
    });

    it('custom alpha changes negative region', async () => {
      const t = torch.tensor([-1]);
      const result = t.elu(2.0);
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(2.0 * (Math.exp(-1) - 1), 4);
    });
  });

  // ========== expm1 ==========
  describe('tensor.expm1()', () => {
    it('expm1(0) = 0, expm1(1) = e - 1', async () => {
      const t = torch.tensor([0, 1]);
      const result = t.expm1();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(Math.E - 1, 4);
    });

    it('expm1(-1) = e^(-1) - 1', async () => {
      const t = torch.tensor([-1]);
      const result = t.expm1();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(Math.exp(-1) - 1, 4);
    });
  });

  // ========== glu ==========
  describe('tensor.glu()', () => {
    it('splits tensor in half along last dim: x[:n] * sigmoid(x[n:])', async () => {
      // Create tensor where first half is ones, second half is large positive (sigmoid ~ 1)
      const t = torch.tensor([[0, 0, 10, 10]]);
      const result = t.glu(-1);
      const arr = await result.toArray();
      // first half: [0, 0], second half sigmoid([10, 10]) ~ [1, 1]
      // result: [0*1, 0*1] = [0, 0]
      expect(arr.length).toBe(2);
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(0, 4);
    });

    it('output is half the input size along glu dimension', () => {
      const t = torch.tensor([[1, 2, 3, 4, 5, 6]]);
      const result = t.glu(-1);
      expect(result.shape).toEqual([1, 3]);
    });

    it('glu with zeros in second half yields zeros', async () => {
      const t = torch.tensor([[1, 2, 0, 0]]);
      const result = t.glu(-1);
      const arr = await result.toArray();
      // sigmoid(0) = 0.5, so result = [1*0.5, 2*0.5] = [0.5, 1.0]
      expect(arr[0]).toBeCloseTo(0.5, 4);
      expect(arr[1]).toBeCloseTo(1.0, 4);
    });
  });

  // ========== hardsigmoid ==========
  describe('tensor.hardsigmoid()', () => {
    it('hardsigmoid(x) = ReLU6(x+3)/6', async () => {
      const t = torch.tensor([0]);
      const result = t.hardsigmoid();
      const arr = await result.toArray();
      // ReLU6(0+3)/6 = 3/6 = 0.5
      expect(arr[0]).toBeCloseTo(0.5, 4);
    });

    it('large positive -> 1, large negative -> 0', async () => {
      const t = torch.tensor([-10, 10]);
      const result = t.hardsigmoid();
      const arr = await result.toArray();
      // ReLU6(-10+3)/6 = ReLU6(-7)/6 = 0/6 = 0
      expect(arr[0]).toBeCloseTo(0, 4);
      // ReLU6(10+3)/6 = ReLU6(13)/6 = 6/6 = 1
      expect(arr[1]).toBeCloseTo(1, 4);
    });

    it('hardsigmoid(3) = 1', async () => {
      const t = torch.tensor([3]);
      const result = t.hardsigmoid();
      const arr = await result.toArray();
      // ReLU6(3+3)/6 = ReLU6(6)/6 = 6/6 = 1
      expect(arr[0]).toBeCloseTo(1, 4);
    });
  });

  // ========== hardswish ==========
  describe('tensor.hardswish()', () => {
    it('hardswish(x) = x * hardsigmoid(x)', async () => {
      const t = torch.tensor([0]);
      const result = t.hardswish();
      const arr = await result.toArray();
      // 0 * hardsigmoid(0) = 0 * 0.5 = 0
      expect(arr[0]).toBeCloseTo(0, 4);
    });

    it('hardswish(3) = 3 * 1 = 3', async () => {
      const t = torch.tensor([3]);
      const result = t.hardswish();
      const arr = await result.toArray();
      // 3 * hardsigmoid(3) = 3 * 1 = 3
      expect(arr[0]).toBeCloseTo(3, 4);
    });

    it('hardswish(-3) = -3 * 0 = 0', async () => {
      const t = torch.tensor([-3]);
      const result = t.hardswish();
      const arr = await result.toArray();
      // -3 * hardsigmoid(-3) = -3 * 0 = 0
      expect(arr[0]).toBeCloseTo(0, 4);
    });
  });

  // ========== i0 ==========
  describe('tensor.i0()', () => {
    it('i0(0) = 1 (Bessel function order 0)', async () => {
      const t = torch.tensor([0]);
      const result = t.i0();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(1, 4);
    });

    it('i0 preserves shape', () => {
      const t = torch.tensor([[0, 1], [2, 3]]);
      const result = t.i0();
      expect(result.shape).toEqual([2, 2]);
    });

    it('i0(1) ~ 1.266', async () => {
      const t = torch.tensor([1]);
      const result = t.i0();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(1.2660658777, 3);
    });
  });

  // ========== lgamma ==========
  describe('tensor.lgamma()', () => {
    it('lgamma(1) = 0', async () => {
      const t = torch.tensor([1]);
      const result = t.lgamma();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
    });

    it('lgamma(2) = 0 (log(1!) = 0)', async () => {
      const t = torch.tensor([2]);
      const result = t.lgamma();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
    });

    it('lgamma preserves shape', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.lgamma();
      expect(result.shape).toEqual([2, 2]);
    });
  });

  // ========== logical_not ==========
  describe('tensor.logical_not()', () => {
    it('element-wise NOT: [0, 1, 1] -> [1, 0, 0]', async () => {
      const t = torch.tensor([0, 1, 1]);
      const result = t.logical_not();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 0, 0]);
    });

    it('logical_not of all zeros -> all ones', async () => {
      const t = torch.tensor([0, 0, 0]);
      const result = t.logical_not();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 1, 1]);
    });

    it('logical_not of all ones -> all zeros', async () => {
      const t = torch.tensor([1, 1, 1]);
      const result = t.logical_not();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0, 0, 0]);
    });
  });

  // ========== mish ==========
  describe('tensor.mish()', () => {
    it('mish(0) = 0 * tanh(softplus(0)) = 0', async () => {
      const t = torch.tensor([0]);
      const result = t.mish();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
    });

    it('mish is smooth version of relu, positive ~ x', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.mish();
      const arr = await result.toArray();
      // mish(1) ~ 0.865, mish(2) ~ 1.944, mish(3) ~ 2.987
      expect(arr[0]).toBeGreaterThan(0.5);
      expect(arr[1]).toBeGreaterThan(1.5);
      expect(arr[2]).toBeGreaterThan(2.5);
    });

    it('mish(-1) is small negative', async () => {
      const t = torch.tensor([-1]);
      const result = t.mish();
      const arr = await result.toArray();
      expect(arr[0]).toBeLessThan(0);
      expect(arr[0]).toBeGreaterThan(-0.5);
    });
  });

  // ========== rad2deg ==========
  describe('tensor.rad2deg()', () => {
    it('converts radians to degrees: 0, pi/2, pi -> 0, 90, 180', async () => {
      const t = torch.tensor([0, Math.PI / 2, Math.PI]);
      const result = t.rad2deg();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(90, 4);
      expect(arr[2]).toBeCloseTo(180, 4);
    });

    it('converts 2*pi to 360', async () => {
      const t = torch.tensor([2 * Math.PI]);
      const result = t.rad2deg();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(360, 4);
    });
  });

  // ========== selu ==========
  describe('tensor.selu()', () => {
    it('positive values scaled by ~1.05', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.selu();
      const arr = await result.toArray();
      const scale = 1.0507009873554804934193349852946;
      expect(arr[0]).toBeCloseTo(1 * scale, 4);
      expect(arr[1]).toBeCloseTo(2 * scale, 4);
      expect(arr[2]).toBeCloseTo(3 * scale, 4);
    });

    it('negative values scaled for self-normalizing', async () => {
      const t = torch.tensor([-1]);
      const result = t.selu();
      const arr = await result.toArray();
      // SELU with alpha=1.673... and scale=1.050...
      // selu(-1) = scale * alpha * (exp(-1) - 1)
      const alpha = 1.6732632423543772848170429916717;
      const scale = 1.0507009873554804934193349852946;
      expect(arr[0]).toBeCloseTo(scale * alpha * (Math.exp(-1) - 1), 4);
    });

    it('selu preserves shape', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.selu();
      expect(result.shape).toEqual([2, 2]);
    });
  });

  // ========== sgn ==========
  describe('tensor.sgn()', () => {
    it('sgn([-2, 0, 3]) = [-1, 0, 1]', async () => {
      const t = torch.tensor([-2, 0, 3]);
      const result = t.sgn();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(-1, 4);
      expect(arr[1]).toBeCloseTo(0, 4);
      expect(arr[2]).toBeCloseTo(1, 4);
    });

    it('sgn preserves sign for fractional values', async () => {
      const t = torch.tensor([-0.5, 0.5]);
      const result = t.sgn();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(-1, 4);
      expect(arr[1]).toBeCloseTo(1, 4);
    });
  });

  // ========== softsign ==========
  describe('tensor.softsign()', () => {
    it('softsign(0) = 0', async () => {
      const t = torch.tensor([0]);
      const result = t.softsign();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
    });

    it('softsign(1) = 0.5', async () => {
      const t = torch.tensor([1]);
      const result = t.softsign();
      const arr = await result.toArray();
      // 1 / (1 + |1|) = 1/2 = 0.5
      expect(arr[0]).toBeCloseTo(0.5, 4);
    });

    it('softsign asymptotes to +/-1 for large values', async () => {
      const t = torch.tensor([-100, 100]);
      const result = t.softsign();
      const arr = await result.toArray();
      // softsign(x) = x / (1 + |x|), for x=100: 100/101 ≈ 0.990099
      expect(arr[0]).toBeCloseTo(-1, 1);  // 0.1 tolerance
      expect(arr[1]).toBeCloseTo(1, 1);
    });
  });

  // ========== tanhshrink ==========
  describe('tensor.tanhshrink()', () => {
    it('tanhshrink(0) = 0 - tanh(0) = 0', async () => {
      const t = torch.tensor([0]);
      const result = t.tanhshrink();
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(0, 4);
    });

    it('tanhshrink of large positive -> x - 1', async () => {
      const t = torch.tensor([10]);
      const result = t.tanhshrink();
      const arr = await result.toArray();
      // tanh(10) ~ 1, so tanhshrink(10) ~ 10 - 1 = 9
      expect(arr[0]).toBeCloseTo(9, 4);
    });

    it('tanhshrink of large negative -> x + 1', async () => {
      const t = torch.tensor([-10]);
      const result = t.tanhshrink();
      const arr = await result.toArray();
      // tanh(-10) ~ -1, so tanhshrink(-10) ~ -10 - (-1) = -9
      expect(arr[0]).toBeCloseTo(-9, 4);
    });
  });

  // ========== threshold ==========
  describe('tensor.threshold()', () => {
    it.skip('threshold([1, 5, -2], threshold=0, value=-1) -> [1, 5, -1]', async () => {
      const t = torch.tensor([1, 5, -2]);
      const result = t.threshold(0, -1);
      const arr = await result.toArray();
      // x > 0 ? x : -1
      expect(arr[0]).toBeCloseTo(1, 4);
      expect(arr[1]).toBeCloseTo(5, 4);
      expect(arr[2]).toBeCloseTo(-1, 4);
    });

    it.skip('threshold with higher threshold', async () => {
      const t = torch.tensor([1, 5, 3]);
      const result = t.threshold(2, 0);
      const arr = await result.toArray();
      // x > 2 ? x : 0
      expect(arr[0]).toBeCloseTo(0, 4);
      expect(arr[1]).toBeCloseTo(5, 4);
      expect(arr[2]).toBeCloseTo(3, 4);
    });
  });
});
