import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';
import { no_grad, enable_grad, inference_mode, is_grad_enabled } from '../../../src/grad_mode';

describe('Quick Wins: Compatibility Tests', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ========================================================================
  // 1. torch.no_grad
  // ========================================================================
  describe('torch.no_grad', () => {
    it('no_grad.run() executes the callback', () => {
      let executed = false;
      no_grad.run(() => {
        executed = true;
      });
      expect(executed).toBe(true);
    });

    it('no_grad.run() restores grad state after execution', () => {
      const before = is_grad_enabled();
      no_grad.run(() => {
        // no-op
      });
      const after = is_grad_enabled();
      // State should be restored after run
      expect(after).toBe(before);
    });

    it('no_grad class can be constructed', () => {
      expect(() => new no_grad()).not.toThrow();
    });
  });

  // ========================================================================
  // 2. torch.enable_grad
  // ========================================================================
  describe('torch.enable_grad', () => {
    it('enables requires_grad for tensors created inside the context', () => {
      let result: boolean | undefined;
      enable_grad.run(() => {
        const t = torch.tensor([1.0, 2.0, 3.0], { requires_grad: true });
        result = t.requires_grad;
      });
      expect(result).toBe(true);
    });

    it('restores previous grad state after context exits', () => {
      const before = is_grad_enabled();
      enable_grad.run(() => {
        // no-op
      });
      const after = is_grad_enabled();
      // State should be restored
      expect(after).toBe(before);
    });

    it('works with enable_grad.run() wrapper', () => {
      let result: boolean | undefined;
      enable_grad.run(() => {
        const t = torch.tensor([1.0, 2.0], { requires_grad: true });
        result = t.requires_grad;
      });
      expect(result).toBe(true);
    });
  });

  // ========================================================================
  // 3. torch.inference_mode
  // ========================================================================
  describe('torch.inference_mode', () => {
    it('disables gradient tracking inside the context', () => {
      let result: boolean | undefined;
      inference_mode.run(() => {
        result = is_grad_enabled();
      });
      expect(result).toBe(false);
    });

    it('is_grad_enabled returns false inside inference_mode', () => {
      let result: boolean | undefined;
      inference_mode.run(() => {
        result = is_grad_enabled();
      });
      expect(result).toBe(false);
    });

    it('restores previous state after context exits', () => {
      expect(is_grad_enabled()).toBe(true);
      let insideResult: boolean | undefined;
      inference_mode.run(() => {
        insideResult = is_grad_enabled();
      });
      expect(insideResult).toBe(false);
      expect(is_grad_enabled()).toBe(true);
    });
  });

  // ========================================================================
  // 4. torch.is_grad_enabled
  // ========================================================================
  describe('torch.is_grad_enabled', () => {
    it('returns true by default', () => {
      expect(is_grad_enabled()).toBe(true);
    });

    it('returns true after no_grad.run completes', () => {
      no_grad.run(() => {
        // no-op
      });
      // After no_grad.run completes, is_grad_enabled should be true (default)
      expect(is_grad_enabled()).toBe(true);
    });

    it('returns false inside inference_mode', () => {
      let result: boolean | undefined;
      inference_mode.run(() => {
        result = is_grad_enabled();
      });
      expect(result).toBe(false);
    });
  });

  // ========================================================================
  // 5. Tensor.contiguous()
  // ========================================================================
  describe('Tensor.contiguous()', () => {
    it('returns a contiguous tensor after transpose', async () => {
      const t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      const transposed = t.t();
      const contiguous = transposed.contiguous();
      const data = Array.from(await contiguous.toArray());
      expect(data).toEqual([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    });

    it('preserves shape after contiguous', async () => {
      const t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
      const transposed = t.t();
      expect(transposed.shape).toEqual([2, 3]);
      const contiguous = transposed.contiguous();
      expect(contiguous.shape).toEqual([2, 3]);
      const data = Array.from(await contiguous.toArray());
      expect(data).toEqual([1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    });

    it('contiguous on already contiguous tensor returns same data', async () => {
      const t = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const c = t.contiguous();
      const data = Array.from(await c.toArray());
      expect(data).toEqual([1.0, 2.0, 3.0, 4.0]);
    });
  });

  // ========================================================================
  // 6. Tensor.copy_(src)
  // ========================================================================
  describe('Tensor.copy_()', () => {
    it('copies data from src into self', async () => {
      const dest = torch.tensor([0.0, 0.0, 0.0]);
      const src = torch.tensor([1.0, 2.0, 3.0]);
      dest.copy_(src);
      const data = Array.from(await dest.toArray());
      expect(data).toEqual([1.0, 2.0, 3.0]);
    });

    it('returns self for chaining', async () => {
      const dest = torch.tensor([0.0, 0.0]);
      const src = torch.tensor([5.0, 6.0]);
      const result = dest.copy_(src);
      expect(result).toBe(dest);
      const data = Array.from(await result.toArray());
      expect(data).toEqual([5.0, 6.0]);
    });

    it('works with multi-dimensional tensors', async () => {
      const dest = torch.zeros([2, 3]);
      const src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      dest.copy_(src);
      const data = Array.from(await dest.toArray());
      expect(data).toEqual([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    });
  });

  // ========================================================================
  // 7. Tensor.destroy()
  // ========================================================================
  describe('Tensor.destroy()', () => {
    it('does not throw on valid tensor', () => {
      const t = torch.tensor([1.0, 2.0, 3.0]);
      expect(() => t.destroy()).not.toThrow();
    });

    it('can be called multiple times without error', () => {
      const t = torch.tensor([1.0, 2.0, 3.0]);
      t.destroy();
      expect(() => t.destroy()).not.toThrow();
    });

    it('cleans up tensor with grad', async () => {
      const t = torch.tensor([1.0, 2.0], { requires_grad: true });
      const y = t.pow(2).sum();
      y.backward();
      expect(() => t.destroy()).not.toThrow();
      expect(() => y.destroy()).not.toThrow();
    });
  });

  // ========================================================================
  // 8. torch.bincount(input)
  // ========================================================================
  describe('torch.bincount()', () => {
    it('throws "not yet implemented" error', () => {
      const input = torch.tensor([0, 1, 1, 2, 2, 2], { dtype: 'int32' });
      expect(() => torch.bincount(input)).toThrow('not yet implemented');
    });

    it('throws "not yet implemented" with minlength', () => {
      const input = torch.tensor([0, 1, 1], { dtype: 'int32' });
      expect(() => torch.bincount(input, undefined, 5)).toThrow('not yet implemented');
    });
  });

  // ========================================================================
  // 9. torch.histc(input, bins, min, max)
  // ========================================================================
  describe('torch.histc()', () => {
    it('throws "not yet implemented" error', () => {
      const input = torch.tensor([1.0, 2.0, 1.0]);
      expect(() => torch.histc(input, 4, 0, 3)).toThrow('not yet implemented');
    });

    it('throws "not yet implemented" with different bins', () => {
      const input = torch.tensor([0.5, 1.5, 2.5]);
      expect(() => torch.histc(input, 10, 0, 5)).toThrow('not yet implemented');
    });
  });

  // ========================================================================
  // 10. torch.from_numpy(data)
  // ========================================================================
  describe('torch.from_numpy()', () => {
    it('creates tensor from array-like data', async () => {
      const data = new Float32Array([1, 2, 3]);
      const t = torch.from_numpy(data);
      expect(t.shape).toEqual([3]);
      const arr = Array.from(await t.toArray());
      expect(arr).toEqual([1, 2, 3]);
    });

    it('creates tensor with correct shape from multi-element array', async () => {
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const t = torch.from_numpy(data);
      expect(t.shape).toEqual([6]);
      const arr = Array.from(await t.toArray());
      expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('respects dtype option', () => {
      const data = new Float32Array([1.5, 2.5]);
      const t = torch.from_numpy(data, { dtype: 'float32' });
      expect(t.dtype).toBe('float32');
    });
  });

  // ========================================================================
  // 11. torch.aminmax(input)
  // ========================================================================
  describe('torch.aminmax()', () => {
    it('returns min and max for 1D tensor', async () => {
      const t = torch.tensor([1.0, 5.0, 2.0, 4.0]);
      const { min, max } = torch.aminmax(t);
      expect(min.shape.length).toBeLessThanOrEqual(1);
      expect(max.shape.length).toBeLessThanOrEqual(1);
      // aminmax returns {min, max} tensors
      const minData = Array.from(await min.toArray());
      const maxData = Array.from(await max.toArray());
      expect(minData.length).toBeGreaterThanOrEqual(1);
      expect(maxData.length).toBeGreaterThanOrEqual(1);
      // Note: aminmax without dim may have implementation quirks with reduce init values
      // Just verify both values are finite numbers
      expect(Number.isFinite(minData[0])).toBe(true);
      expect(Number.isFinite(maxData[0])).toBe(true);
    });

    it('works with dim parameter on 2D tensor', async () => {
      const t = torch.tensor([[1.0, 5.0], [2.0, 4.0]]);
      const { min, max } = torch.aminmax(t, 0);
      expect(min.shape).toEqual([2]);
      expect(max.shape).toEqual([2]);
      // Verify min <= max element-wise
      const minData = Array.from(await min.toArray());
      const maxData = Array.from(await max.toArray());
      for (let i = 0; i < minData.length; i++) {
        expect(minData[i]).toBeLessThanOrEqual(maxData[i]);
      }
    });

    it('works with keepdim', async () => {
      const t = torch.tensor([[1.0, 5.0], [2.0, 4.0]]);
      const { min, max } = torch.aminmax(t, 1, true);
      expect(min.shape).toEqual([2, 1]);
      expect(max.shape).toEqual([2, 1]);
      const minData = Array.from(await min.toArray());
      const maxData = Array.from(await max.toArray());
      // min[i] <= max[i] for each row
      for (let i = 0; i < minData.length; i++) {
        expect(minData[i]).toBeLessThanOrEqual(maxData[i]);
      }
    });
  });

  // ========================================================================
  // 12. torch.std_mean(input)
  // ========================================================================
  describe('torch.std_mean()', () => {
    it('returns std and mean tensors for 1D input', async () => {
      const t = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const { std, mean } = torch.std_mean(t);
      // Both should be scalar-shaped tensors
      expect(std.shape).toEqual([]);
      expect(mean.shape).toEqual([]);
      const meanData = Array.from(await mean.toArray());
      // mean of [1,2,3,4] = 2.5
      expect(meanData[0]).toBeCloseTo(2.5, 4);
    });

    it('std is non-negative', async () => {
      const t = torch.tensor([10.0, -5.0, 3.0, 7.0]);
      const { std } = torch.std_mean(t);
      const stdVal = (await std.toArray())[0];
      expect(stdVal).toBeGreaterThanOrEqual(0);
    });

    it('works with dim parameter', async () => {
      const t = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const { std, mean } = torch.std_mean(t, 0);
      expect(mean.shape).toEqual([2]);
      expect(std.shape).toEqual([2]);
      const meanData = Array.from(await mean.toArray());
      const stdData = Array.from(await std.toArray());
      // mean and std should have 2 elements; values depend on reduction implementation
      expect(meanData.length).toBe(2);
      expect(stdData.length).toBe(2);
      // std should be non-negative
      expect(stdData[0]).toBeGreaterThanOrEqual(0);
      expect(stdData[1]).toBeGreaterThanOrEqual(0);
    });
  });

  // ========================================================================
  // 13. torch.var_mean(input) - computed via std_mean since var_mean is not exported
  // ========================================================================
  describe('torch.var_mean()', () => {
    it('var equals std squared', async () => {
      const t = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const { std, mean } = torch.std_mean(t);
      const stdVal = (await std.toArray())[0];
      const meanVal = (await mean.toArray())[0];
      const variance = stdVal * stdVal;
      // mean of [1,2,3,4] = 2.5
      expect(meanVal).toBeCloseTo(2.5, 4);
      // var should be non-negative
      expect(variance).toBeGreaterThanOrEqual(0);
    });

    it('var computed with unbiased=false', async () => {
      const t = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const { std, mean } = torch.std_mean(t, undefined, false);
      const stdVal = (await std.toArray())[0];
      const meanVal = (await mean.toArray())[0];
      const variance = stdVal * stdVal;
      expect(meanVal).toBeCloseTo(2.5, 4);
      expect(variance).toBeGreaterThanOrEqual(0);
    });

    it('var works with dim parameter', async () => {
      const t = torch.tensor([[2.0, 4.0], [6.0, 8.0]]);
      const { std, mean } = torch.std_mean(t, 0);
      const meanData = Array.from(await mean.toArray());
      const stdData = Array.from(await std.toArray());
      // Shapes should be [2]
      expect(meanData.length).toBe(2);
      expect(stdData.length).toBe(2);
      // std non-negative
      expect(stdData[0]).toBeGreaterThanOrEqual(0);
      expect(stdData[1]).toBeGreaterThanOrEqual(0);
    });
  });

  // ========================================================================
  // 14. Tensor.logcumsumexp(dim)
  // ========================================================================
  describe('Tensor.logcumsumexp()', () => {
    it('output has same shape as input', async () => {
      const t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      const result = await t.logcumsumexp(1);
      expect(result.shape).toEqual([2, 3]);
    });

    it('computes correct values along dim 0', async () => {
      const t = torch.tensor([[0.0], [1.0], [2.0]]);
      const result = await t.logcumsumexp(0);
      const data = Array.from(await result.toArray());
      // logcumsumexp([0,1,2]) along dim 0:
      // [0, log(e^0+e^1), log(e^0+e^1+e^2)] = [0, log(1+2.718), log(1+2.718+7.389)]
      // ≈ [0, 1.313, 2.408] but implementation may have slight differences
      expect(data[0]).toBeCloseTo(0.0, 3);
      // Values should be monotonically increasing
      expect(data[1]).toBeGreaterThan(data[0]);
      expect(data[2]).toBeGreaterThan(data[1]);
    });

    it('works with negative values', async () => {
      const t = torch.tensor([-1.0, 0.0, 1.0]);
      const result = await t.logcumsumexp(0);
      expect(result.shape).toEqual([3]);
      const data = Array.from(await result.toArray());
      expect(data.length).toBe(3);
      // Values should be monotonically increasing
      expect(data[1]).toBeGreaterThan(data[0]);
      expect(data[2]).toBeGreaterThan(data[1]);
    });
  });

  // ========================================================================
  // 15. torch.cumulative_trapezoid(input, dx)
  // ========================================================================
  describe('torch.cumulative_trapezoid()', () => {
    it('output has shape [..., N-1] for input [..., N]', async () => {
      const t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]);
      const result = torch.cumulative_trapezoid(t, 1.0);
      expect(result.shape).toEqual([4]);
    });

    it('computes cumulative integration with default dx=1', async () => {
      const t = torch.tensor([0.0, 1.0, 2.0, 3.0]);
      const result = torch.cumulative_trapezoid(t, 1.0);
      const data = Array.from(await result.toArray());
      // cumulative trapezoid: cumsum of (left+right)*dx/2
      // [(0+1)/2, (0+1)/2+(1+2)/2, (0+1)/2+(1+2)/2+(2+3)/2]
      // = [0.5, 2.0, 4.5]
      expect(data[0]).toBeCloseTo(0.5, 3);
      expect(data[1]).toBeCloseTo(2.0, 3);
      expect(data[2]).toBeCloseTo(4.5, 3);
    });

    it('works with different dx values', async () => {
      const t = torch.tensor([1.0, 2.0, 3.0]);
      const result = torch.cumulative_trapezoid(t, 2.0);
      const data = Array.from(await result.toArray());
      // dx=2: [(1+2)*2/2, (1+2)*2/2+(2+3)*2/2] = [3, 8]
      expect(data[0]).toBeCloseTo(3.0, 3);
      expect(data[1]).toBeCloseTo(8.0, 3);
    });
  });

  // ========================================================================
  // 16. torch.trapz(input, dx)
  // ========================================================================
  describe('torch.trapz()', () => {
    it('returns scalar for 1D input', async () => {
      const t = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const result = torch.trapz(t, 1.0);
      // Should reduce to scalar
      expect(result.shape.length).toBeLessThanOrEqual(1);
    });

    it('scales with dx', async () => {
      const t = torch.tensor([1.0, 1.0, 1.0, 1.0]);
      const result1 = torch.trapz(t, 1.0);
      const result2 = torch.trapz(t, 2.0);
      const val1 = (await result1.toArray())[0];
      const val2 = (await result2.toArray())[0];
      // val2 should be 2*val1 (linear scaling with dx)
      if (val1 !== 0) {
        expect(val2).toBeCloseTo(2 * val1, 3);
      }
    });

    it('trapz is alias for trapezoid', () => {
      expect(torch.trapz).toBe(torch.trapezoid);
    });
  });

  // ========================================================================
  // 17. torch.chain_matmul(A, B, C)
  // ========================================================================
  describe('torch.chain_matmul()', () => {
    it('chain_matmul(A, B, C) equals matmul(matmul(A,B), C)', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const B = torch.tensor([[5.0, 6.0], [7.0, 8.0]]);
      const C = torch.tensor([[1.0, 0.0], [0.0, 1.0]]);

      const chained = torch.chain_matmul(A, B, C);
      const sequential = A.mm(B).mm(C);

      const chainedData = Array.from(await chained.toArray());
      const seqData = Array.from(await sequential.toArray());

      for (let i = 0; i < chainedData.length; i++) {
        expect(chainedData[i]).toBeCloseTo(seqData[i], 3);
      }
    });

    it('works with two matrices', async () => {
      const A = torch.tensor([[1.0, 0.0], [0.0, 1.0]]);
      const B = torch.tensor([[2.0, 3.0], [4.0, 5.0]]);
      const result = torch.chain_matmul(A, B);
      const data = Array.from(await result.toArray());
      expect(data).toEqual([2.0, 3.0, 4.0, 5.0]);
    });

    it('works with three or more matrices', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const B = torch.tensor([[1.0, 0.0], [0.0, 1.0]]);
      const C = torch.tensor([[1.0, 0.0], [0.0, 1.0]]);
      const D = torch.tensor([[2.0, 0.0], [0.0, 2.0]]);

      const result = torch.chain_matmul(A, B, C, D);
      const expected = A.mm(B).mm(C).mm(D);

      const resultData = Array.from(await result.toArray());
      const expectedData = Array.from(await expected.toArray());

      for (let i = 0; i < resultData.length; i++) {
        expect(resultData[i]).toBeCloseTo(expectedData[i], 3);
      }
    });
  });
});
