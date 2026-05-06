import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Shape & Manipulation APIs', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.atleast_1d()
  // ==========================================================================
  describe('torch.atleast_1d', () => {
    it('converts scalar to 1D', async () => {
      const scalar = torch.tensor(42);
      expect(scalar.dim()).toBe(0);
      const result = torch.atleast_1d(scalar);
      expect(result.shape).toEqual([1]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([42]);
    });

    it('leaves 1D tensor unchanged', async () => {
      const t = torch.tensor([2, 3]);
      const result = torch.atleast_1d(t);
      expect(result.shape).toEqual([2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 3]);
    });

    it('leaves 2D tensor unchanged', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = torch.atleast_1d(t);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  // ==========================================================================
  // torch.atleast_2d()
  // ==========================================================================
  describe('torch.atleast_2d', () => {
    it('converts 1D to 2D', async () => {
      const t = torch.tensor([3]);
      const result = torch.atleast_2d(t);
      expect(result.shape).toEqual([1, 1]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([3]);
    });

    it('leaves 2D tensor unchanged', async () => {
      const t = torch.tensor([[2, 3], [4, 5]]);
      const result = torch.atleast_2d(t);
      expect(result.shape).toEqual([2, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 3, 4, 5]);
    });

    it('converts scalar to 2D', async () => {
      const scalar = torch.tensor(7);
      const result = torch.atleast_2d(scalar);
      expect(result.shape).toEqual([1, 1]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([7]);
    });
  });

  // ==========================================================================
  // torch.atleast_3d()
  // ==========================================================================
  describe('torch.atleast_3d', () => {
    it('converts 2D to 3D', async () => {
      const t = torch.tensor([[2, 3]]);
      const result = torch.atleast_3d(t);
      // PyTorch: 2D (H, W) → 3D (H, W, 1)
      expect(result.shape).toEqual([1, 2, 1]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 3]);
    });

    it('leaves 3D tensor unchanged', () => {
      const t = torch.zeros([2, 3, 4]);
      const result = torch.atleast_3d(t);
      expect(result.shape).toEqual([2, 3, 4]);
    });

    it('converts 1D to 3D with shape [1, N, 1]', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = torch.atleast_3d(t);
      expect(result.shape).toEqual([1, 3, 1]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.cartesian_prod()
  // ==========================================================================
  describe('torch.cartesian_prod', () => {
    it('computes Cartesian product of two 1D tensors', async () => {
      const a = torch.tensor([1, 2]);
      const b = torch.tensor([3, 4]);
      const result = await torch.cartesian_prod(a, b);
      expect(result.shape).toEqual([4, 2]);
      const arr = await result.toArray();
      const rows = [];
      for (let i = 0; i < 4; i++) {
        rows.push([arr[i * 2], arr[i * 2 + 1]]);
      }
      expect(rows).toEqual([[1, 3], [1, 4], [2, 3], [2, 4]]);
    });

    it('computes Cartesian product of three tensors', async () => {
      const a = torch.tensor([1]);
      const b = torch.tensor([2]);
      const c = torch.tensor([3]);
      const result = await torch.cartesian_prod(a, b, c);
      expect(result.shape).toEqual([1, 3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.combinations()
  // ==========================================================================
  describe('torch.combinations', () => {
    it('computes combinations of 3 elements taken 2 at a time', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = await torch.combinations(t, 2);
      expect(result.shape).toEqual([3, 2]);
      const arr = await result.toArray();
      const rows = [];
      for (let i = 0; i < 3; i++) {
        rows.push([arr[i * 2], arr[i * 2 + 1]]);
      }
      expect(rows).toEqual([[1, 2], [1, 3], [2, 3]]);
    });

    it('computes combinations with replacement', async () => {
      const t = torch.tensor([1, 2]);
      const result = await torch.combinations(t, 2, true);
      expect(result.shape).toEqual([3, 2]);
      const arr = await result.toArray();
      const rows = [];
      for (let i = 0; i < 3; i++) {
        rows.push([arr[i * 2], arr[i * 2 + 1]]);
      }
      expect(rows).toEqual([[1, 1], [1, 2], [2, 2]]);
    });

    it('default r=2', async () => {
      const t = torch.tensor([10, 20, 30]);
      const result = await torch.combinations(t);
      expect(result.shape).toEqual([3, 2]);
      const arr = await result.toArray();
      const rows = [];
      for (let i = 0; i < 3; i++) {
        rows.push([arr[i * 2], arr[i * 2 + 1]]);
      }
      expect(rows).toEqual([[10, 20], [10, 30], [20, 30]]);
    });
  });

  // ==========================================================================
  // torch.meshgrid()
  // ==========================================================================
  describe('torch.meshgrid', () => {
    it('creates 2D coordinate grids', async () => {
      const x = torch.tensor([1, 2]);
      const y = torch.tensor([3, 4]);
      const [gx, gy] = torch.meshgrid(x, y);
      expect(gx.shape).toEqual([2, 2]);
      expect(gy.shape).toEqual([2, 2]);
      const gxArr = await gx.toArray();
      const gyArr = await gy.toArray();
      // With 'ij' indexing (default): gx repeats rows, gy repeats columns
      expect(Array.from(gxArr)).toEqual([1, 1, 2, 2]);
      expect(Array.from(gyArr)).toEqual([3, 4, 3, 4]);
    });

    it('creates grids with xy indexing', async () => {
      const x = torch.tensor([1, 2, 3]);
      const y = torch.tensor([4, 5]);
      const [gx, gy] = torch.meshgrid(x, y, { indexing: 'xy' });
      expect(gx.shape).toEqual([2, 3]);
      expect(gy.shape).toEqual([2, 3]);
      const gxArr = await gx.toArray();
      const gyArr = await gy.toArray();
      // With 'xy': first output varies along columns, second along rows
      expect(Array.from(gxArr)).toEqual([1, 2, 3, 1, 2, 3]);
      expect(Array.from(gyArr)).toEqual([4, 4, 4, 5, 5, 5]);
    });
  });

  // ==========================================================================
  // torch.movedim()
  // ==========================================================================
  describe('torch.movedim', () => {
    it('moves dimension 0 to position 2', () => {
      const t = torch.zeros([2, 3, 4]);
      const result = torch.movedim(t, 0, 2);
      expect(result.shape).toEqual([3, 4, 2]);
    });

    it('moves multiple dimensions', () => {
      const t = torch.zeros([2, 3, 4, 5]);
      const result = torch.movedim(t, [0, 1], [2, 3]);
      expect(result.shape).toEqual([4, 5, 2, 3]);
    });

    it('handles negative dimensions', () => {
      const t = torch.zeros([2, 3, 4]);
      const result = torch.movedim(t, -1, 0);
      expect(result.shape).toEqual([4, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.rot90()
  // ==========================================================================
  describe('torch.rot90', () => {
    it('rotates 2D tensor by 90 degrees', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = await torch.rot90(t, 1);
      expect(result.shape).toEqual([2, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 4, 1, 3]);
    });

    it('rotates by 180 degrees (k=2)', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = await torch.rot90(t, 2);
      expect(result.shape).toEqual([2, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 3, 2, 1]);
    });

    it('rotates by 270 degrees (k=3)', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = await torch.rot90(t, 3);
      expect(result.shape).toEqual([2, 2]);
      const arr = await result.toArray();
      // 270 deg = 90 deg counterclockwise
      expect(Array.from(arr)).toEqual([3, 1, 4, 2]);
    });

    it('k=0 returns unchanged', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = await torch.rot90(t, 0);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3, 4]);
    });
  });

  // ==========================================================================
  // torch.swapdims()
  // ==========================================================================
  describe('torch.swapdims', () => {
    it('swaps dimensions 0 and 1', () => {
      const t = torch.zeros([2, 3, 4]);
      const result = torch.swapdims(t, 0, 1);
      expect(result.shape).toEqual([3, 2, 4]);
    });

    it('swaps dimensions in 2D tensor (like transpose)', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = torch.swapdims(t, 0, 1);
      expect(result.shape).toEqual([2, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 3, 2, 4]);
    });
  });

  // ==========================================================================
  // torch.tensor_split()
  // ==========================================================================
  describe('torch.tensor_split', () => {
    it('splits tensor at specified indices', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const chunks = torch.tensor_split(t, [2, 4]);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([2]);
      expect(chunks[1].shape).toEqual([2]);
      expect(chunks[2].shape).toEqual([1]);
      const arr0 = await chunks[0].toArray();
      expect(Array.from(arr0)).toEqual([1, 2]);
      const arr1 = await chunks[1].toArray();
      expect(Array.from(arr1)).toEqual([3, 4]);
      const arr2 = await chunks[2].toArray();
      expect(Array.from(arr2)).toEqual([5]);
    });

    it('splits with single index', async () => {
      const t = torch.tensor([10, 20, 30, 40]);
      const chunks = torch.tensor_split(t, [2]);
      expect(chunks.length).toBe(2);
      expect(chunks[0].shape).toEqual([2]);
      expect(chunks[1].shape).toEqual([2]);
    });

    it('splits 2D tensor along specific dimension', () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const chunks = torch.tensor_split(t, [1], 1);
      expect(chunks.length).toBe(2);
      expect(chunks[0].shape).toEqual([2, 1]);
      expect(chunks[1].shape).toEqual([2, 2]);
    });
  });

  // ==========================================================================
  // torch.trace()
  // ==========================================================================
  describe('torch.trace', () => {
    it('computes trace of 2x2 matrix', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = torch.trace(t);
      const val = (await result.toArray())[0];
      expect(val).toBe(5);
    });

    it('computes trace of 3x3 matrix', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const result = torch.trace(t);
      const val = (await result.toArray())[0];
      expect(val).toBe(15); // 1 + 5 + 9
    });

    it('computes trace of non-square matrix', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const result = torch.trace(t);
      const val = (await result.toArray())[0];
      expect(val).toBe(6); // 1 + 5
    });
  });

  // ==========================================================================
  // torch.unflatten()
  // ==========================================================================
  describe('torch.unflatten', () => {
    it('unflattens a dimension', () => {
      const t = torch.zeros([4, 6]);
      const result = torch.unflatten(t, 1, [2, 3]);
      expect(result.shape).toEqual([4, 2, 3]);
    });

    it('unflattens first dimension', () => {
      const t = torch.zeros([12, 5]);
      const result = torch.unflatten(t, 0, [3, 4]);
      expect(result.shape).toEqual([3, 4, 5]);
    });

    it('throws when product does not match dimension size', () => {
      const t = torch.zeros([4, 6]);
      expect(() => torch.unflatten(t, 1, [2, 4])).toThrow();
    });
  });

  // ==========================================================================
  // torch.unravel_index()
  // ==========================================================================
  describe('torch.unravel_index', () => {
    it('converts flat indices to multi-indices', async () => {
      const indices = torch.tensor([0, 1, 2], { dtype: 'int32' });
      const result = await torch.unravel_index(indices, [2, 3]);
      expect(result.shape).toEqual([3, 2]);
      expect(result.dtype).toBe('int32');
      const arr = await result.toArray();
      const rows = [];
      for (let i = 0; i < 3; i++) {
        rows.push([arr[i * 2], arr[i * 2 + 1]]);
      }
      expect(rows).toEqual([[0, 0], [0, 1], [0, 2]]);
    });

    it('converts larger flat indices', async () => {
      const indices = torch.tensor([3, 4, 5], { dtype: 'int32' });
      const result = await torch.unravel_index(indices, [2, 3]);
      expect(result.shape).toEqual([3, 2]);
      const arr = await result.toArray();
      const rows = [];
      for (let i = 0; i < 3; i++) {
        rows.push([arr[i * 2], arr[i * 2 + 1]]);
      }
      expect(rows).toEqual([[1, 0], [1, 1], [1, 2]]);
    });
  });

  // ==========================================================================
  // torch.row_stack()
  // ==========================================================================
  describe('torch.row_stack', () => {
    it('stacks 1D tensors as rows (alias of vstack)', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const result = torch.row_stack([a, b]);
      expect(result.shape).toEqual([2, 3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('produces same result as vstack', async () => {
      const a = torch.tensor([1, 2]);
      const b = torch.tensor([3, 4]);
      const c = torch.tensor([5, 6]);
      const rowResult = torch.row_stack([a, b, c]);
      const vResult = torch.vstack([a, b, c]);
      const rowArr = await rowResult.toArray();
      const vArr = await vResult.toArray();
      expect(Array.from(rowArr)).toEqual(Array.from(vArr));
    });
  });

  // ==========================================================================
  // Tensor.dim()
  // ==========================================================================
  describe('Tensor.dim()', () => {
    it('returns number of dimensions for 1D tensor', () => {
      const t = torch.tensor([2, 3]);
      expect(t.dim()).toBe(1);
    });

    it('returns number of dimensions for 3D tensor', () => {
      const t = torch.zeros([2, 3, 4]);
      expect(t.dim()).toBe(3);
    });

    it('returns 0 for scalar tensor', () => {
      const t = torch.tensor(42);
      expect(t.dim()).toBe(0);
    });
  });

  // ==========================================================================
  // Tensor.expand()
  // ==========================================================================
  describe('Tensor.expand()', () => {
    it('expands singleton dimension', async () => {
      const t = torch.tensor([5]);
      const result = t.expand([3]);
      expect(result.shape).toEqual([3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([5, 5, 5]);
    });

    it('expands to higher dimension', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.expand([2, 3]);
      expect(result.shape).toEqual([2, 3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3, 1, 2, 3]);
    });

    it('expands scalar-like to 2D', async () => {
      const t = torch.tensor([7]).reshape([]);
      const result = t.expand([2, 3]);
      expect(result.shape).toEqual([2, 3]);
      const arr = await result.toArray();
      expect(Array.from(arr).every((v: number) => v === 7)).toBe(true);
    });
  });

  // ==========================================================================
  // Tensor.movedim() - method version
  // ==========================================================================
  describe('Tensor.movedim()', () => {
    it('moves dimension 0 to position 2', () => {
      const t = torch.zeros([2, 3, 4]);
      const result = t.movedim(0, 2);
      expect(result.shape).toEqual([3, 4, 2]);
    });

    it('moves multiple dimensions', () => {
      const t = torch.zeros([2, 3, 4, 5]);
      const result = t.movedim([0, 1], [2, 3]);
      expect(result.shape).toEqual([4, 5, 2, 3]);
    });
  });

  // ==========================================================================
  // Tensor.rot90() - method version
  // ==========================================================================
  describe('Tensor.rot90()', () => {
    it('rotates 2D tensor by 90 degrees', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = await t.rot90(1);
      expect(result.shape).toEqual([2, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 4, 1, 3]);
    });

    it('rotates by 180 degrees', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = await t.rot90(2);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 3, 2, 1]);
    });
  });

  // ==========================================================================
  // Tensor.swapdims() - method version
  // ==========================================================================
  describe('Tensor.swapdims()', () => {
    it('swaps dimensions 0 and 1', () => {
      const t = torch.zeros([2, 3, 4]);
      const result = t.swapdims(0, 1);
      expect(result.shape).toEqual([3, 2, 4]);
    });

    it('swaps dimensions in 2D tensor', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const result = t.swapdims(0, 1);
      expect(result.shape).toEqual([3, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 4, 2, 5, 3, 6]);
    });
  });

  // ==========================================================================
  // Tensor.unflatten() - method version
  // ==========================================================================
  describe('Tensor.unflatten()', () => {
    it('unflattens a dimension', () => {
      const t = torch.zeros([4, 6]);
      const result = t.unflatten(1, [2, 3]);
      expect(result.shape).toEqual([4, 2, 3]);
    });

    it('unflattens with different sizes', () => {
      const t = torch.zeros([12]);
      const result = t.unflatten(0, [3, 4]);
      expect(result.shape).toEqual([3, 4]);
    });
  });

  // ==========================================================================
  // Tensor.atan2()
  // ==========================================================================
  describe('Tensor.atan2()', () => {
    it('computes element-wise arctan2', async () => {
      const y = torch.tensor([1, 0]);
      const x = torch.tensor([0, 1]);
      const result = y.atan2(x);
      const arr = await result.toArray();
      const values = Array.from(arr);
      // atan2(1, 0) = pi/2, atan2(0, 1) = 0
      expect(values[0]).toBeCloseTo(Math.PI / 2, 5);
      expect(values[1]).toBeCloseTo(0, 5);
    });

    it('computes arctan2 for various quadrants', async () => {
      const y = torch.tensor([1, -1, -1, 1]);
      const x = torch.tensor([1, 1, -1, -1]);
      const result = y.atan2(x);
      const arr = await result.toArray();
      const values = Array.from(arr);
      // atan2(1, 1) = pi/4
      expect(values[0]).toBeCloseTo(Math.PI / 4, 5);
      // atan2(-1, 1) = -pi/4
      expect(values[1]).toBeCloseTo(-Math.PI / 4, 5);
    });
  });

  // ==========================================================================
  // Tensor.logaddexp()
  // ==========================================================================
  describe('Tensor.logaddexp()', () => {
    it('computes log(exp(x) + exp(y))', async () => {
      const a = torch.tensor([0]);
      const b = torch.tensor([0]);
      const result = a.logaddexp(b);
      const arr = await result.toArray();
      // log(exp(0) + exp(0)) = log(2) ≈ 0.693
      expect(arr[0]).toBeCloseTo(Math.log(2), 5);
    });

    it('computes logaddexp for different values', async () => {
      const a = torch.tensor([1]);
      const b = torch.tensor([2]);
      const result = a.logaddexp(b);
      const arr = await result.toArray();
      // log(exp(1) + exp(2)) ≈ 2.313
      expect(arr[0]).toBeCloseTo(Math.log(Math.exp(1) + Math.exp(2)), 4);
    });

    it('computes logaddexp element-wise on tensors', async () => {
      const a = torch.tensor([0, 1]);
      const b = torch.tensor([0, 2]);
      const result = a.logaddexp(b);
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(Math.log(2), 5);
      expect(arr[1]).toBeCloseTo(Math.log(Math.exp(1) + Math.exp(2)), 4);
    });
  });

  // ==========================================================================
  // Tensor.masked_fill()
  // ==========================================================================
  describe('Tensor.masked_fill()', () => {
    it('fills values where mask is true', async () => {
      const t = torch.tensor([1, 2, 3]);
      const mask = torch.tensor([true, false, true]);
      const result = t.masked_fill(mask, 0);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0, 2, 0]);
    });

    it('fills with non-zero value', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const mask = torch.tensor([[true, false], [false, true]]);
      const result = t.masked_fill(mask, 99);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([99, 2, 3, 99]);
    });

    it('no mask true returns unchanged', async () => {
      const t = torch.tensor([1, 2, 3]);
      const mask = torch.tensor([false, false, false]);
      const result = t.masked_fill(mask, 0);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.logaddexp() - function version
  // ==========================================================================
  describe('torch.logaddexp', () => {
    it('computes log(exp(x) + exp(y)) as function', async () => {
      const a = torch.tensor([0]);
      const b = torch.tensor([0]);
      const result = torch.logaddexp(a, b);
      const arr = await result.toArray();
      expect(arr[0]).toBeCloseTo(Math.log(2), 5);
    });

    it('matches Tensor.logaddexp method', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const fnResult = torch.logaddexp(a, b);
      const methodResult = a.logaddexp(b);
      const fnArr = await fnResult.toArray();
      const methodArr = await methodResult.toArray();
      expect(Array.from(fnArr)).toEqual(Array.from(methodArr));
    });
  });
});
