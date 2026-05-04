import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import { save, load } from '../../../src/serialization/browser';

const torch = createTorch(save, load);

describe('torch.sparse', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ---------------------------------------------------------------------------
  // sparse_coo_tensor
  // ---------------------------------------------------------------------------
  describe('sparse_coo_tensor', () => {
    it('creates sparse tensor from indices and values', async () => {
      const indices = torch.tensor([[0, 1], [1, 0], [2, 2]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0, 3.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [3, 3]);

      expect(sparse.nnz).toBe(3);
      expect(sparse.ndim).toBe(2);
      expect(sparse.size).toEqual([3, 3]);
    });

    it('creates sparse tensor with explicit dtype', () => {
      const indices = torch.tensor([[0, 0], [1, 1]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0], { dtype: 'float32' });
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2], 'float32');
      expect(sparse.dtype).toBe('float32');
      expect(sparse.nnz).toBe(2);
    });

    it('creates sparse tensor with duplicate indices', () => {
      const indices = torch.tensor([[0, 0], [0, 0]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);
      expect(sparse.nnz).toBe(2);
      // After coalesce, these should merge to a single entry
    });
  });

  // ---------------------------------------------------------------------------
  // to_sparse (dense -> sparse)
  // ---------------------------------------------------------------------------
  describe('to_sparse', () => {
    it('converts dense tensor to sparse', async () => {
      const dense = torch.tensor([[1.0, 0.0], [0.0, 2.0]]);
      const sparse = await torch.sparse.to_sparse(dense);
      expect(sparse.size).toEqual([2, 2]);
      expect(sparse.nnz).toBe(2);
    });

    it('to_sparse with threshold ignores small values', async () => {
      const dense = torch.tensor([[1.0, 0.01], [0.0, 2.0]]);
      const sparse = await torch.sparse.to_sparse(dense, 0.05);
      expect(sparse.size).toEqual([2, 2]);
      expect(sparse.nnz).toBe(2); // 0.01 should be excluded
    });

    it('to_sparse with all-zeros returns empty sparse tensor', async () => {
      const dense = torch.tensor([[0.0, 0.0], [0.0, 0.0]]);
      const sparse = await torch.sparse.to_sparse(dense);
      expect(sparse.nnz).toBe(0);
      expect(sparse.size).toEqual([2, 2]);
    });

    it('to_sparse on 1D tensor', async () => {
      const dense = torch.tensor([0.0, 3.0, 0.0, 5.0]);
      const sparse = await torch.sparse.to_sparse(dense);
      expect(sparse.size).toEqual([4]);
      expect(sparse.nnz).toBe(2);
    });
  });

  // ---------------------------------------------------------------------------
  // sparse_eye
  // ---------------------------------------------------------------------------
  describe('sparse_eye', () => {
    it('creates identity sparse tensor', async () => {
      const sparse = torch.sparse.sparse_eye(3);
      expect(sparse.size).toEqual([3, 3]);
      expect(sparse.nnz).toBe(3);
      expect(sparse.ndim).toBe(2);
    });

    it('sparse_eye values are all 1', async () => {
      const sparse = torch.sparse.sparse_eye(4);
      const valuesArr = Array.from(await sparse.values.toArray());
      valuesArr.forEach((v) => expect(v).toBe(1.0));
    });

    it('sparse_eye reconstructs to dense identity', async () => {
      const sparse = torch.sparse.sparse_eye(3);
      const dense = await sparse.to_dense();
      const denseArr = Array.from(await dense.toArray());
      // [1,0,0, 0,1,0, 0,0,1]
      expect(denseArr[0]).toBe(1);
      expect(denseArr[1]).toBe(0);
      expect(denseArr[2]).toBe(0);
      expect(denseArr[4]).toBe(1);
      expect(denseArr[8]).toBe(1);
    });
  });

  // ---------------------------------------------------------------------------
  // coalesce
  // ---------------------------------------------------------------------------
  describe('coalesce', () => {
    it('coalesced tensor is marked as coalesced', async () => {
      const indices = torch.tensor([[0, 1], [1, 0]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);
      const coalesced = await sparse.coalesce();
      expect(coalesced.is_coalesced()).toBe(true);
    });

    it('coalesce sorts indices lexicographically', async () => {
      // Create with unsorted indices: [1,0] before [0,1]
      const indices = torch.tensor([[1, 0], [0, 1], [1, 1]], { dtype: 'int32' });
      const values = torch.tensor([2.0, 1.0, 3.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);
      const coalesced = await sparse.coalesce();

      const idxArr = Array.from(await coalesced.indices.toArray());
      // After sorting: [0,1], [1,0], [1,1]
      expect(idxArr[0]).toBe(0); // first row index
      expect(idxArr[1]).toBe(1); // first col index
      expect(idxArr[2]).toBe(1); // second row index
      expect(idxArr[3]).toBe(0); // second col index
    });

    it('coalesce merges duplicate indices by summing', async () => {
      const indices = torch.tensor([[0, 0], [0, 0], [1, 1]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0, 3.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);
      const coalesced = await sparse.coalesce();

      expect(coalesced.nnz).toBe(2); // duplicates merged
      const valsArr = Array.from(await coalesced.values.toArray());
      expect(valsArr[0]).toBe(3.0); // 1+2
      expect(valsArr[1]).toBe(3.0);
    });
  });

  // ---------------------------------------------------------------------------
  // to_dense (reconstruction)
  // ---------------------------------------------------------------------------
  describe('to_dense', () => {
    it('to_dense reconstructs original dense tensor', async () => {
      const indices = torch.tensor([[0, 0], [1, 1], [2, 2]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0, 3.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [3, 3]);
      const dense = await sparse.to_dense();
      const denseArr = Array.from(await dense.toArray());

      expect(denseArr[0]).toBe(1.0);
      expect(denseArr[4]).toBe(2.0);
      expect(denseArr[8]).toBe(3.0);
      expect(denseArr[1]).toBe(0.0);
      expect(denseArr[2]).toBe(0.0);
    });

    it('to_dense handles off-diagonal entries', async () => {
      const indices = torch.tensor([[0, 1], [1, 0]], { dtype: 'int32' });
      const values = torch.tensor([5.0, -3.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);
      const dense = await sparse.to_dense();
      const denseArr = Array.from(await dense.toArray());

      expect(denseArr[0]).toBe(0.0);
      expect(denseArr[1]).toBe(5.0);
      expect(denseArr[2]).toBe(-3.0);
      expect(denseArr[3]).toBe(0.0);
    });

    it('to_dense of empty sparse tensor returns zeros', async () => {
      const emptyIndices = torch.tensor([], { dtype: 'int32' }).reshape([0, 2]);
      const emptyValues = torch.tensor([]);
      const sparse = torch.sparse.sparse_coo_tensor(emptyIndices, emptyValues, [2, 2]);
      const dense = await sparse.to_dense();
      const denseArr = Array.from(await dense.toArray());
      denseArr.forEach((v) => expect(v).toBe(0));
    });
  });

  // ---------------------------------------------------------------------------
  // sparse_add
  // ---------------------------------------------------------------------------
  describe('sparse_add', () => {
    it('adds two sparse tensors with disjoint indices', async () => {
      const idxA = torch.tensor([[0, 0]], { dtype: 'int32' });
      const valA = torch.tensor([1.0]);
      const a = torch.sparse.sparse_coo_tensor(idxA, valA, [2, 2]);

      const idxB = torch.tensor([[1, 1]], { dtype: 'int32' });
      const valB = torch.tensor([2.0]);
      const b = torch.sparse.sparse_coo_tensor(idxB, valB, [2, 2]);

      const result = await torch.sparse.sparse_add(a, b);
      expect(result.nnz).toBe(2);

      const dense = await result.to_dense();
      const denseArr = Array.from(await dense.toArray());
      expect(denseArr[0]).toBe(1.0);
      expect(denseArr[3]).toBe(2.0);
    });

    it('adds two sparse tensors with overlapping indices (sums values)', async () => {
      const idxA = torch.tensor([[0, 0]], { dtype: 'int32' });
      const valA = torch.tensor([3.0]);
      const a = torch.sparse.sparse_coo_tensor(idxA, valA, [2, 2]);

      const idxB = torch.tensor([[0, 0]], { dtype: 'int32' });
      const valB = torch.tensor([4.0]);
      const b = torch.sparse.sparse_coo_tensor(idxB, valB, [2, 2]);

      const result = await torch.sparse.sparse_add(a, b);
      expect(result.nnz).toBe(1);

      const valsArr = Array.from(await result.values.toArray());
      expect(valsArr[0]).toBe(7.0);
    });

    it('throws on shape mismatch', async () => {
      const idxA = torch.tensor([[0, 0]], { dtype: 'int32' });
      const valA = torch.tensor([1.0]);
      const a = torch.sparse.sparse_coo_tensor(idxA, valA, [2, 2]);

      const idxB = torch.tensor([[0, 0, 0]], { dtype: 'int32' });
      const valB = torch.tensor([1.0]);
      const b = torch.sparse.sparse_coo_tensor(idxB, valB, [3, 3, 3]);

      await expect(torch.sparse.sparse_add(a, b)).rejects.toThrow();
    });
  });

  // ---------------------------------------------------------------------------
  // sparse_mul (scalar)
  // ---------------------------------------------------------------------------
  describe('sparse_mul', () => {
    it('multiplies sparse values by scalar', async () => {
      const indices = torch.tensor([[0, 0], [1, 1]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);

      const result = torch.sparse.sparse_mul(sparse, 3.0);
      expect(result.nnz).toBe(2);

      const valsArr = Array.from(await result.values.toArray());
      expect(valsArr[0]).toBe(3.0);
      expect(valsArr[1]).toBe(6.0);
    });

    it('sparse_mul preserves indices', () => {
      const indices = torch.tensor([[0, 1], [1, 0]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);

      const result = torch.sparse.sparse_mul(sparse, 0.5);
      const idxArr = Array.from(result.indices.toArray());
      const origIdxArr = Array.from(indices.toArray());
      expect(idxArr).toEqual(origIdxArr);
    });

    it('sparse_mul by zero keeps structure', async () => {
      const indices = torch.tensor([[0, 0]], { dtype: 'int32' });
      const values = torch.tensor([5.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);

      const result = torch.sparse.sparse_mul(sparse, 0.0);
      const valsArr = Array.from(await result.values.toArray());
      expect(valsArr[0]).toBe(0.0);
    });
  });

  // ---------------------------------------------------------------------------
  // sparse_matmul
  // ---------------------------------------------------------------------------
  describe('sparse_matmul', () => {
    it('sparse @ dense vector', async () => {
      // sparse = [[1, 0], [0, 2]]
      const indices = torch.tensor([[0, 0], [1, 1]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);

      const dense = torch.tensor([3.0, 4.0]);
      const result = await torch.sparse.sparse_matmul(sparse, dense);

      expect(result.shape).toEqual([2]);
      const resArr = Array.from(await result.toArray());
      // [1*3, 2*4] = [3, 8]
      expect(resArr[0]).toBe(3.0);
      expect(resArr[1]).toBe(8.0);
    });

    it('sparse @ dense matrix', async () => {
      // sparse = [[1, 0], [0, 2]]  (2x2)
      const indices = torch.tensor([[0, 0], [1, 1]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);

      // dense = [[1, 2], [3, 4]]  (2x2)
      const dense = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const result = await torch.sparse.sparse_matmul(sparse, dense);

      expect(result.shape).toEqual([2, 2]);
      const resArr = Array.from(await result.toArray());
      // [[1*1, 1*2], [2*3, 2*4]] = [[1, 2], [6, 8]]
      expect(resArr[0]).toBeCloseTo(1.0, 5);
      expect(resArr[1]).toBeCloseTo(2.0, 5);
      expect(resArr[2]).toBeCloseTo(6.0, 5);
      expect(resArr[3]).toBeCloseTo(8.0, 5);
    });

    it('sparse @ zero vector returns zeros', async () => {
      const indices = torch.tensor([[0, 0], [1, 1]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);

      const dense = torch.tensor([0.0, 0.0]);
      const result = await torch.sparse.sparse_matmul(sparse, dense);
      const resArr = Array.from(await result.toArray());
      expect(resArr[0]).toBe(0);
      expect(resArr[1]).toBe(0);
    });

    it('empty sparse @ dense returns zeros', async () => {
      const emptyIndices = torch.tensor([], { dtype: 'int32' }).reshape([0, 2]);
      const emptyValues = torch.tensor([]);
      const sparse = torch.sparse.sparse_coo_tensor(emptyIndices, emptyValues, [2, 2]);

      const dense = torch.tensor([1.0, 2.0]);
      const result = await torch.sparse.sparse_matmul(sparse, dense);
      const resArr = Array.from(await result.toArray());
      expect(resArr[0]).toBe(0);
      expect(resArr[1]).toBe(0);
    });

    it('throws on dimension mismatch', async () => {
      const indices = torch.tensor([[0, 0]], { dtype: 'int32' });
      const values = torch.tensor([1.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);

      const dense = torch.tensor([1.0, 2.0, 3.0]); // 3D vector, K=3 doesn't match sparse K=2
      await expect(torch.sparse.sparse_matmul(sparse, dense)).rejects.toThrow();
    });

    it('throws on non-2D sparse tensor', async () => {
      const indices = torch.tensor([[0], [1]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2]);

      const dense = torch.tensor([1.0]);
      await expect(torch.sparse.sparse_matmul(sparse, dense)).rejects.toThrow();
    });
  });

  // ---------------------------------------------------------------------------
  // SparseTensor properties
  // ---------------------------------------------------------------------------
  describe('SparseTensor properties', () => {
    it('nnz returns number of non-zero entries', () => {
      const indices = torch.tensor([[0, 0], [1, 1], [2, 2]], { dtype: 'int32' });
      const values = torch.tensor([1.0, 2.0, 3.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [3, 3]);
      expect(sparse.nnz).toBe(3);
    });

    it('ndim returns number of dimensions', () => {
      const indices2d = torch.tensor([[0, 0]], { dtype: 'int32' });
      const values = torch.tensor([1.0]);
      const sparse2d = torch.sparse.sparse_coo_tensor(indices2d, values, [2, 2]);
      expect(sparse2d.ndim).toBe(2);
    });

    it('is_coalesced returns true for single-entry sparse', () => {
      const indices = torch.tensor([[0, 0]], { dtype: 'int32' });
      const values = torch.tensor([1.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2]);
      expect(sparse.is_coalesced()).toBe(true);
    });

    it('size returns dense shape', () => {
      const indices = torch.tensor([[0, 0]], { dtype: 'int32' });
      const values = torch.tensor([1.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [5, 7]);
      expect(sparse.size).toEqual([5, 7]);
    });

    it('dtype returns data type', () => {
      const indices = torch.tensor([[0, 0]], { dtype: 'int32' });
      const values = torch.tensor([1.0]);
      const sparse = torch.sparse.sparse_coo_tensor(indices, values, [2, 2], 'float32');
      expect(sparse.dtype).toBe('float32');
    });
  });
});
