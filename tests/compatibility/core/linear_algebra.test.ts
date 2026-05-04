import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Linear Algebra', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.matmul()
  // ==========================================================================
  describe('matmul', () => {
    it('matrix multiplies 2D tensors', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([[5, 6], [7, 8]]);
      const c = torch.matmul(a, b);
      expect(c.shape).toEqual([2, 2]);
      const arr = await c.toArray();
      // [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
      expect(arr).toEqual([19, 22, 43, 50]);
    });

    it('matrix-vector multiplication', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([5, 6]);
      const c = torch.matmul(a, b);
      expect(c.shape).toEqual([2]);
      const arr = await c.toArray();
      expect(arr).toEqual([17, 39]);
    });

    it('vector-matrix multiplication', async () => {
      const a = torch.tensor([1, 2]);
      const b = torch.tensor([[3, 4], [5, 6]]);
      const c = torch.matmul(a, b);
      expect(c.shape).toEqual([2]);
      const arr = await c.toArray();
      expect(arr).toEqual([13, 16]);
    });

    it('dot product of vectors', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.matmul(a, b);
      expect(c.shape).toEqual([]);
      const arr = await c.toArray();
      expect(arr[0]).toBe(32); // 1*4 + 2*5 + 3*6
    });

    it('batched matmul', async () => {
      const a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const b = torch.tensor([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]);
      const c = torch.matmul(a, b);
      expect(c.shape).toEqual([2, 2, 2]);
    });
  });

  // ==========================================================================
  // torch.mm()
  // ==========================================================================
  describe('mm', () => {
    it('matrix multiplies 2D tensors', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([[5, 6], [7, 8]]);
      const c = torch.mm(a, b);
      expect(c.shape).toEqual([2, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([19, 22, 43, 50]);
    });

    it('mm with identity matrix', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.eye(2);
      const c = torch.mm(a, b);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 2, 3, 4]);
    });
  });

  // ==========================================================================
  // torch.bmm()
  // ==========================================================================
  describe('bmm', () => {
    it('batch matrix multiply', async () => {
      const a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const b = torch.tensor([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]);
      const c = torch.bmm(a, b);
      expect(c.shape).toEqual([2, 2, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    });

    it('bmm with non-identity', async () => {
      const a = torch.tensor([[[1, 2], [3, 4]]]);
      const b = torch.tensor([[[5, 6], [7, 8]]]);
      const c = torch.bmm(a, b);
      expect(c.shape).toEqual([1, 2, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([19, 22, 43, 50]);
    });
  });

  // ==========================================================================
  // torch.dot()
  // ==========================================================================
  describe('dot', () => {
    it('computes dot product', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.dot(a, b);
      expect(c.shape).toEqual([]);
      const arr = await c.toArray();
      expect(arr[0]).toBe(32);
    });

    it('dot with zeros', async () => {
      const a = torch.tensor([1, 0, 0]);
      const b = torch.tensor([0, 1, 0]);
      const c = torch.dot(a, b);
      const arr = await c.toArray();
      expect(arr[0]).toBe(0);
    });
  });

  // ==========================================================================
  // torch.vdot()
  // ==========================================================================
  describe('vdot', () => {
    it('computes vector dot product', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.vdot(a, b);
      expect(c.shape).toEqual([]);
      const arr = await c.toArray();
      expect(arr[0]).toBe(32);
    });
  });

  // ==========================================================================
  // torch.outer()
  // ==========================================================================
  describe('outer', () => {
    it('computes outer product', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5]);
      const c = torch.outer(a, b);
      expect(c.shape).toEqual([3, 2]);
      const arr = await c.toArray();
      // [1*4, 1*5, 2*4, 2*5, 3*4, 3*5] = [4, 5, 8, 10, 12, 15]
      expect(arr).toEqual([4, 5, 8, 10, 12, 15]);
    });

    it('outer with ones', async () => {
      const a = torch.tensor([1, 1]);
      const b = torch.tensor([1, 1, 1]);
      const c = torch.outer(a, b);
      expect(c.shape).toEqual([2, 3]);
      const arr = await c.toArray();
      expect(arr.every((v: number) => v === 1)).toBe(true);
    });
  });

  // ==========================================================================
  // torch.ger()
  // ==========================================================================
  describe('ger', () => {
    it('computes outer product (alias of outer)', async () => {
      const a = torch.tensor([2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.ger(a, b);
      expect(c.shape).toEqual([2, 3]);
      const arr = await c.toArray();
      expect(arr).toEqual([8, 10, 12, 12, 15, 18]);
    });
  });

  // ==========================================================================
  // torch.addmm()
  // ==========================================================================
  describe('addmm', () => {
    it('computes beta*input + alpha*mat1 @ mat2', async () => {
      const input = torch.tensor([[1, 1], [1, 1]]);
      const mat1 = torch.tensor([[1, 2], [3, 4]]);
      const mat2 = torch.tensor([[5, 6], [7, 8]]);
      const c = torch.addmm(input, mat1, mat2);
      expect(c.shape).toEqual([2, 2]);
      // input + mat1 @ mat2 = [[1,1],[1,1]] + [[19,22],[43,50]] = [[20,23],[44,51]]
      const arr = await c.toArray();
      expect(arr).toEqual([20, 23, 44, 51]);
    });

    it('addmm with beta=0', async () => {
      const input = torch.tensor([[100, 100], [100, 100]]);
      const mat1 = torch.tensor([[1, 2], [3, 4]]);
      const mat2 = torch.tensor([[5, 6], [7, 8]]);
      const c = torch.addmm(input, mat1, mat2, 0);
      // 0*input + mat1 @ mat2 = [[19,22],[43,50]]
      const arr = await c.toArray();
      expect(arr).toEqual([19, 22, 43, 50]);
    });

    it('addmm with alpha scaling', async () => {
      const input = torch.tensor([[0, 0], [0, 0]]);
      const mat1 = torch.tensor([[1, 0], [0, 1]]);
      const mat2 = torch.tensor([[2, 0], [0, 2]]);
      const c = torch.addmm(input, mat1, mat2, 1, 3);
      // 0 + 3 * (mat1 @ mat2) = 3 * [[2,0],[0,2]] = [[6,0],[0,6]]
      const arr = await c.toArray();
      expect(arr).toEqual([6, 0, 0, 6]);
    });
  });

  // ==========================================================================
  // torch.addbmm()
  // ==========================================================================
  describe('addbmm', () => {
    it('adds batched matrix multiply result', async () => {
      const input = torch.tensor([[1, 0], [0, 1]]);
      const batch1 = torch.tensor([[[1, 2], [3, 4]]]);
      const batch2 = torch.tensor([[[5, 6], [7, 8]]]);
      const c = torch.addbmm(input, batch1, batch2);
      expect(c.shape).toEqual([2, 2]);
      // input + sum of batch matmuls
      const arr = await c.toArray();
      expect(arr).toEqual([20, 22, 43, 51]);
    });
  });

  // ==========================================================================
  // torch.baddbmm()
  // ==========================================================================
  describe('baddbmm', () => {
    it('batched addmm', async () => {
      const input = torch.tensor([[[1, 0], [0, 1]]]);
      const batch1 = torch.tensor([[[1, 2], [3, 4]]]);
      const batch2 = torch.tensor([[[5, 6], [7, 8]]]);
      const c = torch.baddbmm(input, batch1, batch2);
      expect(c.shape).toEqual([1, 2, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([20, 22, 43, 51]);
    });
  });

  // ==========================================================================
  // torch.mv()
  // ==========================================================================
  describe('mv', () => {
    it('matrix-vector multiply', async () => {
      const mat = torch.tensor([[1, 2], [3, 4]]);
      const vec = torch.tensor([5, 6]);
      const c = torch.mv(mat, vec);
      expect(c.shape).toEqual([2]);
      const arr = await c.toArray();
      expect(arr).toEqual([17, 39]);
    });

    it('mv with identity', async () => {
      const mat = torch.eye(2);
      const vec = torch.tensor([3, 4]);
      const c = torch.mv(mat, vec);
      const arr = await c.toArray();
      expect(arr).toEqual([3, 4]);
    });
  });

  // ==========================================================================
  // torch.addmv()
  // ==========================================================================
  describe('addmv', () => {
    it('adds matrix-vector multiply result', async () => {
      const input = torch.tensor([1, 1]);
      const mat = torch.tensor([[1, 2], [3, 4]]);
      const vec = torch.tensor([5, 6]);
      const c = torch.addmv(input, mat, vec);
      expect(c.shape).toEqual([2]);
      // input + mat @ vec = [1,1] + [17,39] = [18,40]
      const arr = await c.toArray();
      expect(arr).toEqual([18, 40]);
    });

    it('addmv with beta=0', async () => {
      const input = torch.tensor([100, 100]);
      const mat = torch.tensor([[1, 0], [0, 1]]);
      const vec = torch.tensor([3, 4]);
      const c = torch.addmv(input, mat, vec, 0);
      const arr = await c.toArray();
      expect(arr).toEqual([3, 4]);
    });
  });

  // ==========================================================================
  // torch.addr()
  // ==========================================================================
  describe('addr', () => {
    it('adds outer product of two vectors', async () => {
      const input = torch.tensor([[1, 1], [1, 1]]);
      const vec1 = torch.tensor([2, 3]);
      const vec2 = torch.tensor([4, 5]);
      const c = torch.addr(input, vec1, vec2);
      expect(c.shape).toEqual([2, 2]);
      // input + outer(vec1, vec2) = [[1,1],[1,1]] + [[8,10],[12,15]] = [[9,11],[13,16]]
      const arr = await c.toArray();
      expect(arr).toEqual([9, 11, 13, 16]);
    });
  });

  // ==========================================================================
  // torch.einsum()
  // ==========================================================================
  describe('einsum', () => {
    it('matrix multiply with einsum', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([[5, 6], [7, 8]]);
      const c = await torch.einsum('ij,jk->ik', a, b);
      expect(c.shape).toEqual([2, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([19, 22, 43, 50]);
    });

    it('dot product with einsum', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = await torch.einsum('i,i->', a, b);
      expect(c.shape).toEqual([]);
      const arr = await c.toArray();
      expect(arr[0]).toBe(32);
    });

    it('transpose with einsum', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const c = await torch.einsum('ij->ji', a);
      expect(c.shape).toEqual([2, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 3, 2, 4]);
    });

    it('sum with einsum', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const c = await torch.einsum('ij->', a);
      expect(c.shape).toEqual([]);
      const arr = await c.toArray();
      expect(arr[0]).toBe(10);
    });

    it('batched matmul with einsum', async () => {
      const a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const b = torch.tensor([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]);
      const c = await torch.einsum('bij,bjk->bik', a, b);
      expect(c.shape).toEqual([2, 2, 2]);
    });
  });

  // ==========================================================================
  // torch.hypot()
  // ==========================================================================
  describe('hypot', () => {
    it('computes hypotenuse', async () => {
      const a = torch.tensor([3, 5]);
      const b = torch.tensor([4, 12]);
      const c = torch.hypot(a, b);
      const arr = await c.toArray();
      expect(arr[0]).toBeCloseTo(5);   // sqrt(9+16)
      expect(arr[1]).toBeCloseTo(13);  // sqrt(25+144)
    });

    it('hypot with zeros', async () => {
      const a = torch.tensor([0, 0]);
      const b = torch.tensor([3, 4]);
      const c = torch.hypot(a, b);
      const arr = await c.toArray();
      expect(arr).toEqual([3, 4]);
    });
  });

  // ==========================================================================
  // torch.trapezoid()
  // ==========================================================================
  describe('trapezoid', () => {
    it('computes trapezoidal integration', async () => {
      const y = torch.tensor([1, 2, 3]);
      const r = torch.trapezoid(y, 1);
      const arr = await r.toArray();
      // dx=1: (1+2)/2 + (2+3)/2 = 1.5 + 2.5 = 4
      expect(arr[0]).toBeCloseTo(4);
    });

    it('trapezoid with different dx', async () => {
      const y = torch.tensor([0, 1, 4]);
      const r = torch.trapezoid(y, 2);
      const arr = await r.toArray();
      // dx=2: 2*(0+1)/2 + 2*(1+4)/2 = 1 + 5 = 6
      expect(arr[0]).toBeCloseTo(6);
    });
  });
});
