import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('torch.linalg norms', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('torch.linalg.norm', () => {
    it('computes Frobenius norm (ord=None) of a matrix', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const n = torch.linalg.norm(A);
      const data = Array.from(await n.toArray());
      // sqrt(1+4+9+16) = sqrt(30)
      expect(data[0]).toBeCloseTo(Math.sqrt(30), 4);
    });

    it('computes norm along specified dimension', async () => {
      const x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      const n = torch.linalg.norm(x, 2, 1);
      const data = Array.from(await n.toArray());
      // row norms: sqrt(1+4+9)=sqrt(14), sqrt(16+25+36)=sqrt(77)
      expect(data[0]).toBeCloseTo(Math.sqrt(14), 4);
      expect(data[1]).toBeCloseTo(Math.sqrt(77), 4);
    });

    it('computes L1 norm (ord=1)', async () => {
      const x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]]);
      const n = torch.linalg.norm(x, 1);
      const data = Array.from(await n.toArray());
      // sqrt(1+4+9+16) = sqrt(30)
      expect(data[0]).toBeCloseTo(Math.sqrt(30), 4);
    });

    it('computes L2 norm (ord=2)', async () => {
      const x = torch.tensor([3.0, 4.0]);
      const n = torch.linalg.norm(x, 2);
      const data = Array.from(await n.toArray());
      expect(data[0]).toBeCloseTo(5.0, 4);
    });

    it('computes infinity norm (ord=inf)', async () => {
      const x = torch.tensor([[-5.0, 3.0], [2.0, -8.0]]);
      const n = torch.linalg.norm(x, Infinity);
      const data = Array.from(await n.toArray());
      // max absolute value = 8
      expect(data[0]).toBeCloseTo(8.0, 4);
    });

    it('computes norm with keepdim', async () => {
      const x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const n = torch.linalg.norm(x, 2, 1, true);
      const shape = n.shape;
      const data = Array.from(await n.toArray());
      expect(shape).toEqual([2, 1]);
      expect(data[0]).toBeCloseTo(Math.sqrt(5), 4);
      expect(data[1]).toBeCloseTo(5.0, 4);
    });
  });

  describe('torch.linalg.vector_norm', () => {
    it('computes L0 norm (count of non-zero elements)', async () => {
      const x = torch.tensor([0.0, 1.0, 0.0, 3.0, 0.0]);
      const n = torch.linalg.vector_norm(x, 0);
      const data = Array.from(await n.toArray());
      expect(data[0]).toBeCloseTo(2.0, 4);
    });

    it('computes L1 norm', async () => {
      const x = torch.tensor([-1.0, 2.0, -3.0, 4.0]);
      const n = torch.linalg.vector_norm(x, 1);
      const data = Array.from(await n.toArray());
      expect(data[0]).toBeCloseTo(10.0, 4);
    });

    it('computes L2 norm (Euclidean)', async () => {
      const x = torch.tensor([3.0, 4.0, 0.0]);
      const n = torch.linalg.vector_norm(x, 2);
      const data = Array.from(await n.toArray());
      expect(data[0]).toBeCloseTo(5.0, 4);
    });

    it('computes infinity norm', async () => {
      const x = torch.tensor([1.0, -7.0, 3.0, -2.0]);
      const n = torch.linalg.vector_norm(x, Infinity);
      const data = Array.from(await n.toArray());
      expect(data[0]).toBeCloseTo(7.0, 4);
    });

    it('computes negative infinity norm', async () => {
      const x = torch.tensor([1.0, 7.0, 3.0, 2.0]);
      const n = torch.linalg.vector_norm(x, -Infinity);
      const data = Array.from(await n.toArray());
      expect(data[0]).toBeCloseTo(1.0, 4);
    });

    it('computes vector norm along batch dimension', async () => {
      const x = torch.tensor([[3.0, 4.0], [6.0, 8.0]]);
      const n = torch.linalg.vector_norm(x, 2, 1);
      const data = Array.from(await n.toArray());
      expect(data[0]).toBeCloseTo(5.0, 4);
      expect(data[1]).toBeCloseTo(10.0, 4);
    });
  });

  describe('torch.linalg.matrix_norm', () => {
    it('computes Frobenius norm', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const n = torch.linalg.matrix_norm(A, 'fro');
      const data = Array.from(await n.toArray());
      expect(data[0]).toBeCloseTo(Math.sqrt(30), 4);
    });

    it('computes max column sum norm (ord=1)', async () => {
      const A = torch.tensor([[1.0, -5.0], [3.0, 2.0]]);
      const n = torch.linalg.matrix_norm(A, 1);
      const data = Array.from(await n.toArray());
      // col sums: |1|+|3|=4, |5|+|2|=7 => max=7
      expect(data[0]).toBeCloseTo(7.0, 4);
    });

    it('computes max row sum norm (ord=inf)', async () => {
      const A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      const n = torch.linalg.matrix_norm(A, Infinity);
      const data = Array.from(await n.toArray());
      // row sums: 6, 15 => max=15
      expect(data[0]).toBeCloseTo(15.0, 4);
    });
  });
});
