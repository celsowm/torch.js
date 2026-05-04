import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

/**
 * Helper: compute max absolute difference between two flat arrays.
 */
function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

describe('torch.linalg decomposition and solvers', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('torch.linalg.svd', () => {
    it('decomposes MxN matrix into U, S, Vh', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
      const { U, S, Vh } = await torch.linalg.svd(A, false);
      const uShape = U.shape;
      const sShape = S.shape;
      const vhShape = Vh.shape;
      // M=3, N=2, K=2 => U:[3,2], S:[2], Vh:[2,2]
      expect(uShape).toEqual([3, 2]);
      expect(sShape).toEqual([2]);
      expect(vhShape).toEqual([2, 2]);
    });

    it('reconstructs matrix: A ~ U @ diag(S) @ Vh', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const { U, S, Vh } = await torch.linalg.svd(A, false);
      // diag(S)
      const sData = Array.from(await S.toArray());
      const diagS = torch.diag(torch.tensor(sData));
      // U @ diag(S) @ Vh
      const reconstructed = torch.matmul(torch.matmul(U, diagS), Vh);
      const original = Array.from(await A.toArray()) as Float32Array;
      const recon = Array.from(await reconstructed.toArray()) as Float32Array;
      expect(maxAbsDiff(original, recon)).toBeLessThan(1e-3);
    });

    it('singular values are sorted descending', async () => {
      const A = torch.tensor([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]]);
      const { S } = await torch.linalg.svd(A, false);
      const sData = Array.from(await S.toArray());
      expect(sData[0]).toBeGreaterThanOrEqual(sData[1]);
      expect(sData[1]).toBeGreaterThanOrEqual(sData[2]);
    });
  });

  describe('torch.linalg.eigh', () => {
    it('computes eigenvalues and eigenvectors of symmetric matrix', async () => {
      const A = torch.tensor([[4.0, 1.0], [1.0, 3.0]]);
      const { eigenvalues, eigenvectors } = await torch.linalg.eigh(A);
      const evalShape = eigenvalues.shape;
      const evecShape = eigenvectors.shape;
      expect(evalShape).toEqual([2]);
      expect(evecShape).toEqual([2, 2]);
    });

    it('eigenvalues are in ascending order', async () => {
      const A = torch.tensor([[4.0, 1.0], [1.0, 3.0]]);
      const { eigenvalues } = await torch.linalg.eigh(A);
      const evals = Array.from(await eigenvalues.toArray());
      expect(evals[0]).toBeLessThanOrEqual(evals[1]);
    });

    it('reconstructs matrix: A = V @ diag(eigenvalues) @ V^T', async () => {
      const A = torch.tensor([[2.0, 1.0], [1.0, 2.0]]);
      const { eigenvalues, eigenvectors } = await torch.linalg.eigh(A);
      const evals = Array.from(await eigenvalues.toArray());
      const diagEvals = torch.diag(torch.tensor(evals));
      const Vt = eigenvectors.transpose(-2, -1);
      const reconstructed = torch.matmul(torch.matmul(eigenvectors, diagEvals), Vt);
      const original = Array.from(await A.toArray()) as Float32Array;
      const recon = Array.from(await reconstructed.toArray()) as Float32Array;
      expect(maxAbsDiff(original, recon)).toBeLessThan(1e-3);
    });
  });

  describe('torch.linalg.qr', () => {
    it('computes reduced QR decomposition', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
      const { Q, R } = await torch.linalg.qr(A, 'reduced');
      const qShape = Q.shape;
      const rShape = R.shape;
      // M=3, N=2 => Q:[3,2], R:[2,2]
      expect(qShape).toEqual([3, 2]);
      expect(rShape).toEqual([2, 2]);
    });

    it('Q is orthogonal (Q^T Q = I)', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
      const { Q } = await torch.linalg.qr(A, 'reduced');
      const Qt = Q.transpose(-2, -1);
      const QtQ = torch.matmul(Qt, Q);
      const expected = Array.from(await torch.eye(2).toArray()) as Float32Array;
      const actual = Array.from(await QtQ.toArray()) as Float32Array;
      expect(maxAbsDiff(expected, actual)).toBeLessThan(1e-4);
    });

    it('R is upper triangular', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
      const { R } = await torch.linalg.qr(A, 'reduced');
      const rData = Array.from(await R.toArray()) as Float32Array;
      // R is 2x2, lower-left element should be ~0
      expect(Math.abs(rData[1])).toBeLessThan(1e-6); // R[1,0] in row-major
    });

    it('computes complete QR decomposition', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const { Q, R } = await torch.linalg.qr(A, 'complete');
      const qShape = Q.shape;
      const rShape = R.shape;
      expect(qShape).toEqual([2, 2]);
      expect(rShape).toEqual([2, 2]);
    });
  });

  describe('torch.linalg.solve', () => {
    it('solves Ax = b for single RHS', async () => {
      const A = torch.tensor([[2.0, 1.0], [1.0, 3.0]]);
      const b = torch.tensor([[5.0], [7.0]]);
      const x = await torch.linalg.solve(A, b);
      // Verify: A @ x should equal b
      const Ax = torch.matmul(A, x);
      const bData = Array.from(await b.toArray()) as Float32Array;
      const axData = Array.from(await Ax.toArray()) as Float32Array;
      expect(maxAbsDiff(bData, axData)).toBeLessThan(1e-4);
    });

    it('solves Ax = b for multiple RHS columns', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const B = torch.tensor([[5.0, 6.0], [7.0, 8.0]]);
      const x = await torch.linalg.solve(A, B);
      const Ax = torch.matmul(A, x);
      const bData = Array.from(await B.toArray()) as Float32Array;
      const axData = Array.from(await Ax.toArray()) as Float32Array;
      expect(maxAbsDiff(bData, axData)).toBeLessThan(1e-4);
    });

    it('solution is correct for known system', async () => {
      // A = [[3,1],[1,2]], x = [1,2] => b = [5,5]
      const A = torch.tensor([[3.0, 1.0], [1.0, 2.0]]);
      const b = torch.tensor([[5.0], [5.0]]);
      const x = await torch.linalg.solve(A, b);
      const xData = Array.from(await x.toArray());
      expect(xData[0]).toBeCloseTo(1.0, 3);
      expect(xData[1]).toBeCloseTo(2.0, 3);
    });
  });

  describe('torch.linalg.pinv', () => {
    it('computes Moore-Penrose pseudo-inverse', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
      const Apin = await torch.linalg.pinv(A);
      const shape = Apin.shape;
      // A:[3,2] => pinv:[2,3]
      expect(shape).toEqual([2, 3]);
    });

    it('A @ pinv(A) approximates I (for full row rank)', async () => {
      const A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
      const Apin = await torch.linalg.pinv(A);
      const AApinv = torch.matmul(A, Apin);
      const expected = Array.from(await torch.eye(2).toArray()) as Float32Array;
      const actual = Array.from(await AApinv.toArray()) as Float32Array;
      expect(maxAbsDiff(expected, actual)).toBeLessThan(1e-4);
    });

    it('pinv(A) @ A approximates I (for full column rank)', async () => {
      const A = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]);
      const Apin = await torch.linalg.pinv(A);
      const ApinvA = torch.matmul(Apin, A);
      const expected = Array.from(await torch.eye(2).toArray()) as Float32Array;
      const actual = Array.from(await ApinvA.toArray()) as Float32Array;
      expect(maxAbsDiff(expected, actual)).toBeLessThan(1e-4);
    });
  });

  describe('torch.linalg.cholesky', () => {
    it('computes lower triangular Cholesky factor', async () => {
      // A = [[4,2],[2,3]] is SPD
      const A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]);
      const L = torch.linalg.cholesky(A);
      const lShape = L.shape;
      expect(lShape).toEqual([2, 2]);
      // Lower triangular: L[0,1] should be ~0
      const lData = Array.from(await L.toArray()) as Float32Array;
      expect(Math.abs(lData[1])).toBeLessThan(1e-6);
    });

    it('reconstructs matrix: A ~ L @ L^T', async () => {
      const A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]);
      const L = torch.linalg.cholesky(A);
      const Lt = L.transpose(-2, -1);
      const reconstructed = torch.matmul(L, Lt);
      const original = Array.from(await A.toArray()) as Float32Array;
      const recon = Array.from(await reconstructed.toArray()) as Float32Array;
      expect(maxAbsDiff(original, recon)).toBeLessThan(1e-4);
    });

    it('computes upper triangular Cholesky factor', async () => {
      const A = torch.tensor([[4.0, 2.0], [2.0, 3.0]]);
      const U = torch.linalg.cholesky(A, true);
      const uData = Array.from(await U.toArray()) as Float32Array;
      // Upper triangular: U[1,0] should be ~0
      expect(Math.abs(uData[2])).toBeLessThan(1e-6);
    });
  });

  describe('torch.linalg.inv', () => {
    it('computes matrix inverse', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const Ainv = torch.linalg.inv(A);
      const shape = Ainv.shape;
      expect(shape).toEqual([2, 2]);
    });

    it('A @ inv(A) approximates I', async () => {
      const A = torch.tensor([[1.0, 0.0], [0.0, 2.0]]);
      const Ainv = torch.linalg.inv(A);
      const product = torch.matmul(A, Ainv);
      const expected = Array.from(await torch.eye(2).toArray()) as Float32Array;
      const actual = Array.from(await product.toArray()) as Float32Array;
      expect(maxAbsDiff(expected, actual)).toBeLessThan(1e-4);
    });

    it('inv(inv(A)) approximates A', async () => {
      const A = torch.tensor([[2.0, 1.0], [1.0, 3.0]]);
      const Ainv = torch.linalg.inv(A);
      const Ainvinv = torch.linalg.inv(Ainv);
      const original = Array.from(await A.toArray()) as Float32Array;
      const recon = Array.from(await Ainvinv.toArray()) as Float32Array;
      expect(maxAbsDiff(original, recon)).toBeLessThan(1e-3);
    });
  });

  describe('torch.linalg.cross', () => {
    it('computes cross product of 3D vectors', async () => {
      const a = torch.tensor([1.0, 0.0, 0.0]);
      const b = torch.tensor([0.0, 1.0, 0.0]);
      const c = torch.linalg.cross(a, b);
      const data = Array.from(await c.toArray());
      // i x j = k => [0,0,1]
      expect(data[0]).toBeCloseTo(0.0, 4);
      expect(data[1]).toBeCloseTo(0.0, 4);
      expect(data[2]).toBeCloseTo(1.0, 4);
    });

    it('result is perpendicular to both input vectors', async () => {
      const a = torch.tensor([1.0, 2.0, 3.0]);
      const b = torch.tensor([4.0, 5.0, 6.0]);
      const c = torch.linalg.cross(a, b);
      // dot(a, c) and dot(b, c) should be ~0
      const dotAC = a.mul(c).sum().toArray();
      const dotBC = b.mul(c).sum().toArray();
      expect(Math.abs(Array.from(await dotAC)[0])).toBeLessThan(1e-4);
      expect(Math.abs(Array.from(await dotBC)[0])).toBeLessThan(1e-4);
    });

    it('computes batched cross products', async () => {
      const a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
      const b = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
      const c = torch.linalg.cross(a, b);
      const data = Array.from(await c.toArray());
      // [i x j = k, j x k = i] => [0,0,1, 1,0,0]
      expect(data[0]).toBeCloseTo(0.0, 4);
      expect(data[1]).toBeCloseTo(0.0, 4);
      expect(data[2]).toBeCloseTo(1.0, 4);
      expect(data[3]).toBeCloseTo(1.0, 4);
      expect(data[4]).toBeCloseTo(0.0, 4);
      expect(data[5]).toBeCloseTo(0.0, 4);
    });
  });

  describe('torch.linalg.matrix_power', () => {
    it('computes A^n for positive n', async () => {
      const A = torch.tensor([[1.0, 1.0], [0.0, 1.0]]);
      const A2 = torch.linalg.matrix_power(A, 2);
      const data = Array.from(await A2.toArray());
      // [[1,1],[0,1]]^2 = [[1,2],[0,1]]
      expect(data[0]).toBeCloseTo(1.0, 4);
      expect(data[1]).toBeCloseTo(2.0, 4);
      expect(data[2]).toBeCloseTo(0.0, 4);
      expect(data[3]).toBeCloseTo(1.0, 4);
    });

    it('computes A^0 = I', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const A0 = torch.linalg.matrix_power(A, 0);
      const expected = Array.from(await torch.eye(2).toArray()) as Float32Array;
      const actual = Array.from(await A0.toArray()) as Float32Array;
      expect(maxAbsDiff(expected, actual)).toBeLessThan(1e-4);
    });

    it('computes A^n for negative n (inverse power)', async () => {
      const A = torch.tensor([[2.0, 0.0], [0.0, 3.0]]);
      const Aneg1 = torch.linalg.matrix_power(A, -1);
      const data = Array.from(await Aneg1.toArray());
      // inv([[2,0],[0,3]]) = [[0.5,0],[0,1/3]]
      expect(data[0]).toBeCloseTo(0.5, 4);
      expect(data[1]).toBeCloseTo(0.0, 4);
      expect(data[2]).toBeCloseTo(0.0, 4);
      expect(data[3]).toBeCloseTo(1 / 3, 4);
    });
  });

  describe('torch.linalg.vecdot', () => {
    it('computes dot product of two vectors', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0]);
      const y = torch.tensor([4.0, 5.0, 6.0]);
      const result = torch.linalg.vecdot(x, y);
      const data = Array.from(await result.toArray());
      // 1*4 + 2*5 + 3*6 = 32
      expect(data[0]).toBeCloseTo(32.0, 4);
    });

    it('computes batched vector dot products', async () => {
      const x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const y = torch.tensor([[5.0, 6.0], [7.0, 8.0]]);
      const result = torch.linalg.vecdot(x, y);
      const data = Array.from(await result.toArray());
      // [1*5+2*6, 3*7+4*8] = [17, 53]
      expect(data[0]).toBeCloseTo(17.0, 4);
      expect(data[1]).toBeCloseTo(53.0, 4);
    });

    it('computes dot product along specified dimension', async () => {
      const x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]);
      const y = torch.tensor([[1.0, 1.0], [1.0, 1.0]]);
      const result = torch.linalg.vecdot(x, y, 0);
      const data = Array.from(await result.toArray());
      // along dim 0: [1*1+3*1, 2*1+4*1] = [4, 6]
      expect(data[0]).toBeCloseTo(4.0, 4);
      expect(data[1]).toBeCloseTo(6.0, 4);
    });
  });
});
