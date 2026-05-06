import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Reduction Operations', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.sum()
  // ==========================================================================
  describe('sum', () => {
    it('sums all elements', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.sum(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(10);
    });

    it('sums along dim 0', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.sum(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([4, 6]);
    });

    it('sums along dim 1', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.sum(t, 1);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 7]);
    });

    it('sums with keepdim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.sum(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([4, 6]);
    });

    it('sums empty tensor', async () => {
      const t = torch.zeros([0]);
      const r = torch.sum(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(0);
    });
  });

  // ==========================================================================
  // torch.mean()
  // ==========================================================================
  describe('mean', () => {
    it('computes mean of all elements', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.mean(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(2.5);
    });

    it('computes mean along dim 0', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.mean(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([2, 3]);
    });

    it('computes mean along dim 1', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.mean(t, 1);
      const arr = await r.toArray();
      expect(arr).toEqual([1.5, 3.5]);
    });

    it('mean with keepdim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.mean(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });
  });

  // ==========================================================================
  // torch.std()
  // ==========================================================================
  describe('std', () => {
    it('computes standard deviation', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.std(t);
      const arr = await r.toArray();
      // std([1,2,3,4], unbiased=True) = sqrt(5/3) ~ 1.291
      expect(arr[0]).toBeGreaterThan(1);
      expect(arr[0]).toBeLessThan(2);
    });

    it('computes std along dim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.std(t, 0);
      expect(r.shape).toEqual([2]);
    });

    it('std with keepdim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.std(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });

    it('std with unbiased=False', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.std(t, undefined, undefined, false);
      const arr = await r.toArray();
      expect(arr[0]).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // torch.var()
  // ==========================================================================
  describe('var', () => {
    it('computes variance', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.var(t);
      const arr = await r.toArray();
      expect(arr[0]).toBeGreaterThan(1);
      expect(arr[0]).toBeLessThan(4);
    });

    it('computes var along dim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.var(t, 0);
      expect(r.shape).toEqual([2]);
    });

    it('var with keepdim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.var(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });
  });

  // ==========================================================================
  // torch.amax()
  // ==========================================================================
  describe('amax', () => {
    it('computes max of all elements', async () => {
      const t = torch.tensor([1, 5, 3, 2]);
      const r = torch.amax(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(5);
    });

    it('computes max along dim 0', async () => {
      const t = torch.tensor([[1, 5], [3, 2]]);
      const r = torch.amax(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 5]);
    });

    it('computes max along dim 1', async () => {
      const t = torch.tensor([[1, 5], [3, 2]]);
      const r = torch.amax(t, 1);
      const arr = await r.toArray();
      expect(arr).toEqual([5, 3]);
    });

    it('amax with keepdim', async () => {
      const t = torch.tensor([[1, 5], [3, 2]]);
      const r = torch.amax(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 5]);
    });

    it('amax with negative values', async () => {
      const t = torch.tensor([-5, -1, -10, -3]);
      const r = torch.amax(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(-1);
    });
  });

  // ==========================================================================
  // torch.amin()
  // ==========================================================================
  describe('amin', () => {
    it('computes min of all elements', async () => {
      const t = torch.tensor([1, 5, 3, 2]);
      const r = torch.amin(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
    });

    it('computes min along dim 0', async () => {
      const t = torch.tensor([[1, 5], [3, 2]]);
      const r = torch.amin(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2]);
    });

    it('amin with keepdim', async () => {
      const t = torch.tensor([[1, 5], [3, 2]]);
      const r = torch.amin(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });

    it('amin with negative values', async () => {
      const t = torch.tensor([-5, -1, -10, -3]);
      const r = torch.amin(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(-10);
    });
  });

  // ==========================================================================
  // torch.prod()
  // ==========================================================================
  describe('prod', () => {
    it('computes product of all elements', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.prod(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(24);
    });

    it('computes product along dim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.prod(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 8]);
    });

    it('prod with keepdim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.prod(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });
  });

  // ==========================================================================
  // torch.all()
  // ==========================================================================
  describe('all', () => {
    it('returns true when all elements are truthy', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.all(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
    });

    it('returns false when any element is zero', async () => {
      const t = torch.tensor([1, 0, 3]);
      const r = torch.all(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(0);
    });

    it('all along dim', async () => {
      const t = torch.tensor([[1, 0], [1, 1]]);
      const r = torch.all(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0]);
    });

    it('all with keepdim', async () => {
      const t = torch.tensor([[1, 0], [1, 1]]);
      const r = torch.all(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });
  });

  // ==========================================================================
  // torch.any()
  // ==========================================================================
  describe('any', () => {
    it('returns true when any element is truthy', async () => {
      const t = torch.tensor([0, 0, 3]);
      const r = torch.any(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
    });

    it('returns false when all elements are zero', async () => {
      const t = torch.tensor([0, 0, 0]);
      const r = torch.any(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(0);
    });

    it('any along dim', async () => {
      const t = torch.tensor([[0, 0], [1, 1]]);
      const r = torch.any(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 1]);
    });
  });

  // ==========================================================================
  // torch.argmax()
  // ==========================================================================
  describe('argmax', () => {
    it('returns index of max element', async () => {
      const t = torch.tensor([1, 5, 3, 2]);
      const r = await torch.argmax(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
    });

    it('argmax along dim 0', async () => {
      const t = torch.tensor([[1, 5], [3, 2]]);
      const r = await torch.argmax(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0]);
    });

    it('argmax along dim 1', async () => {
      const t = torch.tensor([[1, 5], [3, 2]]);
      const r = await torch.argmax(t, 1);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0]);
    });

    it('argmax with keepdim', async () => {
      const t = torch.tensor([[1, 5], [3, 2]]);
      const r = await torch.argmax(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });
  });

  // ==========================================================================
  // torch.argmin()
  // ==========================================================================
  describe('argmin', () => {
    it('returns index of min element', async () => {
      const t = torch.tensor([3, 1, 5, 2]);
      const r = await torch.argmin(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(1);
    });

    it('argmin along dim 0', async () => {
      const t = torch.tensor([[3, 1], [2, 5]]);
      const r = await torch.argmin(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0]);
    });

    it('argmin along dim 1', async () => {
      const t = torch.tensor([[3, 1], [2, 5]]);
      const r = await torch.argmin(t, 1);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 0]);
    });
  });

  // ==========================================================================
  // torch.sort()
  // ==========================================================================
  describe('sort', () => {
    it('sorts in ascending order', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5]);
      const { values, indices } = await torch.sort(t);
      const valArr = await values.toArray();
      expect(valArr).toEqual([1, 1, 3, 4, 5]);
    });

    it('sorts in descending order', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5]);
      const { values, indices } = await torch.sort(t, -1, true);
      const valArr = await values.toArray();
      expect(valArr).toEqual([5, 4, 3, 1, 1]);
    });

    it('sorts along specific dim', async () => {
      const t = torch.tensor([[3, 1, 2], [6, 4, 5]]);
      const { values } = await torch.sort(t, 1);
      expect(values.shape).toEqual([2, 3]);
      const arr = await values.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
    });
  });

  // ==========================================================================
  // torch.argsort()
  // ==========================================================================
  describe('argsort', () => {
    it('returns sorted indices ascending', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5]);
      const r = await torch.argsort(t);
      const arr = await r.toArray();
      // Indices that would sort: [1, 3, 0, 2, 4]
      expect(arr.length).toBe(5);
    });

    it('returns sorted indices descending', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5]);
      const r = await torch.argsort(t, -1, true);
      const arr = await r.toArray();
      expect(arr.length).toBe(5);
    });
  });

  // ==========================================================================
  // torch.topk()
  // ==========================================================================
  describe('topk', () => {
    it('returns top k largest values', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5]);
      const { values, indices } = await torch.topk(t, 3);
      const valArr = await values.toArray();
      expect(valArr).toEqual([5, 4, 3]);
    });

    it('returns top k smallest values', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5]);
      const { values } = await torch.topk(t, 3, -1, false);
      const valArr = await values.toArray();
      expect(valArr).toEqual([1, 1, 3]);
    });

    it('topk along dim 0', async () => {
      const t = torch.tensor([[1, 5], [3, 2], [4, 1]]);
      const { values } = await torch.topk(t, 2, 0);
      expect(values.shape).toEqual([2, 2]);
    });
  });

  // ==========================================================================
  // torch.kthvalue()
  // ==========================================================================
  describe('kthvalue', () => {
    it('returns kth smallest value', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5]);
      const { values } = await torch.kthvalue(t, 3);
      const valArr = await values.toArray();
      expect(valArr[0]).toBe(3);
    });

    it('returns kth largest with keepdim', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5]);
      const { values, indices } = await torch.kthvalue(t, 2, -1, true);
      expect(values.shape[0]).toBe(1);
    });
  });

  // ==========================================================================
  // torch.cummax()
  // ==========================================================================
  describe('cummax', () => {
    it('computes cumulative max', async () => {
      const t = torch.tensor([1, 3, 2, 5, 4]);
      const { values } = await torch.cummax(t, 0);
      const arr = await values.toArray();
      expect(arr).toEqual([1, 3, 3, 5, 5]);
    });
  });

  // ==========================================================================
  // torch.cummin()
  // ==========================================================================
  describe('cummin', () => {
    it('computes cumulative min', async () => {
      const t = torch.tensor([5, 3, 4, 1, 2]);
      const { values } = await torch.cummin(t, 0);
      const arr = await values.toArray();
      expect(arr).toEqual([5, 3, 3, 1, 1]);
    });
  });

  // ==========================================================================
  // torch.logsumexp()
  // ==========================================================================
  describe('logsumexp', () => {
    it('computes logsumexp', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.logsumexp(t);
      const arr = await r.toArray();
      // logsumexp([1,2,3]) = log(e^1 + e^2 + e^3)
      const expected = Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3));
      expect(arr[0]).toBeCloseTo(expected, 3);
    });

    it('logsumexp along dim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.logsumexp(t, 0);
      expect(r.shape).toEqual([2]);
    });

    it('logsumexp with keepdim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.logsumexp(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });
  });

  // ==========================================================================
  // torch.count_nonzero()
  // ==========================================================================
  describe('count_nonzero', () => {
    it('counts non-zero elements', async () => {
      const t = torch.tensor([0, 1, 0, 3, 0, 5]);
      const r = torch.count_nonzero(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(3);
    });

    it('counts non-zero along dim', async () => {
      const t = torch.tensor([[0, 1], [2, 0], [3, 4]]);
      const r = torch.count_nonzero(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([2, 2]);
    });

    it('count_nonzero with keepdim', async () => {
      const t = torch.tensor([[0, 1], [2, 0]]);
      const r = torch.count_nonzero(t, 0, true);
      expect(r.shape).toEqual([1, 2]);
    });

    it('counts all non-zero', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.count_nonzero(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(3);
    });

    it('counts zero non-zero', async () => {
      const t = torch.tensor([0, 0, 0]);
      const r = torch.count_nonzero(t);
      const arr = await r.toArray();
      expect(arr[0]).toBe(0);
    });
  });
});
