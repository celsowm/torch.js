import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Tensor Sorting Methods', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ========== sort() ==========
  describe('tensor.sort()', () => {
    it('sorts in ascending order', async () => {
      const t = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6]);
      const result = await t.sort(-1, false);
      const valArr = await result.values.toArray();
      const idxArr = await result.indices.toArray();
      expect(Array.from(valArr)).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
      // Check that indices are valid
      expect(idxArr.length).toBe(8);
    });

    it('sorts in descending order', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.sort(-1, true);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([4, 3, 2, 1]);
    });

    it('preserves shape', async () => {
      const t = torch.tensor([[3, 1, 2], [6, 5, 4]]);
      const result = await t.sort(-1, false);
      expect(result.values.shape).toEqual([2, 3]);
      expect(result.indices.shape).toEqual([2, 3]);
    });

    it('sorts along specific dimension', async () => {
      const t = torch.tensor([[3, 1], [2, 4]]);
      const result = await t.sort(0, false);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([2, 1, 3, 4]);
    });

    it('already sorted tensor', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const result = await t.sort(-1, false);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([1, 2, 3, 4, 5]);
    });

    it('single element sort', async () => {
      const t = torch.tensor([42]);
      const result = await t.sort(-1, false);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([42]);
      const idxArr = await result.indices.toArray();
      expect(Array.from(idxArr)).toEqual([0]);
    });

    it('duplicate values', async () => {
      const t = torch.tensor([2, 2, 1, 1, 3, 3]);
      const result = await t.sort(-1, false);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([1, 1, 2, 2, 3, 3]);
    });

    it('negative values', async () => {
      const t = torch.tensor([-3, 1, -5, 2, 0]);
      const result = await t.sort(-1, false);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([-5, -3, 0, 1, 2]);
    });
  });

  // ========== argsort() ==========
  describe('tensor.argsort()', () => {
    it('returns indices for ascending sort', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.argsort(-1, false);
      const arr = await result.toArray();
      // Sorted: [1, 2, 3, 4] at original indices [1, 3, 0, 2]
      expect(Array.from(arr)).toEqual([1, 3, 0, 2]);
    });

    it('returns indices for descending sort', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.argsort(-1, true);
      const arr = await result.toArray();
      // Sorted desc: [4, 3, 2, 1] at original indices [2, 0, 3, 1]
      expect(Array.from(arr)).toEqual([2, 0, 3, 1]);
    });

    it('preserves shape', async () => {
      const t = torch.tensor([[3, 1], [4, 2]]);
      const result = await t.argsort(-1, false);
      expect(result.shape).toEqual([2, 2]);
    });

    it('argsort along specific dimension', async () => {
      const t = torch.tensor([[3, 1], [2, 4]]);
      const result = await t.argsort(0, false);
      expect(result.shape).toEqual([2, 2]);
    });

    it('all equal values', async () => {
      const t = torch.tensor([5, 5, 5]);
      const result = await t.argsort(-1, false);
      const arr = await result.toArray();
      // All indices should be valid (0, 1, 2 in some order)
      expect(arr.length).toBe(3);
      const sortedIdx = Array.from(arr).sort((a, b) => a - b);
      expect(sortedIdx).toEqual([0, 1, 2]);
    });

    it('single element', async () => {
      const t = torch.tensor([42]);
      const result = await t.argsort(-1, false);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0]);
    });
  });

  // ========== topk() ==========
  describe('tensor.topk()', () => {
    it('returns k largest values', async () => {
      const t = torch.tensor([3, 1, 4, 2, 5]);
      const result = await t.topk(3, -1, true);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([5, 4, 3]);
    });

    it('returns k smallest values', async () => {
      const t = torch.tensor([3, 1, 4, 2, 5]);
      const result = await t.topk(3, -1, false);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([1, 2, 3]);
    });

    it('preserves k in output shape', async () => {
      const t = torch.tensor([3, 1, 4, 2, 5]);
      const result = await t.topk(3, -1, true);
      expect(result.values.shape).toEqual([3]);
      expect(result.indices.shape).toEqual([3]);
    });

    it('k equals tensor size', async () => {
      const t = torch.tensor([3, 1, 4]);
      const result = await t.topk(3, -1, true);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([4, 3, 1]);
    });

    it('k equals 1', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.topk(1, -1, true);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([4]);
    });

    it('topk along specific dimension', async () => {
      const t = torch.tensor([[1, 2, 3], [6, 5, 4]]);
      const result = await t.topk(2, -1, true);
      expect(result.values.shape).toEqual([2, 2]);
    });

    it('throws when k > dimension size', async () => {
      const t = torch.tensor([1, 2, 3]);
      try {
        await t.topk(5, -1, true);
        expect.fail('Should have thrown');
      } catch (e: any) {
        expect(e.message).toContain('k');
      }
    });
  });

  // ========== kthvalue() ==========
  describe('tensor.kthvalue()', () => {
    it('returns 1st smallest value', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.kthvalue(1, -1);
      expect((await result.values.item())).toBeCloseTo(1, 5);
    });

    it('returns 3rd smallest value', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.kthvalue(3, -1);
      expect((await result.values.item())).toBeCloseTo(3, 5);
    });

    it('returns largest value', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.kthvalue(4, -1);
      expect((await result.values.item())).toBeCloseTo(4, 5);
    });

    it('returns index along with value', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.kthvalue(2, -1);
      expect((await result.values.item())).toBeCloseTo(2, 5);
      const idx = await result.indices.item();
      expect(idx).toBeCloseTo(3, 5);
    });

    it('kthvalue with keepdim', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.kthvalue(2, -1, true);
      expect(result.values.shape).toEqual([1]);
      expect(result.indices.shape).toEqual([1]);
    });

    it('kthvalue without keepdim', async () => {
      const t = torch.tensor([3, 1, 4, 2]);
      const result = await t.kthvalue(2, -1, false);
      expect(result.values.shape).toEqual([]);
      expect(result.indices.shape).toEqual([]);
    });

    it('2D tensor kthvalue', async () => {
      const t = torch.tensor([[1, 2, 3], [6, 5, 4]]);
      const result = await t.kthvalue(2, -1);
      expect(result.values.shape).toEqual([2]);
      const arr = await result.values.toArray();
      expect(Array.from(arr)).toEqual([2, 5]);
    });
  });

  // ========== cummax() ==========
  describe('tensor.cummax()', () => {
    it('cumulative maximum along dimension', async () => {
      const t = torch.tensor([1, 3, 2, 5, 4]);
      const result = await t.cummax(0);
      const valArr = await result.values.toArray();
      const idxArr = await result.indices.toArray();
      expect(Array.from(valArr)).toEqual([1, 3, 3, 5, 5]);
      expect(Array.from(idxArr)).toEqual([0, 1, 1, 3, 3]);
    });

    it('cummax of already sorted', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const result = await t.cummax(0);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([1, 2, 3, 4, 5]);
    });

    it('cummax of reverse sorted', async () => {
      const t = torch.tensor([5, 4, 3, 2, 1]);
      const result = await t.cummax(0);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([5, 5, 5, 5, 5]);
    });

    it('cummax preserves shape', async () => {
      const t = torch.tensor([1, 3, 2, 5]);
      const result = await t.cummax(0);
      expect(result.values.shape).toEqual([4]);
      expect(result.indices.shape).toEqual([4]);
    });

    it('cummax with negative values', async () => {
      const t = torch.tensor([-3, -1, -2, 0]);
      const result = await t.cummax(0);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([-3, -1, -1, 0]);
    });

    it('cummax 2D along specific dimension', async () => {
      const t = torch.tensor([[1, 3, 2], [4, 2, 5]]);
      const result = await t.cummax(1);
      expect(result.values.shape).toEqual([2, 3]);
      const arr = await result.values.toArray();
      expect(Array.from(arr)).toEqual([1, 3, 3, 4, 4, 5]);
    });
  });

  // ========== cummin() ==========
  describe('tensor.cummin()', () => {
    it('cumulative minimum along dimension', async () => {
      const t = torch.tensor([5, 3, 4, 2, 1]);
      const result = await t.cummin(0);
      const valArr = await result.values.toArray();
      const idxArr = await result.indices.toArray();
      expect(Array.from(valArr)).toEqual([5, 3, 3, 2, 1]);
      expect(Array.from(idxArr)).toEqual([0, 1, 1, 3, 4]);
    });

    it('cummin of already sorted ascending', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const result = await t.cummin(0);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([1, 1, 1, 1, 1]);
    });

    it('cummin preserves shape', async () => {
      const t = torch.tensor([5, 3, 4, 2]);
      const result = await t.cummin(0);
      expect(result.values.shape).toEqual([4]);
      expect(result.indices.shape).toEqual([4]);
    });

    it('cummin with negative values', async () => {
      const t = torch.tensor([-1, -3, -2, 0]);
      const result = await t.cummin(0);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([-1, -3, -3, -3]);
    });

    it('cummin 2D along specific dimension', async () => {
      const t = torch.tensor([[3, 1, 2], [4, 6, 5]]);
      const result = await t.cummin(1);
      expect(result.values.shape).toEqual([2, 3]);
      const arr = await result.values.toArray();
      expect(Array.from(arr)).toEqual([3, 1, 1, 4, 4, 4]);
    });

    it('single element cummin', async () => {
      const t = torch.tensor([42]);
      const result = await t.cummin(0);
      const valArr = await result.values.toArray();
      expect(Array.from(valArr)).toEqual([42]);
      const idxArr = await result.indices.toArray();
      expect(Array.from(idxArr)).toEqual([0]);
    });
  });
});
