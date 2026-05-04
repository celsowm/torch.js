import { describe, it, expect, beforeAll } from 'vitest';
import torch from '../src';

describe('torch.js Creation Operations', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('empty', () => {
    it('creates a tensor with uninitialized data', () => {
      const t = torch.empty([2, 3]);
      expect(t.shape).toEqual([2, 3]);
      expect(t.dtype).toBe('float32');
    });

    it('respects dtype option', () => {
      const t = torch.empty([4], { dtype: 'int32' });
      expect(t.shape).toEqual([4]);
      expect(t.dtype).toBe('int32');
    });
  });

  describe('randint', () => {
    it('creates random integers in [low, high)', async () => {
      const t = torch.randint(0, 10, [1000]);
      expect(t.shape).toEqual([1000]);
      const arr = await t.toArray();
      for (const v of arr) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(10);
        expect(Number.isInteger(v)).toBe(true);
      }
    });

    it('works with requires_grad', () => {
      const t = torch.randint(0, 5, [3], { requires_grad: true });
      expect(t.requires_grad).toBe(true);
    });
  });

  describe('randperm', () => {
    it('creates a random permutation of 0..n-1', async () => {
      const n = 10;
      const t = torch.randperm(n);
      expect(t.shape).toEqual([n]);
      const arr = await t.toArray();
      expect(arr.length).toBe(n);
      const sorted = [...arr].sort((a, b) => a - b);
      for (let i = 0; i < n; i++) {
        expect(sorted[i]).toBe(i);
      }
    });

    it('produces different permutations on successive calls', async () => {
      const t1 = await torch.randperm(20).toArray();
      const t2 = await torch.randperm(20).toArray();
      const same = t1.every((v: number, i: number) => v === t2[i]);
      expect(same).toBe(false);
    });
  });

  describe('normal', () => {
    it('creates tensor with specified mean and std', async () => {
      const t = torch.normal(5, 2, [1000]);
      expect(t.shape).toEqual([1000]);
      const arr = await t.toArray();
      const mean = arr.reduce((a: number, b: number) => a + b, 0) / arr.length;
      expect(mean).toBeGreaterThan(3);
      expect(mean).toBeLessThan(7);
    });
  });

  describe('scalar_tensor', () => {
    it('creates a scalar tensor', () => {
      const t = torch.scalar_tensor(42);
      expect(t.shape).toEqual([]);
      expect(t.numel()).toBe(1);
    });
  });

  describe('as_tensor', () => {
    it('converts data to tensor like tensor()', () => {
      const t = torch.as_tensor([1, 2, 3]);
      expect(t.shape).toEqual([3]);
    });
  });

  describe('split', () => {
    it('splits a tensor into chunks', () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const chunks = torch.split(t, 2);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([2]);
    });
  });
});
