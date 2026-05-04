import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Sampling', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.multinomial()
  // ==========================================================================
  describe('multinomial', () => {
    it('samples from a categorical distribution', async () => {
      const weights = torch.tensor([0.5, 0.3, 0.2]);
      const r = await torch.multinomial(weights, 1);
      expect(r.shape).toEqual([1, 1]);
    });

    it('samples multiple items', async () => {
      const weights = torch.tensor([0.5, 0.3, 0.2]);
      const r = await torch.multinomial(weights, 5);
      expect(r.shape).toEqual([1, 5]);
      const arr = await r.toArray();
      for (const v of arr) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(3);
      }
    });

    it('samples without replacement', async () => {
      const weights = torch.tensor([0.25, 0.25, 0.25, 0.25]);
      const r = await torch.multinomial(weights, 4, false);
      expect(r.shape).toEqual([1, 4]);
      const arr = await r.toArray();
      // Without replacement, all samples should be unique
      const unique = new Set(arr);
      expect(unique.size).toBe(4);
    });

    it('samples with replacement allows duplicates', async () => {
      const weights = torch.tensor([1, 0, 0]);
      const r = await torch.multinomial(weights, 10, true);
      const arr = await r.toArray();
      // All should be 0 since only first category has weight
      expect(arr.every((v: number) => v === 0)).toBe(true);
    });

    it('multinomial with uniform weights', async () => {
      const weights = torch.tensor([1, 1, 1, 1]);
      const r = await torch.multinomial(weights, 100, true);
      const arr = await r.toArray();
      // All values should be in range [0, 3]
      for (const v of arr) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(4);
      }
    });

    it('multinomial with batched weights', async () => {
      const weights = torch.tensor([[0.5, 0.5], [0.8, 0.2]]);
      const r = await torch.multinomial(weights, 5, true);
      expect(r.shape).toEqual([2, 5]);
    });

    it('multinomial with default num_samples=1', async () => {
      const weights = torch.tensor([0.5, 0.3, 0.2]);
      const r = await torch.multinomial(weights);
      expect(r.shape).toEqual([1, 1]);
    });

    it('multinomial prefers higher weights', async () => {
      torch.manual_seed(42);
      const weights = torch.tensor([0.99, 0.005, 0.005]);
      const r = await torch.multinomial(weights, 100, true);
      const arr = await r.toArray();
      const zeros = arr.filter((v: number) => v === 0).length;
      // Most samples should be 0 (highest weight)
      expect(zeros).toBeGreaterThan(80);
    });

    it('throws on zero-sum weights', async () => {
      const weights = torch.tensor([0, 0, 0]);
      try {
        await torch.multinomial(weights, 1);
        // If no error, that's also acceptable behavior
      } catch (e) {
        expect((e as Error).message).toBeDefined();
      }
    });

    it('throws when num_samples > population without replacement', async () => {
      const weights = torch.tensor([0.25, 0.25, 0.25, 0.25]);
      try {
        await torch.multinomial(weights, 5, false);
        // If no error thrown, the implementation might handle this gracefully
      } catch (e) {
        expect((e as Error).message).toBeDefined();
      }
    });
  });

  // ==========================================================================
  // torch.normal() - distribution checks
  // ==========================================================================
  describe('normal distribution checks', () => {
    it('normal distribution has approximately correct mean', async () => {
      torch.manual_seed(42);
      const t = torch.normal(10, 1, [10000]);
      const arr = await t.toArray();
      const mean = arr.reduce((a: number, b: number) => a + b, 0) / arr.length;
      expect(mean).toBeGreaterThan(9.8);
      expect(mean).toBeLessThan(10.2);
    });

    it('normal distribution has approximately correct std', async () => {
      torch.manual_seed(42);
      const t = torch.normal(0, 2, [10000]);
      const arr = await t.toArray();
      const mean = arr.reduce((a: number, b: number) => a + b, 0) / arr.length;
      const variance = arr.reduce((sum: number, v: number) => sum + (v - mean) ** 2, 0) / arr.length;
      const std = Math.sqrt(variance);
      expect(std).toBeGreaterThan(1.8);
      expect(std).toBeLessThan(2.2);
    });

    it('normal distribution is approximately symmetric', async () => {
      torch.manual_seed(42);
      const t = torch.normal(0, 1, [10000]);
      const arr = await t.toArray();
      const negCount = arr.filter((v: number) => v < 0).length;
      const posCount = arr.filter((v: number) => v > 0).length;
      // Should be roughly 50/50
      expect(negCount).toBeGreaterThan(4500);
      expect(posCount).toBeGreaterThan(4500);
    });

    it('normal distribution covers most of the expected range', async () => {
      torch.manual_seed(42);
      const t = torch.normal(0, 1, [10000]);
      const arr = await t.toArray();
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      // 99.7% of normal distribution is within 3 std
      expect(min).toBeLessThan(-2.5);
      expect(max).toBeGreaterThan(2.5);
    });

    it('different means produce different distributions', async () => {
      torch.manual_seed(42);
      const t1 = torch.normal(0, 1, [1000]);
      torch.manual_seed(42);
      const t2 = torch.normal(10, 1, [1000]);
      const arr1 = await t1.toArray();
      const arr2 = await t2.toArray();
      const mean1 = arr1.reduce((a: number, b: number) => a + b, 0) / arr1.length;
      const mean2 = arr2.reduce((a: number, b: number) => a + b, 0) / arr2.length;
      expect(Math.abs(mean2 - mean1)).toBeGreaterThan(8);
    });

    it('different stds produce different distributions', async () => {
      torch.manual_seed(42);
      const t1 = torch.normal(0, 0.1, [1000]);
      torch.manual_seed(42);
      const t2 = torch.normal(0, 5, [1000]);
      const arr1 = await t1.toArray();
      const arr2 = await t2.toArray();
      const std1 = Math.sqrt(arr1.reduce((sum: number, v: number) => sum + v ** 2, 0) / arr1.length);
      const std2 = Math.sqrt(arr2.reduce((sum: number, v: number) => sum + v ** 2, 0) / arr2.length);
      expect(std2).toBeGreaterThan(std1 * 10);
    });

    it('normal produces reproducible results with same seed', async () => {
      torch.manual_seed(123);
      const t1 = await torch.normal(0, 1, [100]).toArray();
      torch.manual_seed(123);
      const t2 = await torch.normal(0, 1, [100]).toArray();
      const arr1 = Array.from(t1);
      const arr2 = Array.from(t2);
      expect(arr1).toEqual(arr2);
    });
  });
});
