import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Tensor Creation', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.tensor()
  // ==========================================================================
  describe('tensor', () => {
    it('creates a 1D tensor from array', () => {
      const t = torch.tensor([1, 2, 3]);
      expect(t.shape).toEqual([3]);
      expect(t.dtype).toBe('float32');
    });

    it('creates a 2D tensor from nested array', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      expect(t.shape).toEqual([2, 2]);
    });

    it('creates a scalar (0D) tensor', () => {
      const t = torch.tensor(42);
      expect(t.shape).toEqual([]);
      expect(t.numel()).toBe(1);
    });

    it('respects dtype option', () => {
      const t = torch.tensor([1, 2, 3], { dtype: 'int32' });
      expect(t.dtype).toBe('int32');
    });

    it('creates tensor with bool dtype', () => {
      const t = torch.tensor([true, false, true], { dtype: 'bool' });
      expect(t.dtype).toBe('bool');
    });

    it('creates empty tensor (0 elements)', () => {
      const t = torch.tensor([]);
      expect(t.shape).toEqual([0]);
    });

    it('creates tensor with requires_grad', () => {
      const t = torch.tensor([1.0, 2.0], { requires_grad: true });
      expect(t.requires_grad).toBe(true);
    });
  });

  // ==========================================================================
  // torch.zeros()
  // ==========================================================================
  describe('zeros', () => {
    it('creates a tensor of zeros', async () => {
      const t = torch.zeros([2, 3]);
      expect(t.shape).toEqual([2, 3]);
      const arr = await t.toArray();
      expect(arr.every((v: number) => v === 0)).toBe(true);
    });

    it('creates zeros with int32 dtype', async () => {
      const t = torch.zeros([3], { dtype: 'int32' });
      expect(t.dtype).toBe('int32');
      const arr = await t.toArray();
      expect(arr.every((v: number) => v === 0)).toBe(true);
    });

    it('creates a scalar zero', async () => {
      const t = torch.zeros([]);
      expect(t.shape).toEqual([]);
      const arr = await t.toArray();
      expect(arr[0]).toBe(0);
    });

    it('creates empty zeros tensor', () => {
      const t = torch.zeros([0]);
      expect(t.shape).toEqual([0]);
    });

    it('creates large zeros tensor', () => {
      const t = torch.zeros([100, 100]);
      expect(t.shape).toEqual([100, 100]);
      expect(t.numel()).toBe(10000);
    });
  });

  // ==========================================================================
  // torch.ones()
  // ==========================================================================
  describe('ones', () => {
    it('creates a tensor of ones', async () => {
      const t = torch.ones([2, 3]);
      expect(t.shape).toEqual([2, 3]);
      const arr = await t.toArray();
      expect(arr.every((v: number) => v === 1)).toBe(true);
    });

    it('creates ones with int32 dtype', () => {
      const t = torch.ones([3], { dtype: 'int32' });
      expect(t.dtype).toBe('int32');
    });

    it('creates a scalar one', async () => {
      const t = torch.ones([]);
      const arr = await t.toArray();
      expect(arr[0]).toBe(1);
    });

    it('creates empty ones tensor', () => {
      const t = torch.ones([0]);
      expect(t.shape).toEqual([0]);
    });
  });

  // ==========================================================================
  // torch.full()
  // ==========================================================================
  describe('full', () => {
    it('creates a tensor filled with a specific value', async () => {
      const t = torch.full([2, 3], 7.5);
      expect(t.shape).toEqual([2, 3]);
      const arr = await t.toArray();
      expect(arr.every((v: number) => v === 7.5)).toBe(true);
    });

    it('creates full tensor with int32 dtype', async () => {
      const t = torch.full([3], 42, { dtype: 'int32' });
      expect(t.dtype).toBe('int32');
      const arr = await t.toArray();
      expect(arr.every((v: number) => v === 42)).toBe(true);
    });

    it('creates scalar full tensor', async () => {
      const t = torch.full([], 5.0);
      expect(t.shape).toEqual([]);
      const arr = await t.toArray();
      expect(arr[0]).toBe(5);
    });

    it('creates full tensor with negative value', async () => {
      const t = torch.full([4], -3.14);
      const arr = await t.toArray();
      // float32 precision - -3.14 becomes approximately -3.14
      expect(arr.every((v: number) => Math.abs(v - (-3.14)) < 0.001)).toBe(true);
    });

    it('creates empty full tensor', () => {
      const t = torch.full([0], 1.0);
      expect(t.shape).toEqual([0]);
    });
  });

  // ==========================================================================
  // torch.empty()
  // ==========================================================================
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

    it('creates empty tensor with 0 elements', () => {
      const t = torch.empty([0]);
      expect(t.shape).toEqual([0]);
    });

    it('creates large empty tensor', () => {
      const t = torch.empty([50, 50]);
      expect(t.numel()).toBe(2500);
    });
  });

  // ==========================================================================
  // torch.zeros_like()
  // ==========================================================================
  describe('zeros_like', () => {
    it('creates zeros tensor with same shape', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const z = torch.zeros_like(t);
      expect(z.shape).toEqual([2, 2]);
      const arr = await z.toArray();
      expect(arr.every((v: number) => v === 0)).toBe(true);
    });

    it('preserves dtype by default', () => {
      const t = torch.tensor([1, 2, 3], { dtype: 'int32' });
      const z = torch.zeros_like(t);
      expect(z.dtype).toBe('int32');
    });

    it('overrides dtype when specified', () => {
      const t = torch.tensor([1, 2, 3], { dtype: 'int32' });
      const z = torch.zeros_like(t, { dtype: 'float32' });
      expect(z.dtype).toBe('float32');
    });
  });

  // ==========================================================================
  // torch.ones_like()
  // ==========================================================================
  describe('ones_like', () => {
    it('creates ones tensor with same shape', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const o = torch.ones_like(t);
      expect(o.shape).toEqual([2, 2]);
      const arr = await o.toArray();
      expect(arr.every((v: number) => v === 1)).toBe(true);
    });

    it('preserves dtype by default', () => {
      const t = torch.tensor([1, 2, 3], { dtype: 'int32' });
      const o = torch.ones_like(t);
      expect(o.dtype).toBe('int32');
    });
  });

  // ==========================================================================
  // torch.full_like()
  // ==========================================================================
  describe('full_like', () => {
    it('creates full tensor with same shape', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const f = torch.full_like(t, 9.0);
      expect(f.shape).toEqual([2, 2]);
      const arr = await f.toArray();
      expect(arr.every((v: number) => v === 9)).toBe(true);
    });

    it('preserves dtype by default', () => {
      const t = torch.tensor([1, 2, 3], { dtype: 'int32' });
      const f = torch.full_like(t, 5);
      expect(f.dtype).toBe('int32');
    });
  });

  // ==========================================================================
  // torch.empty_like()
  // ==========================================================================
  describe('empty_like', () => {
    it('creates empty tensor with same shape', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const e = torch.empty_like(t);
      expect(e.shape).toEqual([2, 2]);
    });

    it('preserves dtype by default', () => {
      const t = torch.tensor([1, 2, 3], { dtype: 'int32' });
      const e = torch.empty_like(t);
      expect(e.dtype).toBe('int32');
    });
  });

  // ==========================================================================
  // torch.rand()
  // ==========================================================================
  describe('rand', () => {
    it('creates random tensor in [0, 1)', async () => {
      const t = torch.rand([100]);
      expect(t.shape).toEqual([100]);
      const arr = await t.toArray();
      expect(arr.every((v: number) => v >= 0 && v < 1)).toBe(true);
    });

    it('creates 2D random tensor', () => {
      const t = torch.rand([3, 4]);
      expect(t.shape).toEqual([3, 4]);
    });

    it('creates random tensor with int32 dtype', () => {
      const t = torch.rand([5], { dtype: 'int32' });
      expect(t.dtype).toBe('int32');
    });

    it('produces different values on successive calls', async () => {
      const t1 = await torch.rand([50]).toArray();
      const t2 = await torch.rand([50]).toArray();
      const arr1 = Array.from(t1);
      const arr2 = Array.from(t2);
      const same = arr1.every((v, i) => v === arr2[i]);
      expect(same).toBe(false);
    });
  });

  // ==========================================================================
  // torch.randn()
  // ==========================================================================
  describe('randn', () => {
    it('creates tensor from normal distribution', async () => {
      const t = torch.randn([1000]);
      expect(t.shape).toEqual([1000]);
      const arr = await t.toArray();
      const mean = arr.reduce((a: number, b: number) => a + b, 0) / arr.length;
      // Mean should be approximately 0
      expect(Math.abs(mean)).toBeLessThan(0.2);
    });

    it('creates 2D tensor', () => {
      const t = torch.randn([3, 4]);
      expect(t.shape).toEqual([3, 4]);
    });

    it('values are spread around 0', async () => {
      const t = torch.randn([500]);
      const arr = await t.toArray();
      const negCount = arr.filter((v: number) => v < 0).length;
      const posCount = arr.filter((v: number) => v > 0).length;
      // Roughly half should be negative, half positive
      expect(negCount).toBeGreaterThan(100);
      expect(posCount).toBeGreaterThan(100);
    });
  });

  // ==========================================================================
  // torch.randint()
  // ==========================================================================
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

    it('works with non-zero low value', async () => {
      const t = torch.randint(5, 15, [500]);
      const arr = await t.toArray();
      for (const v of arr) {
        expect(v).toBeGreaterThanOrEqual(5);
        expect(v).toBeLessThan(15);
      }
    });

    it('works with requires_grad', () => {
      const t = torch.randint(0, 5, [3], { requires_grad: true });
      expect(t.requires_grad).toBe(true);
    });

    it('creates 2D tensor', () => {
      const t = torch.randint(0, 10, [3, 4]);
      expect(t.shape).toEqual([3, 4]);
    });
  });

  // ==========================================================================
  // torch.randperm()
  // ==========================================================================
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
      const arr1 = Array.from(t1);
      const arr2 = Array.from(t2);
      const same = arr1.every((v, i) => v === arr2[i]);
      expect(same).toBe(false);
    });

    it('works with n=1', async () => {
      const t = torch.randperm(1);
      const arr = await t.toArray();
      expect(arr[0]).toBe(0);
    });

    it('works with large n', async () => {
      const n = 100;
      const t = torch.randperm(n);
      const arr = await t.toArray();
      expect(arr.length).toBe(n);
      const sorted = [...arr].sort((a, b) => a - b);
      for (let i = 0; i < n; i++) {
        expect(sorted[i]).toBe(i);
      }
    });
  });

  // ==========================================================================
  // torch.normal()
  // ==========================================================================
  describe('normal', () => {
    it('creates tensor with specified mean and std', async () => {
      const t = torch.normal(5, 2, [1000]);
      expect(t.shape).toEqual([1000]);
      const arr = await t.toArray();
      const mean = arr.reduce((a: number, b: number) => a + b, 0) / arr.length;
      expect(mean).toBeGreaterThan(3);
      expect(mean).toBeLessThan(7);
    });

    it('creates tensor with zero mean', async () => {
      const t = torch.normal(0, 1, [1000]);
      const arr = await t.toArray();
      const mean = arr.reduce((a: number, b: number) => a + b, 0) / arr.length;
      expect(Math.abs(mean)).toBeLessThan(0.2);
    });

    it('creates tensor with different std', () => {
      const t = torch.normal(0, 5, [100]);
      expect(t.shape).toEqual([100]);
    });

    it('creates 2D tensor', () => {
      const t = torch.normal(0, 1, [3, 4]);
      expect(t.shape).toEqual([3, 4]);
    });
  });

  // ==========================================================================
  // torch.eye()
  // ==========================================================================
  describe('eye', () => {
    it('creates an identity matrix', async () => {
      const t = torch.eye(3);
      expect(t.shape).toEqual([3, 3]);
      const arr = await t.toArray();
      // Diagonal elements should be 1
      expect(arr[0]).toBe(1);
      expect(arr[4]).toBe(1);
      expect(arr[8]).toBe(1);
      // Off-diagonal should be 0
      expect(arr[1]).toBe(0);
      expect(arr[2]).toBe(0);
      expect(arr[3]).toBe(0);
    });

    it('creates rectangular identity matrix', async () => {
      const t = torch.eye(2, 4);
      expect(t.shape).toEqual([2, 4]);
      const arr = await t.toArray();
      expect(arr[0]).toBe(1);
      expect(arr[5]).toBe(1);
    });

    it('creates 1x1 identity', async () => {
      const t = torch.eye(1);
      const arr = await t.toArray();
      expect(arr[0]).toBe(1);
    });
  });

  // ==========================================================================
  // torch.arange()
  // ==========================================================================
  describe('arange', () => {
    it('creates a range from 0 to end', async () => {
      const t = torch.arange(5);
      expect(t.shape).toEqual([5]);
      const arr = await t.toArray();
      expect(arr).toEqual([0, 1, 2, 3, 4]);
    });

    it('creates a range from start to end', async () => {
      const t = torch.arange(2, 6);
      const arr = await t.toArray();
      expect(arr).toEqual([2, 3, 4, 5]);
    });

    it('creates a range with step', async () => {
      const t = torch.arange(0, 10, 2);
      const arr = await t.toArray();
      expect(arr).toEqual([0, 2, 4, 6, 8]);
    });

    it('creates range with float step', async () => {
      const t = torch.arange(0, 2, 0.5);
      const arr = await t.toArray();
      expect(arr).toEqual([0, 0.5, 1, 1.5]);
    });

    it('creates empty range when start >= end with positive step', () => {
      const t = torch.arange(5, 3);
      expect(t.shape[0]).toBe(0);
    });
  });

  // ==========================================================================
  // torch.linspace()
  // ==========================================================================
  describe('linspace', () => {
    it('creates evenly spaced values', async () => {
      const t = torch.linspace(0, 10, 5);
      expect(t.shape).toEqual([5]);
      const arr = await t.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo(2.5);
      expect(arr[2]).toBeCloseTo(5);
      expect(arr[3]).toBeCloseTo(7.5);
      expect(arr[4]).toBeCloseTo(10);
    });

    it('creates with 2 points', async () => {
      const t = torch.linspace(0, 1, 2);
      const arr = await t.toArray();
      expect(arr[0]).toBeCloseTo(0);
      expect(arr[1]).toBeCloseTo(1);
    });

    it('creates with 1 point', async () => {
      const t = torch.linspace(5, 10, 1);
      const arr = await t.toArray();
      expect(arr[0]).toBeCloseTo(5);
    });

    it('creates with negative range', async () => {
      const t = torch.linspace(-5, 5, 5);
      const arr = await t.toArray();
      expect(arr[0]).toBeCloseTo(-5);
      expect(arr[2]).toBeCloseTo(0);
      expect(arr[4]).toBeCloseTo(5);
    });
  });

  // ==========================================================================
  // torch.logspace()
  // ==========================================================================
  describe('logspace', () => {
    it('creates logarithmically spaced values', async () => {
      const t = torch.logspace(0, 2, 3);
      expect(t.shape).toEqual([3]);
      const arr = await t.toArray();
      expect(arr[0]).toBeCloseTo(1);   // 10^0
      expect(arr[1]).toBeCloseTo(10);  // 10^1
      expect(arr[2]).toBeCloseTo(100); // 10^2
    });

    it('creates with negative exponents', async () => {
      const t = torch.logspace(-2, 0, 3);
      const arr = await t.toArray();
      expect(arr[0]).toBeCloseTo(0.01);
      expect(arr[1]).toBeCloseTo(0.1);
      expect(arr[2]).toBeCloseTo(1);
    });
  });

  // ==========================================================================
  // torch.scalar_tensor()
  // ==========================================================================
  describe('scalar_tensor', () => {
    it('creates a scalar tensor', () => {
      const t = torch.scalar_tensor(42);
      expect(t.shape).toEqual([]);
      expect(t.numel()).toBe(1);
    });

    it('creates scalar with specific dtype', async () => {
      const t = torch.scalar_tensor(3.14, { dtype: 'float32' });
      expect(t.dtype).toBe('float32');
      const arr = await t.toArray();
      expect(arr[0]).toBeCloseTo(3.14);
    });

    it('creates scalar with int32 dtype', async () => {
      const t = torch.scalar_tensor(7, { dtype: 'int32' });
      expect(t.dtype).toBe('int32');
      const arr = await t.toArray();
      expect(arr[0]).toBe(7);
    });
  });

  // ==========================================================================
  // torch.as_tensor()
  // ==========================================================================
  describe('as_tensor', () => {
    it('converts array to tensor', () => {
      const t = torch.as_tensor([1, 2, 3]);
      expect(t.shape).toEqual([3]);
    });

    it('creates tensor with dtype', () => {
      const t = torch.as_tensor([1, 2, 3], { dtype: 'int32' });
      expect(t.dtype).toBe('int32');
    });

    it('creates scalar from number', () => {
      const t = torch.as_tensor(5);
      expect(t.shape).toEqual([]);
    });
  });

  // ==========================================================================
  // torch.manual_seed()
  // ==========================================================================
  describe('manual_seed', () => {
    it('sets seed for reproducibility', async () => {
      torch.manual_seed(42);
      const t1 = await torch.rand([10]).toArray();
      torch.manual_seed(42);
      const t2 = await torch.rand([10]).toArray();
      const arr1 = Array.from(t1);
      const arr2 = Array.from(t2);
      expect(arr1).toEqual(arr2);
    });

    it('different seeds produce different results', async () => {
      torch.manual_seed(1);
      const t1 = await torch.rand([20]).toArray();
      torch.manual_seed(2);
      const t2 = await torch.rand([20]).toArray();
      const arr1 = Array.from(t1);
      const arr2 = Array.from(t2);
      const same = arr1.every((v, i) => v === arr2[i]);
      expect(same).toBe(false);
    });

    it('seed affects randn as well', async () => {
      torch.manual_seed(123);
      const t1 = await torch.randn([10]).toArray();
      torch.manual_seed(123);
      const t2 = await torch.randn([10]).toArray();
      const arr1 = Array.from(t1);
      const arr2 = Array.from(t2);
      expect(arr1).toEqual(arr2);
    });

    it('seed affects randint as well', async () => {
      torch.manual_seed(99);
      const t1 = await torch.randint(0, 100, [10]).toArray();
      torch.manual_seed(99);
      const t2 = await torch.randint(0, 100, [10]).toArray();
      const arr1 = Array.from(t1);
      const arr2 = Array.from(t2);
      expect(arr1).toEqual(arr2);
    });
  });
});
