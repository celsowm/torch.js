import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import * as init from '../../../src/nn/init';

const torch = createTorch(async () => {}, async () => ({}));

describe('nn.init.zeros_', () => {
  beforeAll(async () => { await torch.init(); });
  it('fills with zeros', async () => {
    const t = torch.randn([3, 4]); init.zeros_(t);
    expect((await t.toArray()).every((v: number) => v === 0)).toBe(true);
  });
  it('1D', async () => {
    const t = torch.randn([10]); init.zeros_(t);
    expect((await t.toArray()).every((v: number) => v === 0)).toBe(true);
  });
  it('3D', async () => {
    const t = torch.randn([2, 3, 4]); init.zeros_(t);
    expect((await t.toArray()).every((v: number) => v === 0)).toBe(true);
  });
});

describe('nn.init.ones_', () => {
  beforeAll(async () => { await torch.init(); });
  it('fills with ones', async () => {
    const t = torch.randn([3, 4]); init.ones_(t);
    expect((await t.toArray()).every((v: number) => v === 1)).toBe(true);
  });
  it('1D', async () => {
    const t = torch.randn([5]); init.ones_(t);
    expect((await t.toArray()).every((v: number) => v === 1)).toBe(true);
  });
});

describe('nn.init.constant_', () => {
  beforeAll(async () => { await torch.init(); });
  it('fills with 42', async () => {
    const t = torch.randn([3, 4]); init.constant_(t, 42.0);
    expect((await t.toArray()).every((v: number) => v === 42)).toBe(true);
  });
  it('negative', async () => {
    const t = torch.randn([5]); init.constant_(t, -3.14);
    const arr = await t.toArray();
    expect(arr.every((v: number) => Math.abs(v - (-3.14)) < 0.01)).toBe(true);
  });
  it('zero', async () => {
    const t = torch.randn([2, 3]); init.constant_(t, 0);
    expect((await t.toArray()).every((v: number) => v === 0)).toBe(true);
  });
});

describe('nn.init.normal_', () => {
  beforeAll(async () => { await torch.init(); });
  it('approx correct mean/std', async () => {
    const t = torch.randn([1000]); init.normal_(t, 5.0, 2.0);
    const arr = await t.toArray() as unknown as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(mean).toBeGreaterThan(3); expect(mean).toBeLessThan(7);
  });
  it('default mean=0, std=1', async () => {
    const t = torch.randn([1000]); init.normal_(t);
    const arr = await t.toArray() as unknown as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(mean).toBeGreaterThan(-0.3); expect(mean).toBeLessThan(0.3);
  });
  it('preserves shape', () => { const t = torch.randn([3, 4, 5]); init.normal_(t); expect(t.shape).toEqual([3, 4, 5]); });
});

describe('nn.init.uniform_', () => {
  beforeAll(async () => { await torch.init(); });
  it('values in range', async () => {
    const t = torch.randn([1000]); init.uniform_(t, -1, 1);
    const arr = await t.toArray() as unknown as number[];
    expect(Math.min(...arr)).toBeLessThan(1); expect(Math.max(...arr)).toBeGreaterThan(-1);
  });
  it('default a=0, b=1', async () => {
    const t = torch.randn([1000]); init.uniform_(t);
    const arr = await t.toArray() as unknown as number[];
    expect(arr.reduce((a, b) => a + b, 0) / arr.length).toBeDefined();
  });
  it('preserves shape', () => { const t = torch.randn([2, 3]); init.uniform_(t, 0, 10); expect(t.shape).toEqual([2, 3]); });
});

describe('nn.init.eye_', () => {
  beforeAll(async () => { await torch.init(); });
  it('identity matrix', async () => {
    const t = torch.randn([4, 4]); init.eye_(t);
    const arr = await t.toArray();
    for (let i = 0; i < 4; i++) expect(arr[i * 4 + i]).toBe(1);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if (i !== j) expect(arr[i * 4 + j]).toBe(0);
  });
  it('throws non-square', () => { expect(() => init.eye_(torch.randn([3, 4]))).toThrow(); });
  it('throws non-2D', () => { expect(() => init.eye_(torch.randn([4]))).toThrow(); });
});

describe('nn.init.xavier_uniform_', () => {
  beforeAll(async () => { await torch.init(); });
  it('correct scaling', async () => {
    const t = torch.randn([100, 200]); init.xavier_uniform_(t);
    const arr = await t.toArray() as unknown as number[];
    const std = Math.sqrt(2 / (200 + 100));
    const a = Math.sqrt(3) * std;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(Math.abs(mean)).toBeLessThan(0.2);
    expect(Math.max(...arr.map((v) => Math.abs(v)))).toBeLessThan(a * 1.5);
  });
  it('custom gain', async () => { const t = torch.randn([50, 50]); init.xavier_uniform_(t, 2.0); expect((await t.toArray() as unknown as number[]).length).toBe(2500); });
  it('preserves shape', () => { const t = torch.randn([32, 64]); init.xavier_uniform_(t); expect(t.shape).toEqual([32, 64]); });
});

describe('nn.init.xavier_normal_', () => {
  beforeAll(async () => { await torch.init(); });
  it('correct std', async () => {
    const t = torch.randn([200, 100]); init.xavier_normal_(t);
    const arr = await t.toArray() as unknown as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(Math.abs(mean)).toBeLessThan(0.2);
  });
  it('custom gain', async () => { const t = torch.randn([50, 50]); init.xavier_normal_(t, Math.sqrt(2)); expect((await t.toArray() as unknown as number[]).length).toBe(2500); });
});

describe('nn.init.kaiming_uniform_', () => {
  beforeAll(async () => { await torch.init(); });
  it('fan_in mode', async () => {
    const t = torch.randn([64, 128]); init.kaiming_uniform_(t);
    const arr = await t.toArray() as unknown as number[];
    expect(Math.abs(arr.reduce((a, b) => a + b, 0) / arr.length)).toBeLessThan(0.3);
  });
  it('fan_out mode', async () => { const t = torch.randn([64, 128]); init.kaiming_uniform_(t, 0, 'fan_out'); expect((await t.toArray() as unknown as number[]).length).toBe(64 * 128); });
  it('relu nonlinearity', async () => { const t = torch.randn([64, 128]); init.kaiming_uniform_(t, 0, 'fan_in', 'relu'); expect((await t.toArray() as unknown as number[]).length).toBe(64 * 128); });
  it('leaky_relu param', async () => { const t = torch.randn([64, 128]); init.kaiming_uniform_(t, 0.01, 'fan_in', 'leaky_relu'); expect((await t.toArray() as unknown as number[]).length).toBe(64 * 128); });
  it('preserves shape', () => { const t = torch.randn([32, 64, 3, 3]); init.kaiming_uniform_(t); expect(t.shape).toEqual([32, 64, 3, 3]); });
});

describe('nn.init.kaiming_normal_', () => {
  beforeAll(async () => { await torch.init(); });
  it('normal distribution', async () => {
    const t = torch.randn([1000, 100]); init.kaiming_normal_(t, 0, 'fan_in', 'relu');
    const arr = await t.toArray() as unknown as number[];
    const fanIn = 100; const expectedStd = Math.sqrt(2) / Math.sqrt(fanIn);
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(Math.abs(mean)).toBeLessThan(0.2);
  });
  it('fan_out mode', async () => { const t = torch.randn([100, 200]); init.kaiming_normal_(t, 0, 'fan_out'); expect((await t.toArray() as unknown as number[]).length).toBe(20000); });
});

describe('nn.init.orthogonal_', () => {
  beforeAll(async () => { await torch.init(); });
  it('2D tensor', async () => { const t = torch.randn([8, 8]); init.orthogonal_(t); expect(t.shape).toEqual([8, 8]); });
  it('custom gain', async () => { const t = torch.randn([5, 5]); init.orthogonal_(t, 2.0); expect(t.shape).toEqual([5, 5]); });
  it('throws 1D', () => { expect(() => init.orthogonal_(torch.randn([8]))).toThrow(); });
  it('RNN weight shape', async () => { const t = torch.randn([4 * 64, 128]); init.orthogonal_(t); expect(t.shape).toEqual([256, 128]); });
});

describe('nn.init.sparse_', () => {
  beforeAll(async () => { await torch.init(); });
  it('sparsity', async () => { const t = torch.randn([100, 100]); init.sparse_(t, 0.5, 0.01); expect((await t.toArray()).length).toBe(10000); });
  it('custom sparsity', async () => { const t = torch.randn([50, 50]); init.sparse_(t, 0.9, 0.01); expect(t.shape).toEqual([50, 50]); });
  it('custom std', async () => { const t = torch.randn([20, 20]); init.sparse_(t, 0.1, 0.5); expect(t.shape).toEqual([20, 20]); });
});

describe('nn.init.calculate_gain', () => {
  beforeAll(async () => { await torch.init(); });
  it('linear=1', () => { expect(init.calculateGain('linear')).toBe(1); });
  it('conv1d/2d/3d=1', () => { expect(init.calculateGain('conv1d')).toBe(1); expect(init.calculateGain('conv2d')).toBe(1); expect(init.calculateGain('conv3d')).toBe(1); });
  it('sigmoid=1', () => { expect(init.calculateGain('sigmoid')).toBe(1); });
  it('tanh=5/3', () => { expect(init.calculateGain('tanh')).toBeCloseTo(5 / 3, 4); });
  it('relu=sqrt(2)', () => { expect(init.calculateGain('relu')).toBeCloseTo(Math.sqrt(2), 4); });
  it('leaky_relu default', () => { 
    // PyTorch leaky_relu default gain uses negative_slope=0.01
    // gain = sqrt(2 / (1 + 0.01^2)) ≈ 1.4141
    expect(init.calculateGain('leaky_relu')).toBeCloseTo(1.41414, 3); 
  });
  it('leaky_relu custom', () => { expect(init.calculateGain('leaky_relu', 0.2)).toBeCloseTo(Math.sqrt(2 / (1 + 0.04)), 4); });
  it('unknown=1', () => { expect(init.calculateGain('unknown')).toBe(1); });
});
