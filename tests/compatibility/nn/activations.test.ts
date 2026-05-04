import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import {
  ReLU, Sigmoid, Tanh, GELU, Softmax, LogSoftmax,
  PReLU, CELU, RReLU, Hardtanh, Hardshrink, Softshrink,
  LogSigmoid, Softmin,
} from '../../../src/nn/activation';
import { Dropout, Dropout1d, Dropout2d, Dropout3d, AlphaDropout } from '../../../src/nn/dropout';

const torch = createTorch(async () => {}, async () => ({}));

describe('nn.ReLU', () => {
  beforeAll(async () => { await torch.init(); });
  it('preserves shape', async () => { expect(new ReLU().forward(torch.randn([3, 4, 5])).shape).toEqual([3, 4, 5]); });
  it('clamps negatives to zero', async () => {
    const out = await new ReLU().forward(torch.tensor([-1.0, 2.0, -3.0, 4.0])).toArray();
    expect(out).toEqual([0, 2, 0, 4]);
  });
});

describe('nn.Sigmoid', () => {
  beforeAll(async () => { await torch.init(); });
  it('values in (0,1)', async () => {
    const arr = await new Sigmoid().forward(torch.tensor([-10.0, 0.0, 10.0])).toArray();
    expect(arr[0]).toBeGreaterThan(0); expect(arr[0]).toBeLessThan(0.5);
    expect(arr[1]).toBeCloseTo(0.5, 2);
    expect(arr[2]).toBeGreaterThan(0.5); expect(arr[2]).toBeLessThan(1);
  });
  it('shape', async () => { expect(new Sigmoid().forward(torch.randn([2, 3, 4])).shape).toEqual([2, 3, 4]); });
});

describe('nn.Tanh', () => {
  beforeAll(async () => { await torch.init(); });
  it('values in (-1,1)', async () => {
    const arr = await new Tanh().forward(torch.tensor([-5.0, 0.0, 5.0])).toArray();
    expect(arr[0]).toBeGreaterThan(-1); expect(arr[2]).toBeLessThan(1);
  });
  it('shape', async () => { expect(new Tanh().forward(torch.randn([3, 5])).shape).toEqual([3, 5]); });
});

describe('nn.GELU', () => {
  beforeAll(async () => { await torch.init(); });
  it('shape', async () => { expect(new GELU().forward(torch.randn([4, 8])).shape).toEqual([4, 8]); });
  it('large positive ~ x', async () => {
    const arr = await new GELU().forward(torch.tensor([5.0])).toArray();
    expect(arr[0]).toBeGreaterThan(4);
  });
});

describe('nn.Softmax', () => {
  beforeAll(async () => { await torch.init(); });
  it('sums to 1', async () => {
    const out = new Softmax(-1).forward(torch.tensor([[1.0, 2.0, 3.0]]));
    const s = await out.sum().toArray();
    expect(s[0]).toBeCloseTo(1, 4);
  });
  it('shape', async () => { expect(new Softmax(-1).forward(torch.randn([2, 3, 4])).shape).toEqual([2, 3, 4]); });
  it('all positive', async () => {
    const arr = await new Softmax(-1).forward(torch.tensor([[-1.0, -2.0, -3.0]])).toArray();
    expect(arr.every((v: number) => v > 0)).toBe(true);
  });
});

describe('nn.LogSoftmax', () => {
  beforeAll(async () => { await torch.init(); });
  it('values <= 0', async () => {
    const arr = await new LogSoftmax(-1).forward(torch.tensor([[1.0, 2.0, 3.0]])).toArray();
    expect(arr.every((v: number) => v <= 0)).toBe(true);
  });
  it('exp sums to 1', async () => {
    const s = await new LogSoftmax(-1).forward(torch.tensor([[2.0, 1.0, 0.1]])).exp().sum().toArray();
    expect(s[0]).toBeCloseTo(1, 4);
  });
  it('shape', async () => { expect(new LogSoftmax(-1).forward(torch.randn([2, 5])).shape).toEqual([2, 5]); });
});

describe('nn.PReLU', () => {
  beforeAll(async () => { await torch.init(); });
  it('learnable weight', () => {
    const p = new PReLU(1, 0.25);
    expect(p.weight.shape).toEqual([1]);
    expect(Array.from(p.parameters()).length).toBe(1);
  });
  it('positive unchanged', async () => {
    expect(await new PReLU(1, 0.25).forward(torch.tensor([1.0, 2.0, 3.0])).toArray()).toEqual([1, 2, 3]);
  });
  it('negative scaled', async () => {
    const arr = await new PReLU(1, 0.5).forward(torch.tensor([-2.0, -4.0])).toArray();
    expect(arr[0]).toBeCloseTo(-1, 3);
    expect(arr[1]).toBeCloseTo(-2, 3);
  });
});

describe('nn.CELU', () => {
  beforeAll(async () => { await torch.init(); });
  it('positive unchanged', async () => {
    expect(await new CELU(1.0).forward(torch.tensor([1.0, 2.0, 3.0])).toArray()).toEqual([1, 2, 3]);
  });
  it('negative transformed', async () => {
    const arr = await new CELU(1.0).forward(torch.tensor([-1.0])).toArray();
    expect(arr[0]).toBeCloseTo(-0.632, 2);
  });
  it('custom alpha', () => { expect(new CELU(2.0).alpha).toBe(2.0); });
});

describe('nn.RReLU', () => {
  beforeAll(async () => { await torch.init(); });
  it('training mode: negatives scaled randomly', async () => {
    const r = new RReLU(0.1, 0.3); r.train();
    const arr = await r.forward(torch.tensor([-1.0, -2.0])).toArray();
    expect(arr[0]).toBeLessThan(0); expect(arr[1]).toBeLessThan(0);
  });
  it('eval mode: avg slope', async () => {
    const r = new RReLU(0.1, 0.3); r.eval();
    const arr = await r.forward(torch.tensor([-2.0])).toArray();
    expect(arr[0]).toBeCloseTo(-0.4, 2);
  });
});

describe('nn.Hardtanh', () => {
  beforeAll(async () => { await torch.init(); });
  it('clamps', async () => {
    expect(await new Hardtanh(-1, 1).forward(torch.tensor([-2.0, 0.0, 2.0])).toArray()).toEqual([-1, 0, 1]);
  });
  it('custom range', async () => {
    expect(await new Hardtanh(0, 6).forward(torch.tensor([-1.0, 3.0, 7.0])).toArray()).toEqual([0, 3, 6]);
  });
});

describe('nn.Hardshrink', () => {
  beforeAll(async () => { await torch.init(); });
  it('zeroes |x| <= lambd', async () => {
    const arr = await new Hardshrink(0.5).forward(torch.tensor([-1.0, -0.3, 0.0, 0.3, 1.0])).toArray();
    expect(arr).toEqual([-1, 0, 0, 0, 1]);
  });
});

describe('nn.Softshrink', () => {
  beforeAll(async () => { await torch.init(); });
  it('shrinks toward zero', async () => {
    const arr = await new Softshrink(0.5).forward(torch.tensor([-1.0, -0.3, 0.0, 0.3, 1.0])).toArray();
    expect(arr[0]).toBeCloseTo(-0.5, 4); expect(arr[1]).toBe(0); expect(arr[4]).toBeCloseTo(0.5, 4);
  });
});

describe('nn.LogSigmoid', () => {
  beforeAll(async () => { await torch.init(); });
  it('values <= 0', async () => {
    const arr = await new LogSigmoid().forward(torch.tensor([0.0, 1.0, -1.0])).toArray();
    expect(arr.every((v: number) => v <= 0)).toBe(true);
  });
  it('shape', async () => { expect(new LogSigmoid().forward(torch.randn([2, 3])).shape).toEqual([2, 3]); });
});

describe('nn.Softmin', () => {
  beforeAll(async () => { await torch.init(); });
  it('sums to 1, higher prob for smaller values', async () => {
    const arr = await new Softmin(-1).forward(torch.tensor([[1.0, 2.0, 3.0]])).toArray() as unknown as number[];
    expect(arr.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
    expect(arr[0]).toBeGreaterThan(arr[2]);
  });
});

describe('nn.Dropout', () => {
  beforeAll(async () => { await torch.init(); });
  it('training: scales by (1-p)', async () => {
    const d = new Dropout(0.5); d.train();
    const arr = await d.forward(torch.ones([100])).toArray() as unknown as number[];
    expect(arr.reduce((a, b) => a + b, 0) / arr.length).toBeCloseTo(0.5, 1);
  });
  it('eval: identity', async () => {
    const d = new Dropout(0.5); d.eval();
    const arr = await d.forward(torch.ones([10])).toArray() as unknown as number[];
    expect(arr.every((v) => v === 1)).toBe(true);
  });
  it('p=0: identity', async () => {
    const arr = await new Dropout(0).forward(torch.ones([5])).toArray();
    expect(arr).toEqual([1, 1, 1, 1, 1]);
  });
});

describe('nn.Dropout1d', () => {
  beforeAll(async () => { await torch.init(); });
  it('3D input training', async () => { expect(new Dropout1d(0.5).forward(torch.ones([2, 4, 8])).shape).toEqual([2, 4, 8]); });
  it('3D input eval', async () => {
    const d = new Dropout1d(0.5); d.eval();
    const arr = await d.forward(torch.ones([2, 4, 8])).toArray();
    expect(arr.length).toBe(64);
  });
  it('2D input', async () => { expect(new Dropout1d(0.3).forward(torch.ones([4, 8])).shape).toEqual([4, 8]); });
});

describe('nn.Dropout2d', () => {
  beforeAll(async () => { await torch.init(); });
  it('training', async () => { expect(new Dropout2d(0.5).forward(torch.ones([2, 3, 8, 8])).shape).toEqual([2, 3, 8, 8]); });
  it('eval: identity', async () => {
    const d = new Dropout2d(0.5); d.eval();
    const arr = await d.forward(torch.ones([1, 2, 4, 4])).toArray();
    expect(arr.every((v: number) => v === 1)).toBe(true);
  });
});

describe('nn.Dropout3d', () => {
  beforeAll(async () => { await torch.init(); });
  it('training', async () => { expect(new Dropout3d(0.5).forward(torch.ones([1, 2, 4, 4, 4])).shape).toEqual([1, 2, 4, 4, 4]); });
  it('eval', async () => { expect(new Dropout3d(0.5).forward(torch.ones([1, 2, 3, 3, 3])).shape).toEqual([1, 2, 3, 3, 3]); });
});

describe('nn.AlphaDropout', () => {
  beforeAll(async () => { await torch.init(); });
  it('training', async () => { expect(new AlphaDropout(0.5).forward(torch.randn([100])).shape).toEqual([100]); });
  it('eval: identity', async () => {
    const d = new AlphaDropout(0.5); d.eval();
    const arr = await d.forward(torch.ones([10])).toArray();
    expect(arr.every((v: number) => v === 1)).toBe(true);
  });
  it('p=0: identity', async () => {
    expect(await new AlphaDropout(0).forward(torch.tensor([1.0, 2.0, 3.0])).toArray()).toEqual([1, 2, 3]);
  });
});
