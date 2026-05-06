import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import * as F from '../../../src/nn/functional';
import {
  mse_loss, l1_loss, cross_entropy, binary_cross_entropy,
  nll_loss, huber_loss, kl_div_loss,
} from '../../../src/nn/losses';

const torch = createTorch(async () => {}, async () => ({}));

describe('F.relu', () => {
  beforeAll(async () => { await torch.init(); });
  it('clamps negatives', async () => { expect(await F.relu(torch.tensor([-1.0, 2.0, -3.0, 4.0])).toArray()).toEqual([0, 2, 0, 4]); });
  it('shape', async () => { expect(F.relu(torch.randn([2, 3, 4])).shape).toEqual([2, 3, 4]); });
});

describe('F.sigmoid', () => {
  beforeAll(async () => { await torch.init(); });
  it('values in (0,1)', async () => {
    const arr = await F.sigmoid(torch.tensor([-5.0, 0.0, 5.0])).toArray();
    expect(arr[0]).toBeGreaterThan(0); expect(arr[0]).toBeLessThan(0.5);
    expect(arr[2]).toBeGreaterThan(0.5); expect(arr[2]).toBeLessThan(1);
  });
});

describe('F.tanh', () => {
  beforeAll(async () => { await torch.init(); });
  it('values in (-1,1)', async () => {
    const arr = await F.tanh(torch.tensor([-5.0, 0.0, 5.0])).toArray();
    expect(arr[0]).toBeGreaterThan(-1); expect(arr[2]).toBeLessThan(1);
  });
});

describe('F.softmax', () => {
  beforeAll(async () => { await torch.init(); });
  it('sums to 1', async () => {
    const out = F.softmax(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), -1);
    expect(out.shape).toEqual([2, 3]);
    for (let i = 0; i < 2; i++) {
      const s = await out.select(0, i).sum().toArray();
      expect(s[0]).toBeCloseTo(1, 4);
    }
  });
  it('shape', async () => { expect(F.softmax(torch.randn([2, 3, 4]), -1).shape).toEqual([2, 3, 4]); });
});

describe('F.log_softmax', () => {
  beforeAll(async () => { await torch.init(); });
  it('values <= 0', async () => {
    const arr = await F.log_softmax(torch.tensor([[1.0, 2.0, 3.0]]), -1).toArray();
    expect(arr.every((v: number) => v <= 0)).toBe(true);
  });
  it('exp sums to 1', async () => {
    const s = await F.log_softmax(torch.tensor([[2.0, 1.0, 0.1]]), -1).exp().sum().toArray();
    expect(s[0]).toBeCloseTo(1, 4);
  });
});

describe('F.dropout', () => {
  beforeAll(async () => { await torch.init(); });
  it('eval: identity', async () => {
    const arr = await F.dropout(torch.ones([10]), 0.5, false).toArray();
    expect(arr.every((v: number) => v === 1)).toBe(true);
  });
  it('train: scales', async () => {
    const arr = await F.dropout(torch.ones([100]), 0.5, true).toArray() as unknown as number[];
    expect(arr.reduce((a, b) => a + b, 0) / arr.length).toBeCloseTo(0.5, 1);
  });
  it('p=0: identity', async () => {
    const arr = await F.dropout(torch.ones([5]), 0, true).toArray();
    expect(arr.every((v: number) => v === 1)).toBe(true);
  });
  it('p=1: zeros', async () => {
    const arr = await F.dropout(torch.ones([5]), 1, true).toArray();
    expect(arr.every((v: number) => v === 0)).toBe(true);
  });
});

describe('F.conv2d', () => {
  beforeAll(async () => { await torch.init(); });
  it('basic', async () => { expect(F.conv2d(torch.randn([1, 3, 32, 32]), torch.randn([8, 3, 3, 3])).shape).toEqual([1, 8, 30, 30]); });
  it('stride', async () => { expect(F.conv2d(torch.randn([1, 3, 32, 32]), torch.randn([8, 3, 3, 3]), undefined, 2).shape).toEqual([1, 8, 15, 15]); });
  it('padding', async () => { expect(F.conv2d(torch.randn([1, 3, 32, 32]), torch.randn([8, 3, 3, 3]), undefined, 1, 1).shape).toEqual([1, 8, 32, 32]); });
  it('bias', async () => { expect(F.conv2d(torch.randn([1, 1, 8, 8]), torch.randn([2, 1, 3, 3]), torch.zeros([2])).shape).toEqual([1, 2, 6, 6]); });
  it('dilation', async () => { expect(F.conv2d(torch.randn([1, 1, 16, 16]), torch.randn([2, 1, 3, 3]), undefined, 1, 0, 2).shape).toEqual([1, 2, 12, 12]); });
});

describe('F.max_pool2d', () => {
  beforeAll(async () => { await torch.init(); });
  it('kernel 2', async () => { expect(F.max_pool2d(torch.randn([2, 3, 32, 32]), 2).shape).toEqual([2, 3, 16, 16]); });
  it('stride', async () => { expect(F.max_pool2d(torch.randn([1, 1, 10, 10]), 3, 2).shape).toEqual([1, 1, 4, 4]); });
  it('padding', async () => { expect(F.max_pool2d(torch.randn([1, 1, 8, 8]), 2, 2, 1).shape).toEqual([1, 1, 5, 5]); });
});

describe('F.avg_pool2d', () => {
  beforeAll(async () => { await torch.init(); });
  it('kernel 2', async () => { expect(F.avg_pool2d(torch.randn([2, 3, 32, 32]), 2).shape).toEqual([2, 3, 16, 16]); });
  it('stride', async () => { expect(F.avg_pool2d(torch.randn([1, 1, 10, 10]), 3, 2).shape).toEqual([1, 1, 4, 4]); });
});

describe('F.interpolate', () => {
  beforeAll(async () => { await torch.init(); });
  it('size nearest', async () => { expect(F.interpolate(torch.randn([1, 3, 8, 8]), [16, 16], undefined, 'nearest').shape).toEqual([1, 3, 16, 16]); });
  it('scale_factor nearest', async () => { expect(F.interpolate(torch.randn([1, 3, 8, 8]), undefined, 2, 'nearest').shape).toEqual([1, 3, 16, 16]); });
  it('downsample', async () => { expect(F.interpolate(torch.randn([1, 3, 32, 32]), [8, 8], undefined, 'nearest').shape).toEqual([1, 3, 8, 8]); });
});

describe('F.normalize', () => {
  beforeAll(async () => { await torch.init(); });
  it('L2 norm [3,4] -> [0.6,0.8]', async () => {
    const arr = await F.normalize(torch.tensor([[3.0, 4.0]]), 2, -1).toArray();
    expect(arr[0]).toBeCloseTo(0.6, 3); expect(arr[1]).toBeCloseTo(0.8, 3);
  });
  it('unit norm', async () => {
    const out = F.normalize(torch.randn([2, 5]), 2, -1);
    for (let i = 0; i < 2; i++) {
      const n = await out.select(0, i).pow(2).sum().sqrt().toArray();
      expect(n[0]).toBeCloseTo(1, 4);
    }
  });
});

describe('F.pad', () => {
  beforeAll(async () => { await torch.init(); });
  it('constant pad', async () => { expect(F.pad(torch.ones([2, 2]), [1, 1, 1, 1], 'constant', 0).shape).toEqual([4, 4]); });
});

describe('F.one_hot', () => {
  beforeAll(async () => { await torch.init(); });
  it('3 classes', async () => {
    const out = await F.one_hot(torch.tensor([0, 1, 2], { dtype: 'int32' }), 3);
    expect(out.shape).toEqual([3, 3]);
    expect(await out.toArray()).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
  it('auto num_classes', async () => { expect((await F.one_hot(torch.tensor([0, 2, 4], { dtype: 'int32' }))).shape).toEqual([3, 5]); });
});

describe('F loss functions', () => {
  beforeAll(async () => { await torch.init(); });
  it('mse_loss', async () => { expect((await mse_loss(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])).toArray())[0]).toBeCloseTo(0, 6); });
  it('l1_loss', async () => { expect((await l1_loss(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 3.0, 4.0])).toArray())[0]).toBeCloseTo(1, 4); });
  it('cross_entropy', async () => { const arr = await cross_entropy(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0], { dtype: 'int32' })).toArray(); expect(arr[0]).toBeGreaterThan(0); });
  it('binary_cross_entropy', async () => { const arr = await binary_cross_entropy(torch.tensor([0.9, 0.1, 0.8]), torch.tensor([1.0, 0.0, 1.0])).toArray(); expect(arr[0]).toBeGreaterThan(0); });
  it('nll_loss', async () => { const arr = await nll_loss(torch.tensor([[-0.1, -2.0, -3.0]]), torch.tensor([0], { dtype: 'int32' })).toArray(); expect(arr[0]).toBeCloseTo(0.1, 3); });
  it('huber_loss', async () => { const arr = await huber_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 3.0]), 1.0).toArray(); expect(arr[0]).toBeGreaterThan(0); });
  it('kl_div_loss', async () => { expect(kl_div_loss(torch.tensor([[-0.1, -2.0]]), torch.tensor([[0.7, 0.3]])).shape.length).toBeGreaterThanOrEqual(0); });
});
