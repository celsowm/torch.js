import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import {
  CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, MSELoss, L1Loss,
  SmoothL1Loss, NLLLoss, HuberLoss, KLDivLoss,
  HingeEmbeddingLoss, CosineEmbeddingLoss, TripletMarginLoss,
} from '../../../src/nn/losses';

const torch = createTorch(async () => {}, async () => ({}));

describe('nn.CrossEntropyLoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('mean reduction', async () => {
    const l = new CrossEntropyLoss(undefined, 'mean').forward(torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 1.0]]), torch.tensor([0, 1], { dtype: 'int32' }));
    expect(l.shape).toEqual([]); expect((await l.toArray())[0]).toBeGreaterThan(0);
  });
  it('sum reduction', async () => {
    expect(new CrossEntropyLoss(undefined, 'sum').forward(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0], { dtype: 'int32' })).shape).toEqual([]);
  });
  it('none reduction', async () => {
    expect(new CrossEntropyLoss(undefined, 'none').forward(torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 1.0]]), torch.tensor([0, 1], { dtype: 'int32' })).shape).toEqual([2]);
  });
  it('class weights', async () => {
    expect(new CrossEntropyLoss(torch.tensor([1.0, 2.0, 0.5]), 'mean').forward(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([1], { dtype: 'int32' })).shape).toEqual([]);
  });
  it('label smoothing', async () => {
    expect(new CrossEntropyLoss(undefined, 'mean', 0.1).forward(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0], { dtype: 'int32' })).shape).toEqual([]);
  });
  it('correct < wrong prediction', async () => {
    const ce = new CrossEntropyLoss(undefined, 'mean');
    const c = await ce.forward(torch.tensor([[5.0, 1.0, 1.0]]), torch.tensor([0], { dtype: 'int32' })).toArray();
    const w = await ce.forward(torch.tensor([[1.0, 1.0, 5.0]]), torch.tensor([0], { dtype: 'int32' })).toArray();
    expect(c[0]).toBeLessThan(w[0]);
  });
});

describe('nn.BCELoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('computes loss', async () => {
    const l = new BCELoss(undefined, 'mean').forward(torch.tensor([0.9, 0.1, 0.8]), torch.tensor([1.0, 0.0, 1.0]));
    expect(l.shape).toEqual([]); expect((await l.toArray())[0]).toBeGreaterThan(0);
  });
  it('none reduction', async () => {
    expect(new BCELoss(undefined, 'none').forward(torch.tensor([0.9, 0.1, 0.8]), torch.tensor([1.0, 0.0, 1.0])).shape).toEqual([3]);
  });
  it('correct < wrong', async () => {
    const b = new BCELoss(undefined, 'mean');
    const c = await b.forward(torch.tensor([0.9, 0.1]), torch.tensor([1.0, 0.0])).toArray();
    const w = await b.forward(torch.tensor([0.1, 0.9]), torch.tensor([1.0, 0.0])).toArray();
    expect(c[0]).toBeLessThan(w[0]);
  });
});

describe('nn.BCEWithLogitsLoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('from logits', async () => {
    expect(new BCEWithLogitsLoss(undefined, 'mean').forward(torch.tensor([2.0, -2.0, 1.5]), torch.tensor([1.0, 0.0, 1.0])).shape).toEqual([]);
  });
  it('pos_weight', async () => {
    expect(new BCEWithLogitsLoss(undefined, 'mean', torch.tensor([2.0])).forward(torch.tensor([0.5, -0.5]), torch.tensor([1.0, 0.0])).shape).toEqual([]);
  });
  it('none reduction', async () => {
    expect(new BCEWithLogitsLoss(undefined, 'none').forward(torch.tensor([2.0, -2.0, 1.5]), torch.tensor([1.0, 0.0, 1.0])).shape).toEqual([3]);
  });
});

describe('nn.MSELoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('identical -> 0', async () => { expect((await new MSELoss('mean').forward(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])).toArray())[0]).toBeCloseTo(0, 6); });
  it('non-zero', async () => {
    // MSE = (1+4+9)/3 = 14/3
    expect((await new MSELoss('mean').forward(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0])).toArray())[0]).toBeCloseTo(14 / 3, 3);
  });
  it('sum', async () => { expect((await new MSELoss('sum').forward(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])).toArray())[0]).toBeCloseTo(8, 3); });
  it('none', async () => { expect(new MSELoss('none').forward(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 2.0, 5.0])).shape).toEqual([3]); });
});

describe('nn.L1Loss', () => {
  beforeAll(async () => { await torch.init(); });
  it('identical -> 0', async () => { expect((await new L1Loss('mean').forward(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])).toArray())[0]).toBeCloseTo(0, 6); });
  it('non-zero: (1+2+2)/3', async () => { expect((await new L1Loss('mean').forward(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 5.0])).toArray())[0]).toBeCloseTo(5 / 3, 3); });
  it('sum', async () => { expect((await new L1Loss('sum').forward(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])).toArray())[0]).toBeCloseTo(4, 3); });
});

describe('nn.SmoothL1Loss', () => {
  beforeAll(async () => { await torch.init(); });
  it('identical -> 0', async () => { expect((await new SmoothL1Loss('mean').forward(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])).toArray())[0]).toBeCloseTo(0, 6); });
  it('small error: 0.5*0.25=0.125', async () => { expect((await new SmoothL1Loss('mean').forward(torch.tensor([0.5]), torch.tensor([0.0])).toArray())[0]).toBeCloseTo(0.125, 3); });
  it('large error: 3-0.5=2.5', async () => { expect((await new SmoothL1Loss('mean').forward(torch.tensor([3.0]), torch.tensor([0.0])).toArray())[0]).toBeCloseTo(2.5, 3); });
});

describe('nn.NLLLoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('computes loss', async () => {
    const l = new NLLLoss(undefined, 'mean').forward(torch.tensor([[-0.1, -2.0, -3.0], [-3.0, -0.2, -1.0]]), torch.tensor([0, 1], { dtype: 'int32' }));
    expect(l.shape).toEqual([]); expect((await l.toArray())[0]).toBeGreaterThan(0);
  });
  it('none reduction', async () => {
    expect(new NLLLoss(undefined, 'none').forward(torch.tensor([[-0.1, -2.0, -3.0]]), torch.tensor([0], { dtype: 'int32' })).shape).toEqual([1]);
  });
  it('weights', async () => {
    expect(new NLLLoss(torch.tensor([1.0, 2.0, 0.5]), 'mean').forward(torch.tensor([[-0.1, -2.0, -3.0]]), torch.tensor([1], { dtype: 'int32' })).shape).toEqual([]);
  });
});

describe('nn.HuberLoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('identical -> 0', async () => { expect((await new HuberLoss(1.0, 'mean').forward(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])).toArray())[0]).toBeCloseTo(0, 6); });
  it('custom delta', async () => { expect(new HuberLoss(0.5, 'mean').forward(torch.tensor([0.3]), torch.tensor([0.0])).shape).toEqual([]); });
  it('none reduction', async () => { expect(new HuberLoss(1.0, 'none').forward(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 3.0])).shape).toEqual([2]); });
});

describe('nn.KLDivLoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('batchmean', async () => { expect(new KLDivLoss('batchmean').forward(torch.tensor([[-0.1, -2.0, -3.0]]), torch.tensor([[0.7, 0.2, 0.1]])).shape).toEqual([]); });
  it('none reduction', async () => { expect(new KLDivLoss('none').forward(torch.tensor([[-0.1, -2.0, -3.0]]), torch.tensor([[0.7, 0.2, 0.1]])).shape).toEqual([1, 3]); });
  it('log_target', async () => { expect(new KLDivLoss('batchmean', true).forward(torch.tensor([[-0.1, -2.0, -3.0]]), torch.tensor([[-0.357, -1.609, -2.303]])).shape).toEqual([]); });
});

describe('nn.HingeEmbeddingLoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('computes', async () => { expect(new HingeEmbeddingLoss(1.0, 'mean').forward(torch.tensor([1.0, 0.5, -0.5]), torch.tensor([1.0, -1.0, -1.0])).shape).toEqual([]); });
  it('target=1: loss=input', async () => { expect((await new HingeEmbeddingLoss(1.0, 'none').forward(torch.tensor([0.5]), torch.tensor([1.0])).toArray())[0]).toBeCloseTo(0.5, 4); });
  it('target=-1: max(0, margin-input)', async () => { expect((await new HingeEmbeddingLoss(1.0, 'none').forward(torch.tensor([0.5]), torch.tensor([-1.0])).toArray())[0]).toBeCloseTo(0.5, 4); });
});

describe('nn.CosineEmbeddingLoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('computes', async () => { expect(new CosineEmbeddingLoss(0.0, 'mean').forward(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([1.0, -1.0])).shape).toEqual([]); });
  it('target=1 identical -> 0', async () => { expect((await new CosineEmbeddingLoss(0.0, 'none').forward(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), torch.tensor([1.0])).toArray())[0]).toBeCloseTo(0, 4); });
  it('target=-1 identical -> 1', async () => { expect((await new CosineEmbeddingLoss(0.0, 'none').forward(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), torch.tensor([-1.0])).toArray())[0]).toBeCloseTo(1, 4); });
});

describe('nn.TripletMarginLoss', () => {
  beforeAll(async () => { await torch.init(); });
  it('computes', async () => { expect(new TripletMarginLoss(1.0, 2, 1e-6, 'mean').forward(torch.tensor([[1.0, 2.0]]), torch.tensor([[1.1, 2.1]]), torch.tensor([[5.0, 6.0]])).shape).toEqual([]); });
  it('zero when negative far', async () => {
    const arr = await new TripletMarginLoss(1.0, 2, 1e-6, 'none').forward(torch.tensor([[0.0, 0.0]]), torch.tensor([[0.1, 0.0]]), torch.tensor([[10.0, 10.0]])).toArray();
    expect(arr[0]).toBeCloseTo(0, 3);
  });
  it('non-zero when negative close', async () => {
    const arr = await new TripletMarginLoss(1.0, 2, 1e-6, 'none').forward(torch.tensor([[0.0, 0.0]]), torch.tensor([[0.1, 0.0]]), torch.tensor([[0.2, 0.0]])).toArray();
    expect(arr[0]).toBeGreaterThan(0.5);
  });
  it('L1 norm', async () => { expect(new TripletMarginLoss(1.0, 1, 1e-6, 'mean').forward(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]])).shape).toEqual([]); });
});
