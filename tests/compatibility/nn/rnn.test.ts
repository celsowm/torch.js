import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import { RNN, LSTM, GRU } from '../../../src/nn/rnn';

const torch = createTorch(async () => {}, async () => ({}));

describe('nn.RNN', () => {
  beforeAll(async () => { await torch.init(); });

  it('parameters exist', () => {
    const r = new RNN(10, 20, 1);
    expect(Array.from(r.parameters()).length).toBeGreaterThanOrEqual(2);
    expect(r.input_size).toBe(10); expect(r.hidden_size).toBe(20);
  });

  it('default hidden state', async () => {
    const [out, hn] = new RNN(10, 20, 1).forward(torch.randn([5, 3, 10]));
    expect(out.shape).toEqual([5, 3, 20]); expect(hn.shape).toEqual([1, 3, 20]);
  });

  it('provided hidden state', async () => {
    const [out, hn] = new RNN(10, 20, 1).forward(torch.randn([3, 2, 10]), torch.randn([1, 2, 20]));
    expect(out.shape).toEqual([3, 2, 20]); expect(hn.shape).toEqual([1, 2, 20]);
  });

  it('batch_first', async () => {
    const [out, hn] = new RNN(10, 20, 1, 'tanh', true, true).forward(torch.randn([3, 5, 10]));
    expect(out.shape).toEqual([3, 5, 20]); expect(hn.shape).toEqual([1, 3, 20]);
  });

  it('multiple layers', async () => {
    const [out, hn] = new RNN(10, 20, 3).forward(torch.randn([5, 2, 10]));
    expect(out.shape).toEqual([5, 2, 20]); expect(hn.shape).toEqual([3, 2, 20]);
  });

  it('bidirectional', async () => {
    const [out, hn] = new RNN(10, 20, 1, 'tanh', true, false, 0, true).forward(torch.randn([5, 2, 10]));
    expect(hn.shape).toEqual([2, 2, 20]);
  });

  it('relu nonlinearity', () => { expect(new RNN(10, 20, 1, 'relu').nonlinearity).toBe('relu'); });
});

describe('nn.LSTM', () => {
  beforeAll(async () => { await torch.init(); });

  it('parameters exist', () => {
    expect(Array.from(new LSTM(10, 20, 1).parameters()).length).toBeGreaterThanOrEqual(2);
  });

  it('default hidden/cell', async () => {
    const [out, hn, cn] = new LSTM(10, 20, 1).forward(torch.randn([5, 3, 10]));
    expect(out.shape).toEqual([5, 3, 20]); expect(hn.shape).toEqual([1, 3, 20]); expect(cn.shape).toEqual([1, 3, 20]);
  });

  it('provided hidden/cell', async () => {
    const [out, hn, cn] = new LSTM(10, 20, 1).forward(
      torch.randn([3, 2, 10]), [torch.randn([1, 2, 20]), torch.randn([1, 2, 20])]
    );
    expect(out.shape).toEqual([3, 2, 20]); expect(hn.shape).toEqual([1, 2, 20]); expect(cn.shape).toEqual([1, 2, 20]);
  });

  it('batch_first', async () => {
    const [out, hn, cn] = new LSTM(10, 20, 1, true, true).forward(torch.randn([3, 5, 10]));
    expect(out.shape).toEqual([3, 5, 20]); expect(hn.shape).toEqual([1, 3, 20]); expect(cn.shape).toEqual([1, 3, 20]);
  });

  it('multiple layers', async () => {
    const [out, hn, cn] = new LSTM(10, 20, 3).forward(torch.randn([5, 2, 10]));
    expect(hn.shape).toEqual([3, 2, 20]); expect(cn.shape).toEqual([3, 2, 20]);
  });

  it('bidirectional', async () => {
    const [out, hn, cn] = new LSTM(10, 20, 1, true, false, 0, true).forward(torch.randn([5, 2, 10]));
    expect(hn.shape).toEqual([2, 2, 20]); expect(cn.shape).toEqual([2, 2, 20]);
  });

  it('no bias', () => { expect(Array.from(new LSTM(10, 20, 1, false).parameters()).length).toBeGreaterThanOrEqual(1); });
});

describe('nn.GRU', () => {
  beforeAll(async () => { await torch.init(); });

  it('parameters exist', () => {
    expect(Array.from(new GRU(10, 20, 1).parameters()).length).toBeGreaterThanOrEqual(2);
  });

  it('default hidden', async () => {
    const [out, hn] = new GRU(10, 20, 1).forward(torch.randn([5, 3, 10]));
    expect(out.shape).toEqual([5, 3, 20]); expect(hn.shape).toEqual([1, 3, 20]);
  });

  it('provided hidden', async () => {
    const [out, hn] = new GRU(10, 20, 1).forward(torch.randn([3, 2, 10]), torch.randn([1, 2, 20]));
    expect(out.shape).toEqual([3, 2, 20]); expect(hn.shape).toEqual([1, 2, 20]);
  });

  it('batch_first', async () => {
    const [out, hn] = new GRU(10, 20, 1, true, true).forward(torch.randn([3, 5, 10]));
    expect(out.shape).toEqual([3, 5, 20]); expect(hn.shape).toEqual([1, 3, 20]);
  });

  it('multiple layers', async () => {
    const [out, hn] = new GRU(10, 20, 3).forward(torch.randn([5, 2, 10]));
    expect(hn.shape).toEqual([3, 2, 20]);
  });

  it('bidirectional', async () => {
    const [out, hn] = new GRU(10, 20, 1, true, false, 0, true).forward(torch.randn([5, 2, 10]));
    expect(hn.shape).toEqual([2, 2, 20]);
  });
});

describe('RNN dtype', () => {
  beforeAll(async () => { await torch.init(); });
  it('RNN float32', async () => {
    const [out, hn] = new RNN(10, 20, 1).forward(torch.randn([3, 2, 10], { dtype: 'float32' }));
    expect(out.shape).toEqual([3, 2, 20]);
  });
  it('LSTM float32', async () => {
    const [out, hn, cn] = new LSTM(10, 20, 1).forward(torch.randn([3, 2, 10], { dtype: 'float32' }));
    expect(out.shape).toEqual([3, 2, 20]); expect(cn.shape).toEqual([1, 2, 20]);
  });
  it('GRU float32', async () => {
    const [out, hn] = new GRU(10, 20, 1).forward(torch.randn([3, 2, 10], { dtype: 'float32' }));
    expect(out.shape).toEqual([3, 2, 20]);
  });
});
