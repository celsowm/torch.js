import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import {
  TransformerEncoderLayer, TransformerEncoder,
  TransformerDecoderLayer, TransformerDecoder,
} from '../../../src/nn/transformer';
import { MultiheadAttention } from '../../../src/nn/attention';
import { LayerNorm } from '../../../src/nn/layernorm';

const torch = createTorch(async () => {}, async () => ({}));

describe('nn.TransformerEncoderLayer', () => {
  beforeAll(async () => { await torch.init(); });

  it('creates with d_model and nhead', () => {
    expect(new TransformerEncoderLayer({ d_model: 64, nhead: 8 })).toBeDefined();
  });

  it('forward preserves shape', async () => {
    const out = new TransformerEncoderLayer({ d_model: 64, nhead: 8 }).forward(torch.randn([2, 10, 64]));
    expect(out.shape).toEqual([2, 10, 64]);
  });

  it('custom dimFeedforward', async () => {
    expect(new TransformerEncoderLayer({ d_model: 64, nhead: 8, dimFeedforward: 128 }).forward(torch.randn([2, 5, 64])).shape).toEqual([2, 5, 64]);
  });

  it('gelu activation', async () => {
    expect(new TransformerEncoderLayer({ d_model: 64, nhead: 8, activation: 'gelu' }).forward(torch.randn([2, 5, 64])).shape).toEqual([2, 5, 64]);
  });

  it('custom layerNormEps', () => { expect(new TransformerEncoderLayer({ d_model: 64, nhead: 8, layerNormEps: 1e-3 })).toBeDefined(); });

  it('mask support', async () => {
    expect(new TransformerEncoderLayer({ d_model: 64, nhead: 8 }).forward(torch.randn([2, 5, 64]), torch.randn([5, 5])).shape).toEqual([2, 5, 64]);
  });

  it('causal flag', async () => {
    expect(new TransformerEncoderLayer({ d_model: 64, nhead: 8 }).forward(torch.randn([2, 5, 64]), undefined, undefined, true).shape).toEqual([2, 5, 64]);
  });

  it('seq-first (batch_first=false)', async () => {
    expect(new TransformerEncoderLayer({ d_model: 64, nhead: 8, batchFirst: false }).forward(torch.randn([10, 2, 64])).shape).toEqual([10, 2, 64]);
  });

  it('dropout', () => { expect(new TransformerEncoderLayer({ d_model: 64, nhead: 8, dropout: 0.1 })).toBeDefined(); });
});

describe('nn.TransformerEncoder', () => {
  beforeAll(async () => { await torch.init(); });

  it('multiple layers', () => { expect(new TransformerEncoder({ encoderLayer: { d_model: 64, nhead: 8 }, numLayers: 3 })).toBeDefined(); });

  it('forward through layers', async () => {
    expect(new TransformerEncoder({ encoderLayer: { d_model: 64, nhead: 8 }, numLayers: 2 }).forward(torch.randn([2, 10, 64])).shape).toEqual([2, 10, 64]);
  });

  it('LayerNorm at end', async () => {
    expect(new TransformerEncoder({ encoderLayer: { d_model: 64, nhead: 8 }, numLayers: 2, norm: new LayerNorm(64) }).forward(torch.randn([2, 10, 64])).shape).toEqual([2, 10, 64]);
  });

  it('pre-built layer', async () => {
    const layer = new TransformerEncoderLayer({ d_model: 64, nhead: 8 });
    expect(new TransformerEncoder({ encoderLayer: layer, numLayers: 2 }).forward(torch.randn([2, 5, 64])).shape).toEqual([2, 5, 64]);
  });

  it('mask in forward', async () => {
    expect(new TransformerEncoder({ encoderLayer: { d_model: 64, nhead: 8 }, numLayers: 2 }).forward(torch.randn([2, 10, 64]), torch.randn([10, 10])).shape).toEqual([2, 10, 64]);
  });
});

describe('nn.TransformerDecoderLayer', () => {
  beforeAll(async () => { await torch.init(); });

  it('creates with config', () => { expect(new TransformerDecoderLayer({ d_model: 64, nhead: 8 })).toBeDefined(); });

  it('forward preserves shape', async () => {
    const out = new TransformerDecoderLayer({ d_model: 64, nhead: 8 }).forward(torch.randn([2, 10, 64]), torch.randn([2, 5, 64]));
    expect(out.shape).toEqual([2, 10, 64]);
  });

  it('cross-attention with different seq lengths', async () => {
    expect(new TransformerDecoderLayer({ d_model: 64, nhead: 8, dimFeedforward: 128 }).forward(torch.randn([2, 8, 64]), torch.randn([2, 4, 64])).shape).toEqual([2, 8, 64]);
  });

  it('mask parameters', async () => {
    expect(new TransformerDecoderLayer({ d_model: 64, nhead: 8 }).forward(torch.randn([2, 5, 64]), torch.randn([2, 3, 64]), torch.randn([5, 5]), torch.randn([3, 3])).shape).toEqual([2, 5, 64]);
  });

  it('gelu activation', async () => {
    expect(new TransformerDecoderLayer({ d_model: 64, nhead: 8, activation: 'gelu' }).forward(torch.randn([2, 5, 64]), torch.randn([2, 3, 64])).shape).toEqual([2, 5, 64]);
  });

  it('seq-first', async () => {
    expect(new TransformerDecoderLayer({ d_model: 64, nhead: 8, batchFirst: false }).forward(torch.randn([10, 2, 64]), torch.randn([5, 2, 64])).shape).toEqual([10, 2, 64]);
  });
});

describe('nn.TransformerDecoder', () => {
  beforeAll(async () => { await torch.init(); });

  it('multiple layers', () => { expect(new TransformerDecoder({ decoderLayer: { d_model: 64, nhead: 8 }, numLayers: 3 })).toBeDefined(); });

  it('forward', async () => {
    expect(new TransformerDecoder({ decoderLayer: { d_model: 64, nhead: 8 }, numLayers: 2 }).forward(torch.randn([2, 10, 64]), torch.randn([2, 5, 64])).shape).toEqual([2, 10, 64]);
  });

  it('LayerNorm at end', async () => {
    expect(new TransformerDecoder({ decoderLayer: { d_model: 64, nhead: 8 }, numLayers: 2, norm: new LayerNorm(64) }).forward(torch.randn([2, 10, 64]), torch.randn([2, 5, 64])).shape).toEqual([2, 10, 64]);
  });

  it('pre-built layer', async () => {
    const layer = new TransformerDecoderLayer({ d_model: 64, nhead: 8 });
    expect(new TransformerDecoder({ decoderLayer: layer, numLayers: 2 }).forward(torch.randn([2, 5, 64]), torch.randn([2, 3, 64])).shape).toEqual([2, 5, 64]);
  });
});

describe('nn.MultiheadAttention', () => {
  beforeAll(async () => { await torch.init(); });

  it('embed_dim and num_heads', () => {
    const m = new MultiheadAttention(64, 8);
    expect(m.embed_dim).toBe(64); expect(m.num_heads).toBe(8); expect(m.head_dim).toBe(8);
  });

  it('throws if not divisible', () => { expect(() => new MultiheadAttention(64, 7)).toThrow(); });

  it('self-attention forward', async () => {
    const m = new MultiheadAttention(64, 8, 0.0, true, false);
    const q = torch.randn([10, 2, 64]), k = torch.randn([10, 2, 64]), v = torch.randn([10, 2, 64]);
    const [out] = m.forward(q, k, v);
    expect(out.shape).toEqual([10, 2, 64]);
  });

  it('batch_first', async () => {
    const m = new MultiheadAttention(64, 8, 0.0, true, true);
    const [out] = m.forward(torch.randn([2, 10, 64]), torch.randn([2, 10, 64]), torch.randn([2, 10, 64]));
    expect(out.shape).toEqual([2, 10, 64]);
  });

  it('need_weights returns attn', async () => {
    const m = new MultiheadAttention(64, 8, 0.0, true, false);
    const q = torch.randn([5, 2, 64]), k = torch.randn([5, 2, 64]), v = torch.randn([5, 2, 64]);
    const [, attn] = m.forward(q, k, v, null, true);
    expect(attn).not.toBe(null);
  });

  it('cross-attention different seq lengths', async () => {
    const m = new MultiheadAttention(64, 8, 0.0, true, false);
    const [out] = m.forward(torch.randn([10, 2, 64]), torch.randn([5, 2, 64]), torch.randn([5, 2, 64]));
    expect(out.shape).toEqual([10, 2, 64]);
  });

  it('no bias', async () => {
    const m = new MultiheadAttention(64, 8, 0.0, false);
    expect(m.in_proj_bias).toBe(null); expect(m.out_proj_bias).toBe(null);
    const [out] = m.forward(torch.randn([5, 2, 64]), torch.randn([5, 2, 64]), torch.randn([5, 2, 64]));
    expect(out.shape).toEqual([5, 2, 64]);
  });
});
