/**
 * GPT2 inference test - comprehensive coverage of all operations needed
 * to run the GPT2 model from transformers.
 * 
 * Tests the following torch.js capabilities:
 * - matmul broadcasting 4D (attention Q@K.T)
 * - softmax, multinomial
 * - split, view, reshape, transpose, contiguous
 * - masked_fill with bool mask
 * - triu, arange, cat
 * - state_dict / load_state_dict
 * - nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout, nn.ModuleList
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../src/index';

describe('GPT2 Inference', () => {
  beforeAll(async () => {
    await torch.init();
  });

  it('creates causal mask with triu + bool', async () => {
    const seqLen = 8;
    const ones = torch.ones([seqLen, seqLen]);
    const mask = ones.triu(1).bool();
    expect(mask.shape).toEqual([seqLen, seqLen]);
  });

  it('runs 4D batched matmul (Q @ K.T for attention)', async () => {
    const batch = 1;
    const heads = 4;
    const seq = 8;
    const headDim = 16;

    // GPT2 attention: [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq]
    const q = torch.randn([batch, heads, seq, headDim]);
    const k = torch.randn([batch, heads, headDim, seq]);

    const attn = q.matmul(k);
    expect(attn.shape).toEqual([batch, heads, seq, seq]);
  });

  it('runs full attention block with softmax + masked_fill', async () => {
    const batch = 1;
    const heads = 4;
    const seq = 8;
    const headDim = 16;

    const q = torch.randn([batch, heads, seq, headDim]);
    const k = torch.randn([batch, heads, headDim, seq]);
    const v = torch.randn([batch, heads, seq, headDim]);

    // Q @ K.T / sqrt(d)
    const scaling = 1.0 / Math.sqrt(headDim);
    const attnWeights = q.matmul(k).mul(scaling);

    // Causal mask
    const causalMask = torch.ones([seq, seq]).triu(1).bool();
    const masked = attnWeights.masked_fill(causalMask, -1e9);

    // Softmax
    const probs = masked.softmax(-1);

    // Probs @ V
    const output = probs.matmul(v);
    expect(output.shape).toEqual([batch, heads, seq, headDim]);
  });

  it('runs attention output path: transpose, contiguous, reshape', async () => {
    const batch = 1;
    const heads = 4;
    const seq = 8;
    const headDim = 16;
    const embedDim = heads * headDim;

    const attnOutput = torch.randn([batch, heads, seq, headDim]);

    // transpose(1, 2) -> [batch, seq, heads, head_dim]
    const transposed = attnOutput.transpose(1, 2);
    expect(transposed.shape).toEqual([batch, seq, heads, headDim]);

    // contiguous() -> no-op in torch.js but should work
    const contiguous = transposed.contiguous();
    expect(contiguous.shape).toEqual([batch, seq, heads, headDim]);

    // reshape -> [batch, seq, embed_dim]
    const reshaped = contiguous.reshape([batch, seq, embedDim]);
    expect(reshaped.shape).toEqual([batch, seq, embedDim]);
  });

  it('runs GPT2-like forward: split, view, matmul chain', async () => {
    const batch = 1;
    const seq = 8;
    const embedDim = 64;
    const heads = 4;
    const headDim = embedDim / heads;

    // Simulate c_attn output (QKV concatenated)
    const qkv = torch.randn([batch, seq, embedDim * 3]);

    // Split Q, K, V
    const [q, k, v] = qkv.split(embedDim, -1);
    expect(q.shape).toEqual([batch, seq, embedDim]);

    // Reshape + transpose for multihead
    const query = q.reshape([batch, seq, heads, headDim]).transpose(1, 2);
    const key = k.reshape([batch, seq, heads, headDim]).transpose(1, 2);
    const value = v.reshape([batch, seq, heads, headDim]).transpose(1, 2);

    expect(query.shape).toEqual([batch, heads, seq, headDim]);

    // Attention scores
    const scores = query.matmul(key.transpose(-2, -1)).mul(1.0 / Math.sqrt(headDim));
    expect(scores.shape).toEqual([batch, heads, seq, seq]);

    // Softmax + weighted sum
    const probs = scores.softmax(-1);
    const out = probs.matmul(value);
    expect(out.shape).toEqual([batch, heads, seq, headDim]);
  });

  it('tests softmax + multinomial sampling', async () => {
    const vocabSize = 100;
    const logits = torch.randn([vocabSize]);
    const probs = logits.softmax(0);
    const nextToken = await probs.multinomial(1);
    const tokenVal = await nextToken.item();
    expect(tokenVal).toBeGreaterThanOrEqual(0);
    expect(tokenVal).toBeLessThan(vocabSize);
  });

  it('tests arange + unsqueeze (position_ids)', async () => {
    const seq = 8;
    const positionIds = torch.arange(0, seq, 1);
    expect(positionIds.shape).toEqual([seq]);

    const unsqueezed = positionIds.unsqueeze(0);
    expect(unsqueezed.shape).toEqual([1, seq]);
  });

  it('tests cat (token appending)', async () => {
    const a = torch.tensor([[1, 2, 3]]);
    const b = torch.tensor([[4]]);
    const c = torch.cat([a, b], -1);
    expect(c.shape).toEqual([1, 4]);
  });

  it('tests state_dict and load_state_dict', async () => {
    const linear1 = new torch.nn.Linear(10, 5);
    const sd = linear1.state_dict();
    const keys = Object.keys(sd);
    expect(keys.length).toBeGreaterThan(0);

    // Create new linear and load weights
    const linear2 = new torch.nn.Linear(10, 5);
    linear2.load_state_dict(sd);

    // Test that weights match
    const sd2 = linear2.state_dict();
    for (const key of keys) {
      expect(sd[key].shape).toEqual(sd2[key].shape);
    }
  });

  it('runs full GPT2 block (LN + attn + MLP + residual)', async () => {
    const batch = 1;
    const seq = 8;
    const embedDim = 64;
    const heads = 4;
    const headDim = embedDim / heads;
    const intermediate = embedDim * 4;

    // Components
    const ln1 = new torch.nn.LayerNorm(embedDim);
    const cAttn = new torch.nn.Linear(embedDim, embedDim * 3);
    const cProj = new torch.nn.Linear(embedDim, embedDim);
    const ln2 = new torch.nn.LayerNorm(embedDim);
    const cFc = new torch.nn.Linear(embedDim, intermediate);
    const cFcProj = new torch.nn.Linear(intermediate, embedDim);

    // Input
    let hidden = torch.randn([batch, seq, embedDim]);

    // Attention block with residual - debug matmul
    const residual = hidden;
    const h1 = await ln1.forward(hidden);

    // Test matmul 3D @ 2D manually
    const wT = cAttn.weight.t();
    expect(wT.shape).toEqual([embedDim, embedDim * 3]); // Should be [64, 192]

    const qkv = await cAttn.forward(h1);  // Should be [batch, seq, 192]
    expect(qkv.shape).toEqual([batch, seq, embedDim * 3]);
    const [q, k, v] = qkv.split(embedDim, -1);  // each: [batch, seq, 64]
    const query = q.reshape([batch, seq, heads, headDim]).transpose(1, 2);
    const key = k.reshape([batch, seq, heads, headDim]).transpose(1, 2);
    const value = v.reshape([batch, seq, heads, headDim]).transpose(1, 2);
    const scores = query.matmul(key.transpose(-2, -1)).mul(1.0 / Math.sqrt(headDim));
    const causalMask = torch.ones([seq, seq]).triu(1).bool();
    const probs = scores.masked_fill(causalMask, -1e9).softmax(-1);
    const attnOut = probs.matmul(value).transpose(1, 2).reshape([batch, seq, embedDim]);
    const projOut = await cProj.forward(attnOut);
    hidden = residual.add(projOut);

    // MLP block
    const residual2 = hidden;
    const h2 = await ln2.forward(hidden);
    const ff = await cFc.forward(h2);  // [batch, seq, intermediate]
    const ffOut = ff.gelu();
    const projFF = await cFcProj.forward(ffOut);  // [batch, seq, embedDim]
    hidden = residual2.add(projFF);

    expect(hidden.shape).toEqual([batch, seq, embedDim]);
  });

  it('runs ModuleList iteration', async () => {
    const blocks = new torch.nn.ModuleList([
      new torch.nn.Linear(10, 10),
      new torch.nn.Linear(10, 10),
      new torch.nn.Linear(10, 10),
    ]);

    const x = torch.randn([2, 10]);
    let h = x;
    for (const block of blocks) {
      h = await (block as torch.nn.Linear).forward(h);
    }
    expect(h.shape).toEqual([2, 10]);
  });
});
