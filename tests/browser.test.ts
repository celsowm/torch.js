import { describe, it, expect, beforeAll } from 'vitest';
import torch from '../src';

describe('torch.js Browser Smoke Test', () => {
  beforeAll(async () => {
    // In browser tests, vitest-browser already provides WebGPU if supported
    await torch.init();
  });

  it('can create a tensor and perform addition', async () => {
    const a = torch.tensor([1, 2, 3]);
    const b = torch.tensor([4, 5, 6]);
    const c = a.add(b);
    
    expect(c.shape).toEqual([3]);
    expect(await c.toArray()).toEqual([5, 7, 9]);
  });

  it('can perform matrix multiplication', async () => {
    const a = torch.tensor([[1, 2], [3, 4]]);
    const b = torch.tensor([[5, 6], [7, 8]]);
    const c = a.matmul(b);
    
    expect(await c.toNestedArray()).toEqual([[19, 22], [43, 50]]);
  });
});
