import { describe, it, expect } from 'vitest';
import torch from '../src';

describe('torch.js Browser Bootup Performance', () => {
  it('measures cold boot and first-op latency', async () => {
    const start = performance.now();
    
    // 1. Measure Init
    const initStart = performance.now();
    await torch.init();
    const initEnd = performance.now();
    
    // 2. Measure First Op (Shader JIT)
    const opStart = performance.now();
    const a = torch.tensor([1, 2, 3]);
    const b = torch.tensor([4, 5, 6]);
    a.add(b);
    await torch.syncDevice();
    const opEnd = performance.now();
    
    const total = opEnd - start;
    
    console.log(`[Browser Bootup] Init: ${(initEnd - initStart).toFixed(2)}ms`);
    console.log(`[Browser Bootup] First Op: ${(opEnd - opStart).toFixed(2)}ms`);
    console.log(`[Browser Bootup] Total: ${total.toFixed(2)}ms`);
    
    // Sanity check
    expect(total).toBeLessThan(2000); // Should definitely boot in under 2s
  });
});
