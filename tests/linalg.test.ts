import { describe, it, expect, beforeAll } from 'vitest';
import torch from '../src';

describe('torch.linalg', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('cross', () => {
    it('computes cross product of 3-element vectors', async () => {
      const a = torch.tensor([1.0, 2.0, 3.0]);
      const b = torch.tensor([4.0, 5.0, 6.0]);
      const c = torch.linalg.cross(a, b);
      const data = Array.from(await c.toArray());
      // cross([1,2,3], [4,5,6]) = [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
      expect(data[0]).toBeCloseTo(-3);
      expect(data[1]).toBeCloseTo(6);
      expect(data[2]).toBeCloseTo(-3);
    });

    it('computes cross product for batched vectors', async () => {
      const a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      const b = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);
      const c = torch.linalg.cross(a, b);
      const data = Array.from(await c.toArray());
      // first: [2*9-3*8, 3*7-1*9, 1*8-2*7] = [-6, 12, -6]
      expect(data[0]).toBeCloseTo(-6);
      expect(data[1]).toBeCloseTo(12);
      expect(data[2]).toBeCloseTo(-6);
      // second: [5*12-6*11, 6*10-4*12, 4*11-5*10] = [-6, 12, -6]
      expect(data[3]).toBeCloseTo(-6);
      expect(data[4]).toBeCloseTo(12);
      expect(data[5]).toBeCloseTo(-6);
    });
  });
});
