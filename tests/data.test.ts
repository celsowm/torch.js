import { describe, it, expect, beforeAll } from 'vitest';
import torch from '../src';

describe('torch.utils.data', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('TensorDataset', () => {
    it('creates a dataset from tensors', () => {
      const x = torch.tensor([1, 2, 3, 4]);
      const y = torch.tensor([10, 20, 30, 40]);
      const ds = new torch.utils.data.TensorDataset(x, y);
      expect(ds.len()).toBe(4);
    });

    it('throws on mismatched sizes', () => {
      const x = torch.tensor([1, 2, 3]);
      const y = torch.tensor([10, 20]);
      expect(() => new torch.utils.data.TensorDataset(x, y)).toThrow();
    });

    it('throws on empty tensors', () => {
      expect(() => new (torch.utils.data.TensorDataset as any)()).toThrow();
    });

    it('returns [input, target] from get()', async () => {
      const x = torch.tensor([1, 2, 3]);
      const y = torch.tensor([10, 20, 30]);
      const ds = new torch.utils.data.TensorDataset(x, y);
      const [xi, yi] = ds.get(1);
      expect(await xi.item()).toBe(2);
      expect(await yi.item()).toBe(20);
    });
  });

  describe('DataLoader', () => {
    it('iterates over batches', async () => {
      const x = torch.tensor([1, 2, 3, 4]);
      const y = torch.tensor([10, 20, 30, 40]);
      const ds = new torch.utils.data.TensorDataset(x, y);
      const loader = new torch.utils.data.DataLoader(ds, 2);
      let count = 0;
      for await (const [bx, by] of loader) {
        expect(bx.shape[0]).toBe(2);
        expect(by.shape[0]).toBe(2);
        count++;
      }
      expect(count).toBe(2);
    });
  });
});
