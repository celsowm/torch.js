import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Core: Tensor Operations', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ==========================================================================
  // torch.cat()
  // ==========================================================================
  describe('cat', () => {
    it('concatenates two 1D tensors', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.cat([a, b]);
      expect(c.shape).toEqual([6]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('concatenates 2D tensors along dim 0', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([[5, 6]]);
      const c = torch.cat([a, b], 0);
      expect(c.shape).toEqual([3, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('concatenates along dim 1', async () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([[5], [6]]);
      const c = torch.cat([a, b], 1);
      expect(c.shape).toEqual([2, 3]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 2, 5, 3, 4, 6]);
    });

    it('concatenates multiple tensors', async () => {
      const a = torch.tensor([1]);
      const b = torch.tensor([2]);
      const c = torch.tensor([3]);
      const d = torch.cat([a, b, c]);
      const arr = await d.toArray();
      expect(arr).toEqual([1, 2, 3]);
    });

    it('throws on shape mismatch', () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([[5, 6, 7]]);
      expect(() => torch.cat([a, b], 0)).toThrow();
    });
  });

  // ==========================================================================
  // torch.stack()
  // ==========================================================================
  describe('stack', () => {
    it('stacks two 1D tensors into 2D', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.stack([a, b]);
      expect(c.shape).toEqual([2, 3]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('stacks along dim 1', () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.stack([a, b], 1);
      expect(c.shape).toEqual([3, 2]);
    });

    it('stacks multiple tensors', () => {
      const a = torch.tensor([1, 2]);
      const b = torch.tensor([3, 4]);
      const c = torch.tensor([5, 6]);
      const d = torch.stack([a, b, c]);
      expect(d.shape).toEqual([3, 2]);
    });
  });

  // ==========================================================================
  // torch.split()
  // ==========================================================================
  describe('split', () => {
    it('splits a tensor into equal chunks', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const chunks = torch.split(t, 2);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([2]);
      const arr0 = await chunks[0].toArray();
      expect(arr0).toEqual([1, 2]);
    });

    it('splits with remainder', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const chunks = torch.split(t, 2);
      expect(chunks.length).toBe(3);
      expect(chunks[2].shape).toEqual([1]);
      const arr2 = await chunks[2].toArray();
      expect(arr2).toEqual([5]);
    });

    it('splits 2D tensor along dim 0', () => {
      const t = torch.tensor([[1, 2], [3, 4], [5, 6]]);
      const chunks = torch.split(t, 1, 0);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([1, 2]);
    });
  });

  // ==========================================================================
  // torch.chunk()
  // ==========================================================================
  describe('chunk', () => {
    it('splits tensor into N chunks', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const chunks = torch.chunk(t, 3);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([2]);
    });

    it('handles uneven chunks', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const chunks = torch.chunk(t, 2);
      expect(chunks.length).toBe(2);
      expect(chunks[0].shape[0]).toBeGreaterThanOrEqual(chunks[1].shape[0]);
    });
  });

  // ==========================================================================
  // torch.vstack()
  // ==========================================================================
  describe('vstack', () => {
    it('stacks tensors vertically', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.vstack([a, b]);
      expect(c.shape).toEqual([2, 3]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('stacks 2D tensors', () => {
      const a = torch.tensor([[1], [2]]);
      const b = torch.tensor([[3], [4]]);
      const c = torch.vstack([a, b]);
      expect(c.shape).toEqual([4, 1]);
    });
  });

  // ==========================================================================
  // torch.hstack()
  // ==========================================================================
  describe('hstack', () => {
    it('stacks tensors horizontally', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.hstack([a, b]);
      expect(c.shape).toEqual([6]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('stacks 2D tensors horizontally', () => {
      const a = torch.tensor([[1], [2]]);
      const b = torch.tensor([[3], [4]]);
      const c = torch.hstack([a, b]);
      expect(c.shape).toEqual([2, 2]);
    });
  });

  // ==========================================================================
  // torch.dstack()
  // ==========================================================================
  describe('dstack', () => {
    it('stacks tensors along depth (dim 2)', () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.dstack([a, b]);
      expect(c.shape).toEqual([1, 3, 2]);
    });

    it('stacks 2D tensors along depth', () => {
      const a = torch.tensor([[1, 2], [3, 4]]);
      const b = torch.tensor([[5, 6], [7, 8]]);
      const c = torch.dstack([a, b]);
      expect(c.shape).toEqual([2, 2, 2]);
    });
  });

  // ==========================================================================
  // torch.column_stack()
  // ==========================================================================
  describe('column_stack', () => {
    it('stacks 1D tensors as columns', async () => {
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const c = torch.column_stack([a, b]);
      expect(c.shape).toEqual([3, 2]);
      const arr = await c.toArray();
      expect(arr).toEqual([1, 4, 2, 5, 3, 6]);
    });
  });

  // ==========================================================================
  // torch.reshape()
  // ==========================================================================
  describe('reshape', () => {
    it('reshapes a tensor', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const r = torch.reshape(t, [2, 3]);
      expect(r.shape).toEqual([2, 3]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it('flattens a tensor', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.reshape(t, [4]);
      expect(r.shape).toEqual([4]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3, 4]);
    });

    it('reshapes with -1 dimension', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const r = torch.reshape(t, [2, -1]);
      expect(r.shape).toEqual([2, 3]);
    });
  });

  // ==========================================================================
  // torch.flatten()
  // ==========================================================================
  describe('flatten', () => {
    it('flattens a 2D tensor', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const f = torch.flatten(t);
      expect(f.shape).toEqual([4]);
      const arr = await f.toArray();
      expect(arr).toEqual([1, 2, 3, 4]);
    });

    it('flattens from start_dim', async () => {
      const t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const f = torch.flatten(t, 1);
      expect(f.shape).toEqual([2, 4]);
    });

    it('flattens partially with start_dim and end_dim', () => {
      const t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
      const f = torch.flatten(t, 0, 1);
      expect(f.shape).toEqual([4, 2]);
    });
  });

  // ==========================================================================
  // torch.squeeze()
  // ==========================================================================
  describe('squeeze', () => {
    it('removes all dimensions of size 1', async () => {
      const t = torch.tensor([[[1, 2, 3]]]);
      expect(t.shape).toEqual([1, 1, 3]);
      const s = torch.squeeze(t);
      expect(s.shape).toEqual([3]);
      const arr = await s.toArray();
      expect(arr).toEqual([1, 2, 3]);
    });

    it('squeezes specific dimension', () => {
      const t = torch.tensor([[[1, 2, 3]]]);
      const s = torch.squeeze(t, 0);
      expect(s.shape).toEqual([1, 3]);
    });

    it('does nothing when no dim is 1', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const s = torch.squeeze(t);
      expect(s.shape).toEqual([2, 2]);
    });
  });

  // ==========================================================================
  // torch.unsqueeze()
  // ==========================================================================
  describe('unsqueeze', () => {
    it('adds a dimension at position 0', () => {
      const t = torch.tensor([1, 2, 3]);
      const u = torch.unsqueeze(t, 0);
      expect(u.shape).toEqual([1, 3]);
    });

    it('adds a dimension at position 1', () => {
      const t = torch.tensor([1, 2, 3]);
      const u = torch.unsqueeze(t, 1);
      expect(u.shape).toEqual([3, 1]);
    });

    it('adds dimension to scalar', () => {
      const t = torch.tensor(42);
      expect(t.shape).toEqual([]);
      const u = torch.unsqueeze(t, 0);
      expect(u.shape).toEqual([1]);
    });
  });

  // ==========================================================================
  // torch.transpose()
  // ==========================================================================
  describe('transpose', () => {
    it('transposes a 2D tensor', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const tr = torch.transpose(t, 0, 1);
      expect(tr.shape).toEqual([3, 2]);
      const arr = await tr.toArray();
      expect(arr).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it('transposes specific dimensions in 3D tensor', () => {
      const t = torch.zeros([2, 3, 4]);
      const tr = torch.transpose(t, 1, 2);
      expect(tr.shape).toEqual([2, 4, 3]);
    });
  });

  // ==========================================================================
  // torch.permute()
  // ==========================================================================
  describe('permute', () => {
    it('permutes dimensions', () => {
      const t = torch.zeros([2, 3, 4]);
      const p = torch.permute(t, [2, 0, 1]);
      expect(p.shape).toEqual([4, 2, 3]);
    });

    it('permutes 2D tensor (same as transpose)', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const p = torch.permute(t, [1, 0]);
      expect(p.shape).toEqual([2, 2]);
      const arr = await p.toArray();
      expect(arr).toEqual([1, 3, 2, 4]);
    });
  });

  // ==========================================================================
  // torch.moveaxis()
  // ==========================================================================
  describe('moveaxis', () => {
    it('moves axis from source to destination', () => {
      const t = torch.zeros([2, 3, 4]);
      const m = torch.moveaxis(t, 0, 2);
      expect(m.shape).toEqual([3, 4, 2]);
    });

    it('moves axis backward', () => {
      const t = torch.zeros([2, 3, 4]);
      const m = torch.moveaxis(t, 2, 0);
      expect(m.shape).toEqual([4, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.swapaxes()
  // ==========================================================================
  describe('swapaxes', () => {
    it('swaps two axes', () => {
      const t = torch.zeros([2, 3, 4]);
      const s = torch.swapaxes(t, 0, 2);
      expect(s.shape).toEqual([4, 3, 2]);
    });

    it('swaps axes in 2D tensor', () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const s = torch.swapaxes(t, 0, 1);
      expect(s.shape).toEqual([3, 2]);
    });
  });

  // ==========================================================================
  // torch.flip()
  // ==========================================================================
  describe('flip', () => {
    it('flips tensor along all dims', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const f = torch.flip(t, [0, 1]);
      expect(f.shape).toEqual([2, 2]);
      const arr = await f.toArray();
      expect(arr).toEqual([4, 3, 2, 1]);
    });

    it('flips along single dim', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const f = torch.flip(t, [0]);
      const arr = await f.toArray();
      expect(arr).toEqual([4, 3, 2, 1]);
    });

    it('flips 2D along dim 0 only', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const f = torch.flip(t, [0]);
      const arr = await f.toArray();
      expect(arr).toEqual([3, 4, 1, 2]);
    });
  });

  // ==========================================================================
  // torch.fliplr()
  // ==========================================================================
  describe('fliplr', () => {
    it('flips tensor left-right', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const f = torch.fliplr(t);
      const arr = await f.toArray();
      expect(arr).toEqual([3, 2, 1, 6, 5, 4]);
    });
  });

  // ==========================================================================
  // torch.flipud()
  // ==========================================================================
  describe('flipud', () => {
    it('flips tensor up-down', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const f = torch.flipud(t);
      const arr = await f.toArray();
      expect(arr).toEqual([3, 4, 1, 2]);
    });
  });

  // ==========================================================================
  // torch.roll()
  // ==========================================================================
  describe('roll', () => {
    it('rolls tensor by shift', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const r = await torch.roll(t, 2);
      const arr = await r.toArray();
      expect(arr).toEqual([4, 5, 1, 2, 3]);
    });

    it('rolls with negative shift', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const r = await torch.roll(t, -2);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 4, 5, 1, 2]);
    });

    it('rolls along specific dim', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const r = await torch.roll(t, 1, 1);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 1, 2, 6, 4, 5]);
    });
  });

  // ==========================================================================
  // torch.tril()
  // ==========================================================================
  describe('tril', () => {
    it('extracts lower triangular part', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const l = torch.tril(t);
      const arr = await l.toArray();
      expect(arr).toEqual([1, 0, 0, 4, 5, 0, 7, 8, 9]);
    });

    it('tril with diagonal offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const l = torch.tril(t, 1);
      const arr = await l.toArray();
      expect(arr).toEqual([1, 2, 0, 4, 5, 6, 7, 8, 9]);
    });

    it('tril with negative diagonal offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const l = torch.tril(t, -1);
      const arr = await l.toArray();
      expect(arr).toEqual([0, 0, 0, 4, 0, 0, 7, 8, 0]);
    });
  });

  // ==========================================================================
  // torch.triu()
  // ==========================================================================
  describe('triu', () => {
    it('extracts upper triangular part', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const u = torch.triu(t);
      const arr = await u.toArray();
      expect(arr).toEqual([1, 2, 3, 0, 5, 6, 0, 0, 9]);
    });

    it('triu with diagonal offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const u = torch.triu(t, 1);
      const arr = await u.toArray();
      expect(arr).toEqual([0, 2, 3, 0, 0, 6, 0, 0, 0]);
    });
  });

  // ==========================================================================
  // torch.diag()
  // ==========================================================================
  describe('diag', () => {
    it('extracts diagonal from matrix', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const d = torch.diag(t);
      const arr = await d.toArray();
      expect(arr).toEqual([1, 5, 9]);
    });

    it('creates diagonal matrix from vector', async () => {
      const t = torch.tensor([1, 2, 3]);
      const d = torch.diag(t);
      expect(d.shape).toEqual([3, 3]);
      const arr = await d.toArray();
      expect(arr).toEqual([1, 0, 0, 0, 2, 0, 0, 0, 3]);
    });

    it('extracts diagonal with offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const d = torch.diag(t, 1);
      const arr = await d.toArray();
      expect(arr).toEqual([2, 6]);
    });
  });

  // ==========================================================================
  // torch.diagonal()
  // ==========================================================================
  describe('diagonal', () => {
    it('extracts diagonal from matrix', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const d = torch.diagonal(t);
      const arr = await d.toArray();
      expect(arr).toEqual([1, 5, 9]);
    });

    it('extracts diagonal with offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const d = torch.diagonal(t, 1);
      const arr = await d.toArray();
      expect(arr).toEqual([2, 6]);
    });
  });

  // ==========================================================================
  // torch.broadcast_to()
  // ==========================================================================
  describe('broadcast_to', () => {
    it('broadcasts scalar to shape', async () => {
      const t = torch.tensor(5);
      const b = torch.broadcast_to(t, [3]);
      expect(b.shape).toEqual([3]);
      const arr = await b.toArray();
      expect(arr.every((v: number) => v === 5)).toBe(true);
    });

    it('broadcasts 1D to 2D', async () => {
      const t = torch.tensor([1, 2, 3]);
      const b = torch.broadcast_to(t, [2, 3]);
      expect(b.shape).toEqual([2, 3]);
      const arr = await b.toArray();
      expect(arr).toEqual([1, 2, 3, 1, 2, 3]);
    });
  });

  // ==========================================================================
  // torch.tile()
  // ==========================================================================
  describe('tile', () => {
    it('tiles a tensor', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = torch.tile(t, [2]);
      expect(r.shape).toEqual([6]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3, 1, 2, 3]);
    });

    it('tiles 2D tensor', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.tile(t, [2, 1]);
      expect(r.shape).toEqual([4, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3, 4, 1, 2, 3, 4]);
    });
  });

  // ==========================================================================
  // torch.repeat_interleave()
  // ==========================================================================
  describe('repeat_interleave', () => {
    it('repeats each element', async () => {
      const t = torch.tensor([1, 2, 3]);
      const r = await torch.repeat_interleave(t, 2);
      expect(r.shape).toEqual([6]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 1, 2, 2, 3, 3]);
    });

    it('repeats with dim', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = await torch.repeat_interleave(t, 2, 0);
      expect(r.shape).toEqual([4, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 1, 2, 3, 4, 3, 4]);
    });
  });

  // ==========================================================================
  // torch.unbind()
  // ==========================================================================
  describe('unbind', () => {
    it('unbinds along dim 0', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const parts = torch.unbind(t);
      expect(parts.length).toBe(2);
      const arr0 = await parts[0].toArray();
      expect(arr0).toEqual([1, 2]);
    });

    it('unbinds along dim 1', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const parts = torch.unbind(t, 1);
      expect(parts.length).toBe(2);
      const arr0 = await parts[0].toArray();
      expect(arr0).toEqual([1, 3]);
    });
  });

  // ==========================================================================
  // torch.narrow()
  // ==========================================================================
  describe('narrow', () => {
    it('narrows along dim 0', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const n = torch.narrow(t, 0, 1, 2);
      expect(n.shape).toEqual([2, 3]);
      const arr = await n.toArray();
      expect(arr).toEqual([4, 5, 6, 7, 8, 9]);
    });

    it('narrows along dim 1', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const n = torch.narrow(t, 1, 1, 2);
      expect(n.shape).toEqual([2, 2]);
      const arr = await n.toArray();
      expect(arr).toEqual([2, 3, 5, 6]);
    });
  });

  // ==========================================================================
  // torch.select()
  // ==========================================================================
  describe('select', () => {
    it('selects row from 2D tensor', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const s = torch.select(t, 0, 1);
      expect(s.shape).toEqual([3]);
      const arr = await s.toArray();
      expect(arr).toEqual([4, 5, 6]);
    });

    it('selects column from 2D tensor', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const s = torch.select(t, 1, 2);
      expect(s.shape).toEqual([2]);
      const arr = await s.toArray();
      expect(arr).toEqual([3, 6]);
    });
  });

  // ==========================================================================
  // torch.take()
  // ==========================================================================
  describe('take', () => {
    it('takes elements by flattened index', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const idx = torch.tensor([0, 3]);
      const r = await torch.take(t, idx);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 4]);
    });
  });

  // ==========================================================================
  // torch.gather()
  // ==========================================================================
  describe('gather', () => {
    it('gathers values along dim 0', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const idx = torch.tensor([[1, 0], [0, 1]], { dtype: 'int32' });
      const r = await torch.gather(t, 0, idx);
      expect(r.shape).toEqual([2, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([3, 2, 1, 4]);
    });

    it('gathers values along dim 1', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const idx = torch.tensor([[1, 0], [0, 1]], { dtype: 'int32' });
      const r = await torch.gather(t, 1, idx);
      expect(r.shape).toEqual([2, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([2, 1, 3, 4]);
    });
  });

  // ==========================================================================
  // torch.scatter()
  // ==========================================================================
  describe('scatter', () => {
    it('scatters values along dim 1', async () => {
      const src = torch.tensor([[1, 2], [3, 4]]);
      const idx = torch.tensor([[0, 1], [1, 0]], { dtype: 'int32' });
      const out = torch.zeros([2, 2]);
      const r = await torch.scatter(out, 1, idx, src);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 4, 3]);
    });
  });

  // ==========================================================================
  // torch.scatter_add()
  // ==========================================================================
  describe('scatter_add', () => {
    it('scatters and adds values', async () => {
      const src = torch.tensor([[1, 2], [3, 4]]);
      const idx = torch.tensor([[0, 1], [0, 1]], { dtype: 'int32' });
      const out = torch.zeros([2, 2]);
      const r = await torch.scatter_add(out, 0, idx, src);
      const arr = await r.toArray();
      expect(arr).toEqual([4, 0, 0, 6]);
    });
  });

  // ==========================================================================
  // torch.where()
  // ==========================================================================
  describe('where', () => {
    it('selects elements based on condition', async () => {
      const cond = torch.tensor([true, false, true]);
      const a = torch.tensor([1, 2, 3]);
      const b = torch.tensor([4, 5, 6]);
      const r = torch.where(cond, a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 5, 3]);
    });

    it('works with all true condition', async () => {
      const cond = torch.tensor([true, true, true]);
      const a = torch.tensor([10, 20, 30]);
      const b = torch.tensor([0, 0, 0]);
      const r = torch.where(cond, a, b);
      const arr = await r.toArray();
      expect(arr).toEqual([10, 20, 30]);
    });
  });

  // ==========================================================================
  // torch.masked_select()
  // ==========================================================================
  describe('masked_select', () => {
    it('selects elements where mask is true', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const mask = torch.tensor([true, false, true, false, true], { dtype: 'bool' });
      const r = await torch.masked_select(t, mask);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 3, 5]);
    });

    it('selects from 2D tensor', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const mask = torch.tensor([[true, false], [false, true]], { dtype: 'bool' });
      const r = await torch.masked_select(t, mask);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 4]);
    });
  });

  // ==========================================================================
  // torch.nonzero()
  // ==========================================================================
  describe('nonzero', () => {
    it('returns indices of non-zero elements', async () => {
      const t = torch.tensor([0, 1, 0, 3, 0]);
      const r = await torch.nonzero(t);
      expect(r.shape[1]).toBe(1);
      expect(r.shape[0]).toBe(2);
    });

    it('returns indices for 2D tensor', async () => {
      const t = torch.tensor([[0, 1], [2, 0]]);
      const r = await torch.nonzero(t);
      expect(r.shape[1]).toBe(2);
      expect(r.shape[0]).toBe(2);
    });
  });

  // ==========================================================================
  // torch.index_select()
  // ==========================================================================
  describe('index_select', () => {
    it('selects rows by index', async () => {
      const t = torch.tensor([[1, 2], [3, 4], [5, 6]]);
      const idx = torch.tensor([0, 2], { dtype: 'int32' });
      const r = torch.index_select(t, 0, idx);
      expect(r.shape).toEqual([2, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 5, 6]);
    });

    it('selects columns by index', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const idx = torch.tensor([0, 2], { dtype: 'int32' });
      const r = torch.index_select(t, 1, idx);
      expect(r.shape).toEqual([2, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 3, 4, 6]);
    });
  });

  // ==========================================================================
  // torch.cumsum()
  // ==========================================================================
  describe('cumsum', () => {
    it('computes cumulative sum', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.cumsum(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 3, 6, 10]);
    });

    it('cumsum along rows', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.cumsum(t, 0);
      expect(r.shape).toEqual([2, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 4, 6]);
    });

    it('cumsum along cols', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.cumsum(t, 1);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 3, 3, 7]);
    });
  });

  // ==========================================================================
  // torch.cumprod()
  // ==========================================================================
  describe('cumprod', () => {
    it('computes cumulative product', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const r = torch.cumprod(t, 0);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 6, 24]);
    });

    it('cumprod along dimension', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const r = torch.cumprod(t, 0);
      expect(r.shape).toEqual([2, 2]);
      const arr = await r.toArray();
      expect(arr).toEqual([1, 2, 3, 8]);
    });
  });

  // ==========================================================================
  // torch.heaviside()
  // ==========================================================================
  describe('heaviside', () => {
    it('computes heaviside step function', async () => {
      const t = torch.tensor([-1.0, 0.0, 1.0, 2.0]);
      const values = torch.tensor([0.5]);
      const r = torch.heaviside(t, values);
      const arr = await r.toArray();
      expect(arr[0]).toBe(0);
      expect(arr[1]).toBe(0.5);
      expect(arr[2]).toBe(1);
      expect(arr[3]).toBe(1);
    });
  });
});
