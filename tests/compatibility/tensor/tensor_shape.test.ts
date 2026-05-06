import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('Tensor Shape Methods', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ========== reshape ==========
  describe('tensor.reshape()', () => {
    it('reshapes a tensor to new dimensions', () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const result = t.reshape([2, 3]);
      expect(result.shape).toEqual([2, 3]);
      expect(result.numel()).toBe(6);
    });

    it('reshapes preserving total elements', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.reshape([4]);
      expect(result.shape).toEqual([4]);
    });

    it('reshapes to 3D', () => {
      const t = torch.arange(24);
      const result = t.reshape([2, 3, 4]);
      expect(result.shape).toEqual([2, 3, 4]);
    });

    it('single element tensor reshaped to scalar-like', () => {
      const t = torch.tensor([42]);
      const result = t.reshape([]);
      expect(result.shape).toEqual([]);
      expect(result.numel()).toBe(1);
    });

    it('infer dimension with -1', () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const result = t.reshape([2, -1]);
      expect(result.shape).toEqual([2, 3]);
    });
  });

  // ========== view ==========
  describe('tensor.view()', () => {
    it('creates a view with new shape (same as reshape)', () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const result = t.view(2, 3);
      expect(result.shape).toEqual([2, 3]);
    });

    it('view with array argument', () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const result = t.view([2, 2]);
      expect(result.shape).toEqual([2, 2]);
    });

    it('preserves data', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const result = t.view([2, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3, 4]);
    });
  });

  // ========== flatten ==========
  describe('tensor.flatten()', () => {
    it('flattens all dimensions', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.flatten();
      expect(result.shape).toEqual([4]);
    });

    it('flattens from specific start dimension', () => {
      const t = torch.ones([2, 3, 4]);
      const result = t.flatten(1);
      expect(result.shape).toEqual([2, 12]);
    });

    it('flattens between specific dimensions', () => {
      const t = torch.ones([2, 3, 4, 5]);
      const result = t.flatten(1, 2);
      expect(result.shape).toEqual([2, 12, 5]);
    });

    it('flattening 1D is identity', () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.flatten();
      expect(result.shape).toEqual([3]);
    });
  });

  // ========== unsqueeze ==========
  describe('tensor.unsqueeze()', () => {
    it('adds dimension at front', () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.unsqueeze(0);
      expect(result.shape).toEqual([1, 3]);
    });

    it('adds dimension at end', () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.unsqueeze(-1);
      expect(result.shape).toEqual([3, 1]);
    });

    it('adds dimension in middle', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.unsqueeze(1);
      expect(result.shape).toEqual([2, 1, 2]);
    });

    it('multiple unsqueeze', () => {
      const t = torch.tensor([1, 2]);
      const result = t.unsqueeze(0).unsqueeze(0);
      expect(result.shape).toEqual([1, 1, 2]);
    });

    it('scalar-like tensor unsqueeze', () => {
      const t = torch.tensor([42]).reshape([]);
      const result = t.unsqueeze(0);
      expect(result.shape).toEqual([1]);
    });
  });

  // ========== squeeze ==========
  describe('tensor.squeeze()', () => {
    it('removes all dimensions of size 1', () => {
      const t = torch.ones([1, 3, 1, 2, 1]);
      const result = t.squeeze();
      expect(result.shape).toEqual([3, 2]);
    });

    it('squeezes specific dimension', () => {
      const t = torch.ones([1, 3, 1, 2]);
      const result = t.squeeze(0);
      expect(result.shape).toEqual([3, 1, 2]);
    });

    it('squeeze non-unit dimension returns same tensor', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.squeeze(1);
      expect(result.shape).toEqual([2, 2]);
    });

    it('squeeze negative dimension', () => {
      const t = torch.ones([3, 2, 1]);
      const result = t.squeeze(-1);
      expect(result.shape).toEqual([3, 2]);
    });
  });

  // ========== transpose ==========
  describe('tensor.transpose()', () => {
    it('swaps two dimensions', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.transpose(0, 1);
      expect(result.shape).toEqual([2, 2]);
    });

    it('preserves data after transpose', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.transpose(0, 1);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 3, 2, 4]);
    });

    it('transpose with negative dimension', () => {
      const t = torch.ones([2, 3, 4]);
      const result = t.transpose(0, -1);
      expect(result.shape).toEqual([4, 3, 2]);
    });

    it('transpose same dimension returns self', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.transpose(0, 0);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  // ========== permute ==========
  describe('tensor.permute()', () => {
    it('reorders dimensions', () => {
      const t = torch.ones([2, 3, 4]);
      const result = t.permute([2, 0, 1]);
      expect(result.shape).toEqual([4, 2, 3]);
    });

    it('permute with negative dimensions', () => {
      const t = torch.ones([2, 3, 4]);
      const result = t.permute([-1, -2, -3]);
      expect(result.shape).toEqual([4, 3, 2]);
    });

    it('identity permute', () => {
      const t = torch.ones([2, 3]);
      const result = t.permute([0, 1]);
      expect(result.shape).toEqual([2, 3]);
    });

    it('permute 2D is transpose', () => {
      const t = torch.tensor([[1, 2], [3, 4], [5, 6]]);
      const result = t.permute([1, 0]);
      expect(result.shape).toEqual([2, 3]);
    });
  });

  // ========== t() ==========
  describe('tensor.t()', () => {
    it('transposes 2D tensor', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const result = t.t();
      expect(result.shape).toEqual([3, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it('T property is alias for t()', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.T;
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 3, 2, 4]);
    });
  });

  // ========== broadcast_to ==========
  describe('tensor.broadcast_to()', () => {
    it('broadcasts scalar-like tensor', async () => {
      const t = torch.tensor([5]);
      const result = t.broadcast_to([3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([5, 5, 5]);
    });

    it('broadcasts to higher dimension', () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.broadcast_to([2, 3]);
      expect(result.shape).toEqual([2, 3]);
    });

    it('preserves original shape when already compatible', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.broadcast_to([2, 2]);
      expect(result.shape).toEqual([2, 2]);
    });
  });

  // ========== tile ==========
  describe('tensor.tile()', () => {
    it('tiles along each dimension', async () => {
      const t = torch.tensor([1, 2]);
      const result = t.tile([3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 1, 2, 1, 2]);
    });

    it('tiles 2D tensor', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.tile([2, 2]);
      expect(result.shape).toEqual([4, 4]);
    });

    it('tile with 1 is identity', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.tile([1]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3]);
    });
  });

  // ========== repeat ==========
  describe('tensor.repeat()', () => {
    it('repeats tensor', async () => {
      const t = torch.tensor([1, 2]);
      const result = t.repeat([3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 1, 2, 1, 2]);
    });

    it('repeat adds leading dimensions', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.repeat([2, 1]);
      expect(result.shape).toEqual([2, 3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3, 1, 2, 3]);
    });
  });

  // ========== split ==========
  describe('tensor.split()', () => {
    it('splits tensor into equal chunks', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const chunks = t.split(2);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([2]);
      const arr0 = await chunks[0].toArray();
      expect(Array.from(arr0)).toEqual([1, 2]);
    });

    it('splits along specific dimension', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const chunks = t.split(1, 1);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([2, 1]);
    });

    it('last chunk may be smaller', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const chunks = t.split(2);
      expect(chunks.length).toBe(3);
      expect(chunks[2].shape).toEqual([1]);
    });
  });

  // ========== chunk ==========
  describe('tensor.chunk()', () => {
    it('splits tensor into N chunks', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5, 6]);
      const chunks = t.chunk(3);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([2]);
    });

    it('chunk with uneven division', async () => {
      const t = torch.tensor([1, 2, 3, 4, 5]);
      const chunks = t.chunk(2);
      expect(chunks.length).toBe(2);
      expect(chunks[0].shape).toEqual([3]);
      expect(chunks[1].shape).toEqual([2]);
    });

    it('chunk along specific dimension', () => {
      const t = torch.ones([2, 6]);
      const chunks = t.chunk(3, 1);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([2, 2]);
    });
  });

  // ========== unbind ==========
  describe('tensor.unbind()', () => {
    it('removes a dimension, returning tuple of slices', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const chunks = t.unbind(0);
      expect(chunks.length).toBe(2);
      expect(chunks[0].shape).toEqual([2]);
      const arr = await chunks[0].toArray();
      expect(Array.from(arr)).toEqual([1, 2]);
    });

    it('unbind along column dimension', () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const chunks = t.unbind(1);
      expect(chunks.length).toBe(2);
      expect(chunks[0].shape).toEqual([2]);
    });

    it('unbind 1D tensor gives scalars', () => {
      const t = torch.tensor([1, 2, 3]);
      const chunks = t.unbind(0);
      expect(chunks.length).toBe(3);
      expect(chunks[0].shape).toEqual([]);
    });
  });

  // ========== narrow ==========
  describe('tensor.narrow()', () => {
    it('narrow along a dimension', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const result = t.narrow(1, 0, 2);
      expect(result.shape).toEqual([2, 2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 4, 5]);
    });

    it('narrow with length 1 is like select', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const result = t.narrow(0, 1, 1);
      expect(result.shape).toEqual([1, 3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 5, 6]);
    });

    it('narrow with negative dimension', () => {
      const t = torch.ones([2, 3, 4]);
      const result = t.narrow(-1, 0, 2);
      expect(result.shape).toEqual([2, 3, 2]);
    });
  });

  // ========== select ==========
  describe('tensor.select()', () => {
    it('selects index along dimension, removing it', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const result = t.select(0, 1);
      expect(result.shape).toEqual([3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 5, 6]);
    });

    it('select column', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6]]);
      const result = t.select(1, 0);
      expect(result.shape).toEqual([2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 4]);
    });
  });

  // ========== take ==========
  describe('tensor.take()', () => {
    it('takes elements by flat indices', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const indices = torch.tensor([0, 3], { dtype: 'int32' });
      const result = await t.take(indices);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 4]);
    });

    it('take with single index', async () => {
      const t = torch.tensor([10, 20, 30]);
      const indices = torch.tensor([1], { dtype: 'int32' });
      const result = await t.take(indices);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([20]);
    });
  });

  // ========== diagonal ==========
  describe('tensor.diagonal()', () => {
    it('extracts main diagonal of 2D tensor', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const result = t.diagonal();
      expect(result.shape).toEqual([3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 5, 9]);
    });

    it('extracts diagonal with offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const result = t.diagonal(1);
      expect(result.shape).toEqual([2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 6]);
    });

    it('extracts diagonal with negative offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const result = t.diagonal(-1);
      expect(result.shape).toEqual([2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 8]);
    });
  });

  // ========== flip ==========
  describe('tensor.flip()', () => {
    it('flips along a dimension', async () => {
      const t = torch.tensor([1, 2, 3, 4]);
      const result = t.flip([0]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 3, 2, 1]);
    });

    it('flips 2D tensor along rows', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.flip([0]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([3, 4, 1, 2]);
    });

    it('flips along multiple dimensions', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.flip([0, 1]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([4, 3, 2, 1]);
    });

    it('flip with negative dimension', async () => {
      const t = torch.tensor([[1, 2], [3, 4]]);
      const result = t.flip([-1]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([2, 1, 4, 3]);
    });
  });

  // ========== triu ==========
  describe('tensor.triu()', () => {
    it('extracts upper triangular part', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const result = t.triu();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 2, 3, 0, 5, 6, 0, 0, 9]);
    });

    it('upper triangular with diagonal offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const result = t.triu(1);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0, 2, 3, 0, 0, 6, 0, 0, 0]);
    });
  });

  // ========== tril ==========
  describe('tensor.tril()', () => {
    it('extracts lower triangular part', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const result = t.tril();
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 0, 0, 4, 5, 0, 7, 8, 9]);
    });

    it('lower triangular with diagonal offset', async () => {
      const t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
      const result = t.tril(-1);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([0, 0, 0, 4, 0, 0, 7, 8, 0]);
    });
  });

  // ========== diag ==========
  describe('tensor.diag()', () => {
    it('vector to diagonal matrix', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.diag();
      expect(result.shape).toEqual([3, 3]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 0, 0, 0, 2, 0, 0, 0, 3]);
    });

    it('matrix diagonal extraction', async () => {
      const t = torch.tensor([[1, 2], [3, 4], [5, 6]]);
      const result = t.diag();
      expect(result.shape).toEqual([2]);
      const arr = await result.toArray();
      expect(Array.from(arr)).toEqual([1, 4]);
    });

    it('vector to diagonal matrix with offset', async () => {
      const t = torch.tensor([1, 2, 3]);
      const result = t.diag(1);
      expect(result.shape).toEqual([4, 4]);
    });
  });
});
