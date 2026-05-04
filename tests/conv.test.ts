import { describe, it, expect, beforeAll } from 'vitest';
import torch from '../src';

describe('torch.js Convolution Operations', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('Conv2d module', () => {
    it('creates a Conv2d layer with correct weight shape', () => {
      const conv = new torch.nn.Conv2d(3, 16, 3);
      expect(conv.weight.shape).toEqual([16, 3, 3, 3]);
      expect(conv.bias?.shape).toEqual([16]);
    });

    it('performs forward pass with correct output shape', async () => {
      const conv = new torch.nn.Conv2d(3, 16, 3, 1, 1);
      const input = torch.randn([2, 3, 32, 32]);
      const output = conv.forward(input);
      expect(output.shape).toEqual([2, 16, 32, 32]);
      await torch.syncDevice();
    });

    it('performs forward pass with stride', async () => {
      const conv = new torch.nn.Conv2d(3, 16, 3, 2, 1);
      const input = torch.randn([2, 3, 32, 32]);
      const output = conv.forward(input);
      expect(output.shape).toEqual([2, 16, 16, 16]);
      await torch.syncDevice();
    });

    it('computes gradients for weight and input', async () => {
      const conv = new torch.nn.Conv2d(3, 16, 3, 1, 1);
      const input = torch.randn([2, 3, 32, 32], { requires_grad: true });
      const output = conv.forward(input);
      const loss = output.mean();
      await loss.backward();
      
      expect(input.grad).toBeDefined();
      expect(conv.weight.grad).toBeDefined();
      expect(conv.bias?.grad).toBeDefined();
      await torch.syncDevice();
    });
  });

  describe('ConvTranspose2d module', () => {
    it('creates a ConvTranspose2d layer with correct weight shape', () => {
      const conv = new torch.nn.ConvTranspose2d(3, 16, 3);
      expect(conv.weight.shape).toEqual([3, 16, 3, 3]);
      expect(conv.bias?.shape).toEqual([16]);
    });

    it('performs forward pass with correct output shape', async () => {
      const conv = new torch.nn.ConvTranspose2d(3, 16, 3, 1, 1);
      const input = torch.randn([2, 3, 32, 32]);
      const output = conv.forward(input);
      // Output shape should be [2, 16, 32, 32] with stride=1, padding=1
      expect(output.shape).toEqual([2, 16, 32, 32]);
      await torch.syncDevice();
    });
  });

  describe('MaxPool2d module', () => {
    it('creates a MaxPool2d layer', () => {
      const pool = new torch.nn.MaxPool2d(2, 2);
      expect(pool).toBeDefined();
    });

    it('performs forward pass with correct output shape', async () => {
      const pool = new torch.nn.MaxPool2d(2, 2);
      const input = torch.randn([2, 3, 32, 32]);
      const output = pool.forward(input);
      expect(output.shape).toEqual([2, 3, 16, 16]);
      await torch.syncDevice();
    });
  });

  describe('AvgPool2d module', () => {
    it('creates an AvgPool2d layer', () => {
      const pool = new torch.nn.AvgPool2d(2, 2);
      expect(pool).toBeDefined();
    });

    it('performs forward pass with correct output shape', async () => {
      const pool = new torch.nn.AvgPool2d(2, 2);
      const input = torch.randn([2, 3, 32, 32]);
      const output = pool.forward(input);
      expect(output.shape).toEqual([2, 3, 16, 16]);
      await torch.syncDevice();
    });
  });

  describe('Functional conv2d', () => {
    it('performs 2D convolution', async () => {
      const input = torch.randn([1, 3, 32, 32]);
      const weight = torch.randn([16, 3, 3, 3]);
      const bias = torch.randn([16]);
      const output = torch.conv2d(input, weight, bias, 1, 1);
      expect(output.shape).toEqual([1, 16, 32, 32]);
      await torch.syncDevice();
    });

    it('computes gradients for conv2d', async () => {
      const input = torch.randn([1, 3, 32, 32], { requires_grad: true });
      const weight = torch.randn([16, 3, 3, 3], { requires_grad: true });
      const bias = torch.randn([16], { requires_grad: true });
      const output = torch.conv2d(input, weight, bias, 1, 1);
      const loss = output.mean();
      await loss.backward();
      
      expect(input.grad).toBeDefined();
      expect(weight.grad).toBeDefined();
      expect(bias.grad).toBeDefined();
      await torch.syncDevice();
    });
  });
});
