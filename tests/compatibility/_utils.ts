/**
 * Shared test utilities for compatibility tests.
 */

import { createTorch } from '../../src/create_torch';

export const torch = createTorch();

/**
 * Convert tensor data to array for assertions.
 */
export async function toArray(tensor: any): Promise<Float32Array> {
  return await tensor.toArray();
}

/**
 * Convert array to tensor using torch.tensor().
 */
export function tensor(data: any, options?: any) {
  return torch.tensor(data, options);
}

/**
 * Compare two tensors with tolerance.
 */
export async function allclose(
  a: any,
  b: any,
  rtol = 1e-5,
  atol = 1e-8,
  equalNan = false
): Promise<boolean> {
  if (a.shape.join(',') !== b.shape.join(',')) return false;
  const aData = await toArray(a);
  const bData = await toArray(b);
  for (let i = 0; i < aData.length; i++) {
    if (equalNan && Number.isNaN(aData[i]) && Number.isNaN(bData[i])) continue;
    if (Math.abs(aData[i] - bData[i]) > atol + rtol * Math.abs(bData[i])) return false;
  }
  return true;
}

/**
 * Assert two tensors are close with helpful error message.
 */
export async function expectClose(
  a: any,
  b: any,
  rtol = 1e-4,
  atol = 1e-6,
  msg?: string
) {
  const { expect } = await import('vitest');
  const close = await allclose(a, b, rtol, atol);
  const aData = await toArray(a);
  const bData = await toArray(b);
  expect(
    close,
    msg || `Expected tensors to be close.\nGot: ${Array.from(aData)}\nExpected: ${Array.from(bData)}`
  ).toBe(true);
}

/**
 * Create a tensor with given data and check shape matches.
 */
export function makeTensor(data: number[], shape: number[]) {
  return torch.tensor(data, { shape });
}

/**
 * Check that tensor has expected shape.
 */
export function expectShape(tensor: any, expected: number[], msg?: string) {
  const { expect } = require('vitest');
  expect(tensor.shape, msg || `Shape mismatch`).toEqual(expected);
}

/**
 * Run a test with async setup and sync assertions.
 */
export async function tensorTest(name: string, fn: () => Promise<void>) {
  const { it } = await import('vitest');
  it(name, fn);
}

/**
 * Verify that a function throws with specific error.
 */
export function expectThrows(fn: () => any, errorType: string | RegExp, msg?: string) {
  const { expect } = require('vitest');
  expect(fn).toThrow(errorType);
}
