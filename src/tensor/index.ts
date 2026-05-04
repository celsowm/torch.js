/**
 * Tensor module exports.
 */

export { Tensor, Slice } from './Tensor';
export type { IndexType } from './Tensor';
export type { GradFn, TensorOptions, TensorData, SliceSpec } from './types';

/**
 * Helper function to create a slice for advanced indexing.
 * Equivalent to Python's slice(start, stop, step).
 * @example
 * s(0, 10, 2)    // 0:10:2
 * s(null, 5)     // :5
 * s(5, null)     // 5:
 * s(null, null, 2) // ::2
 */
export async function createSlice(start: number | null = null, stop: number | null = null, step: number | null = null) {
  const { Slice } = await import('./Tensor');
  return new Slice(start, stop, step);
}
