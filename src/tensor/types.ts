/**
 * Tensor types and interfaces.
 */

import type { DType } from '../dtype';
import type { Tensor } from './Tensor';

/**
 * Gradient function for autograd.
 * backward can be async for operations that need GPU readback.
 */
export type GradFn = {
  backward(gradOutput: Tensor): void | Promise<void>;
};

/**
 * Options for tensor creation.
 */
export interface TensorOptions {
  dtype?: DType;
  device?: 'webgpu' | 'cpu';
  requires_grad?: boolean;
}

/**
 * Internal tensor data structure.
 */
export interface TensorData {
  buffer: GPUBuffer;
  shape: readonly number[];
  dtype: DType;
  device: 'webgpu';
  requires_grad: boolean;
  grad_fn?: GradFn;
}

/**
 * Slice specification for tensor indexing.
 * Supports Python-style slicing: start:stop:step
 */
export interface SliceSpec {
  start?: number;
  stop?: number;
  step?: number;
}
