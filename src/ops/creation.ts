/**
 * Tensor creation operations.
 * @status partial
 */

import { Tensor, TensorOptions } from '../tensor';
import { DType, getDTypeBytes, getTypedArrayConstructor } from '../dtype';
import {
  getDevice,
  createBufferWithData,
  createStorageBuffer,
  getOrCreatePipeline,
  calculateWorkgroups,
  FILL_SHADER,
  RANDOM_SHADER,
  TRIL_SHADER,
  BufferUsage,
} from '../backend';
import { numel, validateShape } from '../utils/shape';

type NestedArray = number | NestedArray[];

/**
 * Infer shape from nested array.
 */
function inferShapeFromData(data: NestedArray): number[] {
  if (typeof data === 'number') {
    return [];
  }
  if (!Array.isArray(data) || data.length === 0) {
    return [0];
  }
  const first = data[0];
  const restShape = inferShapeFromData(first);
  return [data.length, ...restShape];
}

/**
 * Flatten nested array to 1D.
 */
function flattenData(data: NestedArray): number[] {
  if (typeof data === 'number') {
    return [data];
  }
  const result: number[] = [];
  for (const item of data) {
    result.push(...flattenData(item));
  }
  return result;
}

/**
 * Create a tensor from array data.
 * @status implemented
 * @pytorch torch.tensor()
 */
export function tensor(
  data: NestedArray | number[],
  options: TensorOptions = {}
): Tensor {
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  const shape = inferShapeFromData(data);
  const flat = flattenData(data);

  // Convert to typed array
  const TypedArrayCtor = getTypedArrayConstructor(dtype);
  const typedData = new TypedArrayCtor(flat);

  const buffer = createBufferWithData(typedData, dtype);

  return new Tensor({
    buffer,
    shape,
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}

/**
 * Create a tensor filled with zeros.
 * @status implemented
 * @pytorch torch.zeros()
 */
export function zeros(shape: number[], options: TensorOptions = {}): Tensor {
  validateShape(shape);
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  return _fill(shape, 0, dtype, requires_grad);
}

/**
 * Create a tensor filled with ones.
 * @status implemented
 * @pytorch torch.ones()
 */
export function ones(shape: number[], options: TensorOptions = {}): Tensor {
  validateShape(shape);
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  return _fill(shape, 1, dtype, requires_grad);
}

/**
 * Create a tensor filled with a specific value.
 * @status implemented
 * @pytorch torch.full()
 */
export function full(
  shape: number[],
  fillValue: number,
  options: TensorOptions = {}
): Tensor {
  validateShape(shape);
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  return _fill(shape, fillValue, dtype, requires_grad);
}

/**
 * Create a tensor with zeros like another tensor.
 * @status implemented
 * @pytorch torch.zeros_like()
 */
export function zeros_like(tensor: Tensor, options: TensorOptions = {}): Tensor {
  return zeros([...tensor.shape], {
    dtype: options.dtype ?? tensor.dtype,
    requires_grad: options.requires_grad ?? tensor.requires_grad,
  });
}

/**
 * Create a tensor with ones like another tensor.
 * @status implemented
 * @pytorch torch.ones_like()
 */
export function ones_like(tensor: Tensor, options: TensorOptions = {}): Tensor {
  return ones([...tensor.shape], {
    dtype: options.dtype ?? tensor.dtype,
    requires_grad: options.requires_grad ?? tensor.requires_grad,
  });
}

/**
 * Create a tensor filled with a scalar value, with the same size as input.
 * @status implemented
 * @pytorch torch.full_like()
 */
export function full_like(
  input: Tensor,
  fillValue: number,
  options: TensorOptions = {}
): Tensor {
  return full([...input.shape], fillValue, {
    dtype: options.dtype ?? input.dtype,
    requires_grad: options.requires_grad ?? input.requires_grad,
  });
}

/**
 * Returns an uninitialized tensor with the same size as input.
 * @status implemented
 * @pytorch torch.empty_like()
 */
export function empty_like(
  input: Tensor,
  options: TensorOptions = {}
): Tensor {
  return zeros_like(input, options);
}

/**
 * Create a tensor with random values from uniform distribution [0, 1).
 * @status implemented
 * @pytorch torch.rand()
 */
export function rand(shape: number[], options: TensorOptions = {}): Tensor {
  validateShape(shape);
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  return _random(shape, 'rand', dtype, requires_grad);
}

/**
 * Create a tensor with random values from standard normal distribution.
 * @status implemented
 * @pytorch torch.randn()
 */
export function randn(shape: number[], options: TensorOptions = {}): Tensor {
  validateShape(shape);
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  return _random(shape, 'randn', dtype, requires_grad);
}

/**
 * Create an identity matrix.
 * @status implemented
 * @pytorch torch.eye()
 */
export function eye(n: number, m?: number, options: TensorOptions = {}): Tensor {
  m = m ?? n;
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  // Create on CPU and upload (simple for now, could optimize with shader)
  const data = new Float32Array(n * m);
  const minDim = Math.min(n, m);
  for (let i = 0; i < minDim; i++) {
    data[i * m + i] = 1;
  }

  const buffer = createBufferWithData(data, dtype);

  return new Tensor({
    buffer,
    shape: [n, m],
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}

/**
 * Create a tensor with values in a range.
 * @status implemented
 * @pytorch torch.arange()
 */
export function arange(
  start: number,
  end?: number,
  step: number = 1,
  options: TensorOptions = {}
): Tensor {
  // Handle single argument case
  if (end === undefined) {
    end = start;
    start = 0;
  }

  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  const length = Math.ceil((end - start) / step);
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    data[i] = start + i * step;
  }

  const buffer = createBufferWithData(data, dtype);

  return new Tensor({
    buffer,
    shape: [length],
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}

/**
 * Create a tensor with linearly spaced values.
 * @status implemented
 * @pytorch torch.linspace()
 */
export function linspace(
  start: number,
  end: number,
  steps: number,
  options: TensorOptions = {}
): Tensor {
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  const data = new Float32Array(steps);
  if (steps === 1) {
    data[0] = start;
  } else {
    const stepSize = (end - start) / (steps - 1);
    for (let i = 0; i < steps; i++) {
      data[i] = start + i * stepSize;
    }
  }

  const buffer = createBufferWithData(data, dtype);

  return new Tensor({
    buffer,
    shape: [steps],
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}

/**
 * Create a tensor with logarithmically spaced values.
 * @status implemented
 * @pytorch torch.logspace()
 */
export function logspace(
  start: number,
  end: number,
  steps: number,
  base: number = 10.0,
  options: TensorOptions = {}
): Tensor {
  const result = linspace(start, end, steps, options);
  if (base === Math.E) {
    return result.exp();
  }
  return result.mul(Math.log(base)).exp();
}

/**
 * Lower triangular part of a matrix.
 * @status implemented
 * @pytorch torch.tril()
 */
export function tril(input: Tensor, diagonal: number = 0): Tensor {
  if (input.shape.length !== 2) {
    throw new Error('tril currently only supports 2D tensors');
  }

  const [rows, cols] = input.shape;
  const device = getDevice();
  const outputBuffer = createStorageBuffer(input.numel() * getDTypeBytes(input.dtype));

  const paramsData = new ArrayBuffer(16);
  new Uint32Array(paramsData, 0, 2).set([rows, cols]);
  new Int32Array(paramsData, 8, 1)[0] = diagonal;

  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  const pipeline = getOrCreatePipeline(TRIL_SHADER, 'main');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(input.numel()));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  return new Tensor({
    buffer: outputBuffer,
    shape: [rows, cols],
    dtype: input.dtype,
    device: 'webgpu',
    requires_grad: false,
  });
}

/**
 * Concatenate tensors along a dimension.
 * @status implemented
 * @pytorch torch.cat()
 */
export function cat(tensors: Tensor[], dim: number = 0): Tensor {
  return Tensor.cat(tensors, dim);
}

/**
 * Stack tensors along a new dimension.
 * @status implemented
 * @pytorch torch.stack()
 */
export function stack(tensors: Tensor[], dim: number = 0): Tensor {
  if (tensors.length === 0) {
    throw new Error('stack requires at least one tensor');
  }
  const expanded = tensors.map(t => t.unsqueeze(dim));
  return cat(expanded, dim);
}

/**
 * Stack tensors vertically (row wise).
 * @status implemented
 * @pytorch torch.vstack()
 */
export function vstack(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new Error('vstack requires at least one tensor');
  }
  const first = tensors[0];
  if (first.dim() === 1) {
    return stack(tensors, 0);
  }
  return cat(tensors, 0);
}
export const row_stack = vstack;

/**
 * Stack tensors horizontally (column wise).
 * @status implemented
 * @pytorch torch.hstack()
 */
export function hstack(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new Error('hstack requires at least one tensor');
  }
  const first = tensors[0];
  if (first.dim() === 1) {
    return cat(tensors, 0);
  }
  return cat(tensors, 1);
}

/**
 * Stack tensors depthwise (along third axis).
 * @status implemented
 * @pytorch torch.dstack()
 */
export function dstack(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new Error('dstack requires at least one tensor');
  }
  const processed = tensors.map(t => {
    if (t.dim() === 1) return t.unsqueeze(0).unsqueeze(1);
    if (t.dim() === 2) return t.unsqueeze(2);
    return t;
  });
  return cat(processed, 2);
}

/**
 * Stack 1-D arrays as columns into a 2-D array.
 * @status implemented
 * @pytorch torch.column_stack()
 */
export function column_stack(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new Error('column_stack requires at least one tensor');
  }
  const processed = tensors.map(t => {
    if (t.dim() === 1) return t.unsqueeze(1);
    return t;
  });
  return hstack(processed);
}

// ============ Private Helpers ============

function _fill(
  shape: number[],
  value: number,
  dtype: DType,
  requires_grad: boolean
): Tensor {
  const device = getDevice();
  const n = numel(shape);
  const outputBuffer = createStorageBuffer(n * getDTypeBytes(dtype));

  // Create params uniform buffer
  const paramsData = new ArrayBuffer(8);
  new Float32Array(paramsData, 0, 1)[0] = value;
  new Uint32Array(paramsData, 4, 1)[0] = n;
  const paramsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  const pipeline = getOrCreatePipeline(FILL_SHADER, 'fill');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(n));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  return new Tensor({
    buffer: outputBuffer,
    shape,
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}

// Global seed for random number generation
let globalSeed = Math.floor(Math.random() * 2147483647);

/**
 * Set the random seed for reproducibility.
 */
export function manual_seed(seed: number): void {
  globalSeed = seed;
}

function _random(
  shape: number[],
  op: 'rand' | 'randn',
  dtype: DType,
  requires_grad: boolean
): Tensor {
  const device = getDevice();
  const n = numel(shape);
  const outputBuffer = createStorageBuffer(n * getDTypeBytes(dtype));

  // Increment seed for each call
  globalSeed = (globalSeed * 1103515245 + 12345) & 0x7fffffff;

  // Create RNG params uniform buffer
  const paramsData = new Uint32Array([globalSeed, n]);
  const paramsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  const pipeline = getOrCreatePipeline(RANDOM_SHADER, op);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(n));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  return new Tensor({
    buffer: outputBuffer,
    shape,
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}
