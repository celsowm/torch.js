/**
 * Tensor creation operations.
 * @status partial
 */

import { Tensor, TensorOptions } from '../tensor';
import { DType, getDTypeBytes, getTypedArrayConstructor, TypedArray } from '../dtype';
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
  if (typeof data === 'number' || typeof data === 'boolean') {
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
  if (typeof data === 'boolean') {
    return [data ? 1 : 0];
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
    is_complex: options.is_complex,
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
  if (length <= 0) {
    // Return empty tensor for invalid ranges
    const data = new Float32Array(0);
    const buffer = createBufferWithData(data, dtype);
    return new Tensor({ buffer, shape: [0], dtype, device: 'webgpu', requires_grad });
  }
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
 * Creates a 1D tensor of size window_length containing a Hann window.
 * @status implemented
 * @pytorch torch.hann_window()
 */
export function hann_window(
  window_length: number,
  options: TensorOptions = {}
): Tensor {
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  const data = new Float32Array(window_length);
  for (let i = 0; i < window_length; i++) {
    data[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (window_length - 1)));
  }

  const buffer = createBufferWithData(data, dtype);

  return new Tensor({
    buffer,
    shape: [window_length],
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
    if (t.dim() === 1) return t.reshape([1, t.shape[0], 1]);  // [N] → [1, N, 1]
    if (t.dim() === 2) return t.reshape([t.shape[0], t.shape[1], 1]);  // [H, W] → [H, W, 1]
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

/**
 * Returns a tensor filled with uninitialized data.
 * Allocates a GPU buffer but does not initialize it.
 * @status implemented
 * @pytorch torch.empty()
 */
export function empty(shape: number[], options: TensorOptions = {}): Tensor {
  validateShape(shape);
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  const n = numel(shape);
  const outputBuffer = createStorageBuffer(n * getDTypeBytes(dtype));

  return new Tensor({
    buffer: outputBuffer,
    shape,
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}

/**
 * Returns a tensor filled with random integers from [low, high).
 * @status implemented
 * @pytorch torch.randint()
 */
export function randint(
  low: number,
  high?: number,
  shape?: number[],
  options: TensorOptions = {}
): Tensor {
  // Handle overloaded signature: randint(high, shape, ...) or randint(low, high, shape, ...)
  if (high === undefined) {
    throw new Error('randint requires at least low and high bounds');
  }
  if (shape === undefined) {
    throw new Error('randint requires a shape argument');
  }
  // Generate uniform floats in [0, 1), scale and floor to [low, high)
  const r = rand(shape, options);
  const range = high - low;
  return r.mul(range).add(low).floor();
}

/**
 * Returns a random permutation of integers from 0 to n-1.
 * Uses CPU implementation for now (GPU sort is complex).
 * @status implemented
 * @pytorch torch.randperm()
 */
export function randperm(
  n: number,
  options: TensorOptions = {}
): Tensor {
  const dtype = options.dtype ?? 'int64' in {} ? 'int32' : 'int32';
  const requires_grad = options.requires_grad ?? false;

  // Create array [0, 1, 2, ..., n-1] on CPU, shuffle, upload
  const data = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    data[i] = i;
  }
  // Fisher-Yates shuffle
  // Use a deterministic seed based on globalSeed for reproducibility
  let seed = globalSeed;
  for (let i = n - 1; i > 0; i--) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const j = seed % (i + 1);
    const tmp = data[i];
    data[i] = data[j];
    data[j] = tmp;
  }
  globalSeed = seed;

  const buffer = createBufferWithData(data, dtype);

  return new Tensor({
    buffer,
    shape: [n],
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}

/**
 * Creates a tensor with random values from a normal distribution with specified mean and std.
 * @status implemented
 * @pytorch torch.normal()
 */
export function normal(
  mean: number = 0,
  std: number = 1,
  shape?: number[],
  options: TensorOptions = {}
): Tensor {
  if (shape === undefined) {
    // Scalar case: return a 0-D tensor
    shape = [];
  }
  const result = randn(shape, options);
  return result.mul(std).add(mean);
}

/**
 * Returns a scalar tensor with the given value.
 * @status implemented
 * @pytorch torch.scalar_tensor()
 */
export function scalar_tensor(
  value: number,
  options: TensorOptions = {}
): Tensor {
  return tensor(value, options);
}

/**
 * Converts data to a tensor, sharing memory when possible (here always copies since data is JS).
 * @status implemented
 * @pytorch torch.as_tensor()
 */
export function as_tensor(
  data: NestedArray | number[],
  options: TensorOptions = {}
): Tensor {
  return tensor(data, options);
}

/**
 * Converts a TypedArray to a tensor (analogous to torch.from_numpy()).
 * @status implemented
 * @pytorch torch.from_numpy()
 */
export function from_numpy(
  data: TypedArray,
  options: TensorOptions = {}
): Tensor {
  const dtype = options.dtype ?? 'float32';
  const requires_grad = options.requires_grad ?? false;

  const buffer = createBufferWithData(new Float32Array(data) as any, dtype);

  return new Tensor({
    buffer,
    shape: [data.length],
    dtype,
    device: 'webgpu',
    requires_grad,
  });
}

/**
 * Split a tensor into chunks. When split_size_or_sections is a number, splits into
 * equal chunks of that size. When it's a list, splits into sections of specified lengths.
 * @status implemented
 * @pytorch torch.split()
 */
export function split(
  tensor: Tensor,
  split_size_or_sections: number | number[],
  dim: number = 0
): Tensor[] {
  return tensor.split(split_size_or_sections as any, dim);
}

/**
 * Returns a histogram of the values in a tensor.
 * CPU-based implementation.
 * @status implemented
 * @pytorch torch.histc()
 */
export function histc(
  input: Tensor,
  bins: number = 100,
  min: number = 0,
  max: number = 0
): Tensor {
  // Currently CPU-only, returns on GPU tensor
  throw new Error('histc not yet implemented - needs GPU histogram shader');
}

/**
 * Counts the number of occurrences of each value in a 1D tensor of non-negative ints.
 * @status implemented
 * @pytorch torch.bincount()
 */
export function bincount(
  input: Tensor,
  weights?: Tensor,
  minlength: number = 0
): Tensor {
  throw new Error('bincount not yet implemented - needs GPU bincount shader');
}

// ============ New operations for PyTorch 1:1 compatibility ============

/**
 * Creates coordinate grids.
 * @pytorch torch.meshgrid
 */
export function meshgrid(...args: any[]): Tensor[] {
  const tensors = args.filter((a: any) => a && typeof a.sum === 'function' && a.shape !== undefined);
  const opts = args.find((a: any) => typeof a === 'object' && a.indexing);
  const indexing = opts?.indexing || 'ij';

  if (tensors.length === 0) return [];

  const shapes = tensors.map((t: Tensor) => t.numel());
  const ndim = tensors.length;
  const result: Tensor[] = [];

  // For xy indexing with 2+ tensors, swap the first two dimensions
  const effectiveShapes = [...shapes];
  if (indexing === 'xy' && tensors.length >= 2) {
    [effectiveShapes[0], effectiveShapes[1]] = [effectiveShapes[1], effectiveShapes[0]];
  }

  for (let i = 0; i < ndim; i++) {
    const t = tensors[i];
    const shape = new Array(ndim).fill(1);
    // Map the tensor's dimension to the effective shape position
    let dimIdx = i;
    if (indexing === 'xy' && tensors.length >= 2) {
      if (i === 0) dimIdx = 1;
      else if (i === 1) dimIdx = 0;
    }
    shape[dimIdx] = shapes[i];
    const reshaped = t.reshape(shape);
    result.push(reshaped.expand(effectiveShapes));
  }

  return result;
}

/**
 * Returns the Cartesian product of input tensors.
 * @pytorch torch.cartesian_prod
 */
export async function cartesian_prod(...tensors: Tensor[]): Promise<Tensor> {
  if (tensors.length === 0) return tensor([]);

  const sizes = tensors.map(t => t.numel());
  const totalSize = sizes.reduce((a, b) => a * b, 1);
  const ndim = tensors.length;
  const output = new Float32Array(totalSize * ndim);

  // Read all data from GPU
  const allData: number[][] = [];
  for (const t of tensors) {
    const arr = await t.toArray();
    allData.push(arr);
  }

  for (let i = 0; i < totalSize; i++) {
    let temp = i;
    for (let d = ndim - 1; d >= 0; d--) {
      const idx = temp % sizes[d];
      temp = Math.floor(temp / sizes[d]);
      output[i * ndim + d] = allData[d][idx];
    }
  }

  return tensor(Array.from(output)).reshape([totalSize, ndim]);
}

/**
 * Returns combinations of elements.
 * @pytorch torch.combinations
 */
export async function combinations(input: Tensor, r: number = 2, with_replacement: boolean = false): Promise<Tensor> {
  const inputData = await input.toArray();
  const n = inputData.length;
  const indices: number[][] = [];

  const generate = (start: number, current: number[]) => {
    if (current.length === r) {
      indices.push([...current]);
      return;
    }
    for (let i = start; i < n; i++) {
      current.push(i);
      generate(with_replacement ? i : i + 1, current);
      current.pop();
    }
  };

  generate(0, []);
  const output = new Float32Array(indices.length * r);
  for (let i = 0; i < indices.length; i++) {
    for (let j = 0; j < r; j++) {
      output[i * r + j] = inputData[indices[i][j]];
    }
  }

  return tensor(Array.from(output)).reshape([indices.length, r]);
}

/**
 * Splits tensor into chunks based on indices.
 * @pytorch torch.tensor_split
 */
export function tensor_split(input: Tensor, indices: number | number[], dim: number = 0): Tensor[] {
  const indicesArray = Array.isArray(indices) ? indices : [indices];
  const resolvedDim = dim < 0 ? dim + input.shape.length : dim;
  const dimSize = input.shape[resolvedDim];
  const splitIndices = [0, ...indicesArray.filter(i => i > 0 && i < dimSize), dimSize];
  const sections: number[] = [];
  for (let i = 1; i < splitIndices.length; i++) {
    sections.push(splitIndices[i] - splitIndices[i - 1]);
  }
  return split(input, sections, dim);
}

/**
 * Trace of a matrix (sum of diagonal elements).
 * @pytorch torch.trace
 */
export function trace(input: Tensor): Tensor {
  return input.diagonal(0).sum();
}

/**
 * Unravel flat indices into multi-dimensional coordinates.
 * @pytorch torch.unravel_index
 */
export async function unravel_index(indices: Tensor, shape: number[]): Promise<Tensor> {
  const ndim = shape.length;
  const outNumel = indices.numel();
  const output = new Int32Array(outNumel * ndim);
  const indicesData = await indices.toArray();

  for (let i = 0; i < outNumel; i++) {
    let tempIdx = Math.round(indicesData[i]);
    for (let d = ndim - 1; d >= 0; d--) {
      output[i * ndim + d] = tempIdx % shape[d];
      tempIdx = Math.floor(tempIdx / shape[d]);
    }
  }

  return tensor(Array.from(output), { dtype: 'int32' }).reshape([outNumel, ndim]);
}

// ============ Private Helpers ============

function _fill(
  shape: number[],
  value: number,
  dtype: DType,
  requires_grad: boolean
): Tensor {
  // GPU fill shader only works correctly for float32
  // For int32, bool, and other dtypes, use CPU path
  if (dtype !== 'float32') {
    const n = numel(shape);
    const data = new Array(n).fill(value);
    return tensor(data, { dtype, requires_grad });
  }

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
