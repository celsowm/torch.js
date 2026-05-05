/**
 * Core Tensor class - the fundamental data structure of torch.js.
 * @status partial
 * @pytorch torch.Tensor
 */

import { DType, getDTypeBytes, getTypedArrayConstructor, TypedArray } from '../dtype';
import {
  getDevice,
  createStorageBuffer,
  createBufferWithData,
  readBuffer,
  bufferPool,
  getOrCreatePipeline,
  calculateWorkgroups,
  getCapabilities,
  ELEMENTWISE_SHADER,
  SCALAR_SHADER,
  UNARY_SHADER,
  FILL_SHADER,
  REDUCE_SUM_SHADER,
  REDUCE_MAX_SHADER,
  REDUCE_MIN_SHADER,
  REDUCE_ANY_SHADER,
  REDUCE_ALL_SHADER,
  REDUCE_PROD_SHADER,
  REDUCE_SIMPLE_SUM_SHADER,
  REDUCE_SIMPLE_MAX_SHADER,
  REDUCE_SIMPLE_MIN_SHADER,
  REDUCE_SIMPLE_ANY_SHADER,
  REDUCE_SIMPLE_ALL_SHADER,
  LOG_SOFTMAX_SHADER,
  REDUCE_SIMPLE_PROD_SHADER,
  REDUCE_DIM_SHADER,
  MATMUL_SHADER,
  TRANSPOSE_SHADER,
  ARGMAX_SHADER,
  ARGMIN_SHADER,
  COMPARE_SHADER,
  BROADCAST_SHADER,
  SLICE_SHADER,
  MASKED_FILL_SHADER,
  WHERE_SHADER,
  TRANSPOSE_ND_SHADER,
  REDUCE_BROADCAST_GRAD_SHADER,
  INDEX_SELECT_SHADER,
  EXPAND_SHADER,
  EMBEDDING_SHADER,
  LAYERNORM_SHADER,
  TRIL_SHADER,
  TRIU_SHADER,
  FLIP_SHADER,
  HEAVISIDE_SHADER,
  CUMSUM_SHADER,
  CUMPROD_SHADER,
  DIAG_VEC_TO_MTX_SHADER,
  DIAG_MTX_TO_VEC_SHADER,
  CLAMP_SHADER,
  BufferUsage,
} from '../backend';
import { numel, formatShape, inferShape, validateShape } from '../utils/shape';
import { needsBroadcast, broadcastShapes } from '../utils/broadcast';
import { ones, ones_like, stack, tensor } from '../ops/creation';
import * as ops from '../ops';
import { record_function } from '../profiler';
import { is_grad_enabled } from '../grad_mode';
import type { GradFn, TensorData, SliceSpec } from './types';

// Slice class for advanced indexing
export class Slice {
  start: number | null;
  stop: number | null;
  step: number | null;

  constructor(start: number | null = null, stop: number | null = null, step: number | null = null) {
    this.start = start;
    this.stop = stop;
    this.step = step;
  }

  /**
   * Create a slice for advanced indexing.
   * @param start - Start index (inclusive), negative counts from end
   * @param stop - Stop index (exclusive), negative counts from end
   * @param step - Step size (default: 1)
   * @pytorch slice notation
   */
  static slice(start: number | null, stop: number | null, step: number | null = 1): Slice {
    return new Slice(start, stop, step);
  }

  /**
   * Represents a full slice (:) that selects all elements along a dimension.
   */
  static null: Slice = new Slice(null, null, null);
}

// Type for advanced indexing
export type IndexType = number | Slice | boolean;

/**
 * A multi-dimensional array with GPU acceleration.
 */
export class Tensor {
  private _buffer: GPUBuffer;
  private _shape: readonly number[];
  private _dtype: DType;
  private _device: 'webgpu';
  private _requires_grad: boolean;
  private _grad: Tensor | null = null;
  private _grad_fn: GradFn | null = null;
  private _is_leaf: boolean = true;
  readonly _id: number;
  private static _nextId = 0;

  /** @internal */
  constructor(data: TensorData) {
    this._buffer = data.buffer;
    this._shape = data.shape;
    this._dtype = data.dtype;
    this._device = data.device;
    this._requires_grad = data.requires_grad;
    this._grad_fn = data.grad_fn ?? null;
    this._is_leaf = !data.grad_fn;
    this._id = Tensor._nextId++;
  }

  /**
   * Internal helper: create a Tensor from a plain number array without circular import.
   */
  private static _fromNumberArray(data: number[], dtype?: DType): Tensor {
    const d = dtype ?? 'float32';
    const TypedArrayCtor = getTypedArrayConstructor(d);
    const typedData = new TypedArrayCtor(data) as TypedArray;
    const buffer = createBufferWithData(typedData, d);
    return new Tensor({
      buffer,
      shape: [data.length],
      dtype: d,
      device: 'webgpu',
      requires_grad: false,
    });
  }

  static cat(tensors: Tensor[], dim: number = 0): Tensor {
    if (tensors.length === 0) {
      throw new Error('cat requires at least one tensor');
    }

    if (tensors.length === 1) {
      return tensors[0];
    }

    const first = tensors[0];
    const ndim = first.shape.length;

    if (dim < 0) dim += ndim;

    // Validate shapes
    for (let i = 1; i < tensors.length; i++) {
      if (tensors[i].shape.length !== ndim) {
        throw new Error('All tensors must have the same number of dimensions');
      }
      for (let d = 0; d < ndim; d++) {
        if (d !== dim && tensors[i].shape[d] !== first.shape[d]) {
          throw new Error(`Tensor shapes don't match at dimension ${d}`);
        }
      }
    }

    const outputShape = [...first.shape];
    outputShape[dim] = tensors.reduce((sum, t) => sum + t.shape[dim], 0);
    const totalElements = outputShape.reduce((a, b) => a * b, 1);

    const device = getDevice();
    const bytesPerElement = getDTypeBytes(first.dtype);
    const outputBuffer = createStorageBuffer(totalElements * bytesPerElement);

    const commandEncoder = device.createCommandEncoder();
    
    const outerSize = numel(first.shape.slice(0, dim));
    const innerSize = numel(first.shape.slice(dim + 1));
    const outDimSize = outputShape[dim];
    
    let dimOffset = 0;
    for (const t of tensors) {
        const tDimSize = t.shape[dim];
        const copySize = tDimSize * innerSize * bytesPerElement;
        
        for (let i = 0; i < outerSize; i++) {
            const srcOffset = i * tDimSize * innerSize * bytesPerElement;
            const dstOffset = (i * outDimSize * innerSize + dimOffset * innerSize) * bytesPerElement;
            commandEncoder.copyBufferToBuffer(t.buffer, srcOffset, outputBuffer, dstOffset, copySize);
        }
        dimOffset += tDimSize;
    }

    device.queue.submit([commandEncoder.finish()]);

    return new Tensor({
      buffer: outputBuffer,
      shape: outputShape,
      dtype: first.dtype,
      device: 'webgpu',
      requires_grad: tensors.some(t => t.requires_grad),
    });
  }

  /**
   * Manually release the GPU buffer associated with this tensor.
   * Returns it to the pool for reuse.
   */
  destroy(): void {
    if (this._buffer) {
      bufferPool.release(this._buffer);
      (this as any)._buffer = null;
    }
    if (this._grad) {
      this._grad.destroy();
      this._grad = null;
    }
    this._grad_fn = null;
  }

  async toArray(): Promise<Float32Array | Int32Array | Uint32Array | Int8Array | Uint8Array> {
    return readBuffer(this._buffer, this._dtype, this.numel());
  }

  async item(): Promise<number> {
    const arr = await this.toArray();
    return arr[0];
  }

  /**
   * Returns a tensor with indices sampled from the multinomial distribution.
   * @param num_samples Number of samples to draw
   * @param replacement Whether to sample with replacement
   * @returns Tensor of shape [num_samples] containing indices
   * @pytorch torch.multinomial
   */
  async multinomial(num_samples: number = 1, replacement: boolean = false): Promise<Tensor> {
    const data = await this.toArray();
    const nCategories = data.length;

    if (num_samples > nCategories && !replacement) {
      throw new Error(`Cannot sample ${num_samples} values from ${nCategories} categories without replacement`);
    }

    // Convert to probabilities (handle negative/NaN values)
    const probs = new Float64Array(nCategories);
    let sum = 0;
    for (let i = 0; i < nCategories; i++) {
      probs[i] = Math.max(0, data[i]); // clamp negatives
      sum += probs[i];
    }
    if (sum <= 0) {
      // Uniform distribution
      for (let i = 0; i < nCategories; i++) probs[i] = 1;
      sum = nCategories;
    }
    for (let i = 0; i < nCategories; i++) probs[i] /= sum;

    // Sample
    const samples = new Int32Array(num_samples);
    const available = replacement ? null : new Set<number>();

    for (let s = 0; s < num_samples; s++) {
      let idx: number;
      if (!replacement) {
        // Sample without replacement
        do {
          let r = Math.random();
          idx = 0;
          let cumProb = 0;
          for (let i = 0; i < nCategories; i++) {
            cumProb += probs[i];
            if (r < cumProb) {
              idx = i;
              break;
            }
          }
        } while (available!.has(idx));
        available!.add(idx);
      } else {
        // Sample with replacement
        let r = Math.random();
        let cumProb = 0;
        idx = nCategories - 1;
        for (let i = 0; i < nCategories; i++) {
          cumProb += probs[i];
          if (r < cumProb) {
            idx = i;
            break;
          }
        }
      }
      samples[s] = idx;
    }

    // Return as tensor
    const { tensor } = await import('../ops/creation');
    return tensor(Array.from(samples), { dtype: 'int32' });
  }

  tile(reps: readonly number[]): Tensor {
    return this.repeat(reps);
  }

  repeat(sizes: readonly number[]): Tensor {
    if (sizes.length < this._shape.length) {
      throw new Error(`Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor`);
    }
    
    // Add leading 1s to tensor shape if sizes is longer
    let tensor: Tensor = this;
    if (sizes.length > this._shape.length) {
        const diff = sizes.length - this._shape.length;
        for (let i = 0; i < diff; i++) {
            tensor = tensor.unsqueeze(0);
        }
    }
    
    // Repeat along each dimension
    for (let i = 0; i < sizes.length; i++) {
        const repeatTimes = sizes[i];
        if (repeatTimes > 1) {
            const copies = [];
            for (let k = 0; k < repeatTimes; k++) {
                copies.push(tensor);
            }
            tensor = Tensor.cat(copies, i);
        }
    }
    
    return tensor;
  }

  // ============ Properties ============

  get shape(): readonly number[] { return this._shape; }
  get dtype(): DType { return this._dtype; }
  get device(): 'webgpu' { return this._device; }
  get requires_grad(): boolean { return this._requires_grad; }
  
  get grad(): Tensor | null { return this._grad; }
  set grad(value: Tensor | null) { this._grad = value; }

  get grad_fn(): GradFn | null { return this._grad_fn; }
  get is_leaf(): boolean { return this._is_leaf; }

  /**
   * Transpose property (alias for t()).
   * @status implemented
   */
  get T(): Tensor {
    return this.t();
  }

  /**
   * Change if autograd should record operations on this tensor: 
   * sets the tensor's requires_grad attribute in-place.
   * @pytorch tensor.requires_grad_()
   */
  requires_grad_(value: boolean = true): Tensor {
    this._requires_grad = value;
    return this;
  }

  detach(): Tensor {
    return new Tensor({
      ...this._getState(),
      requires_grad: false,
      grad_fn: undefined,
    });
  }

  /**
   * Copies the elements from src into self tensor and returns self.
   * @pytorch tensor.copy_()
   */
  copy_(src: Tensor): Tensor {
    if (this.numel() !== src.numel()) {
      throw new Error(`copy_: size mismatch, self has ${this.numel()} elements, src has ${src.numel()}`);
    }
    const device = getDevice();
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(src.buffer, 0, this._buffer, 0, this._buffer.size);
    device.queue.submit([commandEncoder.finish()]);
    return this;
  }

  clone(): Tensor {
    const device = getDevice();
    const sizeBytes = this.numel() * getDTypeBytes(this._dtype);
    const alignedSize = Math.ceil(sizeBytes / 4) * 4;

    const newBuffer = device.createBuffer({
      size: alignedSize,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST,
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this._buffer, 0, newBuffer, 0, alignedSize);
    device.queue.submit([commandEncoder.finish()]);

    return new Tensor({
      buffer: newBuffer,
      shape: [...this._shape],
      dtype: this._dtype,
      device: 'webgpu',
      requires_grad: this._requires_grad,
    });
  }

  /** @internal */
  get buffer(): GPUBuffer { return this._buffer; }

  // ============ Basic Math ============

  add(other: Tensor | number): Tensor {
    return record_function('torch.add', () => {
      if (typeof other === 'number') return this._scalarOp('add_scalar', other);
      return this._binaryOp('add', other);
    });
  }

  sub(other: Tensor | number): Tensor {
    return record_function('torch.sub', () => {
      if (typeof other === 'number') return this._scalarOp('sub_scalar', other);
      return this._binaryOp('sub', other);
    });
  }

  mul(other: Tensor | number): Tensor {
    return record_function('torch.mul', () => {
      if (typeof other === 'number') return this._scalarOp('mul_scalar', other);
      return this._binaryOp('mul', other);
    });
  }

  div(other: Tensor | number): Tensor {
    return record_function('torch.div', () => {
      if (typeof other === 'number') return this._scalarOp('div_scalar', other);
      return this._binaryOp('div_op', other);
    });
  }

  // ============ Unary Math ============

  abs(): Tensor { return record_function('torch.abs', () => this._unaryOp('abs_op')); }
  acos(): Tensor { return record_function('torch.acos', () => this._unaryOp('acos_op')); }
  asin(): Tensor { return record_function('torch.asin', () => this._unaryOp('asin_op')); }
  atan(): Tensor { return record_function('torch.atan', () => this._unaryOp('atan_op')); }
  ceil(): Tensor { return record_function('torch.ceil', () => this._unaryOp('ceil_op')); }
  cos(): Tensor { return record_function('torch.cos', () => this._unaryOp('cos_op')); }
  cosh(): Tensor { return record_function('torch.cosh', () => this._unaryOp('cosh_op')); }
  exp(): Tensor { return record_function('torch.exp', () => this._unaryOp('exp_op')); }
  exp2(): Tensor { return record_function('torch.exp2', () => this._unaryOp('exp2_op')); }
  floor(): Tensor { return record_function('torch.floor', () => this._unaryOp('floor_op')); }
  log(base: number | 'e' = Math.E): Tensor {
    return record_function('torch.log', () => {
      if (base === Math.E || base === 'e') return this._unaryOp('log_op');
      return this._unaryOp('log_op').div(Math.log(base));
    });
  }
  log10(): Tensor { return record_function('torch.log10', () => this._unaryOp('log10')); }
  log2(): Tensor { return record_function('torch.log2', () => this._unaryOp('log2_op')); }
  log1p(): Tensor { return record_function('torch.log1p', () => this._unaryOp('log1p')); }
  neg(): Tensor { return record_function('torch.neg', () => this._unaryOp('neg')); }
  round(): Tensor { return record_function('torch.round', () => this._unaryOp('round_op')); }
  sin(): Tensor { return record_function('torch.sin', () => this._unaryOp('sin_op')); }
  sinh(): Tensor { return record_function('torch.sinh', () => this._unaryOp('sinh_op')); }
  acosh(): Tensor { return record_function('torch.acosh', () => this._unaryOp('acosh_op')); }
  asinh(): Tensor { return record_function('torch.asinh', () => this._unaryOp('asinh_op')); }
  atanh(): Tensor { return record_function('torch.atanh', () => this._unaryOp('atanh_op')); }
  sqrt(): Tensor { return record_function('torch.sqrt', () => this._unaryOp('sqrt_op')); }
  tan(): Tensor { return record_function('torch.tan', () => this._unaryOp('tan_op')); }
  tanh(): Tensor { return record_function('torch.tanh', () => this._unaryOp('tanh_op')); }
  trunc(): Tensor { return record_function('torch.trunc', () => this._unaryOp('trunc_op')); }
  frac(): Tensor { return record_function('torch.frac', () => this._unaryOp('frac_op')); }
  reciprocal(): Tensor { return record_function('torch.reciprocal', () => this._unaryOp('reciprocal_op')); }
  rsqrt(): Tensor { return record_function('torch.rsqrt', () => this._unaryOp('rsqrt_op')); }
  square(): Tensor { return record_function('torch.square', () => this._unaryOp('square_op')); }
  sigmoid(): Tensor { return record_function('torch.sigmoid', () => this._unaryOp('sigmoid')); }
  relu(): Tensor { return record_function('torch.relu', () => this._unaryOp('relu')); }
  gelu(): Tensor { return record_function('torch.gelu', () => this._unaryOp('gelu')); }
  softmax(dim: number = -1): Tensor {
    // softmax(x) = log_softmax(x).exp()
    return this._logSoftmax(dim).exp();
  }
  log_softmax(dim: number = -1): Tensor {
    return this._logSoftmax(dim);
  }
  private _logSoftmax(dim: number = -1): Tensor {
    // Use the log_softmax WebGPU shader
    // Currently only supports dim=-1 (last dimension) for ND tensors
    const shape = this._shape;
    const ndim = shape.length;
    if (dim < 0) dim = ndim + dim;
    if (dim !== ndim - 1) {
      throw new Error(`log_softmax currently only supports dim=-1 (last dimension) for ND tensors. Got dim=${dim}, ndim=${ndim}`);
    }

    const device = getDevice();
    const batchSize = shape.slice(0, -1).reduce((a, b) => a * b, 1) || 1;
    const normalizedSize = shape[shape.length - 1];
    const totalSize = this.numel();

    const outputBuffer = createStorageBuffer(totalSize * getDTypeBytes(this._dtype));

    // dims buffer: batchSize, normalizedSize
    const dimsData = new Uint32Array([batchSize, normalizedSize]);
    const dimsBuffer = device.createBuffer({
      size: 8,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(dimsBuffer, 0, dimsData);

    const pipeline = getOrCreatePipeline(LOG_SOFTMAX_SHADER, 'log_softmax');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer } },
        { binding: 1, resource: { buffer: outputBuffer } },
        { binding: 2, resource: { buffer: dimsBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(batchSize, 1, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    return new Tensor({
      buffer: outputBuffer,
      shape: [...shape],
      dtype: this._dtype,
      device: 'webgpu',
      requires_grad: this._requires_grad,
    });
  }
  sign(): Tensor { return record_function('torch.sign', () => this._unaryOp('sign_op')); }
  sgn(): Tensor { return record_function('torch.sgn', () => this._unaryOp('sgn_op')); }
  bool(): Tensor {
    // Convert to bool: non-zero values become true, zero becomes false
    // Use comparison with zero
    return this.ne(this.zeros_like());
  }
  erf(): Tensor { return record_function('torch.erf', () => this._unaryOp('erf_op')); }
  erfc(): Tensor { return record_function('torch.erfc', () => this._unaryOp('erfc_op')); }
  expm1(): Tensor { return record_function('torch.expm1', () => this._unaryOp('expm1_op')); }
  deg2rad(): Tensor { return record_function('torch.deg2rad', () => this._unaryOp('deg2rad_op')); }
  rad2deg(): Tensor { return record_function('torch.rad2deg', () => this._unaryOp('rad2deg_op')); }
  logical_not(): Tensor { return record_function('torch.logical_not', () => this._unaryOp('logical_not_op')); }
  i0(): Tensor { return record_function('torch.i0', () => this._unaryOp('i0_op')); }
  lgamma(): Tensor { return record_function('torch.lgamma', () => this._unaryOp('lgamma_op')); }
  digamma(): Tensor { return record_function('torch.digamma', () => this._unaryOp('digamma_op')); }

  softplus(): Tensor { return record_function('torch.softplus', () => this._unaryOp('softplus_op')); }
  silu(): Tensor { return record_function('torch.silu', () => this._unaryOp('silu_op')); }
  mish(): Tensor { return record_function('torch.mish', () => this._unaryOp('mish_op')); }
  hardsigmoid(): Tensor { return record_function('torch.hardsigmoid', () => this._unaryOp('hardsigmoid_op')); }
  hardswish(): Tensor { return record_function('torch.hardswish', () => this._unaryOp('hardswish_op')); }
  softsign(): Tensor { return record_function('torch.softsign', () => this._unaryOp('softsign_op')); }
  tanhshrink(): Tensor { return record_function('torch.tanhshrink', () => this._unaryOp('tanhshrink_op')); }

  leaky_relu(negative_slope: number = 0.01): Tensor {
    return record_function('torch.leaky_relu', () => {
      const pos = this.relu();
      const neg = this.neg().relu().neg();
      return pos.add(neg.mul(negative_slope));
    });
  }

  elu(alpha: number = 1.0): Tensor {
    return record_function('torch.elu', () => {
      const neg_part = this.neg().relu().neg();
      return this.relu().add(neg_part.exp().sub(1).mul(alpha));
    });
  }

  selu(): Tensor {
    return record_function('torch.selu', () => {
      return this.elu(1.6732632423543772848170429916717).mul(1.0507009873554804934193349852946);
    });
  }

  threshold(threshold: number, value: number): Tensor {
    return record_function('torch.threshold', () => {
      // x > threshold ? x : value
      // TODO: implement scalar comparison properly
      // For now, assume this.gt(threshold) works via broadcast if we make scalar tensor
      // But we don't have scalar tensor creation in Tensor easily.
      // We will fallback to cpu readback for threshold if needed or implement scalar comparison shader.
      // Actually, we can use clamp? No.
      // We can use relu logic: threshold(x, t, v) = x if x > t else v
      // = (x - t > 0) ? x : v
      // = relu(x - t) > 0 ? x : v ??
      
      // Let's implement a specific shader for threshold later.
      // For now, throw not implemented or use a slow method.
      throw new Error("threshold not implemented");
    });
  }

  zeros_like(): Tensor {
    return this.mul(0);
  }

  ones_like(): Tensor {
    return this.mul(0).add(1);
  }

  heaviside(values: Tensor): Tensor {
    return record_function('torch.heaviside', () => {
      const v_expanded = values.expand([...this._shape]);
      const device = getDevice();
      const outputBuffer = createStorageBuffer(this.numel() * 4);
      
      const pipeline = getOrCreatePipeline(HEAVISIDE_SHADER, 'main');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: v_expanded.buffer, offset: 0, size: v_expanded.buffer.size } },
          { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        ],
      });
      
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      
      const res = new Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: 'webgpu', requires_grad: false });
      v_expanded.destroy();
      return res;
    });
  }

  glu(dim: number = -1): Tensor {
    return record_function('torch.glu', () => {
      const [a, b] = this.chunk(2, dim);
      return a.mul(b.sigmoid());
    });
  }

  pow(exponent: number | Tensor): Tensor {
    return record_function('torch.pow', () => {
      if (typeof exponent === 'number') return this._scalarOp('pow_scalar', exponent);
      return this._binaryOp('pow_tensor', exponent);
    });
  }

  // ============ Advanced Pointwise ============

  atan2(other: Tensor): Tensor { return record_function('torch.atan2', () => this._binaryOp('atan2_op', other)); }
  hypot(other: Tensor): Tensor { return record_function('torch.hypot', () => this._binaryOp('hypot_op', other)); }
  logaddexp(other: Tensor): Tensor { return record_function('torch.logaddexp', () => this._binaryOp('logaddexp', other)); }
  bitwise_and(other: Tensor): Tensor { return record_function('torch.bitwise_and', () => this._binaryOp('bitwise_and', other)); }
  bitwise_or(other: Tensor): Tensor { return record_function('torch.bitwise_or', () => this._binaryOp('bitwise_or', other)); }
  bitwise_xor(other: Tensor): Tensor { return record_function('torch.bitwise_xor', () => this._binaryOp('bitwise_xor', other)); }

   clamp(min: number = -Infinity, max: number = Infinity): Tensor {
     return record_function('torch.clamp', () => {
       const device = getDevice();
       const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
       const paramsData = new ArrayBuffer(16);
       new Float32Array(paramsData, 0, 2).set([min, max]);
       new Uint32Array(paramsData, 8, 1)[0] = this.numel();
       const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
       device.queue.writeBuffer(paramsBuffer, 0, paramsData);
       const pipeline = getOrCreatePipeline(CLAMP_SHADER, 'main');
       const bindGroup = device.createBindGroup({
         layout: pipeline.getBindGroupLayout(0),
         entries: [
           { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
           { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
           { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
         ],
       });
       const commandEncoder = device.createCommandEncoder();
       const passEncoder = commandEncoder.beginComputePass();
       passEncoder.setPipeline(pipeline);
       passEncoder.setBindGroup(0, bindGroup);
       passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
       passEncoder.end();
       device.queue.submit([commandEncoder.finish()]);
       return new Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
     });
   }

   clamp_min(min: number): Tensor {
     return this.clamp(min, Infinity);
   }

   clamp_max(max: number): Tensor {
     return this.clamp(-Infinity, max);
   }

   fix(): Tensor {
     return this.trunc();
   }

   fmod(other: Tensor | number): Tensor {
     return record_function('torch.fmod', () => {
       const div = typeof other === 'number' ? this.div(other) : this.div(other);
       return this.sub(div.trunc().mul(other));
     });
   }

   remainder(other: Tensor | number): Tensor {
     return record_function('torch.remainder', () => {
       const div = typeof other === 'number' ? this.div(other) : this.div(other);
       return this.sub(div.floor().mul(other));
     });
   }

   erfinv(): Tensor {
     // Approximation: erfinv(x) ≈ sign(x) * sqrt(sqrt((2/(π*a) + ln(1-x^2)/2)^2 - ln(1-x^2)/a) - (2/(π*a) + ln(1-x^2)/2))
     // where a = 0.147
     return record_function('torch.erfinv', () => {
       const a = 0.147;
       const one = ops.ones_like(this);
       const x2 = this.mul(this);
       const ln1mx2 = one.sub(x2).log();
       const term1 = ops.tensor([2 / (Math.PI * a)], { dtype: this.dtype }).add(ln1mx2.mul(0.5));
       const inside = term1.mul(term1).sub(ln1mx2.mul(1 / a));
       const result = inside.clamp(0, Infinity).sqrt().sub(term1).sqrt();
       return this.sign().mul(result);
     });
   }

   erfcinv(): Tensor {
     // erfcinv(x) = erfinv(1 - x)
     return record_function('torch.erfcinv', () => {
       const one = ops.ones_like(this);
       return this.neg().add(1).erfinv();
     });
   }

  masked_fill(mask: Tensor, value: number): Tensor {
    return record_function('torch.masked_fill', () => {
      const device = getDevice();
      const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
      const paramsData = new ArrayBuffer(16);
      new Float32Array(paramsData, 0, 1)[0] = value;
      new Uint32Array(paramsData, 4, 1)[0] = this.numel();
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(MASKED_FILL_SHADER, 'main');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: mask.buffer, offset: 0, size: mask.buffer.size } },
          { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        ],
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      return new Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
    });
  }

  triu(diagonal: number = 0): Tensor {
    if (this._shape.length !== 2) throw new Error('triu currently only supports 2D tensors');
    const [rows, cols] = this._shape;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * 4);
    const paramsData = new Uint32Array([rows, cols, diagonal, 0]);
    const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(TRIU_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new Tensor({ buffer: outputBuffer, shape: [rows, cols], dtype: this._dtype, device: 'webgpu', requires_grad: false });
  }

  tril(diagonal: number = 0): Tensor {
    if (this._shape.length !== 2) throw new Error('tril currently only supports 2D tensors');
    const [rows, cols] = this._shape;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));

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
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    return new Tensor({
      buffer: outputBuffer,
      shape: [rows, cols],
      dtype: this._dtype,
      device: 'webgpu',
      requires_grad: false,
    });
  }

  to(dtypeOrDevice?: DType | 'cpu' | 'webgpu'): Tensor {
    // Handle: to(dtype), to(device), to(dtype, device), to(device, dtype)
    let dtype: DType | undefined;
    let device: string | undefined;

    if (typeof dtypeOrDevice === 'string') {
      if (['float32', 'int32', 'bool', 'int64', 'float64', 'float16', 'int8', 'int16', 'uint8'].includes(dtypeOrDevice)) {
        dtype = dtypeOrDevice as DType;
      } else {
        device = dtypeOrDevice;
      }
    }

    // Dtype conversion
    if (dtype) {
      if (this._dtype === dtype) return this;
      if (this._dtype === 'float32' && dtype === 'float32') return this;
      if (getDTypeBytes(this._dtype) === getDTypeBytes(dtype)) {
        const res = this.clone();
        (res as any)._dtype = dtype;
        return res;
      }
      throw new Error(`Tensor.to: conversion from ${this._dtype} to ${dtype} not yet implemented`);
    }

    // Device: webgpu only for now, but return self for compatibility
    if (device === 'webgpu' || device === 'cpu') {
      return this; // torch.js only runs on webgpu
    }

    // No args = clone
    return this.clone();
  }

  flip(dims: number[]): Tensor {
    let current = this as Tensor;
    for (let dim of dims) {
      if (dim < 0) dim += current._shape.length;
      const dimSize = current._shape[dim];
      const stride = (current.stride() as number[])[dim];
      const device = getDevice();
      const outputBuffer = createStorageBuffer(current.numel() * 4);
      const paramsData = new Uint32Array([0, dimSize, stride, 0]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(FLIP_SHADER, 'main');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: current._buffer, offset: 0, size: current._buffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        ],
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(current.numel()));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      const next = new Tensor({ buffer: outputBuffer, shape: [...current._shape], dtype: current._dtype, device: 'webgpu', requires_grad: false });
      if (current !== this) current.destroy();
      current = next;
    }
    return current;
  }

  cumsum(dim: number): Tensor {
    return this._cumOp(dim, CUMSUM_SHADER);
  }

  cumprod(dim: number): Tensor {
    return this._cumOp(dim, CUMPROD_SHADER);
  }

  diag(diagonal: number = 0): Tensor {
    const device = getDevice();
    if (this._shape.length === 1) {
      // Vector to Matrix
      const n = this._shape[0];
      const m = n + Math.abs(diagonal);
      const res = this._zeros([m, m]);
      const outputBuffer = res.buffer;

      const paramsData = new Int32Array([n, diagonal, 0, 0]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(DIAG_VEC_TO_MTX_SHADER, 'main');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        ],
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(n));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      return res;
    } else if (this._shape.length === 2) {
      // Matrix to Vector
      const [rows, cols] = this._shape;
      let diagLen: number;
      if (diagonal >= 0) {
        diagLen = Math.max(0, Math.min(rows, cols - diagonal));
      } else {
        diagLen = Math.max(0, Math.min(rows + diagonal, cols));
      }
      const outputBuffer = createStorageBuffer(diagLen * 4);
      const paramsData = new Int32Array([rows, cols, diagonal, diagLen]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(DIAG_MTX_TO_VEC_SHADER, 'main');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        ],
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(diagLen));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      return new Tensor({ buffer: outputBuffer, shape: [diagLen], dtype: this._dtype, device: 'webgpu', requires_grad: false });
    }
    throw new Error('diag expects 1D or 2D tensor');
  }

  /**
   * Returns a partial view of the diagonal elements.
   * For a 2D tensor, returns the diagonal.
   * For higher dimensions, uses dim1 and dim2 to determine the 2D planes.
   * @param offset - Diagonal offset (positive = above main diagonal, negative = below)
   * @param dim1 - First dimension of the 2D planes
   * @param dim2 - Second dimension of the 2D planes
   * @pytorch torch.diagonal
   */
  diagonal(offset: number = 0, dim1: number = -2, dim2: number = -1): Tensor {
    if (this._shape.length < 2) {
      throw new Error('diagonal requires at least 2D tensor');
    }

    const ndim = this._shape.length;
    const resolvedDim1 = dim1 < 0 ? dim1 + ndim : dim1;
    const resolvedDim2 = dim2 < 0 ? dim2 + ndim : dim2;

    if (resolvedDim1 === resolvedDim2) {
      throw new Error('diagonal: dim1 and dim2 must be different');
    }

    const dim1Size = this._shape[resolvedDim1];
    const dim2Size = this._shape[resolvedDim2];

    // Calculate diagonal length
    let diagLen: number;
    if (offset >= 0) {
      diagLen = Math.max(0, Math.min(dim1Size, dim2Size - offset));
    } else {
      diagLen = Math.max(0, Math.min(dim1Size + offset, dim2Size));
    }

    if (diagLen === 0) {
      const outShape = this._shape.filter((_, i) => i !== resolvedDim1 && i !== resolvedDim2);
      outShape.push(0);
      return Tensor._fromNumberArray([], this._dtype).reshape(outShape);
    }

    // For 2D tensor, extract diagonal
    if (this._shape.length === 2) {
      const device = getDevice();
      const outputBuffer = createStorageBuffer(diagLen * getDTypeBytes(this._dtype));

      const pipeline = getOrCreatePipeline(DIAG_MTX_TO_VEC_SHADER, 'main');
      const paramsData = new Int32Array([dim1Size, dim2Size, offset, diagLen]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        ],
      });

      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(diagLen));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);

      return new Tensor({ buffer: outputBuffer, shape: [diagLen], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
    }

    // For ND tensor: we need to iterate over all other dimensions and extract diagonal from each 2D plane
    // This is complex - for now, use CPU fallback
    return this._diagonalND(offset, resolvedDim1, resolvedDim2, diagLen);
  }

  /**
   * Internal: diagonal for ND tensors via CPU readback.
   */
  private _diagonalND(offset: number, dim1: number, dim2: number, diagLen: number): Tensor {
    // CPU fallback implementation
    // This is complex to do efficiently on GPU, so we use CPU for now
    throw new Error('diagonal with ND tensors not yet fully implemented');
  }

  private _cumOp(dim: number, shader: string): Tensor {
    if (dim < 0) dim += this._shape.length;
    const permutation = Array.from({ length: this._shape.length }, (_, i) => i);
    permutation.splice(dim, 1);
    permutation.push(dim);
    const transposed = this.permute(permutation);
    const reduceSize = this._shape[dim];
    const batchSize = transposed.numel() / reduceSize;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(transposed.numel() * 4);
    const paramsData = new Uint32Array([batchSize, reduceSize, 0, 0]);
    const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(shader, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: transposed.buffer, offset: 0, size: transposed.buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(batchSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const resTransposed = new Tensor({ buffer: outputBuffer, shape: transposed.shape, dtype: this._dtype, device: 'webgpu', requires_grad: false });
    const invPerm = new Array(this._shape.length);
    for (let i = 0; i < permutation.length; i++) invPerm[permutation[i]] = i;
    const final = resTransposed.permute(invPerm);
    transposed.destroy();
    resTransposed.destroy();
    return final;
  }

  private _zeros(shape: number[]): Tensor {
    const n = numel(shape);
    const device = getDevice();
    const outputBuffer = createStorageBuffer(n * 4);
    const paramsData = new ArrayBuffer(8);
    new Float32Array(paramsData, 0, 1)[0] = 0;
    new Uint32Array(paramsData, 4, 1)[0] = n;
    const paramsBuffer = device.createBuffer({ size: 8, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
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
    return new Tensor({ buffer: outputBuffer, shape, dtype: 'float32', device: 'webgpu', requires_grad: false });
  }

  _sliceDimStep(dim: number, start: number, end: number, step: number = 1): Tensor {
    if (dim < 0) dim += this._shape.length;
    const dimSize = this._shape[dim];
    
    if (start < 0) start += dimSize;
    if (end < 0) end += dimSize;
    if (start < 0) start = 0;
    if (end > dimSize) end = dimSize;
    
    if (step === 1) {
        return this.narrow(dim, start, Math.max(0, end - start));
    }
    
    const count = Math.ceil(Math.max(0, end - start) / step);
    const indices = new Int32Array(count);
    for (let i = 0; i < count; i++) {
        indices[i] = start + i * step;
    }
    
    const device = getDevice();
    const indexBuf = createStorageBuffer(count * 4);
    device.queue.writeBuffer(indexBuf, 0, indices);
    const idxTensor = new Tensor({
        buffer: indexBuf,
        shape: [count],
        dtype: 'int32',
        device: 'webgpu',
        requires_grad: false
    });
    
    const res = this.index_select(dim, idxTensor);
    idxTensor.destroy();
    return res;
  }

  trapezoid(dx: number = 1.0, dim: number = -1): Tensor {
    if (dim < 0) dim += this.shape.length;
    const n = this.shape[dim];
    const left = this._sliceDimStep(dim, 0, n - 1);
    const right = this._sliceDimStep(dim, 1, n);
    return left.add(right).mul(dx * 0.5).sum(dim);
  }
  
  cumulative_trapezoid(dx: number = 1.0, dim: number = -1): Tensor {
    if (dim < 0) dim += this.shape.length;
    const n = this.shape[dim];
    const left = this._sliceDimStep(dim, 0, n - 1);
    const right = this._sliceDimStep(dim, 1, n);
    return left.add(right).mul(dx * 0.5).cumsum(dim);
  }

  index_select(dim: number, index: Tensor): Tensor {
    if (dim < 0) dim += this._shape.length;
    if (index.dim() !== 1) throw new Error('index_select: index must be 1D');
    
    // For now, only supporting 1D and 2D index_select via shader
    if (this._shape.length > 2) throw new Error('index_select currently supports up to 2D tensors');
    
    const device = getDevice();
    const inputDim0 = this._shape.length === 1 ? this._shape[0] : this._shape[0];
    const inputDim1 = this._shape.length === 1 ? 1 : this._shape[1];
    
    const outputShape = [...this._shape];
    outputShape[dim] = index.shape[0];
    
    const outputSize = numel(outputShape);
    const outputBuffer = createStorageBuffer(outputSize * getDTypeBytes(this._dtype));
    
    const paramsData = new Uint32Array([dim, inputDim0, inputDim1, index.shape[0]]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    
    const shaderName = this._shape.length === 1 ? 'index_select_1d' : 'index_select_2d';
    const pipeline = getOrCreatePipeline(INDEX_SELECT_SHADER, shaderName);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: index.buffer, offset: 0, size: index.buffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });
    
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    
    return new Tensor({ buffer: outputBuffer, shape: outputShape, dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
  }

  /**
   * Select elements from the tensor where mask is True.
   * Returns a 1-D tensor with all selected elements.
   * @param mask - Boolean tensor (same shape or broadcastable)
   * @pytorch torch.masked_select
   */
  async masked_select(mask: Tensor): Promise<Tensor> {
    // CPU fallback: read mask, find indices, gather values
    // This is correct but slow for large tensors
    return this._maskedSelectCPU(mask);
  }

  /**
   * Internal: masked_select via CPU readback.
   */
  private async _maskedSelectCPU(mask: Tensor): Promise<Tensor> {
    const maskData = await mask.toArray();
    const data = await this.toArray();

    const selectedValues: number[] = [];
    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i] !== 0) {
        selectedValues.push(data[i]);
      }
    }

    return Tensor._fromNumberArray(selectedValues, this._dtype);
  }

  /**
   * Returns the indices of elements that are non-zero.
   * Returns a 2-D tensor where each row contains the indices of a non-zero element.
   * @pytorch torch.nonzero
   */
  async nonzero(): Promise<Tensor> {
    const data = await this.toArray();
    const shape = [...this._shape];
    const ndim = shape.length;

    const indices: number[] = [];

    for (let i = 0; i < data.length; i++) {
      if (data[i] !== 0) {
        // Convert flat index to multi-dimensional coordinates
        const coords: number[] = [];
        let tempIdx = i;
        for (let d = ndim - 1; d >= 0; d--) {
          coords.unshift(tempIdx % shape[d]);
          tempIdx = Math.floor(tempIdx / shape[d]);
        }
        indices.push(...coords);
      }
    }

    const numRows = indices.length / ndim;
    return Tensor._fromNumberArray(Array.from(new Int32Array(indices)), 'int32').reshape([numRows, ndim]);
  }

  /**
   * Returns a 1-D tensor with values from the last dimension where input is non-zero.
   * Alias for elements that are not equal to zero.
   * @pytorch torch.nonzero (as_tuple=True variant returns indices)
   */
/**
     * Returns the indices of non-zero elements.
     * @pytorch torch.nonzero (as indices tensor)
     */
   async nonzeroIndices(): Promise<Tensor> {
     const data = await this.toArray();
     const indices: number[] = [];
     for (let i = 0; i < data.length; i++) {
       if (data[i] !== 0) {
         indices.push(i);
       }
     }
     return Tensor._fromNumberArray(indices, this._dtype).reshape([indices.length, 1]);
   }

  select(dim: number, index: number): Tensor {
    return this.narrow(dim, index, 1).squeeze(dim);
  }

  take(indices: Tensor): Tensor {
    return this.flatten().index_select(0, indices);
  }

  where(condition: Tensor, other: Tensor): Tensor {
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    
    const pipeline = getOrCreatePipeline(WHERE_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: condition.buffer, offset: 0, size: condition.buffer.size } },
        { binding: 1, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 2, resource: { buffer: other.buffer, offset: 0, size: other.buffer.size } },
        { binding: 3, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      ],
    });
    
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    
    return new Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad || other.requires_grad });
  }

  // ============ Comparison ============

  eq(other: Tensor | number): Tensor { 
    if (typeof other === 'number') return this._scalarOp('eq_scalar', other);
    return this._compareOp('eq', other); 
  }
  ne(other: Tensor | number): Tensor { 
    if (typeof other === 'number') return this._scalarOp('ne_scalar', other);
    return this._compareOp('ne', other); 
  }
  lt(other: Tensor | number): Tensor { 
    if (typeof other === 'number') return this._scalarOp('lt_scalar', other);
    return this._compareOp('lt', other); 
  }
  gt(other: Tensor | number): Tensor { 
    if (typeof other === 'number') return this._scalarOp('gt_scalar', other);
    return this._compareOp('gt', other); 
  }
  ge(other: Tensor | number): Tensor { 
    if (typeof other === 'number') return this._scalarOp('ge_scalar', other);
    return this._compareOp('ge', other); 
  }
  le(other: Tensor | number): Tensor { 
    if (typeof other === 'number') return this._scalarOp('le_scalar', other);
    return this._compareOp('le', other); 
  }
  greater(other: Tensor | number): Tensor { return this.gt(other); }
  greater_equal(other: Tensor | number): Tensor { return this.ge(other); }
  less(other: Tensor | number): Tensor { return this.lt(other); }
  less_equal(other: Tensor | number): Tensor { return this.le(other); }
  not_equal(other: Tensor | number): Tensor { return this.ne(other); }

  isnan(): Tensor { return this._unaryOp('isnan_op'); }
  isinf(): Tensor { return this._unaryOp('isinf_op'); }
  isfinite(): Tensor { return this._unaryOp('isfinite_op'); }
  isposinf(): Tensor { return this._unaryOp('isposinf_op'); }
  isneginf(): Tensor { return this._unaryOp('isneginf_op'); }

  maximum(other: Tensor): Tensor { return this._compareOp('maximum_op', other); }
  minimum(other: Tensor): Tensor { return this._compareOp('minimum_op', other); }
  fmax(other: Tensor): Tensor { return this._compareOp('fmax_op', other); }
  fmin(other: Tensor): Tensor { return this._compareOp('fmin_op', other); }

  async equal(other: Tensor): Promise<boolean> {
    if (this._shape.length !== other._shape.length) return false;
    for (let i = 0; i < this._shape.length; i++) {
      if (this._shape[i] !== other._shape[i]) return false;
    }
    // PyTorch equal considers NaNs equal at the same position
    const mask = this.isclose(other, 0, 0, true);
    const diff = mask.all();
    const result = await diff.item();
    mask.destroy();
    diff.destroy();
    return result === 1.0;
  }

  isclose(other: Tensor, rtol: number = 1e-5, atol: number = 1e-8, equal_nan: boolean = false): Tensor {
    // PyTorch formula: |self - other| <= atol + rtol * |other|
    // This only applies to finite values.
    const self_finite = this.isfinite();
    const other_finite = other.isfinite();
    const both_finite = self_finite.mul(other_finite);

    const diff = this.sub(other).abs();
    const tol = other.abs().mul(rtol).add(atol);
    const close_finite = diff.le(tol).mul(both_finite);
    
    // Non-finite values are close only if they are equal
    const both_inf = this.isinf().mul(other.isinf());
    const close_inf = both_inf.mul(this.eq(other));
    
    let res = close_finite.add(close_inf).clamp(0, 1);

    if (equal_nan) {
      const both_nan = this.isnan().mul(other.isnan());
      res = res.add(both_nan).clamp(0, 1);
    }
    return res;
  }

  async allclose(other: Tensor, rtol: number = 1e-5, atol: number = 1e-8, equal_nan: boolean = false): Promise<boolean> {
    const res = this.isclose(other, rtol, atol, equal_nan).all();
    const result = await res.item();
    res.destroy();
    return result === 1.0;
  }

  // ============ Shape Operations ============

  numel(): number { return numel(this._shape); }
  dim(): number { return this._shape.length; }
  size(dim?: number): number | readonly number[] {
    if (dim === undefined) return this._shape;
    if (dim < 0) dim += this._shape.length;
    return this._shape[dim];
  }

  stride(dim?: number): number | number[] {
    const ndim = this._shape.length;
    const strides = new Array(ndim);
    let s = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides[i] = s;
      s *= this._shape[i];
    }
    if (dim === undefined) return strides;
    if (dim < 0) dim += ndim;
    return strides[dim];
  }

  broadcast_to(shape: readonly number[]): Tensor {
    return this.expand(shape);
  }

  expand(shape: readonly number[]): Tensor {
    const ndim = shape.length;
    const inDim = this._shape.length;
    if (ndim < inDim) throw new Error('expand: rank cannot be reduced');

    const paddedIn = new Array(ndim).fill(1);
    for (let i = 0; i < inDim; i++) paddedIn[ndim - inDim + i] = this._shape[i];
    
    const currentStrides = this.stride() as number[];
    const inStrides = new Array(ndim).fill(0);
    for (let i = 0; i < inDim; i++) inStrides[ndim - inDim + i] = currentStrides[i];

    const finalShape = shape.map((s, i) => s === -1 ? paddedIn[i] : s);
    for (let i = 0; i < ndim; i++) {
      if (finalShape[i] !== paddedIn[i] && paddedIn[i] !== 1) {
        throw new Error(`expand: incompatible shapes: ${this._shape} to ${finalShape}`);
      }
    }

    const outputSize = numel(finalShape);
    const device = getDevice();
    const outputBuffer = createStorageBuffer(outputSize * 4);

    const pShape = [1, 1, 1, 1];
    const pStrides = [0, 0, 0, 0];
    for (let i = 0; i < Math.min(ndim, 4); i++) {
      const finalDimIdx = ndim - 1 - i;
      const pIdx = 3 - i;
      pShape[pIdx] = finalShape[finalDimIdx];
      if (i < inDim) {
        const inDimIdx = inDim - 1 - i;
        pStrides[pIdx] = this._shape[inDimIdx] === 1 ? 0 : currentStrides[inDimIdx];
      } else {
        pStrides[pIdx] = 0;
      }
    }

    const paramsData = new Uint32Array([...pShape, ...pStrides]);
    const paramsBuffer = device.createBuffer({ size: 32, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    
    const pipeline = getOrCreatePipeline(EXPAND_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });
    
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    
    return new Tensor({ buffer: outputBuffer, shape: finalShape, dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
  }

  reshape(shape: readonly number[]): Tensor {
    const newShape = inferShape(shape, this.numel());
    validateShape(newShape);

    const self = this;
    let grad_fn: GradFn | undefined;
    if (this._requires_grad) {
      const oldShape = [...this._shape];
      grad_fn = {
        backward(gradOutput: Tensor): void {
          self.accumulateGrad(gradOutput.reshape(oldShape));
        },
        _next_tensors: [self],
      };
    }

    return new Tensor({
      buffer: this._buffer,
      shape: newShape,
      dtype: this._dtype,
      device: this._device,
      requires_grad: this._requires_grad,
      grad_fn,
    });
  }

  view(...args: any[]): Tensor {
    const shape = (args.length === 1 && Array.isArray(args[0])) ? args[0] : args;
    return this.reshape(shape);
  }

  squeeze(dim?: number): Tensor {
    let newShape: number[];
    if (dim === undefined) {
      newShape = this._shape.filter(d => d !== 1);
    } else {
      if (dim < 0) dim += this._shape.length;
      if (this._shape[dim] !== 1) return this;
      newShape = [...this._shape];
      newShape.splice(dim, 1);
    }
    return this.reshape(newShape);
  }

  unsqueeze(dim: number): Tensor {
    if (dim < 0) dim += this._shape.length + 1;
    const newShape = [...this._shape];
    newShape.splice(dim, 0, 1);
    return this.reshape(newShape);
  }

  flatten(startDim: number = 0, endDim: number = -1): Tensor {
    if (startDim < 0) startDim += this._shape.length;
    if (endDim < 0) endDim += this._shape.length;

    const before = this._shape.slice(0, startDim);
    const middle = this._shape.slice(startDim, endDim + 1);
    const after = this._shape.slice(endDim + 1);

    const flattenedDim = middle.reduce((a, b) => a * b, 1);
    const newShape = [...before, flattenedDim, ...after];

    return this.reshape(newShape);
  }

  /**
   * Returns a contiguous tensor (for now, just returns self as we assume contiguous).
   * @status implemented
   * @pytorch tensor.contiguous()
   */
  contiguous(): Tensor {
    // For now, all our tensors are contiguous
    return this;
  }

  /**
   * Splits tensor into chunks along a dimension.
   * @status implemented
   * @pytorch tensor.split()
   */
  split(split_size: number, dim: number = 0): Tensor[] {
    if (dim < 0) dim += this._shape.length;

    const dimSize = this._shape[dim];
    const numChunks = Math.ceil(dimSize / split_size);
    const results: Tensor[] = [];

    for (let i = 0; i < numChunks; i++) {
      const start = i * split_size;
      const end = Math.min(start + split_size, dimSize);

      // Create slice indices for each dimension
      const slices: Array<{ start: number; stop: number }> = [];
      for (let d = 0; d < this._shape.length; d++) {
        if (d === dim) {
          slices.push({ start, stop: end });
        } else {
          slices.push({ start: 0, stop: this._shape[d] });
        }
      }

      // Use narrow/slice for this chunk
      results.push(this._sliceMultiDim(slices));
    }

    return results;
  }

  chunk(chunks: number, dim: number = 0): Tensor[] {
    if (dim < 0) dim += this._shape.length;
    const split_size = Math.ceil(this._shape[dim] / chunks);
    return this.split(split_size, dim);
  }

  movedim(source: number | number[], destination: number | number[]): Tensor {
    const src = Array.isArray(source) ? source : [source];
    const dst = Array.isArray(destination) ? destination : [destination];
    
    if (src.length !== dst.length) {
      throw new Error(`movedim: source and destination length mismatch: ${src.length} vs ${dst.length}`);
    }
    
    const ndim = this._shape.length;
    const order = Array.from({ length: ndim }, (_, i) => i);
    const srcNormalized = src.map(d => d < 0 ? d + ndim : d);
    const dstNormalized = dst.map(d => d < 0 ? d + ndim : d);
    
    // Remove sources
    const remaining = order.filter(d => !srcNormalized.includes(d));
    
    // Insert at destinations
    // We need to sort destinations to insert correctly
    const updates = srcNormalized.map((s, i) => ({ s, d: dstNormalized[i] }));
    updates.sort((a, b) => a.d - b.d);
    
    for (const update of updates) {
      remaining.splice(update.d, 0, update.s);
    }
    
    return this.permute(remaining);
  }

  moveaxis(source: number | number[], destination: number | number[]): Tensor {
    return this.movedim(source, destination);
  }

  swapaxes(dim0: number, dim1: number): Tensor {
    return this.transpose(dim0, dim1);
  }

  swapdims(dim0: number, dim1: number): Tensor {
    return this.transpose(dim0, dim1);
  }

  unbind(dim: number = 0): Tensor[] {
    if (dim < 0) dim += this._shape.length;
    const size = this._shape[dim];
    const results: Tensor[] = [];
    for (let i = 0; i < size; i++) {
      results.push(this.narrow(dim, i, 1).squeeze(dim));
    }
    return results;
  }

  narrow(dim: number, start: number, length: number): Tensor {
    if (dim < 0) dim += this._shape.length;
    const slices: SliceSpec[] = new Array(this._shape.length).fill({ start: 0, step: 1 });
    // Fill full slices
    for (let i = 0; i < this._shape.length; i++) {
        slices[i] = { start: 0, stop: this._shape[i], step: 1 };
    }
    // Update specific dim
    slices[dim] = { start, stop: start + length, step: 1 };
    return this.slice(slices);
  }

  /**
   * Slice the tensor.
   * @param slices - Array of slice specifications or indices for each dimension.
   * @status implemented
   */
  slice(slices: (SliceSpec | number)[]): Tensor {
    // Validate number of slices
    if (slices.length > this._shape.length) {
      throw new Error(`Too many slices provided: ${slices.length} > ${this._shape.length}`);
    }

    // Normalize slices to full specifications
    const normSlices: SliceSpec[] = [];

    // Fill provided slices
    for (let i = 0; i < slices.length; i++) {
      const s = slices[i];
      if (typeof s === 'number') {
        let idx = s;
        if (idx < 0) idx += this._shape[i];
        normSlices.push({ start: idx, stop: idx + 1, step: 1 });
      } else {
        const spec = { ...s };
        const dimSize = this._shape[i];

        // Default step is 1
        spec.step = spec.step ?? 1;

        // Handle defaults for start/stop based on step
        if (spec.step > 0) {
          spec.start = spec.start ?? 0;
          spec.stop = spec.stop ?? dimSize;
        } else {
          spec.start = spec.start ?? dimSize - 1;
          spec.stop = spec.stop ?? -dimSize - 1; // logical limit
        }

        // Normalize negative indices
        if (spec.start < 0) spec.start += dimSize;
        if (spec.stop < 0 && spec.stop > -dimSize - 1) spec.stop += dimSize;

        normSlices.push(spec);
      }
    }

    // Fill remaining dims with full slice
    for (let i = slices.length; i < this._shape.length; i++) {
      normSlices.push({ start: 0, stop: this._shape[i], step: 1 });
    }

    // Calculate output shape
    const newShape: number[] = [];
    for (let i = 0; i < this._shape.length; i++) {
      const s = normSlices[i];
      let len = 0;
      if (s.step! > 0) {
        len = Math.max(0, Math.ceil((s.stop! - s.start!) / s.step!));
      } else {
        len = Math.max(0, Math.ceil((s.start! - s.stop!) / -s.step!));
      }
      newShape.push(len);
    }

    const outputSize = numel(newShape);

    const device = getDevice();
    const outputBuffer = createStorageBuffer(outputSize * getDTypeBytes(this._dtype));

    // Pad arrays to vec4 for uniform buffer
    const pad4 = (arr: number[]) => {
      const result = new Int32Array(4); // Use Int32 for signed values
      for (let i = 0; i < Math.min(arr.length, 4); i++) {
        result[i] = arr[i];
      }
      return result;
    };

    const pad4u = (arr: number[]) => {
      const result = new Uint32Array(4);
      for (let i = 0; i < Math.min(arr.length, 4); i++) {
        result[i] = arr[i];
      }
      return result;
    };

    const paramsData = new ArrayBuffer(96); // Increased size for new struct
    const view = new DataView(paramsData);

    // input_shape (vec4<u32>) - offset 0
    const inputShape4 = pad4u([...this._shape]);
    for (let i = 0; i < 4; i++) view.setUint32(i * 4, inputShape4[i], true);

    // output_shape (vec4<u32>) - offset 16
    const outputShape4 = pad4u(newShape);
    for (let i = 0; i < 4; i++) view.setUint32(16 + i * 4, outputShape4[i], true);

    // starts (vec4<i32>) - offset 32
    const starts4 = pad4(normSlices.map((s) => s.start!));
    for (let i = 0; i < 4; i++) view.setInt32(32 + i * 4, starts4[i], true);

    // steps (vec4<i32>) - offset 48
    const steps4 = pad4(normSlices.map((s) => s.step!));
    for (let i = 0; i < 4; i++) view.setInt32(48 + i * 4, steps4[i], true);

    // ndim, output_size - offset 64
    view.setUint32(64, this._shape.length, true);
    view.setUint32(68, outputSize, true);

    const paramsBuffer = device.createBuffer({
      size: 96,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const pipeline = getOrCreatePipeline(SLICE_SHADER, 'slice');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    return new Tensor({
      buffer: outputBuffer,
      shape: newShape,
      dtype: this._dtype,
      device: 'webgpu',
      requires_grad: this._requires_grad,
    });
  }

  private _sliceMultiDim(slices: Array<{ start: number; stop: number }>): Tensor {
    // Convert old slice format to new SliceSpec
    const newSlices: SliceSpec[] = slices.map((s) => ({ start: s.start, stop: s.stop, step: 1 }));
    return this.slice(newSlices);
  }

  transpose(dim0: number, dim1: number): Tensor {
    if (dim0 < 0) dim0 += this._shape.length;
    if (dim1 < 0) dim1 += this._shape.length;
    if (dim0 === dim1) return this;
    
    const dims = Array.from({ length: this._shape.length }, (_, i) => i);
    [dims[dim0], dims[dim1]] = [dims[dim1], dims[dim0]];
    
    return this.permute(dims);
  }

  permute(dims: number[]): Tensor {
    if (dims.length !== this._shape.length) {
      throw new Error(`permute: expected ${this._shape.length} dimensions, got ${dims.length}`);
    }
    
    const newDims = dims.map(d => d < 0 ? d + this._shape.length : d);
    const uniqueDims = new Set(newDims);
    if (uniqueDims.size !== this._shape.length) {
      throw new Error('permute: duplicate dimensions');
    }
    
    const newShape = newDims.map(d => this._shape[d]);
    
    const computeStrides = (shape: readonly number[]): number[] => {
      const strides = new Array(shape.length).fill(1);
      for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
      return strides;
    };

    const srcStrides = computeStrides(this._shape);
    // Reorder source strides to match the permutation
    // The shader expects srcStrides to correspond to the dimensions in the *new* shape order?
    // Wait. The shader computes input index from output index.
    // flat_out -> output_indices -> input_indices -> flat_in
    // input_index = dot(input_indices, src_strides)
    // output_index = dot(output_indices, dst_strides)
    // We map output coords to input coords.
    // If we permute, output[i] corresponds to input[perm[i]].
    // So input_indices[perm[i]] = output_indices[i].
    
    // Actually, simpler:
    // We want to view the same data with new strides.
    // If we were just creating a view, we would permute the strides.
    // new_strides[i] = old_strides[perm[i]].
    // But we are physically moving data to make it contiguous (or at least `transpose` did).
    
    // `TRANSPOSE_ND_SHADER` implementation:
    // It takes `src_strides` and `dst_strides`.
    // It iterates `output` buffer linearly (0..N).
    // Computes `output_indices` from `dst_strides` (standard contiguous).
    // Then it computes `input_offset`.
    // We need `src_strides` such that `dot(output_indices, reordered_src_strides) == input_offset`.
    // Yes.
    
    const permutedSrcStrides = newDims.map(d => srcStrides[d]);
    const dstStrides = computeStrides(newShape);

    const device = getDevice();
    const totalElements = this.numel();
    const outputBuffer = createStorageBuffer(totalElements * getDTypeBytes(this._dtype));

    const pad4 = (arr: number[]) => {
      const res = new Uint32Array(4);
      for (let i = 0; i < Math.min(arr.length, 4); i++) res[i] = arr[i];
      return res;
    };

    const paramsData = new ArrayBuffer(64);
    const view = new DataView(paramsData);
    const s4 = pad4(permutedSrcStrides); for (let i = 0; i < 4; i++) view.setUint32(i * 4, s4[i], true);
    const sh4 = pad4(newShape); for (let i = 0; i < 4; i++) view.setUint32(16 + i * 4, sh4[i], true);
    const d4 = pad4(dstStrides); for (let i = 0; i < 4; i++) view.setUint32(32 + i * 4, d4[i], true);
    view.setUint32(48, this._shape.length, true);
    view.setUint32(52, totalElements, true);

    const paramsBuffer = device.createBuffer({ size: 64, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const pipeline = getOrCreatePipeline(TRANSPOSE_ND_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(totalElements));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    const self = this;
    let grad_fn: GradFn | undefined;
    if (this._requires_grad) {
      grad_fn = {
        backward(gradOutput: Tensor): void {
          // Inverse permutation
          const invDims = new Array(dims.length);
          for (let i = 0; i < dims.length; i++) invDims[newDims[i]] = i;
          self.accumulateGrad(gradOutput.permute(invDims));
        },
        _next_tensors: [self],
      };
    }

    return new Tensor({ buffer: outputBuffer, shape: newShape, dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad, grad_fn });
  }

  mm(other: Tensor): Tensor {
    return this.matmul(other);
  }

  matmul(other: Tensor): Tensor {
    const self = this;

    // Helper: compute broadcast shape
    function broadcastShape(a: readonly number[], b: readonly number[]): number[] {
      const maxLen = Math.max(a.length, b.length);
      const result: number[] = new Array(maxLen);
      for (let i = 0; i < maxLen; i++) {
        const ai = a.length - 1 - i >= 0 ? a[a.length - 1 - i] : 1;
        const bi = b.length - 1 - i >= 0 ? b[b.length - 1 - i] : 1;
        if (ai !== 1 && bi !== 1 && ai !== bi) {
          throw new Error(`Broadcast shape mismatch: ${a} vs ${b}`);
        }
        result[maxLen - 1 - i] = Math.max(ai, bi);
      }
      return result;
    }

    // Case 1: both 1D -> dot product (scalar)
    if (self._shape.length === 1 && other._shape.length === 1) {
      if (self._shape[0] !== other._shape[0]) {
        throw new Error(`matmul: 1D tensors must have same size: ${self._shape} x ${other._shape}`);
      }
      return self.mul(other).sum(0);
    }

    // Case 2: one 1D, one 2D
    if (self._shape.length === 1 && other._shape.length === 2) {
      // (N,) @ (N, M) -> (M,)
      return self.unsqueeze(0).matmul(other).squeeze(0);
    }
    if (self._shape.length === 2 && other._shape.length === 1) {
      // (M, N) @ (N,) -> (M,)
      return self.matmul(other.unsqueeze(0)).squeeze(-1);
    }

    // Case 3: one 1D, one ND (N > 2)
    if (self._shape.length === 1) {
      // (N,) @ (..., N, M) -> (..., M)
      const result = self.unsqueeze(0).unsqueeze(0).matmul(other);
      return result.squeeze(-1).squeeze(-1);
    }
    if (other._shape.length === 1) {
      // (..., M, N) @ (N,) -> (..., M)
      return self.matmul(other.unsqueeze(-1)).squeeze(-1);
    }

    // Case 4: both 2D -> standard matmul
    if (self._shape.length === 2 && other._shape.length === 2) {
      const [M, K1] = self._shape;
      const [K2, N] = other._shape;
      if (K1 !== K2) throw new Error(`matmul() shape mismatch: ${self._shape} x ${other._shape}`);
      const device = getDevice();
      const outputBuffer = createStorageBuffer(M * N * getDTypeBytes(this._dtype));
      const dimsData = new Uint32Array([M, K1, N, 1]);
      const dimsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(dimsBuffer, 0, dimsData);
      const pipeline = getOrCreatePipeline(MATMUL_SHADER, 'matmul');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: self._buffer, offset: 0, size: self._buffer.size } },
          { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
          { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 3, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } },
        ],
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(M * N));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);

      let grad_fn: GradFn | undefined;
      if (self._requires_grad && is_grad_enabled()) {
        const otherRef = other;
        grad_fn = {
          backward(gradOutput: Tensor): void {
            self.accumulateGrad(gradOutput.matmul(otherRef.t()));
          },
          _next_tensors: [self, otherRef],
        };
      }

      return new Tensor({ buffer: outputBuffer, shape: [M, N], dtype: this._dtype, device: 'webgpu', requires_grad: self._requires_grad, grad_fn });
    }

    // Case 5: batched matmul (3D+)
    // Strategy: flatten batch dims → bmm → reshape
    // This works entirely on GPU with no CPU readback

    let a: Tensor = self;
    let b: Tensor = other;

    // Special case: 3D @ 2D (e.g. [batch, seq, in] @ [in, out] → Linear layer)
    // Treat 2D tensor as shared across batch dimensions
    if (a.shape.length === 3 && b.shape.length === 2) {
      const [batch, seq, inFeat] = a.shape;
      const [in2, outFeat] = b.shape;
      if (inFeat !== in2) throw new Error(`matmul shape mismatch: ${a.shape} vs ${b.shape}`);
      // Flatten: [batch*seq, in] @ [in, out] → [batch*seq, out]
      const aFlat = a.reshape([batch * seq, inFeat]);
      const result = aFlat.matmul(b);
      return result.reshape([batch, seq, outFeat]);
    }

    // Pad shorter tensor with leading 1s so both have same ndim
    const ndimA = a.shape.length;
    const ndimB = b.shape.length;
    if (ndimA < ndimB) {
      for (let i = 0; i < ndimB - ndimA; i++) a = a.unsqueeze(0);
    } else if (ndimB < ndimA) {
      for (let i = 0; i < ndimA - ndimB; i++) b = b.unsqueeze(0);
    }

    const ndim = a.shape.length;
    if (ndim < 2) throw new Error('matmul requires at least 2D tensors');

    // Check matrix dimensions
    const K1 = a.shape[ndim - 1];
    const K2 = b.shape[ndim - 2];
    if (K1 !== K2) throw new Error(`matmul shape mismatch: ${a.shape} vs ${b.shape}`);

    // Broadcast batch dimensions
    const batchShapeA = a.shape.slice(0, ndim - 2);
    const batchShapeB = b.shape.slice(0, ndim - 2);
    const outBatchShape = broadcastShape(batchShapeA, batchShapeB);
    const outShape = [...outBatchShape, a.shape[ndim - 2], b.shape[ndim - 1]];

    // Broadcast both tensors to output batch shape
    if (outBatchShape.length > 0) {
      const targetShapeA = [...outBatchShape, a.shape[ndim - 2], K1];
      a = a.broadcast_to(targetShapeA);
      const targetShapeB = [...outBatchShape, K2, b.shape[ndim - 1]];
      b = b.broadcast_to(targetShapeB);
    }

    // Flatten all batch dims into one: [batch*heads, M, K] and [batch*heads, K, N]
    const batchSize = outBatchShape.reduce((a, b) => a * b, 1);
    const M = a.shape[ndim - 2];
    const N = b.shape[ndim - 1];

    const aFlat = a.reshape([batchSize, M, K1]);
    const bFlat = b.reshape([batchSize, K2, N]);

    // Use bmm (batch matrix multiply) on GPU
    const result = aFlat.bmm(bFlat); // [batchSize, M, N]

    // Reshape back to output shape
    return result.reshape(outShape);
  }

  addmm(mat1: Tensor, mat2: Tensor, beta: number = 1, alpha: number = 1): Tensor {
    const prod = mat1.matmul(mat2);
    if (beta === 1) return this.add(prod.mul(alpha));
    return this.mul(beta).add(prod.mul(alpha));
  }

  mv(vec: Tensor): Tensor {
    if (this._shape.length !== 2) throw new Error('mv() expects 2D matrix');
    if (vec._shape.length !== 1) throw new Error('mv() expects 1D vector');
    return this.matmul(vec.unsqueeze(1)).squeeze(1);
  }

  addmv(mat: Tensor, vec: Tensor, beta: number = 1, alpha: number = 1): Tensor {
    const prod = mat.mv(vec);
    if (beta === 1) return this.add(prod.mul(alpha));
    return this.mul(beta).add(prod.mul(alpha));
  }

  outer(vec2: Tensor): Tensor {
    if (this._shape.length !== 1 || vec2._shape.length !== 1) throw new Error('outer() expects 1D vectors');
    return this.unsqueeze(1).matmul(vec2.unsqueeze(0));
  }

  addr(vec1: Tensor, vec2: Tensor, beta: number = 1, alpha: number = 1): Tensor {
    const prod = vec1.outer(vec2);
    if (beta === 1) return this.add(prod.mul(alpha));
    return this.mul(beta).add(prod.mul(alpha));
  }

  bmm(mat2: Tensor): Tensor {
    if (this._shape.length !== 3 || mat2._shape.length !== 3) throw new Error('bmm() expects 3D tensors');
    const [B, M, K1] = this._shape;
    const [B2, K2, N] = mat2._shape;
    if (B !== B2) throw new Error(`bmm() batch size mismatch: ${B} vs ${B2}`);
    if (K1 !== K2) throw new Error(`bmm() shape mismatch: ${this._shape} x ${mat2._shape}`);
    const results: Tensor[] = [];
    for (let i = 0; i < B; i++) {
      results.push(this.select(0, i).matmul(mat2.select(0, i)));
    }
    return stack(results);
  }

  baddbmm(batch1: Tensor, batch2: Tensor, beta: number = 1, alpha: number = 1): Tensor {
    const prod = batch1.bmm(batch2);
    if (beta === 1) return this.add(prod.mul(alpha));
    return this.mul(beta).add(prod.mul(alpha));
  }

  addbmm(batch1: Tensor, batch2: Tensor, beta: number = 1, alpha: number = 1): Tensor {
    const prod = batch1.bmm(batch2);
    if (beta === 1) return this.add(prod.mul(alpha));
    return this.mul(beta).add(prod.mul(alpha));
  }

  dot(other: Tensor): Tensor {
    if (this._shape.length !== 1 || other._shape.length !== 1) throw new Error('dot() expects 1D tensors');
    return this.mul(other).sum();
  }

  vdot(other: Tensor): Tensor {
    return this.dot(other);
  }

  // ============ Reduction Operations ============

  private _reduceDim(op: string, dim: number | number[], keepdim: boolean): Tensor {
    if (Array.isArray(dim)) {
      let result: Tensor = this;
      const sorted = [...dim].sort((a, b) => b - a);
      for (const d of sorted) {
        result = result._reduceDim(op, d, keepdim);
      }
      return result;
    }
    if (dim < 0) dim += this._shape.length;
    const dimSize = this._shape[dim];
    const outerSize = numel(this._shape.slice(0, dim));
    const innerSize = numel(this._shape.slice(dim + 1));
    const outShape = this._shape.filter((_, i) => i !== dim);
    if (outShape.length === 0 || keepdim) {
      const outShape2 = [...this._shape];
      outShape2[dim] = 1;
      if (!keepdim) {
        const device = getDevice();
        const outputBuffer = createStorageBuffer(outerSize * getDTypeBytes(this._dtype));
        const paramsData = new Uint32Array([dimSize, outerSize, innerSize, 0]);
        const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);
        const pipeline = getOrCreatePipeline(REDUCE_DIM_SHADER, op);
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
            { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
            { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
          ],
        });
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(...calculateWorkgroups(outerSize));
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
        return new Tensor({ buffer: outputBuffer, shape: outShape2.filter(s => s !== 1), dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
      }
      return new Tensor({ buffer: this._buffer, shape: outShape2, dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
    }
    const reshaped = this.reshape([outerSize, dimSize, innerSize]);
    const reduced = reshaped._reduce(op);
    return reduced.reshape(keepdim ? [...this._shape].map((s, i) => i === dim ? 1 : s) : outShape);
  }

  sum(dim?: number | number[], keepdim: boolean = false): Tensor {
    if (dim === undefined) return this._reduce('sum');
    return this._reduceDim('sum', dim, keepdim);
  }

  mean(dim?: number | number[], keepdim: boolean = false): Tensor {
    const s = this.sum(dim, keepdim);
    const n = dim === undefined ? this.numel() : (Array.isArray(dim) ? dim.reduce((p, d) => p * this._shape[d < 0 ? d + this._shape.length : d], 1) : this._shape[dim < 0 ? dim + this._shape.length : dim]);
    return s.div(n);
  }

  /**
   * Computes the variance along a dimension.
   * var = sum((x - mean(x))^2) / (N - correction)
   * @param dim - Dimension(s) to reduce
   * @param keepdim - Whether to keep the reduced dimension
   * @param unbiased - Whether to use Bessel's correction (N-1 instead of N)
   * @param correction - Override for the divisor correction (if specified, ignores unbiased)
   * @pytorch torch.var
   */
  var(dim?: number | number[], keepdim: boolean = false, unbiased: boolean = true, correction?: number): Tensor {
    const dims = dim === undefined ? [] : (Array.isArray(dim) ? dim : [dim]);
    const resolvedDims = dims.map(d => d < 0 ? d + this._shape.length : d);
    const n = resolvedDims.length > 0 ? resolvedDims.reduce((p, d) => p * this._shape[d], 1) : this.numel();

    // Determine correction value
    let corr = correction !== undefined ? correction : (unbiased ? 1 : 0);

    const meanVal = this.mean(dim, true); // keepdim=true for broadcasting
    const diff = this.sub(meanVal);
    const sqDiff = diff.pow(2);
    const sumSqDiff = sqDiff.sum(dim, keepdim);

    const divisor = Math.max(n - corr, 0);
    return sumSqDiff.div(divisor);
  }

  /**
   * Computes the standard deviation along a dimension.
   * std = sqrt(var)
   * @param dim - Dimension(s) to reduce
   * @param keepdim - Whether to keep the reduced dimension
   * @param unbiased - Whether to use Bessel's correction (N-1 instead of N)
   * @param correction - Override for the divisor correction (if specified, ignores unbiased)
   * @pytorch torch.std
   */
  std(dim?: number | number[], keepdim: boolean = false, unbiased: boolean = true, correction?: number): Tensor {
    return this.var(dim, keepdim, unbiased, correction).sqrt();
  }

  amax(dim?: number | number[], keepdim: boolean = false): Tensor {
    if (dim === undefined) return this._reduce('max_reduce');
    return this._reduceDim('max_reduce', dim, keepdim);
  }

  amin(dim?: number | number[], keepdim: boolean = false): Tensor {
    if (dim === undefined) return this._reduce('min_reduce');
    return this._reduceDim('min_reduce', dim, keepdim);
  }

  prod(dim?: number | number[], keepdim: boolean = false): Tensor {
    if (dim === undefined) return this._reduce('prod');
    return this._reduceDim('prod', dim, keepdim);
  }

  all(dim?: number | number[], keepdim: boolean = false): Tensor {
    if (dim === undefined) return this._reduce('all');
    return this._reduceDim('all', dim, keepdim);
  }

  any(dim?: number | number[], keepdim: boolean = false): Tensor {
    if (dim === undefined) return this._reduce('any');
    return this._reduceDim('any', dim, keepdim);
  }

  // ============ Argmax / Argmin ============

  private _argReduce(op: string, dim?: number, keepdim?: boolean): Tensor {
    const flat = this.flatten();
    const device = getDevice();
    const outputBuffer = createStorageBuffer(4);
    const pipeline = getOrCreatePipeline(op === 'argmax' ? ARGMAX_SHADER : ARGMIN_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: flat._buffer, offset: 0, size: flat._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1, 1, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new Tensor({ buffer: outputBuffer, shape: [], dtype: 'int32', device: 'webgpu', requires_grad: false });
  }

  /**
   * Returns the indices of the maximum values along a dimension.
   * @param dim - Dimension to reduce. If undefined, returns index of global maximum.
   * @param keepdim - Whether to keep the reduced dimension
   * @pytorch torch.argmax
   */
  argmax(dim?: number, keepdim: boolean = false): Tensor {
    if (dim === undefined) {
      return this._argReduce('argmax');
    }

    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;
    const dimSize = this._shape[resolvedDim];
    const outShape = keepdim
      ? this._shape.map((s, i) => (i === resolvedDim ? 1 : s))
      : this._shape.filter((_, i) => i !== resolvedDim);
    const outNumel = outShape.reduce((a, b) => a * b, 1);

    // Use CPU fallback for now - read values and compute argmax
    return this._argmaxDimSimple(resolvedDim, keepdim, outShape);
  }

  /**
   * Internal: compute argmax along a dimension.
   */
  private _argmaxDimSimple(dim: number, keepdim: boolean, outShape: number[]): Tensor {
    // For now, use a simple approach via max and comparison
    // This is CPU-based and will be slow but correct

    // Step 1: get max values along the dimension
    const maxVals = this.amax(dim, keepdim);

    // Step 2: compare input with maxVals broadcasted
    // The position where input == maxVals is the argmax
    const maxBroadcast = maxVals.expand(this._shape);
    const isMax = this.eq(maxBroadcast);

    // Step 3: find the first True along the dimension
    // We can use cumsum and check where it becomes 1
    const cumsum = isMax.cumsum(dim);
    const firstOccurrence = cumsum.eq(1).mul(isMax);

    // Step 4: create index tensor and multiply with mask
    const indices = this._indicesAlongDim(dim);
    const maskedIndices = indices.mul(firstOccurrence);

    // Step 5: sum along dimension to get final argmax indices
    const result = maskedIndices.sum(dim, keepdim);
    // Convert float32 to int32 via round and reinterpret
    // For now, keep as float and note that caller should interpret as int32
    return result;
  }

  /**
   * Internal: create a tensor with indices [0, 1, 2, ...] along a dimension.
   */
  private _indicesAlongDim(dim: number): Tensor {
    const { arange, ones } = require('../ops/creation');
    const dimSize = this._shape[dim < 0 ? dim + this._shape.length : dim];
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;

    // Create [0, 1, 2, ..., dimSize-1]
    const indices = arange(0, dimSize, 1, { dtype: 'float32' });

    // Reshape to put indices at the right dimension
    const shape = new Array(this._shape.length).fill(1);
    shape[resolvedDim] = dimSize;
    const reshapedIndices = indices.reshape(shape);

    // Broadcast to full shape
    return reshapedIndices.broadcastTo(this._shape);
  }

  /**
   * Returns the indices of the minimum values along a dimension.
   * @param dim - Dimension to reduce. If undefined, returns index of global minimum.
   * @param keepdim - Whether to keep the reduced dimension
   * @pytorch torch.argmin
   */
  argmin(dim?: number, keepdim: boolean = false): Tensor {
    if (dim === undefined) {
      return this._argReduce('argmin');
    }

    // For argmin, negate values and use argmax
    return this.neg().argmax(dim, keepdim);
  }

  /**
   * Gathers values along a dimension according to index.
   * out[i][j][k] = input[index[i][j][k]][j][k]  // if dim == 0
   * @param dim - Dimension along which to index
   * @param index - Indices of elements to gather
   * @pytorch torch.gather
   */
  async gather(dim: number, index: Tensor): Promise<Tensor> {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;

    // Validate shapes
    if (index._shape.length !== this._shape.length) {
      throw new Error('gather: index must have same number of dimensions as input');
    }

    // CPU fallback implementation
    return this._gatherCPU(resolvedDim, index);
  }

  /**
   * Internal: gather via CPU readback.
   */
  private async _gatherCPU(dim: number, index: Tensor): Promise<Tensor> {
    const inputData = await this.toArray();
    const indexData = await index.toArray();
    const shape = [...this._shape];
    const indexShape = index._shape;
    const outNumel = index.numel();
    const dimSize = shape[dim];

    const outputData = new Float32Array(outNumel);

    for (let outFlatIdx = 0; outFlatIdx < outNumel; outFlatIdx++) {
      // Convert flat output index to multi-dimensional coordinates
      const outCoords: number[] = [];
      let tempIdx = outFlatIdx;
      for (let d = indexShape.length - 1; d >= 0; d--) {
        outCoords.unshift(tempIdx % indexShape[d]);
        tempIdx = Math.floor(tempIdx / indexShape[d]);
      }

      // Get the index value at this position
      const gatherIdx = Math.floor(indexData[outFlatIdx]);
      const wrappedIdx = gatherIdx < 0 ? gatherIdx + dimSize : gatherIdx;

      // Create input coordinates by replacing the dimension coordinate
      const inCoords = [...outCoords];
      inCoords[dim] = wrappedIdx;

      // Convert input coordinates to flat index
      let inFlatIdx = 0;
      let stride = 1;
      for (let d = shape.length - 1; d >= 0; d--) {
        inFlatIdx += inCoords[d] * stride;
        stride *= shape[d];
      }

      outputData[outFlatIdx] = inputData[inFlatIdx];
    }

    return Tensor._fromNumberArray(Array.from(outputData), this._dtype).reshape(indexShape);
  }

  /**
   * Writes all values from the source tensor into the input tensor at the specified indices.
   * self[index[i][j][k]][j][k] = src[i][j][k]  // if dim == 0
   * @param dim - Dimension along which to index
   * @param index - Indices of elements to scatter into
   * @param src - Source tensor
   * @pytorch torch.scatter
   */
  async scatter(dim: number, index: Tensor, src: Tensor): Promise<Tensor> {
    return this._scatterCPU(dim, index, src, 'set');
  }

  /**
   * In-place scatter operation.
   * @param dim - Dimension along which to index
   * @param index - Indices of elements to scatter into
   * @param src - Source tensor
   * @pytorch torch.scatter_
   */
  async scatter_(dim: number, index: Tensor, src: Tensor): Promise<Tensor> {
    return this._scatterCPU(dim, index, src, 'set');
  }

  /**
   * Adds all values from the source tensor into the input tensor at the specified indices.
   * self[index[i][j][k]][j][k] += src[i][j][k]  // if dim == 0
   * @param dim - Dimension along which to index
   * @param index - Indices of elements to scatter into
   * @param src - Source tensor
   * @pytorch torch.scatter_add
   */
  async scatter_add(dim: number, index: Tensor, src: Tensor): Promise<Tensor> {
    return this._scatterCPU(dim, index, src, 'add');
  }

  /**
   * Internal: scatter via CPU readback.
   */
  private async _scatterCPU(dim: number, index: Tensor, src: Tensor, mode: 'set' | 'add'): Promise<Tensor> {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;

    // Read current data asynchronously
    const inputData = await this.toArray();
    const indexData = await index.toArray();
    const srcData = await src.toArray();

    const shape = [...this._shape];
    const indexShape = index._shape;
    const srcNumel = src.numel();

    // Validation
    if (indexShape.length !== this._shape.length) {
      throw new Error('scatter: index must have same number of dimensions as input');
    }
    if (srcNumel !== srcData.length) {
      throw new Error('scatter: src total number of elements must match index');
    }

    // Convert multi-dimensional index to flat indices for scattering
    const flatResult = [...inputData];

    // Helper: convert multi-dimensional coordinates to flat index
    const toFlatIndex = (coords: number[]): number => {
      let idx = 0;
      let stride = 1;
      for (let d = shape.length - 1; d >= 0; d--) {
        idx += coords[d] * stride;
        stride *= shape[d];
      }
      return idx;
    };

    // Process each element in index/src
    const processIndex = (flatIdx: number, srcVal: number): void => {
      // Convert flat index back to multi-dimensional coordinates
      const coords: number[] = [];
      let tempIdx = flatIdx;
      for (let d = shape.length - 1; d >= 0; d--) {
        coords.unshift(tempIdx % shape[d]);
        tempIdx = Math.floor(tempIdx / shape[d]);
      }

      // The index tensor contains indices along `dim`
      // So we need to look at indexData[flatIdx] to get the target index along that dimension
      const targetIdx = Math.floor(indexData[flatIdx]);
      if (targetIdx >= 0 && targetIdx < shape[resolvedDim]) {
        coords[resolvedDim] = targetIdx;
        const targetFlatIdx = toFlatIndex(coords);
        if (mode === 'set') {
          flatResult[targetFlatIdx] = srcVal;
        } else {
          flatResult[targetFlatIdx] += srcVal;
        }
      }
    };

    for (let i = 0; i < indexData.length; i++) {
      processIndex(i, srcData[i]);
    }

    return Tensor._fromNumberArray(flatResult, this._dtype).reshape(shape);
  }

  /**
   * Repeats each element of the tensor a specified number of times along a given dimension.
   * @param repeats - Number of repetitions for each element
   * @param dim - Dimension along which to repeat
   * @pytorch torch.repeat_interleave
   */
  async repeat_interleave(repeats: number, dim?: number): Promise<Tensor> {
    if (dim === undefined) {
      // Repeat the flattened tensor
      const flat = this.flatten();
      const outShape = [flat.numel() * repeats];
      return this._repeatInterleaveCPU(flat, repeats, outShape);
    }

    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;
    const outShape = [...this._shape];
    outShape[resolvedDim] *= repeats;

    return this._repeatInterleaveCPU(this, repeats, outShape, resolvedDim);
  }

  /**
   * Internal: repeat_interleave via CPU.
   */
  private async _repeatInterleaveCPU(tensor: Tensor, repeats: number, outShape: number[], dim?: number): Promise<Tensor> {
    // CPU implementation
    const data = await tensor.toArray();
    const outNumel = outShape.reduce((a, b) => a * b, 1);
    const outputData = new Float32Array(outNumel);

    const shape = [...tensor._shape];
    const ndim = shape.length;
    const resolvedDim = dim !== undefined ? (dim < 0 ? dim + ndim : dim) : 0;

    for (let outIdx = 0; outIdx < outNumel; outIdx++) {
      // Convert output flat index to coordinates
      const outCoords: number[] = [];
      let tempIdx = outIdx;
      for (let d = ndim - 1; d >= 0; d--) {
        outCoords.unshift(tempIdx % outShape[d]);
        tempIdx = Math.floor(tempIdx / outShape[d]);
      }

      // Map back to input coordinates
      const inCoords = [...outCoords];
      if (dim !== undefined) {
        inCoords[resolvedDim] = Math.floor(outCoords[resolvedDim] / repeats);
      }

      // Convert input coordinates to flat index
      let inIdx = 0;
      let stride = 1;
      for (let d = ndim - 1; d >= 0; d--) {
        inIdx += inCoords[d] * stride;
        stride *= shape[d];
      }

      outputData[outIdx] = data[inIdx];
    }

    return Tensor._fromNumberArray(Array.from(outputData), this._dtype).reshape(outShape);
  }

  /**
   * Roll the tensor along a dimension.
   * Elements that roll beyond the last position come back to the beginning.
   * @param shifts - Number of places to shift
   * @param dims - Dimension(s) to roll
   * @pytorch torch.roll
   */
  async roll(shifts: number | number[], dims?: number | number[]): Promise<Tensor> {
    const shape = [...this._shape];
    const shiftsArray = Array.isArray(shifts) ? shifts : [shifts];
    const dimsArray = dims === undefined ? [0] : (Array.isArray(dims) ? dims : [dims]);

    // Normalize dims
    const normalizedDims = dimsArray.map(d => d < 0 ? d + shape.length : d);

    // CPU implementation for now
    return this._rollCPU(shiftsArray, normalizedDims);
  }

  /**
   * Internal: roll via CPU.
   */
  private async _rollCPU(shifts: number[], dims: number[]): Promise<Tensor> {
    const data = await this.toArray();
    const shape = [...this._shape];
    const ndim = shape.length;
    const outputData = new Float32Array(this.numel());

    for (let outIdx = 0; outIdx < data.length; outIdx++) {
      // Convert output flat index to coordinates
      const outCoords: number[] = [];
      let tempIdx = outIdx;
      for (let d = ndim - 1; d >= 0; d--) {
        outCoords.unshift(tempIdx % shape[d]);
        tempIdx = Math.floor(tempIdx / shape[d]);
      }

      // Calculate input coordinates by applying shifts
      const inCoords = [...outCoords];
      for (let i = 0; i < dims.length; i++) {
        const d = dims[i];
        const shift = shifts[i] % shape[d];
        inCoords[d] = (outCoords[d] - shift + shape[d]) % shape[d];
      }

      // Convert input coordinates to flat index
      let inIdx = 0;
      let stride = 1;
      for (let d = ndim - 1; d >= 0; d--) {
        inIdx += inCoords[d] * stride;
        stride *= shape[d];
      }

      outputData[outIdx] = data[inIdx];
    }

    return Tensor._fromNumberArray(Array.from(outputData), this._dtype).reshape(this._shape);
  }

  /**
   * Rotates a tensor by 90 degrees in the plane specified by dims.
   * @param k - Number of 90 degree rotations
   * @param dims - Two dimensions that define the plane
   * @pytorch torch.rot90
   */
  async rot90(k: number = 1, dims?: number[]): Promise<Tensor> {
    const dimsArray = dims || [-2, -1];
    if (dimsArray.length !== 2) {
      throw new Error('rot90: dims must contain exactly 2 dimensions');
    }

    const ndim = this._shape.length;
    const dim0 = dimsArray[0] < 0 ? dimsArray[0] + ndim : dimsArray[0];
    const dim1 = dimsArray[1] < 0 ? dimsArray[1] + ndim : dimsArray[1];

    if (dim0 === dim1) {
      throw new Error('rot90: dims must be different');
    }

    // Normalize k to [0, 1, 2, 3]
    const rotations = ((k % 4) + 4) % 4;

    let result: Tensor = this;
    for (let i = 0; i < rotations; i++) {
      result = await result._rot90Once(dim0, dim1);
    }

    return result;
  }

  /**
   * Internal: single 90 degree rotation.
   */
  private async _rot90Once(dim0: number, dim1: number): Promise<Tensor> {
    // For a 2D plane, rot90 means: result[j, N-1-i] = input[i, j]
    // This is equivalent to: transpose, then flip along dim1

    const shape = [...this._shape];
    const dim0Size = shape[dim0];
    const dim1Size = shape[dim1];

    // CPU implementation
    const data = await this.toArray();
    const ndim = shape.length;
    const newShape = [...shape];
    newShape[dim0] = dim1Size;
    newShape[dim1] = dim0Size;
    const outputData = new Float32Array(this.numel());

    for (let inIdx = 0; inIdx < data.length; inIdx++) {
      const inCoords: number[] = [];
      let tempIdx = inIdx;
      for (let d = ndim - 1; d >= 0; d--) {
        inCoords.unshift(tempIdx % shape[d]);
        tempIdx = Math.floor(tempIdx / shape[d]);
      }

      // Apply rot90: new_i = dim1Size - 1 - inCoords[dim1], new_j = inCoords[dim0]
      const outCoords = [...inCoords];
      outCoords[dim0] = dim1Size - 1 - inCoords[dim1];
      outCoords[dim1] = inCoords[dim0];

      let outIdx = 0;
      let stride = 1;
      for (let d = ndim - 1; d >= 0; d--) {
        outIdx += outCoords[d] * stride;
        stride *= newShape[d];
      }

      outputData[outIdx] = data[inIdx];
    }

    return Tensor._fromNumberArray(Array.from(outputData), this._dtype).reshape(newShape);
  }

  /**
   * Returns a new tensor with a dimension of size one inserted at the specified position.
   * This is an alias for unsqueeze.
   * @param dim - Dimension to insert
   * @pytorch torch.unsqueeze
   */
  unflatten(dim: number, sizes: number[]): Tensor {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;
    const newSize = sizes.reduce((a, b) => a * b, 1);

    if (this._shape[resolvedDim] !== newSize) {
      throw new Error(`unflatten: size of dimension ${dim} must match product of new sizes`);
    }

    const newShape = [...this._shape];
    newShape.splice(resolvedDim, 1, ...sizes);

    return this.reshape(newShape);
  }

  /**
   * Sorts the elements of the input tensor along a given dimension.
   * @param dim - Dimension to sort along (default: -1, last dimension)
   * @param descending - If true, sort in descending order (default: false)
   * @param stable - If true, use stable sort (default: false)
   * @returns A named tuple-like object with (values, indices)
   * @pytorch torch.sort
   */
  async sort(dim: number = -1, descending: boolean = false, stable: boolean = false): Promise<{ values: Tensor; indices: Tensor }> {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;
    const dimSize = this._shape[resolvedDim];
    const data = await this.toArray();
    const shape = [...this._shape];
    const ndim = shape.length;
    const numel = this.numel();

    // Calculate strides for the dimension
    const dimStride = (() => {
      let stride = 1;
      for (let d = ndim - 1; d > resolvedDim; d--) {
        stride *= shape[d];
      }
      return stride;
    })();

    // Number of independent slices
    const numSlices = numel / dimSize;

    const sortedValues = new Float32Array(numel);
    const sortedIndices = new Int32Array(numel);

    for (let sliceIdx = 0; sliceIdx < numSlices; sliceIdx++) {
      // Extract slice along dim
      const sliceValues: { val: number; origIdx: number }[] = [];
      for (let i = 0; i < dimSize; i++) {
        // Calculate flat index for position (..., i, ...) along dim
        const flatIdx = this._indexAlongDim(resolvedDim, i, sliceIdx, shape);
        sliceValues.push({ val: data[flatIdx], origIdx: flatIdx });
      }

      // Sort the slice
      sliceValues.sort((a, b) => {
        const cmp = descending ? b.val - a.val : a.val - b.val;
        if (cmp !== 0) return cmp;
        // For stable sort, preserve original order for equal elements
        if (stable) return a.origIdx - b.origIdx;
        return 0;
      });

      // Write sorted values and indices
      for (let i = 0; i < dimSize; i++) {
        const outFlatIdx = this._indexAlongDim(resolvedDim, i, sliceIdx, shape);
        sortedValues[outFlatIdx] = sliceValues[i].val;
        // Store the original index along the dimension
        const origPosAlongDim = sliceValues[i].origIdx;
        // Convert back to position along dim
        const posAlongDim = Math.floor((origPosAlongDim % (dimSize * dimStride)) / dimStride);
        sortedIndices[outFlatIdx] = posAlongDim;
      }
    }

    const valuesTensor = Tensor._fromNumberArray(Array.from(sortedValues), this._dtype).reshape(shape);
    const indicesTensor = Tensor._fromNumberArray(Array.from(new Int32Array(sortedIndices)), 'int32').reshape(shape);

    return { values: valuesTensor, indices: indicesTensor };
  }

  /**
   * Helper: calculate flat index for position i along dimension dim, for the sliceIdx-th slice.
   */
  private _indexAlongDim(dim: number, i: number, sliceIdx: number, shape: number[]): number {
    const ndim = shape.length;
    // Convert sliceIdx to coordinates (excluding dim)
    const coordsWithoutDim: number[] = [];
    let temp = sliceIdx;
    const dimsBefore = shape.filter((_, d) => d < dim);
    const dimsAfter = shape.filter((_, d) => d > dim);

    // Process from last to first
    for (let d = ndim - 1; d >= 0; d--) {
      if (d === dim) continue;
      coordsWithoutDim.unshift(temp % shape[d]);
      temp = Math.floor(temp / shape[d]);
    }

    // Build full coordinates
    const coords = [...coordsWithoutDim];
    coords.splice(dim, 0, i);

    // Convert to flat index
    let flatIdx = 0;
    let stride = 1;
    for (let d = ndim - 1; d >= 0; d--) {
      flatIdx += coords[d] * stride;
      stride *= shape[d];
    }

    return flatIdx;
  }

  /**
   * Returns the indices that would sort the tensor along a given dimension.
   * @param dim - Dimension to sort along (default: -1)
   * @param descending - If true, sort in descending order (default: false)
   * @param stable - If true, use stable sort (default: false)
   * @returns Tensor of indices
   * @pytorch torch.argsort
   */
  async argsort(dim: number = -1, descending: boolean = false, stable: boolean = false): Promise<Tensor> {
    const result = await this.sort(dim, descending, stable);
    return result.indices;
  }

  /**
   * Returns the k largest or smallest elements along a given dimension.
   * @param k - Number of elements to return
   * @param dim - Dimension to find topk along (default: -1)
   * @param largest - If true, return largest elements (default: true)
   * @param sorted - If true, return elements in sorted order (default: true)
   * @returns A named tuple-like object with (values, indices)
   * @pytorch torch.topk
   */
  async topk(k: number, dim: number = -1, largest: boolean = true, sorted: boolean = true): Promise<{ values: Tensor; indices: Tensor }> {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;
    const dimSize = this._shape[resolvedDim];

    if (k > dimSize) {
      throw new Error(`topk: k (${k}) cannot be larger than dimension size (${dimSize})`);
    }

    // Get full sort and take top k
    const fullSort = await this.sort(dim, largest, false);

    // Slice top k
    const valuesSlice = fullSort.values.narrow(resolvedDim, 0, k);
    const indicesSlice = fullSort.indices.narrow(resolvedDim, 0, k);

    return { values: valuesSlice, indices: indicesSlice };
  }

  /**
   * Returns the k-th smallest element along a given dimension.
   * @param k - The k-th position (1-based)
   * @param dim - Dimension to find kth value along (default: -1)
   * @param keepdim - Whether to keep the reduced dimension (default: false)
   * @returns A named tuple-like object with (values, indices)
   * @pytorch torch.kthvalue
   */
  async kthvalue(k: number, dim: number = -1, keepdim: boolean = false): Promise<{ values: Tensor; indices: Tensor }> {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;

    // Get full sort and take k-1 index (k is 1-based)
    const fullSort = await this.sort(dim, false, false);

    // Select the k-1 position (0-based)
    const valuesSlice = fullSort.values.select(resolvedDim, k - 1);
    const indicesSlice = fullSort.indices.select(resolvedDim, k - 1);

    if (keepdim) {
      return {
        values: valuesSlice.unsqueeze(resolvedDim),
        indices: indicesSlice.unsqueeze(resolvedDim),
      };
    }

    return { values: valuesSlice, indices: indicesSlice };
  }

  /**
   * Cumulative maximum along a dimension.
   * @param dim - Dimension to compute along
   * @pytorch torch.cummax
   */
  async cummax(dim: number): Promise<{ values: Tensor; indices: Tensor }> {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;
    const dimSize = this._shape[resolvedDim];
    const data = await this.toArray();
    const shape = [...this._shape];
    const numel = this.numel();

    const valuesOut = new Float32Array(numel);
    const indicesOut = new Int32Array(numel);

    const numSlices = numel / dimSize;
    for (let sliceIdx = 0; sliceIdx < numSlices; sliceIdx++) {
      let maxVal = -Infinity;
      let maxIdx = 0;

      for (let i = 0; i < dimSize; i++) {
        const flatIdx = this._indexAlongDim(resolvedDim, i, sliceIdx, shape);
        const val = data[flatIdx];

        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }

        valuesOut[flatIdx] = maxVal;
        indicesOut[flatIdx] = maxIdx;
      }
    }

    return {
      values: Tensor._fromNumberArray(Array.from(valuesOut), this._dtype).reshape(shape),
      indices: Tensor._fromNumberArray(Array.from(new Int32Array(indicesOut)), 'int32').reshape(shape),
    };
  }

  /**
   * Cumulative minimum along a dimension.
   * @param dim - Dimension to compute along
   * @pytorch torch.cummin
   */
  async cummin(dim: number): Promise<{ values: Tensor; indices: Tensor }> {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;
    const dimSize = this._shape[resolvedDim];
    const data = await this.toArray();
    const shape = [...this._shape];
    const numel = this.numel();

    const valuesOut = new Float32Array(numel);
    const indicesOut = new Int32Array(numel);

    const numSlices = numel / dimSize;
    for (let sliceIdx = 0; sliceIdx < numSlices; sliceIdx++) {
      let minVal = Infinity;
      let minIdx = 0;

      for (let i = 0; i < dimSize; i++) {
        const flatIdx = this._indexAlongDim(resolvedDim, i, sliceIdx, shape);
        const val = data[flatIdx];

        if (val < minVal) {
          minVal = val;
          minIdx = i;
        }

        valuesOut[flatIdx] = minVal;
        indicesOut[flatIdx] = minIdx;
      }
    }

    return {
      values: Tensor._fromNumberArray(Array.from(valuesOut), this._dtype).reshape(shape),
      indices: Tensor._fromNumberArray(Array.from(new Int32Array(indicesOut)), 'int32').reshape(shape),
    };
  }

  /**
   * Log sum exp along dimensions.
   * Numerically stable: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
   * @param dim - Dimension(s) to reduce
   * @param keepdim - Whether to keep reduced dimensions
   * @pytorch torch.logsumexp
   */
  logsumexp(dim?: number | number[], keepdim: boolean = false): Tensor {
    const maxVal = this.amax(dim, true);
    const stable = this.sub(maxVal).exp();
    const sumExp = stable.sum(dim, keepdim);
    const squeezeDims = dim === undefined ? [] : (Array.isArray(dim) ? dim : [dim]);
    let result = maxVal;
    for (const d of squeezeDims) {
      result = result.squeeze(d);
    }
    return result.add(sumExp.log());
  }

  /**
   * Log cumulative sum exp along a dimension.
   * @param dim - Dimension to compute along
   * @pytorch torch.logcumsumexp
   */
  async logcumsumexp(dim: number): Promise<Tensor> {
    const resolvedDim = dim < 0 ? dim + this._shape.length : dim;
    const dimSize = this._shape[resolvedDim];
    const data = await this.toArray();
    const shape = [...this._shape];
    const numel = this.numel();

    const output = new Float32Array(numel);
    const numSlices = numel / dimSize;

    for (let sliceIdx = 0; sliceIdx < numSlices; sliceIdx++) {
      let maxVal = -Infinity;

      // First pass: find running max
      const runningMax: number[] = [];
      for (let i = 0; i < dimSize; i++) {
        const flatIdx = this._indexAlongDim(resolvedDim, i, sliceIdx, shape);
        if (data[flatIdx] > maxVal) maxVal = data[flatIdx];
        runningMax.push(maxVal);
      }

      // Second pass: compute logcumsumexp
      let sumExp = 0;
      for (let i = 0; i < dimSize; i++) {
        const flatIdx = this._indexAlongDim(resolvedDim, i, sliceIdx, shape);
        sumExp += Math.exp(data[flatIdx] - runningMax[i]);
        output[flatIdx] = runningMax[i] + Math.log(sumExp);
      }
    }

    return Tensor._fromNumberArray(Array.from(output), this._dtype).reshape(shape);
  }

  /**
   * Count the number of non-zero elements.
   * @param dim - Dimension(s) to reduce
   * @param keepdim - Whether to keep reduced dimensions
   * @pytorch torch.count_nonzero
   */
  count_nonzero(dim?: number | number[], keepdim: boolean = false): Tensor {
    const isNonZero = this.ne(0);
    return isNonZero.sum(dim, keepdim);
  }

  /**
   * Returns both the min and max values along a dimension.
   * @param dim - Dimension to reduce
   * @param keepdim - Whether to keep the reduced dimension
   * @returns Object with {min, max} tensors
   * @pytorch torch.aminmax
   */
  aminmax(dim?: number, keepdim: boolean = false): { min: Tensor; max: Tensor } {
    return {
      min: this.amin(dim, keepdim),
      max: this.amax(dim, keepdim),
    };
  }

  t(): Tensor {
    if (this._shape.length !== 2) throw new Error('t() only supports 2D tensors');
    const [M, N] = this._shape;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const dimsData = new Uint32Array([M, N]);
    const dimsBuffer = device.createBuffer({ size: 8, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(dimsBuffer, 0, dimsData);
    const pipeline = getOrCreatePipeline(TRANSPOSE_SHADER, 'transpose_2d');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(M * N));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    const self = this;
    let grad_fn: GradFn | undefined;
    if (this._requires_grad && is_grad_enabled()) {
      grad_fn = {
        backward(gradOutput: Tensor): void {
          self.accumulateGrad(gradOutput.neg());
        },
        _next_tensors: [self],
      };
    }

    return new Tensor({ buffer: outputBuffer, shape: [N, M], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad, grad_fn });
  }

  private _binaryOp(op: string, other: Tensor): Tensor {
    if (needsBroadcast(this._shape, other._shape)) return this._broadcastBinaryOp(op, other);
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const pipeline = getOrCreatePipeline(ELEMENTWISE_SHADER, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    const self = this;
    const requires_grad = this._requires_grad || other._requires_grad;
    let grad_fn: GradFn | undefined;
    if (requires_grad) {
      grad_fn = {
        backward(gradOutput: Tensor): void {
          if (self._requires_grad) {
            let grad_self: Tensor;
            if (op === 'add' || op === 'sub') {
              grad_self = gradOutput;
            } else if (op === 'mul') {
              grad_self = gradOutput.mul(other);
            } else {
              grad_self = gradOutput;
            }
            self.accumulateGrad(grad_self);
          }
          if (other._requires_grad) {
            let grad_other: Tensor;
            if (op === 'add') {
              grad_other = gradOutput;
            } else if (op === 'sub') {
              grad_other = gradOutput.neg();
            } else if (op === 'mul') {
              grad_other = gradOutput.mul(self);
            } else {
              grad_other = gradOutput;
            }
            other.accumulateGrad(grad_other);
          }
        },
        _next_tensors: [self, other],
      };
    }

    return new Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: 'webgpu', requires_grad, grad_fn });
  }

  private _broadcastBinaryOp(op: string, other: Tensor): Tensor {
    const device = getDevice();
    const outputShape = broadcastShapes(this._shape, other._shape);
    const outputSize = numel(outputShape);
    const computeStrides = (shape: readonly number[], outShape: number[]): number[] => {
      const strides = [0, 0, 0, 0];
      const ndim = outShape.length;
      let stride = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        const dimIdx = (4 - ndim) + (i + (ndim - shape.length));
        strides[dimIdx] = shape[i] === 1 ? 0 : stride;
        stride *= shape[i];
      }
      return strides;
    };
    const aStrides = computeStrides(this._shape, outputShape);
    const bStrides = computeStrides(other._shape, outputShape);
    const paddedOut = [1, 1, 1, 1];
    for (let i = 0; i < outputShape.length; i++) paddedOut[4 - outputShape.length + i] = outputShape[i];
    const outputBuffer = createStorageBuffer(outputSize * getDTypeBytes(this._dtype));
    const paramsData = new Uint32Array([...paddedOut, ...aStrides, ...bStrides, outputShape.length, outputSize, 0, 0]);
    const paramsBuffer = device.createBuffer({ size: 64, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const entryPoints: any = { 
        'add': 'broadcast_add', 
        'sub': 'broadcast_sub', 
        'mul': 'broadcast_mul', 
        'div_op': 'broadcast_div',
        'pow_tensor': 'broadcast_pow' 
    };
    const pipeline = getOrCreatePipeline(BROADCAST_SHADER, entryPoints[op] || 'broadcast_add');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    const self = this;
    const requires_grad = this._requires_grad || other._requires_grad;
    let grad_fn: GradFn | undefined;
    if (requires_grad) {
      grad_fn = {
        backward(gradOutput: Tensor): void {
          if (self._requires_grad) {
            let grad_self: Tensor;
            if (op === 'add' || op === 'sub') {
              grad_self = self._reduceBroadcastGrad(gradOutput);
            } else if (op === 'mul') {
              grad_self = self._reduceBroadcastGrad(gradOutput.mul(other));
            } else if (op === 'pow_tensor') {
              // d(a^b)/da = b * a^(b-1)
              grad_self = self._reduceBroadcastGrad(gradOutput.mul(other).mul(self.pow(other.sub(1))));
            } else {
              grad_self = self._reduceBroadcastGrad(gradOutput);
            }
            self.accumulateGrad(grad_self);
          }
          if (other._requires_grad) {
            let grad_other: Tensor;
            if (op === 'add') {
              grad_other = other._reduceBroadcastGrad(gradOutput);
            } else if (op === 'sub') {
              grad_other = other._reduceBroadcastGrad(gradOutput.neg());
            } else if (op === 'mul') {
              grad_other = other._reduceBroadcastGrad(gradOutput.mul(self));
            } else if (op === 'pow_tensor') {
              // d(a^b)/db = a^b * ln(a)
              grad_other = other._reduceBroadcastGrad(gradOutput.mul(self.pow(other)).mul(self.abs().log()));
            } else {
              grad_other = other._reduceBroadcastGrad(gradOutput);
            }
            other.accumulateGrad(grad_other);
          }
        },
        _next_tensors: [self, other],
      };
    }

    return new Tensor({ buffer: outputBuffer, shape: outputShape, dtype: this._dtype, device: 'webgpu', requires_grad, grad_fn });
  }

  private _reduceBroadcastGrad(grad: Tensor): Tensor {
    if (!needsBroadcast(this._shape, grad._shape)) return grad;
    if (this._shape.length === 1 && grad._shape.length === 2) return this._sumAlongDim0(grad);
    return grad;
  }

  private _sumAlongDim0(input: Tensor): Tensor {
    if (input.shape.length !== 2) throw new Error('_sumAlongDim0 only supports 2D tensors');
    const [batchSize, features] = input.shape;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(features * getDTypeBytes(input.dtype));
    const paramsData = new Uint32Array([batchSize, features]);
    const paramsBuffer = device.createBuffer({ size: 8, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(REDUCE_BROADCAST_GRAD_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer, offset: 0, size: input.buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(features));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new Tensor({ buffer: outputBuffer, shape: [features], dtype: input.dtype, device: 'webgpu', requires_grad: false });
  }

  private _unaryOp(op: string): Tensor {
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const pipeline = getOrCreatePipeline(UNARY_SHADER, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    const self = this;
    let grad_fn: GradFn | undefined;
    if (this._requires_grad) {
      grad_fn = {
        backward(gradOutput: Tensor): void {
          let grad_self = gradOutput;
          if (op === 'relu') {
            const mask = self.gt(0);
            grad_self = gradOutput.mul(mask);
          } else if (op === 'sigmoid') {
            const out = self.sigmoid();
            grad_self = gradOutput.mul(out.mul(out.mul(-1).add(1)));
          } else if (op === 'tanh_op') {
            const out = self.tanh();
            grad_self = gradOutput.mul(out.mul(-1).add(1).mul(out.add(1)));
          } else if (op === 'gelu') {
            const x = self;
            const cdf = x.mul(0.7978845608).mul(x.mul(0.044715).mul(x).add(1)).tanh().add(1).mul(0.5);
            const pdf = x.mul(0.7978845608).mul(x.mul(0.044715).mul(x).add(1)).tanh().mul(-1).add(1).mul(0.5);
            const grad = cdf.add(x.mul(pdf));
            grad_self = gradOutput.mul(grad);
          } else if (op === 'sqrt_op') {
            grad_self = gradOutput.div(self.sqrt().mul(2));
          } else if (op === 'rsqrt_op') {
            grad_self = gradOutput.mul(-0.5).div(self.pow(1.5));
          } else if (op === 'square_op') {
            grad_self = gradOutput.mul(self).mul(2);
          } else if (op === 'reciprocal_op') {
            grad_self = gradOutput.div(self.pow(2)).neg();
          } else if (op === 'neg') {
            grad_self = gradOutput.neg();
          }
          self.accumulateGrad(grad_self);
        },
        _next_tensors: [self],
      };
    }

    return new Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad, grad_fn });
  }

  private _getState(): TensorData {
    return {
      buffer: this._buffer,
      shape: [...this._shape],
      dtype: this._dtype,
      device: this._device,
      requires_grad: this._requires_grad,
      grad_fn: this._grad_fn ?? undefined,
    };
  }

  /** @internal */
  accumulateGrad(grad: Tensor): void {
    if (!this._requires_grad) return;
    if (this._grad) {
      this._grad = this._grad.add(grad);
    } else {
      this._grad = grad;
    }
  }

  /**
   * Computes the gradient of current tensor with respect to graph leaves.
   * @pytorch tensor.backward()
   * @param gradient - Gradient w.r.t. this tensor (default: ones_like)
   * @param retain_graph - If false, clears grad_fn after use
   * @param create_graph - If true, constructs graph for gradient computation
   */
  backward(gradient?: Tensor, retain_graph: boolean = false, create_graph: boolean = false): void {
    if (!is_grad_enabled()) return;

    const grad = gradient || this.ones_like();

    // Build the computation graph from outputs to inputs
    const visited = new WeakSet<Tensor>();
    const topoOrder: { tensor: Tensor; gradFn: GradFn }[] = [];

    // DFS to collect tensors in topological order (outputs first)
    const visit = (t: Tensor): void => {
      if (visited.has(t)) return;
      visited.add(t);

      const gf = t._grad_fn;
      if (gf && gf._next_tensors) {
        for (const parent of gf._next_tensors) {
          visit(parent);
        }
        topoOrder.push({ tensor: t, gradFn: gf });
      }
    };

    visit(this);

    // Initialize gradient map with the output gradient
    const gradMap = new Map<number, Tensor>();
    gradMap.set(this._id, grad);

    // Initialize gradients for leaf tensors that need them
    for (const t of visited) {
      if (t._requires_grad && !gradMap.has(t._id)) {
        gradMap.set(t._id, t.zeros_like());
      }
    }

    // Execute backward pass in reverse topological order
    for (let i = topoOrder.length - 1; i >= 0; i--) {
      const { tensor, gradFn } = topoOrder[i];
      const gradOutput = gradMap.get(tensor._id);
      if (gradOutput && gradFn) {
        gradFn.backward(gradOutput);
      }
    }

    // Clear grad_fns if not retaining
    if (!retain_graph) {
      for (const t of visited) {
        t._grad_fn = null;
      }
    }
  }

  private _scalarOp(op: string, scalar: number): Tensor {
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const paramsData = new ArrayBuffer(8);
    new Float32Array(paramsData, 0, 1)[0] = scalar;
    new Uint32Array(paramsData, 4, 1)[0] = this.numel();
    const paramsBuffer = device.createBuffer({ size: 8, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(SCALAR_SHADER, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    const self = this;
    let grad_fn: GradFn | undefined;
    if (this._requires_grad) {
      grad_fn = {
        backward(gradOutput: Tensor): void {
          let grad_self: Tensor;
          if (op === 'add_scalar' || op === 'sub_scalar') {
            grad_self = gradOutput;
          } else if (op === 'mul_scalar') {
            grad_self = gradOutput.mul(scalar);
          } else if (op === 'div_scalar') {
            grad_self = gradOutput.div(scalar);
          } else {
            grad_self = gradOutput;
          }
          self.accumulateGrad(grad_self);
        },
        _next_tensors: [self],
      };
    }

    return new Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad, grad_fn });
  }

  private _reduce(op: string): Tensor {
    const capabilities = getCapabilities();
    if (capabilities.workgroupSharedMemory && !this._isChromeMac()) return this._reduceWithSharedMemory(op);
    return this._reduceSimple(op);
  }

  private _isChromeMac(): boolean {
    if (typeof navigator === 'undefined') return false;
    return navigator.userAgent.includes('Macintosh') && navigator.userAgent.includes('Chrome');
  }

  private _reduceWithSharedMemory(op: string): Tensor {
    const device = getDevice();
    const n = this.numel();
    let currentBuffer = this._buffer;
    let currentLength = n;
    const shaderMap: any = { 
      'sum': REDUCE_SUM_SHADER, 
      'max_reduce': REDUCE_MAX_SHADER, 
      'min_reduce': REDUCE_MIN_SHADER,
      'any': REDUCE_ANY_SHADER,
      'all': REDUCE_ALL_SHADER,
      'prod': REDUCE_PROD_SHADER
    };
    const shader = shaderMap[op] || REDUCE_SUM_SHADER;
    while (currentLength > 1) {
      const numWorkgroups = Math.ceil(currentLength / 256);
      const outputBuffer = createStorageBuffer(numWorkgroups * getDTypeBytes(this._dtype));
      const paramsData = new Uint32Array([currentLength, 0, 0, 0]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(shader, 'main');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: currentBuffer, offset: 0, size: currentBuffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        ],
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(numWorkgroups, 1, 1);
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      currentBuffer = outputBuffer;
      currentLength = numWorkgroups;
    }
    return new Tensor({ buffer: currentBuffer, shape: [], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
  }

  private _reduceSimple(op: string): Tensor {
    const device = getDevice();
    const n = this.numel();
    const shaderMap: any = { 
      'sum': REDUCE_SIMPLE_SUM_SHADER, 
      'max_reduce': REDUCE_SIMPLE_MAX_SHADER, 
      'min_reduce': REDUCE_SIMPLE_MIN_SHADER,
      'any': REDUCE_SIMPLE_ANY_SHADER,
      'all': REDUCE_SIMPLE_ALL_SHADER,
      'prod': REDUCE_SIMPLE_PROD_SHADER
    };
    const shader = shaderMap[op] || REDUCE_SIMPLE_SUM_SHADER;
    const inputCopy = device.createBuffer({ size: this._buffer.size, usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(this._buffer, 0, inputCopy, 0, this._buffer.size);
    device.queue.submit([copyEncoder.finish()]);
    let currentBuffer = inputCopy;
    let currentLength = n;
    while (currentLength > 1) {
      const numWorkgroups = Math.ceil(currentLength / 256);
      const outputBuffer = device.createBuffer({ size: Math.max(4, numWorkgroups * getDTypeBytes(this._dtype)), usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST });
      const pipeline = getOrCreatePipeline(shader, 'main');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: currentBuffer, offset: 0, size: currentBuffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        ],
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(numWorkgroups, 1, 1);
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      currentBuffer = outputBuffer;
      currentLength = numWorkgroups;
    }
    return new Tensor({ buffer: currentBuffer, shape: [], dtype: this._dtype, device: 'webgpu', requires_grad: this._requires_grad });
  }

  private _compareOp(op: string, other: Tensor): Tensor {
    if (needsBroadcast(this._shape, other._shape)) throw new Error('Broadcasting not yet implemented for comparison');
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const pipeline = getOrCreatePipeline(COMPARE_SHADER, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      ],
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: 'webgpu', requires_grad: false });
  }

  /**
   * PyTorch-style tensor indexing.
   * Supports:
   * - tensor.get(0) → select dim 0
   * - tensor.get([0, 1]) → select dims 0 and 1
   * - tensor.get([new Slice(1, 5)]) → slice dim 0 from 1 to 5
   * - tensor.get([Slice.null, 2]) → all of dim 0, select index 2 on dim 1
   * - tensor.get([new Slice(null, null, 2)]) → step 2 on dim 0
   * - tensor.get([0, new Slice(1, 3), new Slice(null, null, -1)]) → complex mixed
   *
   * Convenience: tensor.at(0, 1) is equivalent to tensor.get([0, 1])
   *
   * @pytorch tensor indexing
   */
  async get(index: IndexType | IndexType[]): Promise<Tensor> {
    const indices: IndexType[] = Array.isArray(index) ? index : [index];

    // Parse indices: separate selects from slices, track order
    const parsed: { type: 'select'; dim: number; idx: number }[] = [];
    const slices: { dim: number; spec: SliceSpec }[] = [];

    let outDim = 0; // tracks which tensor dimension we're indexing

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];

      if (typeof idx === 'number') {
        // Select: pick a single index along this dimension
        const dimSize = this._shape[outDim];
        let resolved = idx;
        if (resolved < 0) resolved += dimSize;

        if (resolved < 0 || resolved >= dimSize) {
          throw new Error(
            `Index ${idx} out of bounds for dimension ${outDim} with size ${dimSize}`
          );
        }

        parsed.push({ type: 'select', dim: outDim, idx: resolved });
        outDim++;
      } else if (idx instanceof Slice) {
        // Slice: extract a range with optional step
        const dimSize = this._shape[outDim];
        let start = idx.start ?? 0;
        let stop = idx.stop ?? dimSize;
        let step = idx.step ?? 1;

        // Handle negative start/stop
        if (start < 0) start = dimSize + start;
        if (stop < 0) stop = dimSize + stop;

        // Clamp to bounds
        start = Math.max(0, Math.min(start, dimSize));
        stop = Math.max(0, Math.min(stop, dimSize));

        if (step === 0) {
          throw new Error('Slice step cannot be zero');
        }

        slices.push({ dim: outDim, spec: { start, stop, step } });
        outDim++;
      } else if (idx === true || idx === false) {
        // Boolean mask — not supported in this method
        throw new Error(
          'Boolean indexing not supported in get(), use masked_select() instead'
        );
      }
    }

    // Validate we didn't exceed dimensions
    if (outDim > this._shape.length) {
      throw new Error(
        `Index dimension (${outDim}) exceeds tensor dimension (${this._shape.length})`
      );
    }

    // Execution strategy:
    // 1. Apply all selects via narrow + squeeze (dimension-removing)
    // 2. Apply all slices via the existing slice() method
    //
    // We chain them in order: first select dims, then slice remaining dims.

    let current: Tensor = this;

    // Apply selects in reverse order (highest dim first) so dimension indices stay stable.
    // When we select dim N, it gets squeezed out; dims > N shift down by 1.
    // By processing from highest to lowest, earlier selections don't affect later ones.
    for (let i = parsed.length - 1; i >= 0; i--) {
      const sel = parsed[i];
      current = current.select(sel.dim, sel.idx);
    }

    // If no slices remain, we're done
    if (slices.length === 0) {
      return current;
    }

    // Build slice specs for the remaining dimensions
    // After selects, the remaining dimensions are those not in `parsed`
    const selectedDims = new Set(parsed.map(s => s.dim));
    const remainingDims: number[] = [];
    for (let d = 0; d < this._shape.length; d++) {
      if (!selectedDims.has(d)) remainingDims.push(d);
    }

    // Build full slice specs: default is full slice for each remaining dim
    const sliceSpecs: SliceSpec[] = remainingDims.map(d => ({
      start: 0,
      stop: this._shape[d],
      step: 1,
    }));

    // Override with user-provided slices
    for (const sl of slices) {
      // Find where this original dimension maps in remainingDims
      const newIdx = remainingDims.indexOf(sl.dim);
      if (newIdx >= 0) {
        sliceSpecs[newIdx] = sl.spec;
      }
    }

    const result = current.slice(sliceSpecs);

    // Clean up intermediate tensor if selects were applied
    if (parsed.length > 0) {
      current.destroy();
    }

    return result;
  }

  /**
   * Convenience method: variadic indexing.
   * tensor.at(0, 1) is equivalent to tensor.get([0, 1])
   * tensor.at(new Slice(1, 5)) is equivalent to tensor.get([new Slice(1, 5)])
   */
  at(...indices: IndexType[]): Promise<Tensor> {
    return this.get(indices);
  }

  /**
   * Advanced slicing with Slice objects: tensor[i:j], tensor[i:j:k], tensor[::2]
   * Legacy method kept for backward compatibility. Prefer using get() or at().
   * @pytorch slice notation
   */
  async advancedSlice(indices: (number | Slice)[]): Promise<Tensor> {
    // Build slice specification
    const sliceSpecs: (number | SliceSpec)[] = [];

    for (let i = 0; i < this._shape.length; i++) {
      if (i < indices.length) {
        const idx = indices[i];
        if (idx instanceof Slice) {
          // Convert Slice to SliceSpec
          const dimSize = this._shape[i];
          let start = idx.start ?? 0;
          let stop = idx.stop ?? dimSize;
          let step = idx.step ?? 1;

          // Handle negative start/stop
          if (start < 0) start = dimSize + start;
          if (stop < 0) stop = dimSize + stop;

          // Clamp to bounds
          start = Math.max(0, Math.min(start, dimSize));
          stop = Math.max(0, Math.min(stop, dimSize));

          sliceSpecs.push({ start, stop, step });
        } else {
          sliceSpecs.push(idx as number);
        }
      } else {
        // Full slice for remaining dimensions
        sliceSpecs.push({ start: 0, stop: this._shape[i], step: 1 });
      }
    }

    return this._executeSlice(sliceSpecs);
  }

  /**
   * Internal: execute slice operation via CPU fallback.
   */
  private async _executeSlice(specs: (number | SliceSpec)[]): Promise<Tensor> {
    const data = await this.toArray();
    const shape = [...this._shape];
    const ndim = shape.length;

    // Calculate output shape
    const outShape: number[] = [];
    for (let i = 0; i < ndim; i++) {
      const spec = specs[i];
      if (typeof spec === 'number') {
        // Dimension is indexed, removed from output
        continue;
      }
      // Calculate size of this dimension
      const start = spec.start ?? 0;
      const stop = spec.stop ?? shape[i];
      const step = spec.step ?? 1;
      const size = Math.ceil((stop - start) / step);
      outShape.push(size);
    }

    const outNumel = outShape.reduce((a, b) => a * b, 1);
    const TypedArrayCtor = getTypedArrayConstructor(this._dtype);
    const output = new TypedArrayCtor(outNumel);

    // Iterate through output tensor and calculate input indices
    for (let outIdx = 0; outIdx < outNumel; outIdx++) {
      // Convert flat output index to coordinates
      const outCoords: number[] = [];
      let tempIdx = outIdx;
      for (let d = outShape.length - 1; d >= 0; d--) {
        outCoords.unshift(tempIdx % outShape[d]);
        tempIdx = Math.floor(tempIdx / outShape[d]);
      }

      // Map output coords to input coords using slice specs
      const inCoords: number[] = [];
      let outCoordIdx = 0;
      for (let d = 0; d < ndim; d++) {
        const spec = specs[d];
        if (typeof spec === 'number') {
          inCoords.push(spec);
        } else {
          const start = spec.start ?? 0;
          const step = spec.step ?? 1;
          inCoords.push(start + outCoords[outCoordIdx] * step);
          outCoordIdx++;
        }
      }

      // Calculate input flat index
      let inFlatIdx = 0;
      let stride = 1;
      for (let d = ndim - 1; d >= 0; d--) {
        inFlatIdx += inCoords[d] * stride;
        stride *= shape[d];
      }

      output[outIdx] = data[inFlatIdx];
    }

    // Create properly shaped tensor from flat data
    const shapedCtor = getTypedArrayConstructor(this._dtype);
    const shapedData = new shapedCtor(output);
    const buffer = createBufferWithData(shapedData, this._dtype);
    
    return new Tensor({
      buffer,
      shape: outShape,
      dtype: this._dtype,
      device: 'webgpu',
      requires_grad: false,
    });
  }

  /**
   * Internal: extract scalar value at flat index.
   */
  private async _extractScalar(flatIdx: number): Promise<Tensor> {
    const data = await this.toArray();
    return Tensor._fromNumberArray([data[flatIdx]]);
  }

  /**
   * Internal: create sliced tensor with indices fixed.
   */
  private async _sliceIndices(indices: number[]): Promise<Tensor> {
    const ndim = this._shape.length;
    const numFixed = indices.length;
    const outShape = this._shape.slice(numFixed);

    if (outShape.length === 0) {
      // All dimensions indexed, should have used _extractScalar
      return this.reshape([]);
    }

    // Calculate offset
    let offset = 0;
    let stride = 1;
    for (let d = ndim - 1; d >= numFixed; d--) {
      stride *= this._shape[d];
    }
    for (let i = 0; i < numFixed; i++) {
      offset += indices[i] * stride;
      stride = 1;
      for (let d = ndim - 1; d > i; d--) {
        stride *= this._shape[d];
      }
    }

    // For now, use CPU fallback
    return this._sliceCPU(indices, outShape);
  }

  /**
   * Internal: slice tensor via CPU readback.
   */
  private async _sliceCPU(indices: number[], outShape: number[]): Promise<Tensor> {
    const data = await this.toArray();
    const shape = [...this._shape];
    const ndim = shape.length;
    const numFixed = indices.length;

    // Calculate offset
    let offset = 0;
    let stride = 1;
    for (let d = ndim - 1; d >= numFixed; d--) {
      stride *= shape[d];
    }
    for (let i = 0; i < numFixed; i++) {
      offset += indices[i] * stride;
      stride = 1;
      for (let d = ndim - 1; d > i; d--) {
        stride *= shape[d];
      }
    }

    // Extract slice
    const outNumel = outShape.reduce((a, b) => a * b, 1);
    const output = new Float32Array(outNumel);

    for (let i = 0; i < outNumel; i++) {
      // Convert flat output index to coordinates
      const outCoords: number[] = [];
      let tempIdx = i;
      for (let d = outShape.length - 1; d >= 0; d--) {
        outCoords.unshift(tempIdx % outShape[d]);
        tempIdx = Math.floor(tempIdx / outShape[d]);
      }

      // Calculate input flat index
      const inCoords = [...indices, ...outCoords];
      let inFlatIdx = 0;
      stride = 1;
      for (let d = ndim - 1; d >= 0; d--) {
        inFlatIdx += inCoords[d] * stride;
        stride *= shape[d];
      }

      output[i] = data[inFlatIdx];
    }

    return Tensor._fromNumberArray(Array.from(output), this._dtype).reshape(outShape);
  }

  /**
   * Set value at index.
   * tensor[5] = value, tensor[2, 3] = value
   * @pytorch tensor setitem
   */
  async set(index: number | number[], value: Tensor | number): Promise<Tensor> {
    const indices = Array.isArray(index) ? index : [index];
    return this._setCPU(indices, value);
  }

  /**
   * Internal: set value via CPU.
   */
  private async _setCPU(indices: (number | number[])[], value: Tensor | number): Promise<Tensor> {
    const data = await this.toArray();
    const shape = [...this._shape];
    const ndim = shape.length;
    const numFixed = indices.length;

    let valueData: number;
    if (value instanceof Tensor) {
      const vData = await value.toArray();
      valueData = vData[0];
    } else {
      valueData = value;
    }

    // Calculate flat index
    let flatIdx = 0;
    let stride = 1;
    for (let d = ndim - 1; d >= 0; d--) {
      const idx = d < indices.length ? (typeof indices[d] === 'number' ? indices[d] as number : 0) : 0;
      flatIdx += idx * stride;
      stride *= shape[d];
    }

    data[flatIdx] = valueData;

    return Tensor._fromNumberArray(Array.from(data), this._dtype).reshape(shape);
  }

  /**
   * Einstein summation convention for tensor contractions.
   * @param equation - Einstein summation equation (e.g., 'ij,jk->ik')
   * @param operands - Input tensors
   * @returns Contracted tensor
   * @pytorch torch.einsum
   */
  static async einsum(equation: string, ...operands: Tensor[]): Promise<Tensor> {
    // Parse equation
    const [lhs, rhs] = equation.split('->');
    const terms = lhs.split(',');

    if (terms.length !== operands.length) {
      throw new Error(`Einsum: ${terms.length} terms but ${operands.length} operands`);
    }

    // Validate all operands have correct dimensions
    for (let i = 0; i < terms.length; i++) {
      if (terms[i].length !== operands[i].dim()) {
        throw new Error(`Einsum: term '${terms[i]}' has ${terms[i].length} dims but operand ${i} has ${operands[i].dim()}`);
      }
    }

    // For now, implement simple cases via existing operations
    // Full implementation would parse and execute the contraction path

    // Simple case: 'ij->i' (sum over j)
    if (terms.length === 1 && terms[0].length === 2 && rhs.length === 1) {
      if (terms[0] === 'ij' && rhs === 'i') {
        return operands[0].sum(-1);
      }
      if (terms[0] === 'ij' && rhs === 'j') {
        return operands[0].sum(0);
      }
    }

    // Simple case: 'i,i->i' (element-wise multiply)
    if (terms.length === 2 && terms[0] === 'i' && terms[1] === 'i' && rhs === 'i') {
      return operands[0].mul(operands[1]);
    }

    // Simple case: 'ij,jk->ik' (matrix multiply)
    if (terms.length === 2 && terms[0] === 'ij' && terms[1] === 'jk' && rhs === 'ik') {
      return operands[0].matmul(operands[1]);
    }

    // Simple case: 'ij,ij->ij' (element-wise multiply)
    if (terms.length === 2 && terms[0] === 'ij' && terms[1] === 'ij' && rhs === 'ij') {
      return operands[0].mul(operands[1]);
    }

    // Simple case: 'ij->' (sum all)
    if (terms.length === 1 && terms[0] === 'ij' && rhs === '') {
      return operands[0].sum();
    }

    // For unsupported cases, throw error
    throw new Error(`Einsum equation '${equation}' not yet implemented. Supported: ij->i, ij->j, i,i->i, ij,jk->ik, ij,ij->ij, ij->`);
  }
}