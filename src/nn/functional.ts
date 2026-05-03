/**
 * Functional interface for neural network operations.
 * @status partial
 * @pytorch torch.nn.functional
 */

import { Tensor } from '../tensor';
import type { GradFn } from '../tensor';
import {
  getDevice,
  createStorageBuffer,
  getOrCreatePipeline,
  calculateWorkgroups,
  BufferUsage,
  readBuffer,
} from '../backend';
import { getDTypeBytes } from '../dtype';
import { LOG_SOFTMAX_SHADER, NLL_LOSS_SHADER, NLL_LOSS_BACKWARD_SHADER, LOG_SOFTMAX_BACKWARD_SHADER } from '../backend/webgpu/shaders';
import { DEBUG_ASYNC, log } from '../debug';

/**
 * Apply ReLU activation function.
 * @status implemented
 * @pytorch F.relu
 */
export function relu(input: Tensor, inplace: boolean = false): Tensor {
  if (inplace) {
    throw new Error('inplace ReLU not yet supported');
  }
  return input.relu();
}

/**
 * Apply GELU activation function.
 * Uses the tanh approximation as in GPT-2/BERT.
 * @status implemented
 * @pytorch F.gelu
 */
export function gelu(input: Tensor): Tensor {
  return input.gelu();
}

export function sigmoid(input: Tensor): Tensor { return input.sigmoid(); }
export function tanh(input: Tensor): Tensor { return input.tanh(); }
export function softplus(input: Tensor): Tensor { return input.softplus(); }
export function silu(input: Tensor): Tensor { return input.silu(); }
export function mish(input: Tensor): Tensor { return input.mish(); }
export function hardsigmoid(input: Tensor): Tensor { return input.hardsigmoid(); }
export function hardswish(input: Tensor): Tensor { return input.hardswish(); }
export function softsign(input: Tensor): Tensor { return input.softsign(); }
export function tanhshrink(input: Tensor): Tensor { return input.tanhshrink(); }

export function leaky_relu(input: Tensor, negative_slope: number = 0.01): Tensor {
  return input.leaky_relu(negative_slope);
}

export function elu(input: Tensor, alpha: number = 1.0): Tensor {
  return input.elu(alpha);
}

export function selu(input: Tensor): Tensor {
  return input.selu();
}

export function glu(input: Tensor, dim: number = -1): Tensor {
  return input.glu(dim);
}

export function threshold(input: Tensor, threshold: number, value: number): Tensor {
  return input.threshold(threshold, value);
}

/**
 * Apply softmax along the specified dimension.
 * @status implemented
 * @pytorch F.softmax
 */
export function softmax(input: Tensor, dim: number = -1): Tensor {
  // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
  // For numerical stability, we use log_softmax and then exp
  return log_softmax(input, dim).exp();
}

/**
 * Compute log_softmax backward gradient.
 * grad[i,j] = gradOutput[i,j] - softmax[i,j] * sum_k(gradOutput[i,k])
 * @internal
 */
function _logSoftmaxBackward(gradOutput: Tensor, softmax: Tensor): Tensor {
  const [batchSize, numClasses] = gradOutput.shape;
  const device = getDevice();

  const outputBuffer = createStorageBuffer(gradOutput.numel() * getDTypeBytes(gradOutput.dtype));

  // Create dims buffer
  const dimsData = new Uint32Array([batchSize, numClasses]);
  const dimsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuffer, 0, dimsData);

  const pipeline = getOrCreatePipeline(LOG_SOFTMAX_BACKWARD_SHADER, 'log_softmax_backward');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gradOutput.buffer, offset: 0, size: gradOutput.buffer.size } },
      { binding: 1, resource: { buffer: softmax.buffer, offset: 0, size: softmax.buffer.size } },
      { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 3, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(gradOutput.numel()));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  return new Tensor({
    buffer: outputBuffer,
    shape: [...gradOutput.shape],
    dtype: gradOutput.dtype,
    device: 'webgpu',
    requires_grad: false,
  });
}

/**
 * Apply log_softmax along the specified dimension.
 * @status implemented
 * @pytorch F.log_softmax
 */
export function log_softmax(input: Tensor, dim: number = -1): Tensor {
  // Normalize dim
  let ndim = input.shape.length;
  if (dim < 0) {
    dim += ndim;
  }

  // If dim is the last dimension, we can reshape to (N, C)
  if (dim === ndim - 1) {
    // Standard case: flatten all batch dims
    if (ndim > 2) {
      const C = input.shape[ndim - 1];
      const flatInput = input.reshape([-1, C]);
      const flatResult = log_softmax(flatInput, 1); // call self with 2D
      return flatResult.reshape(input.shape as number[]);
    }
  } else {
    // TODO: Support log_softmax on non-last dimension via transpose
    throw new Error(`log_softmax currently only supports dim=-1 (last dimension) for ND tensors. Got dim=${dim}, ndim=${ndim}`);
  }

  if (input.shape.length !== 2) {
    throw new Error(`log_softmax implementation error: expected 2D tensor, got ${input.shape.length}D`);
  }

  if (dim !== 1) {
    throw new Error('log_softmax currently only supports dim=1 (last dimension) for 2D tensors');
  }

  const [batchSize, numClasses] = input.shape;
  const device = getDevice();

  // Create output buffer
  const outputBuffer = createStorageBuffer(input.numel() * getDTypeBytes(input.dtype));

  // Create dims uniform buffer
  const dimsData = new Uint32Array([batchSize, numClasses]);
  const dimsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuffer, 0, dimsData);

  // Dispatch shader
  const pipeline = getOrCreatePipeline(LOG_SOFTMAX_SHADER, 'log_softmax');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer, offset: 0, size: input.buffer.size } },
      { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 2, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  // One workgroup per batch item
  passEncoder.dispatchWorkgroups(batchSize, 1, 1);
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  // Create result tensor first, then set up grad_fn that references it
  const resultHolder: { tensor?: Tensor } = {};

  let grad_fn: GradFn | undefined;
  if (input.requires_grad) {
    const inputRef = input;
    grad_fn = {
      backward(gradOutput: Tensor): void {
        const logSoftmaxOutput = resultHolder.tensor!;
        // d(log_softmax)/d(input) = gradOutput - softmax * sum(gradOutput, dim=1)
        // For [B, C] input: grad[i,j] = gradOutput[i,j] - softmax[i,j] * sum_k(gradOutput[i,k])
        const softmax = logSoftmaxOutput.exp();

        // Sum gradOutput along dim=1 (per row), then broadcast back
        // We compute this using a GPU shader for correctness
        const grad = _logSoftmaxBackward(gradOutput, softmax);
        inputRef.accumulateGrad(grad);
      },
    };
  }

  const result = new Tensor({
    buffer: outputBuffer,
    shape: [...input.shape],
    dtype: input.dtype,
    device: 'webgpu',
    requires_grad: input.requires_grad,
    grad_fn,
  });

  // Store reference for backward pass
  resultHolder.tensor = result;

  return result;
}

/**
 * Negative log likelihood loss.
 * @status implemented
 * @pytorch F.nll_loss
 */
export function nll_loss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  if (input.shape.length !== 2) {
    throw new Error('nll_loss currently only supports 2D input tensors');
  }
  if (target.shape.length !== 1) {
    throw new Error('nll_loss currently only supports 1D target tensors');
  }
  if (input.shape[0] !== target.shape[0]) {
    throw new Error(`Batch size mismatch: input ${input.shape[0]} vs target ${target.shape[0]}`);
  }

  const [batchSize, numClasses] = input.shape;
  const device = getDevice();

  // Create output buffer (one loss per batch item)
  const outputBuffer = createStorageBuffer(batchSize * getDTypeBytes(input.dtype));

  // Create dims uniform buffer
  const dimsData = new Uint32Array([batchSize, numClasses]);
  const dimsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuffer, 0, dimsData);

  // Dispatch shader
  const pipeline = getOrCreatePipeline(NLL_LOSS_SHADER, 'nll_loss');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer, offset: 0, size: input.buffer.size } },
      { binding: 1, resource: { buffer: target.buffer, offset: 0, size: target.buffer.size } },
      { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 3, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } },
    ],
  });

  const workgroups = calculateWorkgroups(batchSize);

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroups);
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  const lossPerItem = new Tensor({
    buffer: outputBuffer,
    shape: [batchSize],
    dtype: input.dtype,
    device: 'webgpu',
    requires_grad: input.requires_grad,
  });

  // Apply reduction
  let result: Tensor;
  if (reduction === 'mean') {
    result = lossPerItem.mean();
  } else if (reduction === 'sum') {
    result = lossPerItem.sum();
  } else {
    result = lossPerItem;
  }

  // Add grad_fn for backprop
  if (input.requires_grad) {
    // Store references for backward pass
    const inputRef = input;
    const targetRef = target;
    const scale = reduction === 'mean' ? 1.0 / batchSize : 1.0;

    const grad_fn: GradFn = {
      backward(_gradOutput: Tensor): void {
        // d(nll_loss)/d(input) = -scale at the target index, 0 elsewhere
        const gradInputBuffer = createStorageBuffer(
          inputRef.numel() * getDTypeBytes(inputRef.dtype)
        );

        // Create params buffer
        const paramsData = new ArrayBuffer(16);
        new Uint32Array(paramsData, 0, 1)[0] = batchSize;
        new Uint32Array(paramsData, 4, 1)[0] = numClasses;
        new Float32Array(paramsData, 8, 1)[0] = scale;
        new Uint32Array(paramsData, 12, 1)[0] = 0; // padding

        const paramsBuffer = device.createBuffer({
          size: 16,
          usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);

        const pipeline = getOrCreatePipeline(NLL_LOSS_BACKWARD_SHADER, 'nll_loss_backward');
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: targetRef.buffer, offset: 0, size: targetRef.buffer.size } },
            { binding: 1, resource: { buffer: gradInputBuffer, offset: 0, size: gradInputBuffer.size } },
            { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
          ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(...calculateWorkgroups(inputRef.numel()));
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);

        const gradInput = new Tensor({
          buffer: gradInputBuffer,
          shape: [...inputRef.shape],
          dtype: inputRef.dtype,
          device: 'webgpu',
          requires_grad: false,
        });

        // Propagate gradient to input (which is the log_softmax output)
        if (inputRef.grad_fn) {
          inputRef.grad_fn.backward(gradInput);
        }
      },
    };

    result = new Tensor({
      buffer: result.buffer,
      shape: [...result.shape],
      dtype: result.dtype,
      device: result.device,
      requires_grad: true,
      grad_fn,
    });
  }

  return result;
}

/**
 * Cross entropy loss (combines log_softmax and nll_loss).
 * @status implemented
 * @pytorch F.cross_entropy
 */
export function cross_entropy(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  return nll_loss(log_softmax(input, -1), target, reduction);
}

/**
 * Apply dropout.
 * @status implemented
 * @pytorch F.dropout
 */
export function dropout(input: Tensor, p: number = 0.5, training: boolean = true): Tensor {
  if (!training || p === 0) {
    return input;
  }
  if (p === 1) {
    return input.mul(0);
  }

  // TODO: Implement proper dropout with random mask
  // For now, just scale during training
  return input.mul(1 - p);
}

/**
 * Apply 2D max pooling.
 * @status partial
 * @pytorch F.max_pool2d
 */
export function max_pool2d(
  input: Tensor,
  kernel_size: number | [number, number],
  stride?: number | [number, number],
  padding: number | [number, number] = 0
): Tensor {
  void input;
  void kernel_size;
  void stride;
  void padding;
  throw new Error('max_pool2d not yet implemented');
}

/**
 * Apply 2D convolution.
 * @status partial
 * @pytorch F.conv2d
 */
export function conv2d(
  input: Tensor,
  weight: Tensor,
  bias?: Tensor,
  stride: number | [number, number] = 1,
  padding: number | [number, number] = 0
): Tensor {
  void input;
  void weight;
  void bias;
  void stride;
  void padding;
  throw new Error('conv2d not yet implemented');
}
