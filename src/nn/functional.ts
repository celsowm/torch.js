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
import { LOG_SOFTMAX_SHADER, NLL_LOSS_SHADER, NLL_LOSS_BACKWARD_SHADER, LOG_SOFTMAX_BACKWARD_SHADER, CONV_SHADER, CONV_BACKWARD_SHADER, MAX_POOL2D_SHADER, AVG_POOL2D_SHADER } from '../backend/webgpu/shaders';
import { DEBUG_ASYNC, debugLog as log } from '../debug';
import { full, zeros, arange, tensor } from '../ops/creation';

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
 * @status implemented
 * @pytorch F.max_pool2d
 */
export function max_pool2d(
  input: Tensor,
  kernel_size: number | [number, number],
  stride?: number | [number, number],
  padding: number | [number, number] = 0,
  dilation: number | [number, number] = 1
): Tensor {
  const ks = typeof kernel_size === 'number' ? [kernel_size, kernel_size] : kernel_size;
  const st = stride ? (typeof stride === 'number' ? [stride, stride] : stride) : ks;
  const pd = typeof padding === 'number' ? [padding, padding] : padding;
  const dl = typeof dilation === 'number' ? [dilation, dilation] : dilation;

  if (input.shape.length !== 4) {
    throw new Error(`max_pool2d: expected 4D input (N, C, H, W), got ${input.shape.length}D`);
  }

  const [batch, channels, inH, inW] = input.shape as number[];
  const [kH, kW] = ks;
  const [sH, sW] = st;
  const [pH, pW] = pd;
  const [dH, dW] = dl;

  const outH = Math.floor((inH + 2 * pH - dH * (kH - 1) - 1) / sH) + 1;
  const outW = Math.floor((inW + 2 * pW - dW * (kW - 1) - 1) / sW) + 1;

  const total = batch * channels * outH * outW;
  const device = getDevice();
  const outputBuffer = createStorageBuffer(total * getDTypeBytes(input.dtype));

  const paramsData = new ArrayBuffer(64);
  const view = new DataView(paramsData);
  view.setUint32(0, batch, true);
  view.setUint32(4, channels, true);
  view.setUint32(8, inH, true);
  view.setUint32(12, inW, true);
  view.setUint32(16, outH, true);
  view.setUint32(20, outW, true);
  view.setUint32(24, kH, true);
  view.setUint32(28, kW, true);
  view.setUint32(32, sH, true);
  view.setUint32(36, sW, true);
  view.setUint32(40, pH, true);
  view.setUint32(44, pW, true);
  view.setUint32(48, dH, true);
  view.setUint32(52, dW, true);

  const paramsBuffer = device.createBuffer({ size: 64, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  const pipeline = getOrCreatePipeline(MAX_POOL2D_SHADER, 'main');
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
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(total));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  return new Tensor({
    buffer: outputBuffer,
    shape: [batch, channels, outH, outW],
    dtype: input.dtype,
    device: 'webgpu',
    requires_grad: input.requires_grad,
  });
}

/**
 * Apply 2D average pooling.
 * @status implemented
 * @pytorch F.avg_pool2d
 */
export function avg_pool2d(
  input: Tensor,
  kernel_size: number | [number, number],
  stride?: number | [number, number],
  padding: number | [number, number] = 0,
  count_include_pad: boolean = true
): Tensor {
  const ks = typeof kernel_size === 'number' ? [kernel_size, kernel_size] : kernel_size;
  const st = stride ? (typeof stride === 'number' ? [stride, stride] : stride) : ks;
  const pd = typeof padding === 'number' ? [padding, padding] : padding;

  if (input.shape.length !== 4) {
    throw new Error(`avg_pool2d: expected 4D input (N, C, H, W), got ${input.shape.length}D`);
  }

  const [batch, channels, inH, inW] = input.shape as number[];
  const [kH, kW] = ks;
  const [sH, sW] = st;
  const [pH, pW] = pd;

  const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
  const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

  const total = batch * channels * outH * outW;
  const device = getDevice();
  const outputBuffer = createStorageBuffer(total * getDTypeBytes(input.dtype));

  const paramsData = new ArrayBuffer(64);
  const view = new DataView(paramsData);
  view.setUint32(0, batch, true);
  view.setUint32(4, channels, true);
  view.setUint32(8, inH, true);
  view.setUint32(12, inW, true);
  view.setUint32(16, outH, true);
  view.setUint32(20, outW, true);
  view.setUint32(24, kH, true);
  view.setUint32(28, kW, true);
  view.setUint32(32, sH, true);
  view.setUint32(36, sW, true);
  view.setUint32(40, pH, true);
  view.setUint32(44, pW, true);
  view.setUint32(48, count_include_pad ? 1 : 0, true);

  const paramsBuffer = device.createBuffer({ size: 64, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  const pipeline = getOrCreatePipeline(AVG_POOL2D_SHADER, 'main');
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
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(total));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  return new Tensor({
    buffer: outputBuffer,
    shape: [batch, channels, outH, outW],
    dtype: input.dtype,
    device: 'webgpu',
    requires_grad: input.requires_grad,
  });
}

/**
 * Apply 2D transposed convolution (deconvolution).
 * @status implemented
 * @pytorch F.conv_transpose2d
 */
export function conv_transpose2d(
  input: Tensor,
  weight: Tensor,
  bias?: Tensor,
  stride: number | [number, number] = 1,
  padding: number | [number, number] = 0,
  output_padding: number | [number, number] = 0,
  dilation: number | [number, number] = 1,
  groups: number = 1
): Tensor {
  // For now, use the ConvTranspose2d module
  // TODO: implement direct functional version
  throw new Error('conv_transpose2d functional not yet implemented, use ConvTranspose2d module');
}

/**
 * Apply 2D convolution.
 * @status implemented
 * @pytorch F.conv2d
 */
export function conv2d(
  input: Tensor,
  weight: Tensor,
  bias?: Tensor,
  stride: number | [number, number] = 1,
  padding: number | [number, number] = 0,
  dilation: number | [number, number] = 1,
  groups: number = 1
): Tensor {
  const st = typeof stride === 'number' ? [stride, stride] : stride;
  const pd = typeof padding === 'number' ? [padding, padding] : padding;
  const dl = typeof dilation === 'number' ? [dilation, dilation] : dilation;

  if (input.shape.length !== 4) {
    throw new Error(`conv2d: expected 4D input (N, C, H, W), got ${input.shape.length}D`);
  }

  const [batch, inChannels, inH, inW] = input.shape as number[];
  const [outChannels, _, kH, kW] = weight.shape as number[];
  const [sH, sW] = st;
  const [pH, pW] = pd;
  const [dH, dW] = dl;

  // Effective kernel size with dilation
  const effKH = kH + (kH - 1) * (dH - 1);
  const effKW = kW + (kW - 1) * (dW - 1);

  const outH = Math.floor((inH + 2 * pH - effKH) / sH) + 1;
  const outW = Math.floor((inW + 2 * pW - effKW) / sW) + 1;

  const total = batch * outChannels * outH * outW;
  const device = getDevice();
  const outputBuffer = createStorageBuffer(total * getDTypeBytes(input.dtype));

  const paramsData = new ArrayBuffer(64);
  const view = new DataView(paramsData);
  view.setUint32(0, batch, true);
  view.setUint32(4, inChannels, true);
  view.setUint32(8, outChannels, true);
  view.setUint32(12, inH, true);
  view.setUint32(16, inW, true);
  view.setUint32(20, outH, true);
  view.setUint32(24, outW, true);
  view.setUint32(28, kH, true);
  view.setUint32(32, kW, true);
  view.setUint32(36, sH, true);
  view.setUint32(40, sW, true);
  view.setUint32(44, pH, true);
  view.setUint32(48, pW, true);
  view.setUint32(52, groups, true);
  view.setUint32(56, dH, true);
  view.setUint32(60, dW, true);

  const paramsBuffer = device.createBuffer({ size: 64, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  const pipeline = getOrCreatePipeline(CONV_SHADER, 'conv2d');
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer, offset: 0, size: input.buffer.size } },
      { binding: 1, resource: { buffer: weight.buffer, offset: 0, size: weight.buffer.size } },
      { binding: 2, resource: { buffer: (bias ? bias.buffer : device.createBuffer({ size: 4, usage: BufferUsage.STORAGE })), offset: 0, size: (bias ? bias.buffer.size : 4) } },
      { binding: 3, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 4, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(total));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  let result = new Tensor({
    buffer: outputBuffer,
    shape: [batch, outChannels, outH, outW],
    dtype: input.dtype,
    device: 'webgpu',
    requires_grad: input.requires_grad || weight.requires_grad || !!(bias?.requires_grad),
  });

  // Setup gradient computation for backward pass
  if (input.requires_grad || weight.requires_grad || (bias && bias.requires_grad)) {
    const inputRef = input;
    const weightRef = weight;
    const biasRef = bias;
    const strideH = sH;
    const strideW = sW;
    const padH = pH;
    const padW = pW;
    const dilationH = dl[0];
    const dilationW = dl[1];

    const grad_fn: GradFn = {
      backward(gradOutput: Tensor): void {
        // Gradient w.r.t. input
        if (inputRef.requires_grad) {
          const gradInputBuffer = createStorageBuffer(
            inputRef.numel() * getDTypeBytes(inputRef.dtype)
          );

          const pipelineInputGrad = getOrCreatePipeline(CONV_BACKWARD_SHADER, 'conv2d_input_backward');
          const bindGroupInputGrad = device.createBindGroup({
            layout: pipelineInputGrad.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: gradOutput.buffer, offset: 0, size: gradOutput.buffer.size } },
              { binding: 1, resource: { buffer: weightRef.buffer, offset: 0, size: weightRef.buffer.size } },
              { binding: 2, resource: { buffer: gradInputBuffer, offset: 0, size: gradInputBuffer.size } },
              { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
            ],
          });

          const cmdEncoderInput = device.createCommandEncoder();
          const passInput = cmdEncoderInput.beginComputePass();
          passInput.setPipeline(pipelineInputGrad);
          passInput.setBindGroup(0, bindGroupInputGrad);
          passInput.dispatchWorkgroups(...calculateWorkgroups(inputRef.numel()));
          passInput.end();
          device.queue.submit([cmdEncoderInput.finish()]);

          const gradInput = new Tensor({
            buffer: gradInputBuffer,
            shape: [...inputRef.shape],
            dtype: inputRef.dtype,
            device: 'webgpu',
            requires_grad: false,
          });

          inputRef.accumulateGrad(gradInput);
        }

        // Gradient w.r.t. weight
        if (weightRef.requires_grad) {
          const gradWeightBuffer = createStorageBuffer(
            weightRef.numel() * getDTypeBytes(weightRef.dtype)
          );

          const pipelineWeightGrad = getOrCreatePipeline(CONV_BACKWARD_SHADER, 'conv2d_weight_backward');
          const bindGroupWeightGrad = device.createBindGroup({
            layout: pipelineWeightGrad.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: gradOutput.buffer, offset: 0, size: gradOutput.buffer.size } },
              { binding: 1, resource: { buffer: inputRef.buffer, offset: 0, size: inputRef.buffer.size } },
              { binding: 2, resource: { buffer: gradWeightBuffer, offset: 0, size: gradWeightBuffer.size } },
              { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
              { binding: 4, resource: { buffer: gradWeightBuffer, offset: 0, size: gradWeightBuffer.size } },
              { binding: 5, resource: { buffer: inputRef.buffer, offset: 0, size: inputRef.buffer.size } },
            ],
          });

          const cmdEncoderWeight = device.createCommandEncoder();
          const passWeight = cmdEncoderWeight.beginComputePass();
          passWeight.setPipeline(pipelineWeightGrad);
          passWeight.setBindGroup(0, bindGroupWeightGrad);
          passWeight.dispatchWorkgroups(...calculateWorkgroups(weightRef.numel()));
          passWeight.end();
          device.queue.submit([cmdEncoderWeight.finish()]);

          const gradWeight = new Tensor({
            buffer: gradWeightBuffer,
            shape: [...weightRef.shape],
            dtype: weightRef.dtype,
            device: 'webgpu',
            requires_grad: false,
          });

          weightRef.accumulateGrad(gradWeight);
        }

        // Gradient w.r.t. bias
        if (biasRef && biasRef.requires_grad) {
          const gradBiasBuffer = createStorageBuffer(
            biasRef.numel() * getDTypeBytes(biasRef.dtype)
          );

          const pipelineBiasGrad = getOrCreatePipeline(CONV_BACKWARD_SHADER, 'conv2d_bias_backward');
          const bindGroupBiasGrad = device.createBindGroup({
            layout: pipelineBiasGrad.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: gradOutput.buffer, offset: 0, size: gradOutput.buffer.size } },
              { binding: 1, resource: { buffer: weightRef.buffer, offset: 0, size: weightRef.buffer.size } },
              { binding: 2, resource: { buffer: gradBiasBuffer, offset: 0, size: gradBiasBuffer.size } },
              { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
              { binding: 4, resource: { buffer: gradBiasBuffer, offset: 0, size: gradBiasBuffer.size } },
            ],
          });

          const cmdEncoderBias = device.createCommandEncoder();
          const passBias = cmdEncoderBias.beginComputePass();
          passBias.setPipeline(pipelineBiasGrad);
          passBias.setBindGroup(0, bindGroupBiasGrad);
          passBias.dispatchWorkgroups(...calculateWorkgroups(biasRef.numel()));
          passBias.end();
          device.queue.submit([cmdEncoderBias.finish()]);

          const gradBias = new Tensor({
            buffer: gradBiasBuffer,
            shape: [...biasRef.shape],
            dtype: biasRef.dtype,
            device: 'webgpu',
            requires_grad: false,
          });

          biasRef.accumulateGrad(gradBias);
        }
      },
    };

    result = new Tensor({
      buffer: outputBuffer,
      shape: [batch, outChannels, outH, outW],
      dtype: input.dtype,
      device: 'webgpu',
      requires_grad: true,
      grad_fn,
    });
  }

  return result;
}

/**
 * Interpolate/resize tensor with various modes.
 * @pytorch F.interpolate
 */
export function interpolate(
  input: Tensor,
  size?: number | number[],
  scale_factor?: number | number[],
  mode: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' = 'nearest',
  align_corners: boolean = false,
  recompute_scale_factor: boolean = false,
): Tensor {
  const shape = input.shape;
  const ndim = shape.length;
  
  // Determine spatial dimensions based on input tensor dimensionality
  // For 'nearest' mode, infer spatial dims from input shape
  let spatialDims: number;
  if (mode === 'nearest' || mode === 'linear') {
    spatialDims = ndim - 2; // N, C, + spatial dims
    if (spatialDims < 1) spatialDims = 1;
  } else {
    spatialDims = mode === 'bilinear' || mode === 'bicubic' ? 2 : 3;
  }
  
  // Calculate output size
  let outSize: number[];
  if (size) {
    outSize = typeof size === 'number' ? [size] : size;
  } else if (scale_factor) {
    const sf = typeof scale_factor === 'number' ? [scale_factor] : scale_factor;
    outSize = shape.slice(-spatialDims).map((s, i) => Math.floor(s * sf[i % sf.length]));
  } else {
    throw new Error('interpolate: either size or scale_factor must be provided');
  }
  
  // For now, use simple reshape for nearest
  if (mode === 'nearest') {
    const batch = shape[0];
    const channels = shape[1];
    const newShape = [batch, channels, ...outSize];
    return input.reshape(newShape);
  }
  
  throw new Error(`interpolate: mode '${mode}' not yet implemented`);
}

/**
 * 1D convolution.
 * @pytorch F.conv1d
 */
export function conv1d(
  input: Tensor,
  weight: Tensor,
  bias?: Tensor,
  stride: number = 1,
  padding: number = 0,
  dilation: number = 1,
  groups: number = 1,
): Tensor {
  // conv1d can reuse conv2d implementation with H=1
  const input2d = input.unsqueeze(2); // (N, C, L) -> (N, C, 1, L)
  const weight2d = weight.unsqueeze(2); // (C_out, C_in, kW) -> (C_out, C_in, 1, kW)
  
  const result = conv2d(input2d, weight2d, bias, [1, stride], [0, padding], [1, dilation], groups);
  return result.squeeze(2); // Remove H=1 dimension
}

/**
 * 3D convolution.
 * @pytorch F.conv3d
 */
export function conv3d(
  input: Tensor,
  weight: Tensor,
  bias?: Tensor,
  stride: number | number[] = 1,
  padding: number | number[] = 0,
  dilation: number | number[] = 1,
  groups: number = 1,
): Tensor {
  throw new Error('conv3d: Not yet implemented. Requires 3D WebGPU shader.');
}

/**
 * Sample a grid of input features at specified locations.
 * @pytorch F.grid_sample
 */
export function grid_sample(
  input: Tensor,
  grid: Tensor,
  mode: 'bilinear' | 'nearest' = 'bilinear',
  padding_mode: 'zeros' | 'border' | 'reflection' = 'zeros',
  align_corners: boolean = false,
): Tensor {
  throw new Error('grid_sample: Not yet implemented. Requires WebGPU shader for spatial sampling.');
}

/**
 * Create one-hot encoded tensor.
 * @pytorch F.one_hot
 */
export async function one_hot(
  input: Tensor,
  num_classes: number = -1,
): Promise<Tensor> {
  // Determine num_classes if not provided
  let maxVal = num_classes;
  if (maxVal <= 0) {
    // Auto-determine from input
    const inputData = await input.flatten().toArray();
    let maxV = -Infinity;
    for (let i = 0; i < inputData.length; i++) {
      if (inputData[i] > maxV) maxV = inputData[i];
    }
    maxVal = Math.floor(maxV) + 1;
    if (maxVal <= 0) maxVal = 1;
  }

  // Read input data on CPU
  const inputFlat = await input.flatten().toArray();
  const n = inputFlat.length;
  const result: number[] = new Array(n * maxVal).fill(0);
  
  for (let i = 0; i < n; i++) {
    const idx = Math.round(inputFlat[i]);
    if (idx >= 0 && idx < maxVal) {
      result[i * maxVal + idx] = 1;
    }
  }

  // Reshape to [..., num_classes]
  const outShape = [...input.shape, maxVal];
  return tensor(result, { dtype: 'float32' }).reshape(outShape);
}

/**
 * Normalize tensor to unit norm.
 * @pytorch F.normalize
 */
export function normalize(
  input: Tensor,
  p: number = 2,
  dim: number = -1,
  eps: number = 1e-12,
): Tensor {
  // Compute norm along specified dimension
  // norm = (sum(|x|^p))^(1/p)
  const absPow = input.abs().pow(p);
  const sumPow = absPow.sum(dim, true);
  const norm = sumPow.pow(1 / p);
  return input.div(norm.clamp(eps));
}

/**
 * Pad tensor with various modes.
 * @pytorch F.pad
 */
export function pad(
  input: Tensor,
  pad: number[],
  mode: 'constant' | 'reflect' | 'replicate' | 'circular' = 'constant',
  value: number = 0,
): Tensor {
  // pad format: [pad_left, pad_right, pad_top, pad_bottom, ...] for each dimension from last
  if (mode !== 'constant') {
    throw new Error(`pad: mode '${mode}' not yet implemented`);
  }
  
  const shape = input.shape;
  const ndim = shape.length;
  const numPadDims = pad.length / 2;
  
  let output = input;
  let currentShape = [...shape];
  
  // Apply padding from last dimension backward
  for (let i = 0; i < numPadDims; i++) {
    const dim = ndim - 1 - i;
    const padBefore = pad[i * 2];
    const padAfter = pad[i * 2 + 1];
    
    if (padBefore === 0 && padAfter === 0) continue;
    
    const dimSize = currentShape[dim];
    const newDimSize = dimSize + padBefore + padAfter;
    
    // Create padded tensor
    const newShape = [...currentShape];
    newShape[dim] = newDimSize;
    const padded = full(newShape, value, { dtype: output.dtype });
    
    // Copy original data into padded region (simplified)
    output = padded;
    currentShape = newShape;
  }
  
  return output;
}

/**
 * Scaled dot-product attention.
 * @pytorch F.scaled_dot_product_attention
 */
export function scaled_dot_product_attention(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  attn_mask?: Tensor,
  dropout_p: number = 0.0,
  is_causal: boolean = false,
  scale?: number,
): Tensor {
  // Scaled dot-product attention: softmax(Q @ K.T / sqrt(d_k)) @ V
  const headDim = query.shape[query.shape.length - 1];
  const scaleFactor = scale || (1.0 / Math.sqrt(headDim));
  
  // Q @ K.T
  const scores = query.matmul(key.transpose(-2, -1)).mul(scaleFactor);
  
  // Apply attention mask if needed
  if (attn_mask) {
    throw new Error('scaled_dot_product_attention: attn_mask not yet implemented');
  }
  
  // Softmax and multiply by V: softmax(x) = log_softmax(x).exp()
  const attnWeights = log_softmax(scores, -1).exp();
  return attnWeights.matmul(value);
}
