/**
 * Layer Normalization.
 * @status implemented
 * @pytorch torch.nn.LayerNorm
 */

import { Tensor } from '../tensor';
import { Module } from './module';
import { Parameter } from './parameter';
import { ones, zeros } from '../ops/creation';
import {
  getDevice,
  createStorageBuffer,
  getOrCreatePipeline,
  calculateWorkgroups,
  BufferUsage,
  LAYERNORM_SHADER,
} from '../backend';
import { getDTypeBytes } from '../dtype';

/**
 * Applies Layer Normalization over a mini-batch of inputs.
 * @pytorch torch.nn.LayerNorm
 */
export class LayerNorm extends Module {
  readonly normalized_shape: number[];
  readonly eps: number;
  readonly elementwise_affine: boolean;
  public weight: Parameter;
  public bias: Parameter;

  constructor(
    normalized_shape: number | number[],
    eps: number = 1e-5,
    elementwise_affine: boolean = true
  ) {
    super();
    this.normalized_shape = Array.isArray(normalized_shape)
      ? normalized_shape
      : [normalized_shape];
    this.eps = eps;
    this.elementwise_affine = elementwise_affine;

    if (this.normalized_shape.length !== 1) {
      throw new Error('LayerNorm currently only supports 1D normalized_shape');
    }

    const size = this.normalized_shape[0];

    // Initialize weight (gamma) to ones and bias (beta) to zeros
    const weightTensor = ones([size], { requires_grad: true });
    const biasTensor = zeros([size], { requires_grad: true });

    this.weight = Parameter.create(weightTensor);
    this.bias = Parameter.create(biasTensor);

    this.register_parameter('weight', this.weight);
    this.register_parameter('bias', this.bias);
  }

  forward(input: Tensor): Tensor {
    const device = getDevice();
    const shape = input.shape;
    const normalizedSize = this.normalized_shape[0];

    // Check that last dimension matches
    if (shape[shape.length - 1] !== normalizedSize) {
      throw new Error(
        `Expected last dim ${normalizedSize}, got ${shape[shape.length - 1]}`
      );
    }

    // Compute batch size (product of all dims except last)
    const batchSize = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const totalSize = input.numel();

    const outputBuffer = createStorageBuffer(totalSize * getDTypeBytes('float32'));

    // Create params buffer
    const paramsData = new ArrayBuffer(16);
    new Uint32Array(paramsData, 0, 2).set([batchSize, normalizedSize]);
    new Float32Array(paramsData, 8, 1)[0] = this.eps;
    new Uint32Array(paramsData, 12, 1)[0] = 0; // padding

    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const pipeline = getOrCreatePipeline(LAYERNORM_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: this.weight.buffer } },
        { binding: 2, resource: { buffer: this.bias.buffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(batchSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    return new Tensor({
      buffer: outputBuffer,
      shape: [...shape],
      dtype: 'float32',
      device: 'webgpu',
      requires_grad: input.requires_grad,
    });
  }
}
