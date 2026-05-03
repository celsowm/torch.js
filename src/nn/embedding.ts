/**
 * Embedding layer for token/position embeddings.
 * @status implemented
 * @pytorch torch.nn.Embedding
 */

import { Tensor } from '../tensor';
import { Module } from './module';
import { Parameter } from './parameter';
import { randn } from '../ops/creation';
import {
  getDevice,
  createStorageBuffer,
  getOrCreatePipeline,
  calculateWorkgroups,
  BufferUsage,
  EMBEDDING_SHADER,
} from '../backend';
import { getDTypeBytes } from '../dtype';

/**
 * A lookup table that stores embeddings of a fixed dictionary and size.
 * @pytorch torch.nn.Embedding
 */
export class Embedding extends Module {
  readonly num_embeddings: number;
  readonly embedding_dim: number;
  public weight: Parameter;

  constructor(num_embeddings: number, embedding_dim: number) {
    super();
    this.num_embeddings = num_embeddings;
    this.embedding_dim = embedding_dim;

    // Initialize weight with normal distribution (std=1.0)
    const weight = randn([num_embeddings, embedding_dim], { requires_grad: true });
    this.weight = Parameter.create(weight);
    this.register_parameter('weight', this.weight);
  }

  /**
   * Look up embeddings for the given indices.
   * @param input - Tensor of indices with shape [...], dtype int32
   * @returns Tensor of embeddings with shape [..., embedding_dim]
   */
  forward(input: Tensor): Tensor {
    const device = getDevice();
    const inputShape = input.shape;
    const numIndices = input.numel();

    // Output shape: input_shape + [embedding_dim]
    const outputShape = [...inputShape, this.embedding_dim];
    const outputSize = numIndices * this.embedding_dim;

    const outputBuffer = createStorageBuffer(outputSize * getDTypeBytes('float32'));

    // Create params buffer
    const paramsData = new Uint32Array([
      this.num_embeddings,
      this.embedding_dim,
      numIndices,
      0, // padding
    ]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const pipeline = getOrCreatePipeline(EMBEDDING_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weight.buffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    const result = new Tensor({
      buffer: outputBuffer,
      shape: outputShape,
      dtype: 'float32',
      device: 'webgpu',
      requires_grad: this.weight.requires_grad,
    });

    // TODO: Add backward pass for embedding gradients

    return result;
  }
}
