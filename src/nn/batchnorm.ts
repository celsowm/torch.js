import { Tensor } from '../tensor';
import { zeros, ones } from '../ops/creation';
import {
  getDevice,
  createStorageBuffer,
  getOrCreatePipeline,
  calculateWorkgroups,
  BufferUsage,
  BATCHNORM_SHADER,
} from '../backend';
import { getDTypeBytes } from '../dtype';
import { Module } from './module';
import { Parameter } from './parameter';

function _computeBatchStats(input: Tensor, channels: number, spatialSize: number): [Tensor, Tensor] {
  const batchSize = input.shape[0];
  const totalPerChannel = batchSize * spatialSize;
  // Reshape to [batch * spatial, channels] and compute mean/var per channel
  const flat = input.reshape([-1, channels]);
  const mean = flat.mean(0); // [channels]
  const centered = flat.sub(mean);
  const var_ = centered.pow(2).sum(0).div(totalPerChannel); // [channels]
  return [mean, var_];
}

/**
 * Base batch normalization class.
 * @pytorch torch.nn.BatchNorm1d, torch.nn.BatchNorm2d
 */
class _BatchNorm extends Module {
  public weight: Parameter;
  public bias: Parameter | null;
  public running_mean: Tensor;
  public running_var: Tensor;

  protected num_features: number;
  protected eps: number;
  protected momentum: number;
  protected track_running_stats: boolean;
  private _is_1d: boolean;

  constructor(
    num_features: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true,
    is_1d: boolean = true
  ) {
    super();
    this.num_features = num_features;
    this.eps = eps;
    this.momentum = momentum;
    this.track_running_stats = track_running_stats;
    this._is_1d = is_1d;

    if (affine) {
      const weightData = ones([num_features]);
      this.weight = Parameter.create(weightData);
      this.register_parameter('weight', this.weight);

      const biasData = zeros([num_features]);
      this.bias = Parameter.create(biasData);
      this.register_parameter('bias', this.bias);
    } else {
      this.weight = null as any;
      this.bias = null;
    }

    this.running_mean = zeros([num_features]);
    this.running_var = ones([num_features]);
    this.register_buffer('running_mean', this.running_mean);
    this.register_buffer('running_var', this.running_var);
  }

  forward(input: Tensor): Tensor {
    const ndim = input.shape.length;
    const is1d = this._is_1d;

    if (is1d && ndim !== 2 && ndim !== 3) {
      throw new Error(`BatchNorm1d: expected 2D or 3D input, got ${ndim}D`);
    }
    if (!is1d && ndim !== 4) {
      throw new Error(`BatchNorm2d: expected 4D input, got ${ndim}D`);
    }

    let inp = input;
    let spatialSize: number;

    if (is1d && ndim === 2) {
      inp = input.unsqueeze(2);
      spatialSize = 1;
    } else if (is1d && ndim === 3) {
      spatialSize = input.shape[2] as number;
    } else {
      const [, , H, W] = input.shape as number[];
      spatialSize = (H as number) * (W as number);
    }

    const batchSize = inp.shape[0] as number;
    const channels = is1d ? (ndim === 2 ? input.shape[1] as number : input.shape[1] as number) : input.shape[1] as number;

    // Compute mean and variance
    let mean: Tensor;
    let variance: Tensor;

    if (this.training && this.track_running_stats) {
      [mean, variance] = _computeBatchStats(inp, channels, spatialSize);

      // Update running stats
      this.running_mean = this.running_mean.mul(1 - this.momentum).add(mean.mul(this.momentum));
      this.running_var = this.running_var.mul(1 - this.momentum).add(variance.mul(this.momentum));
    } else if (this.training) {
      [mean, variance] = _computeBatchStats(inp, channels, spatialSize);
    } else {
      mean = this.running_mean;
      variance = this.running_var;
    }

    // Apply normalization via GPU shader
    const total = batchSize * channels * spatialSize;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(total * getDTypeBytes(inp.dtype));

    const invStd = variance.add(this.eps).rsqrt();

    const paramsData = new ArrayBuffer(32);
    const view = new DataView(paramsData);
    view.setUint32(0, batchSize, true);
    view.setUint32(4, channels, true);
    view.setUint32(8, spatialSize, true);
    view.setFloat32(12, this.eps, true);

    const paramsBuffer = device.createBuffer({ size: 32, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const w = this.weight ?? ones([channels]);
    const b = this.bias ?? zeros([channels]);

    const pipeline = getOrCreatePipeline(BATCHNORM_SHADER, 'main');
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inp.buffer, offset: 0, size: inp.buffer.size } },
        { binding: 1, resource: { buffer: w.buffer, offset: 0, size: w.buffer.size } },
        { binding: 2, resource: { buffer: b.buffer, offset: 0, size: b.buffer.size } },
        { binding: 3, resource: { buffer: mean.buffer, offset: 0, size: mean.buffer.size } },
        { binding: 4, resource: { buffer: invStd.buffer, offset: 0, size: invStd.buffer.size } },
        { binding: 5, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 6, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(total));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    const result = new Tensor({
      buffer: outputBuffer,
      shape: [...inp.shape],
      dtype: inp.dtype,
      device: 'webgpu',
      requires_grad: inp.requires_grad,
    });

    // Restore original shape for 2D input to BatchNorm1d
    if (is1d && ndim === 2) {
      return result.squeeze(2);
    }

    return result;
  }
}

/**
 * 1D batch normalization layer.
 * @pytorch torch.nn.BatchNorm1d
 */
export class BatchNorm1d extends _BatchNorm {
  constructor(
    num_features: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super(num_features, eps, momentum, affine, track_running_stats, true);
  }
}

/**
 * 2D batch normalization layer.
 * @pytorch torch.nn.BatchNorm2d
 */
export class BatchNorm2d extends _BatchNorm {
  constructor(
    num_features: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super(num_features, eps, momentum, affine, track_running_stats, false);
  }
}
