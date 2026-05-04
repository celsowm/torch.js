/**
 * Instance Normalization, Group Normalization, and RMS Normalization.
 * @status implemented
 * @pytorch torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
 *          torch.nn.GroupNorm, torch.nn.RMSNorm
 */

import { Tensor } from '../tensor';
import { Module } from './module';
import { Parameter } from './parameter';
import { ones, zeros } from '../ops/creation';

/**
 * Base class for Instance Normalization layers.
 *
 * InstanceNorm computes mean and variance over spatial dimensions for each
 * sample and channel independently:
 *   y = (x - mean) / sqrt(var + eps) * weight + bias
 *
 * Input shape: (N, C, ...) where ... are spatial dimensions.
 * @pytorch torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d
 */
class _InstanceNorm extends Module {
  readonly num_features: number;
  readonly eps: number;
  readonly affine: boolean;
  readonly spatial_dims: number;
  public weight: Parameter;
  public bias: Parameter;

  constructor(
    num_features: number,
    spatial_dims: number,
    eps: number = 1e-5,
    affine: boolean = false
  ) {
    super();
    this.num_features = num_features;
    this.spatial_dims = spatial_dims;
    this.eps = eps;
    this.affine = affine;

    // PyTorch initializes weight to ones, bias to zeros
    const weightData = ones([num_features], { requires_grad: true });
    this.weight = Parameter.create(weightData);
    this.register_parameter('weight', this.weight);

    const biasData = zeros([num_features], { requires_grad: true });
    this.bias = Parameter.create(biasData);
    this.register_parameter('bias', this.bias);
  }

  forward(input: Tensor): Tensor {
    const ndim = input.shape.length;
    const expectedNdim = this.spatial_dims + 2; // +2 for N and C

    if (ndim !== expectedNdim) {
      throw new Error(
        `InstanceNorm${this.spatial_dims}d: expected ${expectedNdim}D input, got ${ndim}D input with shape ${input.shape}`
      );
    }

    const batchSize = input.shape[0] as number;
    const numChannels = input.shape[1] as number;

    if (numChannels !== this.num_features) {
      throw new Error(
        `InstanceNorm${this.spatial_dims}d: expected num_features=${this.num_features}, got ${numChannels}`
      );
    }

    // Spatial dimension indices: 2, 3, ..., spatial_dims+1
    const spatialDims: number[] = [];
    for (let i = 0; i < this.spatial_dims; i++) {
      spatialDims.push(i + 2);
    }

    // Reshape to (N * C, spatial_elements) to compute mean/var per instance
    const spatialSize = spatialDims.reduce((prod, d) => prod * (input.shape[d] as number), 1);

    // Reshape: (N, C, spatial...) -> (N * C, spatial_elements)
    const reshaped = input.reshape([batchSize * numChannels, spatialSize]);

    // Compute mean and var over spatial elements for each (n, c) pair
    const mean = reshaped.mean(1, true); // [N*C, 1]
    const var_ = reshaped.var(1, true, false); // [N*C, 1], biased (population) variance

    // Normalize
    const normalized = reshaped.sub(mean).mul(var_.add(this.eps).rsqrt());

    // Reshape back to original shape
    const normalizedReshaped = normalized.reshape(input.shape);

    // Apply weight and bias if affine
    // weight and bias have shape [C], need to broadcast over (N, C, H, W, ...)
    if (this.affine) {
      // Reshape weight/bias to [1, C, 1, 1, ...] for broadcasting
      const broadcastShape = [1, numChannels];
      for (let i = 0; i < this.spatial_dims; i++) {
        broadcastShape.push(1);
      }

      const w = this.weight.reshape(broadcastShape);
      const b = this.bias.reshape(broadcastShape);

      return normalizedReshaped.mul(w).add(b);
    }

    return normalizedReshaped;
  }
}

/**
 * Applies Instance Normalization over a 3D input (N, C, L).
 * @pytorch torch.nn.InstanceNorm1d
 */
export class InstanceNorm1d extends _InstanceNorm {
  constructor(
    num_features: number,
    eps: number = 1e-5,
    affine: boolean = false
  ) {
    super(num_features, 1, eps, affine);
  }
}

/**
 * Applies Instance Normalization over a 4D input (N, C, H, W).
 * @pytorch torch.nn.InstanceNorm2d
 */
export class InstanceNorm2d extends _InstanceNorm {
  constructor(
    num_features: number,
    eps: number = 1e-5,
    affine: boolean = false
  ) {
    super(num_features, 2, eps, affine);
  }
}

/**
 * Applies Instance Normalization over a 5D input (N, C, D, H, W).
 * @pytorch torch.nn.InstanceNorm3d
 */
export class InstanceNorm3d extends _InstanceNorm {
  constructor(
    num_features: number,
    eps: number = 1e-5,
    affine: boolean = false
  ) {
    super(num_features, 3, eps, affine);
  }
}

/**
 * Applies Group Normalization over an input tensor.
 *
 * Divides channels into `num_groups` groups and normalizes over each group's
 * channels and spatial dimensions.
 *
 * Input shape: (N, C, ...) where C must be divisible by num_groups.
 * @pytorch torch.nn.GroupNorm
 */
export class GroupNorm extends Module {
  readonly num_groups: number;
  readonly num_channels: number;
  readonly eps: number;
  readonly affine: boolean;
  public weight: Parameter;
  public bias: Parameter;

  constructor(
    num_groups: number,
    num_channels: number,
    eps: number = 1e-5,
    affine: boolean = true
  ) {
    super();

    if (num_channels % num_groups !== 0) {
      throw new Error(
        `GroupNorm: num_channels (${num_channels}) must be divisible by num_groups (${num_groups})`
      );
    }

    this.num_groups = num_groups;
    this.num_channels = num_channels;
    this.eps = eps;
    this.affine = affine;

    // PyTorch initializes weight to ones, bias to zeros
    const weightData = ones([num_channels], { requires_grad: true });
    this.weight = Parameter.create(weightData);
    this.register_parameter('weight', this.weight);

    const biasData = zeros([num_channels], { requires_grad: true });
    this.bias = Parameter.create(biasData);
    this.register_parameter('bias', this.bias);
  }

  forward(input: Tensor): Tensor {
    const ndim = input.shape.length;

    if (ndim < 2) {
      throw new Error(
        `GroupNorm: expected at least 2D input, got ${ndim}D input with shape ${input.shape}`
      );
    }

    const batchSize = input.shape[0] as number;
    const numChannels = input.shape[1] as number;

    if (numChannels !== this.num_channels) {
      throw new Error(
        `GroupNorm: expected num_channels=${this.num_channels}, got ${numChannels}`
      );
    }

    const channelsPerGroup = this.num_channels / this.num_groups;
    const spatialDims = ndim - 2;
    const spatialSize = input.shape.slice(2).reduce((prod, d) => prod * (d as number), 1);

    // Process each group independently
    const groups: Tensor[] = [];

    for (let g = 0; g < this.num_groups; g++) {
      const startChannel = g * channelsPerGroup;

      // Slice out the group's channels using narrow
      const groupChannels = input.narrow(1, startChannel, channelsPerGroup);
      const reshaped = groupChannels.reshape([batchSize, channelsPerGroup * spatialSize]);

      // Compute mean and var over the group's dimensions
      const mean = reshaped.mean(1, true); // [N, 1]
      const var_ = reshaped.var(1, true, false); // [N, 1], biased variance

      // Normalize
      const normalized = reshaped.sub(mean).mul(var_.add(this.eps).rsqrt());

      // Reshape back to group shape
      const groupShape = [batchSize, channelsPerGroup];
      for (let i = 0; i < spatialDims; i++) {
        groupShape.push(input.shape[2 + i] as number);
      }
      groups.push(normalized.reshape(groupShape));
    }

    // Concatenate groups along channel dimension
    let normalized = Tensor.cat(groups, 1);

    // Apply weight and bias if affine
    if (this.affine) {
      // Reshape weight/bias to [1, C, 1, 1, ...] for broadcasting
      const broadcastShape = [1, numChannels];
      for (let i = 0; i < spatialDims; i++) {
        broadcastShape.push(1);
      }

      const w = this.weight.reshape(broadcastShape);
      const b = this.bias.reshape(broadcastShape);

      normalized = normalized.mul(w).add(b);
    }

    return normalized;
  }
}

/**
 * Applies Root Mean Square (RMS) Normalization over an input tensor.
 *
 * RMSNorm is a simplified LayerNorm that only uses the root mean square
 * of the input, without subtracting the mean:
 *   y = x / sqrt(mean(x^2) + eps) * weight
 *
 * Normalizes over the last dimension only.
 * Input shape: any shape, normalized over last dim.
 * @pytorch torch.nn.RMSNorm
 */
export class RMSNorm extends Module {
  readonly normalized_shape: number[];
  readonly eps: number;
  public weight: Parameter;

  constructor(
    normalized_shape: number | number[],
    eps: number = 1e-5
  ) {
    super();
    this.normalized_shape = Array.isArray(normalized_shape)
      ? normalized_shape
      : [normalized_shape];
    this.eps = eps;

    if (this.normalized_shape.length !== 1) {
      throw new Error('RMSNorm currently only supports 1D normalized_shape');
    }

    const size = this.normalized_shape[0];

    // PyTorch initializes weight to ones
    const weightData = ones([size], { requires_grad: true });
    this.weight = Parameter.create(weightData);
    this.register_parameter('weight', this.weight);

    // RMSNorm has no bias
    this.register_parameter('bias', null);
  }

  forward(input: Tensor): Tensor {
    const shape = input.shape;
    const normalizedSize = this.normalized_shape[0];

    // Check that last dimension matches
    if (shape[shape.length - 1] !== normalizedSize) {
      throw new Error(
        `RMSNorm: expected last dim ${normalizedSize}, got ${shape[shape.length - 1]}`
      );
    }

    // Compute RMS: sqrt(mean(x^2) + eps)
    const squared = input.pow(2);
    const meanSquared = squared.mean(-1, true); // keep last dim
    const rms = meanSquared.add(this.eps).sqrt();

    // Normalize: x / rms * weight
    const normalized = input.div(rms);

    // Apply weight (reshape for broadcasting: [1, 1, ..., C])
    const broadcastShape: number[] = [];
    for (let i = 0; i < shape.length - 1; i++) {
      broadcastShape.push(1);
    }
    broadcastShape.push(normalizedSize);

    const w = this.weight.reshape(broadcastShape);

    return normalized.mul(w);
  }
}

/**
 * Synchronized Batch Normalization.
 *
 * In a full multi-GPU implementation, this would synchronize batch statistics
 * across multiple GPUs. This simplified single-GPU version behaves identically
 * to BatchNorm but is structured to allow future multi-GPU support.
 *
 * @pytorch torch.nn.SyncBatchNorm
 */
class _SyncBatchNorm extends Module {
  public weight: Parameter;
  public bias: Parameter | null;
  public running_mean: Tensor;
  public running_var: Tensor;
  readonly process_group: number | null;

  protected num_features: number;
  protected eps: number;
  protected momentum: number;
  protected track_running_stats: boolean;
  protected spatial_dims: number;

  constructor(
    num_features: number,
    spatial_dims: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true,
    process_group: number | null = null
  ) {
    super();
    this.num_features = num_features;
    this.spatial_dims = spatial_dims;
    this.eps = eps;
    this.momentum = momentum;
    this.track_running_stats = track_running_stats;
    this.process_group = process_group;

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
    const expectedNdim = this.spatial_dims + 2;
    const ndim = input.shape.length;

    if (ndim !== expectedNdim) {
      throw new Error(
        `SyncBatchNorm${this.spatial_dims}d: expected ${expectedNdim}D input, got ${ndim}D`
      );
    }

    const batchSize = input.shape[0] as number;
    const channels = input.shape[1] as number;

    if (channels !== this.num_features) {
      throw new Error(
        `SyncBatchNorm${this.spatial_dims}d: expected ${this.num_features} channels, got ${channels}`
      );
    }

    // Compute spatial size
    const spatialSize = input.shape.slice(2).reduce((prod, d) => prod * (d as number), 1);

    // Reshape to (N, C, spatial) -> compute stats over N and spatial
    const reshaped = input.reshape([batchSize, channels, spatialSize]);
    const mean = reshaped.mean([0, 2], true).reshape([channels]);
    const var_ = reshaped.sub(mean.reshape([1, channels, 1])).pow(2).mean([0, 2], false).reshape([channels]);

    let normMean: Tensor;
    let normVar: Tensor;

    if (this.training && this.track_running_stats) {
      normMean = mean;
      normVar = var_;

      // Update running stats
      this.running_mean = this.running_mean.mul(1 - this.momentum).add(mean.mul(this.momentum));
      this.running_var = this.running_var.mul(1 - this.momentum).add(var_.mul(this.momentum));
    } else if (this.training) {
      normMean = mean;
      normVar = var_;
    } else {
      normMean = this.running_mean;
      normVar = this.running_var;
    }

    // Normalize
    const invStd = normVar.add(this.eps).rsqrt();
    const normalized = input.sub(normMean.reshape([1, channels, ...Array(this.spatial_dims).fill(1)]))
      .mul(invStd.reshape([1, channels, ...Array(this.spatial_dims).fill(1)]));

    // Apply weight and bias
    if (this.weight) {
      const broadcastShape = [1, channels];
      for (let i = 0; i < this.spatial_dims; i++) {
        broadcastShape.push(1);
      }
      const w = this.weight.reshape(broadcastShape);
      const b = this.bias!.reshape(broadcastShape);
      return normalized.mul(w).add(b);
    }

    return normalized;
  }
}

/**
 * 1D Synchronized Batch Normalization.
 * Simplified single-GPU version.
 * @pytorch torch.nn.SyncBatchNorm (1D)
 */
export class SyncBatchNorm1d extends _SyncBatchNorm {
  constructor(
    num_features: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super(num_features, 1, eps, momentum, affine, track_running_stats);
  }
}

/**
 * 2D Synchronized Batch Normalization.
 * Simplified single-GPU version.
 * @pytorch torch.nn.SyncBatchNorm
 */
export class SyncBatchNorm2d extends _SyncBatchNorm {
  constructor(
    num_features: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super(num_features, 2, eps, momentum, affine, track_running_stats);
  }
}

/**
 * 3D Synchronized Batch Normalization.
 * Simplified single-GPU version.
 * @pytorch torch.nn.SyncBatchNorm (3D)
 */
export class SyncBatchNorm3d extends _SyncBatchNorm {
  constructor(
    num_features: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super(num_features, 3, eps, momentum, affine, track_running_stats);
  }
}

/**
 * Base class for Lazy Batch Normalization layers.
 * Lazily initializes num_features from the first forward pass.
 * @pytorch torch.nn.LazyBatchNorm1d, torch.nn.LazyBatchNorm2d, torch.nn.LazyBatchNorm3d
 */
class _LazyBatchNorm extends Module {
  public weight: Parameter | null;
  public bias: Parameter | null;
  public running_mean: Tensor | null;
  public running_var: Tensor | null;
  private _bn: _SyncBatchNorm | null;

  protected eps: number;
  protected momentum: number;
  protected affine: boolean;
  protected track_running_stats: boolean;
  protected spatial_dims: number;
  private _initialized: boolean;

  constructor(
    spatial_dims: number,
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super();
    this.spatial_dims = spatial_dims;
    this.eps = eps;
    this.momentum = momentum;
    this.affine = affine;
    this.track_running_stats = track_running_stats;
    this._initialized = false;
    this._bn = null;
    this.weight = null;
    this.bias = null;
    this.running_mean = null;
    this.running_var = null;
  }

  private _init(input: Tensor): void {
    if (this._initialized) return;

    const channels = input.shape[1] as number;
    this._bn = new _SyncBatchNorm(
      channels,
      this.spatial_dims,
      this.eps,
      this.momentum,
      this.affine,
      this.track_running_stats
    );

    // Register parameters and buffers
    if (this.affine) {
      this.weight = this._bn.weight;
      this.bias = this._bn.bias;
      this.register_parameter('weight', this.weight);
      this.register_parameter('bias', this.bias);
    }

    this.running_mean = this._bn.running_mean;
    this.running_var = this._bn.running_var;
    this.register_buffer('running_mean', this.running_mean);
    this.register_buffer('running_var', this.running_var);

    this._initialized = true;
  }

  forward(input: Tensor): Tensor {
    if (!this._initialized) {
      this._init(input);
    }
    return this._bn!.forward(input);
  }
}

/**
 * 1D Lazy Batch Normalization.
 * Lazily initializes num_features from the first forward pass.
 * @pytorch torch.nn.LazyBatchNorm1d
 */
export class LazyBatchNorm1d extends _LazyBatchNorm {
  constructor(
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super(1, eps, momentum, affine, track_running_stats);
  }
}

/**
 * 2D Lazy Batch Normalization.
 * Lazily initializes num_features from the first forward pass.
 * @pytorch torch.nn.LazyBatchNorm2d
 */
export class LazyBatchNorm2d extends _LazyBatchNorm {
  constructor(
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super(2, eps, momentum, affine, track_running_stats);
  }
}

/**
 * 3D Lazy Batch Normalization.
 * Lazily initializes num_features from the first forward pass.
 * @pytorch torch.nn.LazyBatchNorm3d
 */
export class LazyBatchNorm3d extends _LazyBatchNorm {
  constructor(
    eps: number = 1e-5,
    momentum: number = 0.1,
    affine: boolean = true,
    track_running_stats: boolean = true
  ) {
    super(3, eps, momentum, affine, track_running_stats);
  }
}
