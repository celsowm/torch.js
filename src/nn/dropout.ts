/**
 * Dropout module.
 * @status partial
 * @pytorch torch.nn.Dropout
 */

import { Tensor } from '../tensor';
import { Module } from './module';
import * as F from './functional';
import { rand, zeros, ones } from '../ops/creation';

/**
 * During training, randomly zeroes some elements of the input tensor.
 */
export class Dropout extends Module {
  private p: number;
  private _inplace: boolean;

  constructor(p: number = 0.5, inplace: boolean = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    return F.dropout(input, this.p, this.training);
  }
}

/**
 * 2D Dropout for channel-wise dropout on 4D tensors.
 * @pytorch torch.nn.Dropout2d
 */
export class Dropout2d extends Module {
  private p: number;
  private _inplace: boolean;

  constructor(p: number = 0.5, inplace: boolean = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    // For now, use same behavior as 1D dropout
    return F.dropout(input, this.p, this.training);
  }
}

/**
 * Randomly zeroes whole channels (as a 3D input is expected).
 * Applies Channel Dropout to a 3D Tensor (mini-batch, channel, sequence).
 * @pytorch torch.nn.Dropout1d
 */
export class Dropout1d extends Module {
  private p: number;
  private _inplace: boolean;

  constructor(p: number = 0.5, inplace: boolean = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    if (!this.training || this.p === 0) {
      return input;
    }

    const ndim = input.shape.length;
    if (ndim < 2 || ndim > 3) {
      throw new Error(`Dropout1d: expected 2D or 3D input, got ${ndim}D`);
    }

    // Ensure 3D shape: (batch, channels, seq)
    let inp = input;
    let needSqueeze = false;
    if (ndim === 2) {
      inp = input.unsqueeze(0);
      needSqueeze = true;
    }

    const batchSize = inp.shape[0] as number;
    const channels = inp.shape[1] as number;

    // Create noise mask of shape (batch, channels, 1) to drop entire channels
    const noiseShape: number[] = [batchSize, channels, 1];
    const noise = rand(noiseShape, { dtype: input.dtype });

    // Create mask: 0 with probability p, 1/(1-p) with probability (1-p)
    const mask = noise.gt(this.p);
    const scale = 1.0 / (1.0 - this.p);
    const scaledMask = mask.mul(scale);

    // Broadcast to (batch, channels, seq)
    const output = inp.mul(scaledMask);

    if (needSqueeze) {
      return output.squeeze(0);
    }

    return output;
  }
}

/**
 * Randomly zeroes whole channels (as a 5D input is expected).
 * Applies Channel Dropout to a 5D Tensor (mini-batch, channel, depth, height, width).
 * @pytorch torch.nn.Dropout3d
 */
export class Dropout3d extends Module {
  private p: number;
  private _inplace: boolean;

  constructor(p: number = 0.5, inplace: boolean = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    if (!this.training || this.p === 0) {
      return input;
    }

    const ndim = input.shape.length;
    if (ndim !== 5) {
      throw new Error(`Dropout3d: expected 5D input, got ${ndim}D`);
    }

    const batchSize = input.shape[0] as number;
    const channels = input.shape[1] as number;

    // Create noise mask of shape (batch, channels, 1, 1, 1) to drop entire channels
    const noiseShape: number[] = [batchSize, channels, 1, 1, 1];
    const noise = rand(noiseShape, { dtype: input.dtype });

    // Create mask: 0 with probability p, 1/(1-p) with probability (1-p)
    const mask = noise.gt(this.p);
    const scale = 1.0 / (1.0 - this.p);
    const scaledMask = mask.mul(scale);

    // Broadcast to full shape
    return input.mul(scaledMask);
  }
}

/**
 * Alpha Dropout for SELU networks.
 * Alpha dropout keeps the mean and variance of the input invariant.
 * Designed to be used with SELU activation.
 * @pytorch torch.nn.AlphaDropout
 */
export class AlphaDropout extends Module {
  private p: number;
  private _inplace: boolean;

  constructor(p: number = 0.5, inplace: boolean = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    if (!this.training || this.p === 0) {
      return input;
    }

    // SELU parameters (alpha and scale)
    const alpha = -1.7580993408473766;
    const scale = 1.0507009873554804934193349852946;

    // Compute alpha_prime for the affine transformation
    const a = Math.sqrt((1 - this.p) / ((1 - this.p) + this.p * alpha * alpha));

    // Generate random mask
    const noise = rand(input.shape as number[], { dtype: input.dtype });
    const mask = noise.gt(this.p);

    // Compute b term for mean preservation
    const bVal = -a * alpha * this.p;
    const b = ones(input.shape as number[], { dtype: input.dtype }).mul(bVal);

    // Apply: a * mask * x + b * (1 - mask) for dropped elements
    // Simplified: output = a * (mask * x + alpha * (1 - mask)) + b
    const dropped = mask.mul(input).add(mask.eq(0).mul(alpha));
    return dropped.mul(a).add(b);
  }
}
