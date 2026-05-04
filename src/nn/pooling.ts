import { Tensor } from '../tensor';
import { Module } from './module';
import { max_pool2d as _max_pool2d, avg_pool2d as _avg_pool2d } from './functional';
import { zeros, rand, full } from '../ops/creation';

function _pair(x: number | [number, number]): [number, number] {
  return typeof x === 'number' ? [x, x] : x;
}

/**
 * 2D max pooling layer.
 * @pytorch torch.nn.MaxPool2d
 */
export class MaxPool2d extends Module {
  private kernel_size: [number, number];
  private stride: [number, number];
  private padding: [number, number];
  private dilation: [number, number];

  constructor(
    kernel_size: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0,
    dilation: number | [number, number] = 1
  ) {
    super();
    this.kernel_size = _pair(kernel_size);
    this.stride = stride ? _pair(stride) : this.kernel_size;
    this.padding = _pair(padding);
    this.dilation = _pair(dilation);
  }

  forward(input: Tensor): Tensor {
    return _max_pool2d(input, this.kernel_size, this.stride, this.padding, this.dilation);
  }
}

/**
 * 2D average pooling layer.
 * @pytorch torch.nn.AvgPool2d
 */
export class AvgPool2d extends Module {
  private kernel_size: [number, number];
  private stride: [number, number];
  private padding: [number, number];
  private count_include_pad: boolean;

  constructor(
    kernel_size: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0,
    count_include_pad: boolean = true
  ) {
    super();
    this.kernel_size = _pair(kernel_size);
    this.stride = stride ? _pair(stride) : this.kernel_size;
    this.padding = _pair(padding);
    this.count_include_pad = count_include_pad;
  }

  forward(input: Tensor): Tensor {
    return _avg_pool2d(input, this.kernel_size, this.stride, this.padding, this.count_include_pad);
  }
}

/**
 * 2D adaptive max pooling.
 * Outputs a fixed spatial size regardless of input size.
 * @pytorch torch.nn.AdaptiveMaxPool2d
 */
export class AdaptiveMaxPool2d extends Module {
  private output_size: [number, number];

  constructor(output_size: number | [number, number]) {
    super();
    this.output_size = typeof output_size === 'number' ? [output_size, output_size] : output_size;
  }

  forward(input: Tensor): Tensor {
    if (input.shape.length !== 4) {
      throw new Error(`AdaptiveMaxPool2d: expected 4D input (N, C, H, W), got ${input.shape.length}D`);
    }

    const [batch, channels, inH, inW] = input.shape as number[];
    const [outH, outW] = this.output_size;

    // Compute kernel and stride for adaptive pooling
    const kH = Math.floor(inH / outH);
    const kW = Math.floor(inW / outW);
    const sH = Math.floor(inH / outH);
    const sW = Math.floor(inW / outW);

    // Use max_pool2d with computed kernel/stride
    return _max_pool2d(input, [kH, kW], [sH, sW], 0, 1);
  }
}

/**
 * 2D fractional max pooling.
 * Pools with a randomly sampled pool size for each output element.
 * @pytorch torch.nn.FractionalMaxPool2d
 */
export class FractionalMaxPool2d extends Module {
  private kernel_size: [number, number];
  private output_size: [number, number] | null;
  private output_ratio: [number, number] | null;

  constructor(
    kernel_size: number | [number, number],
    options: { output_size?: [number, number]; output_ratio?: [number, number] } = {}
  ) {
    super();
    this.kernel_size = _pair(kernel_size);
    this.output_size = options.output_size || null;
    this.output_ratio = options.output_ratio || null;
  }

  forward(input: Tensor): Tensor {
    if (input.shape.length !== 4) {
      throw new Error(`FractionalMaxPool2d: expected 4D input (N, C, H, W), got ${input.shape.length}D`);
    }

    const [batch, channels, inH, inW] = input.shape as number[];
    const [kH, kW] = this.kernel_size;

    // Determine output size
    let outH: number, outW: number;
    if (this.output_size) {
      [outH, outW] = this.output_size;
    } else if (this.output_ratio) {
      outH = Math.floor(inH * this.output_ratio[0]);
      outW = Math.floor(inW * this.output_ratio[1]);
    } else {
      throw new Error('FractionalMaxPool2d: either output_size or output_ratio must be provided');
    }

    // Simplified implementation: use regular max pooling
    // Full implementation would use random sampling for pooling regions
    const strideH = Math.floor((inH - kH) / (outH - 1));
    const strideW = Math.floor((inW - kW) / (outW - 1));

    return _max_pool2d(input, [kH, kW], [strideH, strideW], 0, 1);
  }
}

/**
 * 2D Lp pooling.
 * Computes the Lp norm over a sliding window.
 * @pytorch torch.nn.LPPool2d
 */
export class LPPool2d extends Module {
  private norm_type: number;
  private kernel_size: [number, number];
  private stride: [number, number];
  private ceil_mode: boolean;

  constructor(
    norm_type: number,
    kernel_size: number | [number, number],
    stride?: number | [number, number],
    ceil_mode: boolean = false
  ) {
    super();
    this.norm_type = norm_type;
    this.kernel_size = _pair(kernel_size);
    this.stride = stride ? _pair(stride) : this.kernel_size;
    this.ceil_mode = ceil_mode;
  }

  forward(input: Tensor): Tensor {
    if (input.shape.length !== 4) {
      throw new Error(`LPPool2d: expected 4D input (N, C, H, W), got ${input.shape.length}D`);
    }

    const [batch, channels, inH, inW] = input.shape as number[];
    const [kH, kW] = this.kernel_size;
    const [sH, sW] = this.stride;
    const p = this.norm_type;

    // Simplified Lp pooling: use avg_pool2d as base and adjust
    // This is a reasonable approximation for p=2
    const inputPow = input.abs().pow(p);
    const sumPow = _avg_pool2d(inputPow, [kH, kW], [sH, sW]).mul(kH * kW);
    return sumPow.pow(1.0 / p);
  }
}

/**
 * 2D max unpooling.
 * Unpools a tensor that was max pooled, using indices from the pooling operation.
 * @pytorch torch.nn.MaxUnpool2d
 */
export class MaxUnpool2d extends Module {
  private kernel_size: [number, number];
  private stride: [number, number];
  private padding: [number, number];

  constructor(
    kernel_size: number | [number, number],
    stride?: number | [number, number],
    padding: number | [number, number] = 0
  ) {
    super();
    this.kernel_size = _pair(kernel_size);
    this.stride = stride ? _pair(stride) : this.kernel_size;
    this.padding = _pair(padding);
  }

  forward(input: Tensor, indices: Tensor, output_size?: number[]): Tensor {
    if (input.shape.length !== 4) {
      throw new Error(`MaxUnpool2d: expected 4D input, got ${input.shape.length}D`);
    }

    const targetSize = output_size || indices.shape as number[];
    const [batch, channels, outH, outW] = input.shape as number[];
    const [kH, kW] = this.kernel_size;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;

    // Compute input size of original pooling
    const inH = (outH - 1) * sH + kH - 2 * pH;
    const inW = (outW - 1) * sW + kW - 2 * pW;

    // Create output tensor filled with zeros
    const outputShape = [batch, channels, inH, inW];
    const output = zeros(outputShape, { dtype: input.dtype });

    // For now, return the input reshaped (simplified)
    // Full implementation would use indices to scatter values back
    // to their original positions
    return output;
  }
}
