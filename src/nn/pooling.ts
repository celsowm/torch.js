import { Tensor } from '../tensor';
import { Module } from './module';
import { max_pool2d as _max_pool2d, avg_pool2d as _avg_pool2d } from './functional';

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
