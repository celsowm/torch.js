import { Tensor } from '../tensor';
import { randn, zeros, arange } from '../ops/creation';
import { Module } from './module';
import { Parameter } from './parameter';
import { conv2d as _conv2d } from './functional';

function _pair(x: number | [number, number]): [number, number] {
  return typeof x === 'number' ? [x, x] : x;
}

function _single(x: number): number {
  return x;
}

/**
 * 1D convolution layer.
 * @pytorch torch.nn.Conv1d
 */
export class Conv1d extends Module {
  public weight: Parameter;
  public bias: Parameter | null;

  private in_channels: number;
  private out_channels: number;
  private kernel_size: number;
  private stride: number;
  private padding: number;
  private dilation: number;
  private groups: number;
  private padding_mode: string;

  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number,
    stride: number = 1,
    padding: number = 0,
    dilation: number = 1,
    groups: number = 1,
    bias: boolean = true,
    padding_mode: string = 'zeros'
  ) {
    super();
    this.in_channels = in_channels;
    this.out_channels = out_channels;
    this.kernel_size = kernel_size;
    this.stride = stride;
    this.padding = padding;
    this.dilation = dilation;
    this.groups = groups;
    this.padding_mode = padding_mode;

    const k = Math.sqrt(1.0 / (in_channels * kernel_size));
    const weightData = randn([out_channels, in_channels / groups, kernel_size]).mul(k);
    this.weight = Parameter.create(weightData);
    this.register_parameter('weight', this.weight);

    if (bias) {
      const biasData = zeros([out_channels]);
      this.bias = Parameter.create(biasData);
      this.register_parameter('bias', this.bias);
    } else {
      this.bias = null;
    }
  }

  forward(input: Tensor): Tensor {
    if (input.shape.length === 2) {
      input = input.unsqueeze(0);
    }
    const [batch, inCh, inL] = input.shape as number[];

    const padL = this.padding_mode === 'zeros' ? this.padding : 0;
    const outL = Math.floor((inL + 2 * padL - this.dilation * (this.kernel_size - 1) - 1) / this.stride) + 1;

    const weight4d = this.weight.unsqueeze(2); // [out, in/g, 1, k]
    const input4d = input.unsqueeze(2); // [N, C, 1, L]

    const output4d = _conv2d(
      input4d,
      weight4d,
      this.bias ?? undefined,
      [1, this.stride],
      [0, padL],
      [1, this.dilation],
      this.groups
    );

    const result = output4d.squeeze(2); // [N, C, outL]
    return result;
  }
}

/**
 * 2D convolution layer.
 * @pytorch torch.nn.Conv2d
 */
export class Conv2d extends Module {
  public weight: Parameter;
  public bias: Parameter | null;

  private in_channels: number;
  private out_channels: number;
  private kernel_size: [number, number];
  private stride: [number, number];
  private padding: [number, number];
  private dilation: [number, number];
  private groups: number;
  private padding_mode: string;

  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | [number, number],
    stride: number | [number, number] = 1,
    padding: number | [number, number] = 0,
    dilation: number | [number, number] = 1,
    groups: number = 1,
    bias: boolean = true,
    padding_mode: string = 'zeros'
  ) {
    super();
    this.in_channels = in_channels;
    this.out_channels = out_channels;
    this.kernel_size = _pair(kernel_size);
    this.stride = _pair(stride);
    this.padding = _pair(padding);
    this.dilation = _pair(dilation);
    this.groups = groups;
    this.padding_mode = padding_mode;

    const [kH, kW] = this.kernel_size;
    const k = Math.sqrt(1.0 / (in_channels * kH * kW));
    const weightData = randn([out_channels, in_channels / groups, kH, kW]).mul(k);
    this.weight = Parameter.create(weightData);
    this.register_parameter('weight', this.weight);

    if (bias) {
      const biasData = zeros([out_channels]);
      this.bias = Parameter.create(biasData);
      this.register_parameter('bias', this.bias);
    } else {
      this.bias = null;
    }
  }

  forward(input: Tensor): Tensor {
    return _conv2d(
      input,
      this.weight,
      this.bias ?? undefined,
      this.stride,
      this.padding,
      this.dilation,
      this.groups
    );
  }
}
