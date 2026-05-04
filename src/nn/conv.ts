import { Tensor } from '../tensor';
import { randn, zeros, arange, stack } from '../ops/creation';
import { Module } from './module';
import { Parameter } from './parameter';
import { conv2d } from './functional';

/**
 * 2D transposed convolution layer (also known as deconvolution).
 * @pytorch torch.nn.ConvTranspose2d
 */
export class ConvTranspose2d extends Module {
  public weight: Parameter;
  public bias: Parameter | null;

  private in_channels: number;
  private out_channels: number;
  private kernel_size: [number, number];
  private stride: [number, number];
  private padding: [number, number];
  private output_padding: [number, number];
  private dilation: [number, number];
  private groups: number;
  private padding_mode: string;

  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | [number, number],
    stride: number | [number, number] = 1,
    padding: number | [number, number] = 0,
    output_padding: number | [number, number] = 0,
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
    this.output_padding = _pair(output_padding);
    this.dilation = _pair(dilation);
    this.groups = groups;
    this.padding_mode = padding_mode;

    const [kH, kW] = this.kernel_size;
    const k = Math.sqrt(1.0 / (in_channels * kH * kW));
    const weightData = randn([in_channels, out_channels / groups, kH, kW]).mul(k);
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
    // ConvTranspose2d is implemented as a convolution with swapped input/output channels
    // and appropriate output padding
    const [kH, kW] = this.kernel_size;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;
    const [oH, oW] = this.output_padding;
    const [dH, dW] = this.dilation;

    if (input.shape.length !== 4) {
      throw new Error(`ConvTranspose2d: expected 4D input (N, C, H, W), got ${input.shape.length}D`);
    }

    const [batch, inCh, inH, inW] = input.shape as number[];
    
    // Calculate output shape for ConvTranspose2d
    // out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    const outH_calc = (inH - 1) * sH - 2 * pH + dH * (kH - 1) + oH + 1;
    const outW_calc = (inW - 1) * sW - 2 * pW + dW * (kW - 1) + oW + 1;

    // Use conv2d with swapped in/out channels and transposed weight
    // weight shape for conv2d: [out_channels, in_channels/groups, kH, kW]
    // Our weight is already [in_channels, out_channels/groups, kH, kW]
    // Need to transpose to [out_channels, in_channels/groups, kH, kW]
    const weight_t = this.weight.transpose(0, 1);

    // For convTranspose2d, we need to use different padding
    // The effective padding for the backward pass
    const effective_kH = dH * (kH - 1) + 1;
    const effective_kW = dW * (kW - 1) + 1;
    const convPaddingH = effective_kH - pH - 1;
    const convPaddingW = effective_kW - pW - 1;

    const output = conv2d(
      input,
      weight_t,
      this.bias ?? undefined,
      [sH, sW],
      [convPaddingH, convPaddingW],
      [dH, dW],
      this.groups
    );

    // Note: output shape might not exactly match due to rounding
    // In practice, we might need to crop or pad
    return output;
  }
}

function _pair(x: number | [number, number]): [number, number] {
  return typeof x === 'number' ? [x, x] : x;
}

function _single(x: number): number {
  return x;
}

function _triple(x: number | [number, number, number]): [number, number, number] {
  return typeof x === 'number' ? [x, x, x] : x;
}

/**
 * 1D transposed convolution layer.
 * @pytorch torch.nn.ConvTranspose1d
 */
export class ConvTranspose1d extends Module {
  public weight: Parameter;
  public bias: Parameter | null;

  private in_channels: number;
  private out_channels: number;
  private kernel_size: number;
  private stride: number;
  private padding: number;
  private output_padding: number;
  private dilation: number;
  private groups: number;
  private padding_mode: string;

  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number,
    stride: number = 1,
    padding: number = 0,
    output_padding: number = 0,
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
    this.output_padding = output_padding;
    this.dilation = dilation;
    this.groups = groups;
    this.padding_mode = padding_mode;

    const k = Math.sqrt(1.0 / (in_channels * kernel_size));
    const weightData = randn([in_channels, out_channels / groups, kernel_size]).mul(k);
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

    // Reshape to 4D: [N, C, 1, L]
    const input4d = input.unsqueeze(2);

    // Weight: [in_channels, out_channels/groups, kernel_size] -> [in, out/g, 1, k]
    const weight4d = this.weight.unsqueeze(2);

    // Use ConvTranspose2d with height=1
    const [sH, sW] = [1, this.stride];
    const [pH, pW] = [0, this.padding];
    const [oH, oW] = [0, this.output_padding];
    const [dH, dW] = [1, this.dilation];
    const [kH, kW] = [1, this.kernel_size];

    // Calculate output width
    const outL_calc = (inL - 1) * sW - 2 * pW + dW * (kW - 1) + oW + 1;

    // Transpose weight for conv2d: [out_channels, in_channels/groups, 1, kW]
    const weight_t = weight4d.transpose(0, 1);

    // Effective padding for the backward pass
    const effective_kW = dW * (kW - 1) + 1;
    const convPaddingW = effective_kW - pW - 1;

    const output4d = conv2d(
      input4d,
      weight_t,
      this.bias ?? undefined,
      [sH, sW],
      [0, convPaddingW],
      [dH, dW],
      this.groups
    );

    // Squeeze back to 3D: [N, C, outL]
    return output4d.squeeze(2);
  }
}

/**
 * 3D transposed convolution layer.
 * @pytorch torch.nn.ConvTranspose3d
 */
export class ConvTranspose3d extends Module {
  public weight: Parameter;
  public bias: Parameter | null;

  private in_channels: number;
  private out_channels: number;
  private kernel_size: [number, number, number];
  private stride: [number, number, number];
  private padding: [number, number, number];
  private output_padding: [number, number, number];
  private dilation: [number, number, number];
  private groups: number;
  private padding_mode: string;

  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | [number, number, number],
    stride: number | [number, number, number] = 1,
    padding: number | [number, number, number] = 0,
    output_padding: number | [number, number, number] = 0,
    dilation: number | [number, number, number] = 1,
    groups: number = 1,
    bias: boolean = true,
    padding_mode: string = 'zeros'
  ) {
    super();
    this.in_channels = in_channels;
    this.out_channels = out_channels;
    this.kernel_size = _triple(kernel_size);
    this.stride = _triple(stride);
    this.padding = _triple(padding);
    this.output_padding = _triple(output_padding);
    this.dilation = _triple(dilation);
    this.groups = groups;
    this.padding_mode = padding_mode;

    const [kD, kH, kW] = this.kernel_size;
    const k = Math.sqrt(1.0 / (in_channels * kD * kH * kW));
    const weightData = randn([in_channels, out_channels / groups, kD, kH, kW]).mul(k);
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
    if (input.shape.length === 4) {
      input = input.unsqueeze(0);
    }
    const [batch, inCh, inD, inH, inW] = input.shape as number[];

    if (input.shape.length !== 5) {
      throw new Error(`ConvTranspose3d: expected 5D input (N, C, D, H, W), got ${input.shape.length}D`);
    }

    const [kD, kH, kW] = this.kernel_size;
    const [sD, sH, sW] = this.stride;
    const [pD, pH, pW] = this.padding;
    const [oD, oH, oW] = this.output_padding;
    const [dD, dH, dW] = this.dilation;

    // Calculate output dimensions
    const outD_calc = (inD - 1) * sD - 2 * pD + dD * (kD - 1) + oD + 1;
    const outH_calc = (inH - 1) * sH - 2 * pH + dH * (kH - 1) + oH + 1;
    const outW_calc = (inW - 1) * sW - 2 * pW + dW * (kW - 1) + oW + 1;

    // Reshape 5D to 4D by merging D into batch: [N*D, C, H, W]
    // Then use ConvTranspose2d, then reshape back
    // Transpose weight: [in_channels, out_channels/groups, kD, kH, kW] -> [out_channels, in_channels/groups, kD, kH, kW]
    // We'll iterate over depth dimension

    const outputs: Tensor[] = [];

    for (let d = 0; d < outD_calc; d++) {
      // Collect input slices that contribute to output depth d
      // Input depth i contributes to output depth d if:
      // d is in range [i * sD - pD, i * sD - pD + kD * dD]
      // This is complex; instead, we use a simpler approach:
      // Reshape weight to 4D by summing over depth dimension

      // Simpler approach: treat 3D conv transpose as multiple 2D operations
      // For each output depth, gather contributions from relevant input depths
      const inputSlices: Tensor[] = [];
      const weightSlices: Tensor[] = [];

      for (let i = 0; i < inD; i++) {
        const outStart = i * sD - pD;
        const outEnd = outStart + dD * (kD - 1) + 1;
        if (d >= outStart && d < outEnd) {
          // This input depth contributes
          const depthOffset = d - outStart;
          if (depthOffset % dD === 0) {
            const kIdx = depthOffset / dD;
            inputSlices.push(input.narrow(2, i, 1).squeeze(2)); // [N, C, H, W]
            // Weight slice at depth kIdx
            weightSlices.push(this.weight.narrow(2, kIdx, 1).squeeze(2)); // [in, out/g, H, W]
          }
        }
      }

      if (inputSlices.length === 0) {
        // No contribution, output is zeros
        const zeroOut = zeros([batch, this.out_channels, outH_calc, outW_calc]);
        outputs.push(zeroOut);
        continue;
      }

      // Sum contributions from all relevant input depths
      let depthOutput: Tensor | null = null;
      for (let i = 0; i < inputSlices.length; i++) {
        const weight_t = weightSlices[i].transpose(0, 1); // [out, in/g, H, W]

        const effective_kH = dH * (kH - 1) + 1;
        const effective_kW = dW * (kW - 1) + 1;
        const convPaddingH = effective_kH - pH - 1;
        const convPaddingW = effective_kW - pW - 1;

        const contrib = conv2d(
          inputSlices[i],
          weight_t,
          undefined,
          [sH, sW],
          [convPaddingH, convPaddingW],
          [dH, dW],
          this.groups
        );

        if (depthOutput === null) {
          depthOutput = contrib;
        } else {
          depthOutput = depthOutput.add(contrib);
        }
      }

      outputs.push(depthOutput!);
    }

    // Stack outputs along depth dimension: [N*D_out, C, H, W] -> [N, C, D, H, W]
    // outputs is [outD, [N, outCh, outH, outW]]
    // Stack along new axis
    const stacked = stack(outputs, 2); // [N, outCh, outD, outH, outW]

    // Add bias if present
    if (this.bias !== null) {
      // Bias: [out_channels] -> [1, out_channels, 1, 1, 1]
      const biasView = this.bias.view([1, this.out_channels, 1, 1, 1]);
      return stacked.add(biasView);
    }

    return stacked;
  }
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

    const output4d = conv2d(
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
    return conv2d(
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

