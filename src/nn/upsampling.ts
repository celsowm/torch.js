/**
 * Upsampling layers for spatial data.
 * @status implemented
 * @pytorch torch.nn.Upsample, torch.nn.UpsamplingNearest2d, torch.nn.UpsamplingBilinear2d
 */

import { Module } from './module';
import { Tensor } from '../tensor';
import { tensor } from '../ops/creation';

type SizeType = number | [number, number] | [number, number, number];
type ScaleFactorType = number | [number, number] | [number, number, number];

/**
 * Generic upsampling layer that can use nearest, bilinear, or trilinear interpolation.
 * @pytorch torch.nn.Upsample
 */
export class Upsample extends Module {
  private _size: SizeType | null;
  private _scale_factor: ScaleFactorType | null;
  private _mode: 'nearest' | 'bilinear' | 'trilinear';
  private _align_corners: boolean;

  constructor({
    size,
    scale_factor,
    mode = 'nearest',
    align_corners = false,
  }: {
    size?: SizeType;
    scale_factor?: ScaleFactorType;
    mode?: 'nearest' | 'bilinear' | 'trilinear';
    align_corners?: boolean;
  } = {}) {
    super();
    this._size = size ?? null;
    this._scale_factor = scale_factor ?? null;
    this._mode = mode;
    this._align_corners = align_corners;
  }

  async forward(input: Tensor): Promise<Tensor> {
    const shape = input.shape;
    const ndim = shape.length;

    if (ndim < 3) {
      throw new Error(`Upsample requires at least 3D input, got ${ndim}D`);
    }

    // Determine output spatial size
    const is3D = this._mode === 'trilinear';
    const spatialDims = is3D ? 3 : 2;
    const spatialShape = shape.slice(-spatialDims);

    let outputSize: number[];
    if (this._size !== null) {
      outputSize = typeof this._size === 'number' ? Array(spatialDims).fill(this._size) : [...this._size];
    } else if (this._scale_factor !== null) {
      const factors = typeof this._scale_factor === 'number' ? Array(spatialDims).fill(this._scale_factor) : [...this._scale_factor];
      outputSize = spatialShape.map((s, i) => Math.floor(s * factors[i]));
    } else {
      throw new Error('Upsample requires either size or scale_factor');
    }

    // For now, use CPU fallback for interpolation
    return this._upsampleCpu(input, outputSize);
  }

  private async _upsampleCpu(input: Tensor, outputSize: number[]): Promise<Tensor> {
    const shape = input.shape;
    const ndim = shape.length;
    const is3D = this._mode === 'trilinear';
    const spatialDims = is3D ? 3 : 2;

    // Get input data
    const inputData = await input.toArray();
    const data = Array.from(inputData);

    // Calculate batch and channel dimensions
    const batch_size = shape[0];
    const channels = ndim >= 4 ? shape[1] : 1;
    const inputSpatial = shape.slice(-spatialDims);
    const outputSpatial = outputSize;

    // Calculate total spatial elements
    const inputSpatialSize = inputSpatial.reduce((a, b) => a * b, 1);
    const outputSpatialSize = outputSpatial.reduce((a, b) => a * b, 1);

    const outputData = new Float32Array(batch_size * channels * outputSpatialSize);

    for (let b = 0; b < batch_size; b++) {
      for (let c = 0; c < channels; c++) {
        const batchOffset = b * channels * inputSpatialSize;
        const channelOffset = c * inputSpatialSize;
        const srcOffset = batchOffset + channelOffset;

        const outBatchOffset = b * channels * outputSpatialSize;
        const outChannelOffset = c * outputSpatialSize;

        if (this._mode === 'nearest') {
          this._nearestInterpolation(
            data, inputSpatial, outputData, outputSpatial,
            srcOffset, outBatchOffset + outChannelOffset
          );
        } else if (this._mode === 'bilinear') {
          this._bilinearInterpolation(
            data, inputSpatial, outputData, outputSpatial,
            srcOffset, outBatchOffset + outChannelOffset,
            this._align_corners
          );
        } else if (this._mode === 'trilinear') {
          this._trilinearInterpolation(
            data, inputSpatial, outputData, outputSpatial,
            srcOffset, outBatchOffset + outChannelOffset,
            this._align_corners
          );
        }
      }
    }

    const outputShape = [batch_size, channels, ...outputSpatial];
    return tensor(Array.from(outputData), { dtype: input.dtype }).reshape(outputShape);
  }

  private _nearestInterpolation(
    src: number[],
    srcSize: number[],
    dst: Float32Array,
    dstSize: number[],
    srcOffset: number,
    dstOffset: number
  ): void {
    if (srcSize.length === 2) {
      const [srcH, srcW] = srcSize;
      const [dstH, dstW] = dstSize;

      for (let oy = 0; oy < dstH; oy++) {
        const iy = Math.min(Math.floor((oy / dstH) * srcH), srcH - 1);
        for (let ox = 0; ox < dstW; ox++) {
          const ix = Math.min(Math.floor((ox / dstW) * srcW), srcW - 1);
          dst[dstOffset + oy * dstW + ox] = src[srcOffset + iy * srcW + ix];
        }
      }
    } else if (srcSize.length === 3) {
      const [srcD, srcH, srcW] = srcSize;
      const [dstD, dstH, dstW] = dstSize;

      for (let oz = 0; oz < dstD; oz++) {
        const iz = Math.min(Math.floor((oz / dstD) * srcD), srcD - 1);
        for (let oy = 0; oy < dstH; oy++) {
          const iy = Math.min(Math.floor((oy / dstH) * srcH), srcH - 1);
          for (let ox = 0; ox < dstW; ox++) {
            const ix = Math.min(Math.floor((ox / dstW) * srcW), srcW - 1);
            dst[dstOffset + oz * dstH * dstW + oy * dstW + ox] =
              src[srcOffset + iz * srcH * srcW + iy * srcW + ix];
          }
        }
      }
    }
  }

  private _bilinearInterpolation(
    src: number[],
    srcSize: number[],
    dst: Float32Array,
    dstSize: number[],
    srcOffset: number,
    dstOffset: number,
    alignCorners: boolean
  ): void {
    const [srcH, srcW] = srcSize;
    const [dstH, dstW] = dstSize;

    for (let oy = 0; oy < dstH; oy++) {
      // Calculate source y coordinate
      const iy = alignCorners && dstH > 1
        ? (oy * (srcH - 1)) / (dstH - 1)
        : (oy + 0.5) * srcH / dstH - 0.5;

      const iy0 = Math.floor(Math.max(0, Math.min(iy, srcH - 2)));
      const iy1 = Math.min(iy0 + 1, srcH - 1);
      const fy = iy - iy0;

      for (let ox = 0; ox < dstW; ox++) {
        // Calculate source x coordinate
        const ix = alignCorners && dstW > 1
          ? (ox * (srcW - 1)) / (dstW - 1)
          : (ox + 0.5) * srcW / dstW - 0.5;

        const ix0 = Math.floor(Math.max(0, Math.min(ix, srcW - 2)));
        const ix1 = Math.min(ix0 + 1, srcW - 1);
        const fx = ix - ix0;

        // Bilinear interpolation
        const v00 = src[srcOffset + iy0 * srcW + ix0];
        const v01 = src[srcOffset + iy0 * srcW + ix1];
        const v10 = src[srcOffset + iy1 * srcW + ix0];
        const v11 = src[srcOffset + iy1 * srcW + ix1];

        const v0 = v00 * (1 - fx) + v01 * fx;
        const v1 = v10 * (1 - fx) + v11 * fx;
        const v = v0 * (1 - fy) + v1 * fy;

        dst[dstOffset + oy * dstW + ox] = v;
      }
    }
  }

  private _trilinearInterpolation(
    src: number[],
    srcSize: number[],
    dst: Float32Array,
    dstSize: number[],
    srcOffset: number,
    dstOffset: number,
    alignCorners: boolean
  ): void {
    const [srcD, srcH, srcW] = srcSize;
    const [dstD, dstH, dstW] = dstSize;

    for (let oz = 0; oz < dstD; oz++) {
      const iz = alignCorners && dstD > 1
        ? (oz * (srcD - 1)) / (dstD - 1)
        : (oz + 0.5) * srcD / dstD - 0.5;

      const iz0 = Math.floor(Math.max(0, Math.min(iz, srcD - 2)));
      const iz1 = Math.min(iz0 + 1, srcD - 1);
      const fz = iz - iz0;

      for (let oy = 0; oy < dstH; oy++) {
        const iy = alignCorners && dstH > 1
          ? (oy * (srcH - 1)) / (dstH - 1)
          : (oy + 0.5) * srcH / dstH - 0.5;

        const iy0 = Math.floor(Math.max(0, Math.min(iy, srcH - 2)));
        const iy1 = Math.min(iy0 + 1, srcH - 1);
        const fy = iy - iy0;

        for (let ox = 0; ox < dstW; ox++) {
          const ix = alignCorners && dstW > 1
            ? (ox * (srcW - 1)) / (dstW - 1)
            : (ox + 0.5) * srcW / dstW - 0.5;

          const ix0 = Math.floor(Math.max(0, Math.min(ix, srcW - 2)));
          const ix1 = Math.min(ix0 + 1, srcW - 1);
          const fx = ix - ix0;

          // Trilinear interpolation (8 corners)
          const v000 = src[srcOffset + iz0 * srcH * srcW + iy0 * srcW + ix0];
          const v001 = src[srcOffset + iz0 * srcH * srcW + iy0 * srcW + ix1];
          const v010 = src[srcOffset + iz0 * srcH * srcW + iy1 * srcW + ix0];
          const v011 = src[srcOffset + iz0 * srcH * srcW + iy1 * srcW + ix1];
          const v100 = src[srcOffset + iz1 * srcH * srcW + iy0 * srcW + ix0];
          const v101 = src[srcOffset + iz1 * srcH * srcW + iy0 * srcW + ix1];
          const v110 = src[srcOffset + iz1 * srcH * srcW + iy1 * srcW + ix0];
          const v111 = src[srcOffset + iz1 * srcH * srcW + iy1 * srcW + ix1];

          const v00 = v000 * (1 - fx) + v001 * fx;
          const v01 = v010 * (1 - fx) + v011 * fx;
          const v10 = v100 * (1 - fx) + v101 * fx;
          const v11 = v110 * (1 - fx) + v111 * fx;

          const v0 = v00 * (1 - fy) + v01 * fy;
          const v1 = v10 * (1 - fy) + v11 * fy;

          const v = v0 * (1 - fz) + v1 * fz;

          dst[dstOffset + oz * dstH * dstW + oy * dstW + ox] = v;
        }
      }
    }
  }

  get size(): SizeType | null {
    return this._size;
  }

  get scale_factor(): ScaleFactorType | null {
    return this._scale_factor;
  }

  get mode(): string {
    return this._mode;
  }

  get align_corners(): boolean {
    return this._align_corners;
  }
}

/**
 * 2D nearest neighbor upsampling.
 * Convenience wrapper around Upsample with mode='nearest'.
 * @pytorch torch.nn.UpsamplingNearest2d
 */
export class UpsamplingNearest2d extends Module {
  private _upsample: Upsample;

  constructor({ size, scale_factor }: { size?: number | [number, number]; scale_factor?: number } = {}) {
    super();
    this._upsample = new Upsample({ size, scale_factor, mode: 'nearest' });
  }

  async forward(input: Tensor): Promise<Tensor> {
    return this._upsample.forward(input);
  }

  get size(): SizeType | null {
    return this._upsample.size;
  }

  get scale_factor(): ScaleFactorType | null {
    return this._upsample.scale_factor;
  }
}

/**
 * 2D bilinear interpolation upsampling.
 * Convenience wrapper around Upsample with mode='bilinear'.
 * @pytorch torch.nn.UpsamplingBilinear2d
 */
export class UpsamplingBilinear2d extends Module {
  private _upsample: Upsample;

  constructor({
    size,
    scale_factor,
    align_corners = false,
  }: {
    size?: number | [number, number];
    scale_factor?: number;
    align_corners?: boolean;
  } = {}) {
    super();
    this._upsample = new Upsample({
      size,
      scale_factor,
      mode: 'bilinear',
      align_corners,
    });
  }

  async forward(input: Tensor): Promise<Tensor> {
    return this._upsample.forward(input);
  }

  get size(): SizeType | null {
    return this._upsample.size;
  }

  get scale_factor(): ScaleFactorType | null {
    return this._upsample.scale_factor;
  }

  get align_corners(): boolean {
    return this._upsample.align_corners;
  }
}
