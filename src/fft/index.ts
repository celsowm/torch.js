/**
 * FFT operations - torch.fft module.
 * Implements discrete Fourier transforms.
 * @status partial
 * @pytorch torch.fft
 */

import { Tensor } from '../tensor';
import { tensor } from '../ops/creation';

/**
 * Computes the one dimensional discrete Fourier transform.
 * @pytorch torch.fft.fft
 */
export async function fft(
  input: Tensor,
  n?: number,
  dim: number = -1,
  norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  const ndim = input.dim();
  const d = dim < 0 ? ndim + dim : dim;
  const shape = input.shape;
  const dimSize = shape[d];
  const N = n ?? dimSize;

  const inputData = await input.toArray();
  const outerDimsSize = shape.slice(0, d).reduce((a, b) => a * b, 1);
  const innerDimsSize = shape.slice(d + 1).reduce((a, b) => a * b, 1);
  const batchCount = outerDimsSize * innerDimsSize;

  const outputReal: number[] = [];
  const outputImag: number[] = [];

  for (let batch = 0; batch < batchCount; batch++) {
    const offset = batch * dimSize;
    const signal = new Float64Array(N);
    const copyLen = Math.min(N, dimSize);
    for (let i = 0; i < copyLen; i++) {
      signal[i] = inputData[offset + i];
    }

    const { re, im } = _fft1d(signal);
    outputReal.push(...re);
    outputImag.push(...im);
  }

  const outShape = [...shape.slice(0, d), N, ...shape.slice(d + 1)];
  return _createComplexTensor(outputReal, outputImag, outShape, input.dtype);
}

/**
 * Computes the inverse one dimensional discrete Fourier transform.
 * @pytorch torch.fft.ifft
 */
export async function ifft(
  input: Tensor,
  n?: number,
  dim: number = -1,
  norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  const ndim = input.dim();
  const d = dim < 0 ? ndim + dim : dim;
  const shape = input.shape;
  const dimSize = shape[d];
  const N = n ?? dimSize;

  const inputData = await input.toArray();
  const outerDimsSize = shape.slice(0, d).reduce((a, b) => a * b, 1);
  const innerDimsSize = shape.slice(d + 1).reduce((a, b) => a * b, 1);
  const batchCount = outerDimsSize * innerDimsSize;

  const outputReal: number[] = [];
  const outputImag: number[] = [];

  for (let batch = 0; batch < batchCount; batch++) {
    const offset = batch * dimSize;
    const signalReal = new Float64Array(N);
    const signalImag = new Float64Array(N);

    const copyLen = Math.min(N, dimSize);
    for (let i = 0; i < copyLen; i++) {
      signalReal[i] = inputData[offset + i];
    }

    const { re, im } = _ifft1d(signalReal, signalImag);
    outputReal.push(...re);
    outputImag.push(...im);
  }

  const outShape = [...shape.slice(0, d), N, ...shape.slice(d + 1)];
  return _createComplexTensor(outputReal, outputImag, outShape, input.dtype);
}

/**
 * Computes the 2D discrete Fourier transform.
 * @pytorch torch.fft.fft2
 */
export async function fft2(
  input: Tensor,
  s?: [number, number],
  dim: [number, number] = [-2, -1],
  norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  let result = await fft(input, s?.[0], dim[0], norm);
  result = await fft(result, s?.[1], dim[1], norm);
  return result;
}

/**
 * Computes the N-dimensional discrete Fourier transform.
 * @pytorch torch.fft.fftn
 */
export async function fftn(
  input: Tensor,
  s?: number[],
  dim?: number[],
  norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  let result = input;
  const dims = dim ?? Array.from({ length: input.dim() }, (_, i) => i);
  const sizes = s ?? dims.map(() => 0);

  for (let i = 0; i < dims.length; i++) {
    result = await fft(result, sizes[i] || undefined, dims[i], norm);
  }
  return result;
}

/**
 * Computes the one dimensional discrete Fourier transform for real input.
 * @pytorch torch.fft.rfft
 */
export async function rfft(
  input: Tensor,
  n?: number,
  dim: number = -1,
  norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  const fullFft = await fft(input, n, dim, norm);
  const shape = fullFft.shape;
  const ndim = input.dim();
  const d = dim < 0 ? ndim + dim : dim;
  const N = shape[d];
  const nOut = Math.floor(N / 2) + 1;

  const slices: any[] = [];
  for (let i = 0; i <= ndim; i++) {
    if (i === d) {
      slices.push({ start: 0, stop: nOut, step: 1 });
    } else {
      slices.push({ start: 0, stop: shape[i] ?? 1, step: 1 });
    }
  }

  return fullFft.slice(slices);
}

/**
 * Inverse of rfft.
 * @pytorch torch.fft.irfft
 */
export async function irfft(
  input: Tensor,
  n?: number,
  dim: number = -1,
  norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  return ifft(input, n, dim, norm);
}

/**
 * Shifts zero-frequency component to center.
 * @pytorch torch.fft.fftshift
 */
export async function fftshift(input: Tensor, dim?: number | number[]): Promise<Tensor> {
  const dims = Array.isArray(dim) ? dim : (dim !== undefined ? [dim] : []);
  let result = input;

  const shape = input.shape;
  const ndim = input.dim();
  const dimsToShift = dims.length > 0 ? dims : Array.from({ length: ndim }, (_, i) => i);

  for (const d of dimsToShift) {
    const dimResolved = d < 0 ? ndim + d : d;
    const shift = Math.floor(shape[dimResolved] / 2);
    result = await result.roll(shift, dimResolved);
  }

  return result;
}

/**
 * Inverse of fftshift.
 * @pytorch torch.fft.ifftshift
 */
export async function ifftshift(input: Tensor, dim?: number | number[]): Promise<Tensor> {
  const dims = Array.isArray(dim) ? dim : (dim !== undefined ? [dim] : []);
  let result = input;

  const shape = input.shape;
  const ndim = input.dim();
  const dimsToShift = dims.length > 0 ? dims : Array.from({ length: ndim }, (_, i) => i);

  for (const d of dimsToShift) {
    const dimResolved = d < 0 ? ndim + d : d;
    const shift = Math.ceil(shape[dimResolved] / 2);
    result = await result.roll(shift, dimResolved);
  }

  return result;
}

/**
 * Discrete Fourier transform frequencies.
 * @pytorch torch.fft.fftfreq
 */
export function fftfreq(n: number, d: number = 1.0): Tensor {
  const scale = 1.0 / d;
  const positive = Array.from({ length: Math.ceil(n / 2) }, (_, i) => i * scale / n);
  const negative = Array.from({ length: Math.floor(n / 2) }, (_, i) => -(Math.ceil(n / 2) - i) * scale / n);
  return tensor([...positive, ...negative]);
}

/**
 * Real discrete Fourier transform frequencies.
 * @pytorch torch.fft.rfftfreq
 */
export function rfftfreq(n: number, d: number = 1.0): Tensor {
  const scale = 1.0 / d;
  return tensor(Array.from({ length: Math.floor(n / 2) + 1 }, (_, i) => i * scale / n));
}

// ============= Internal Helpers =============

/**
 * Create complex tensor from real and imaginary parts.
 */
async function _createComplexTensor(
  real: number[],
  imag: number[],
  shape: number[],
  dtype: string
): Promise<Tensor> {
  const complexData: number[] = [];
  for (let i = 0; i < real.length; i++) {
    complexData.push(real[i], imag[i]);
  }
  
  const finalShape = [...shape, 2];
  return tensor(complexData, { dtype: dtype as any }).reshape(finalShape as any);
}

/**
 * 1D FFT using Cooley-Tukey algorithm (power-of-2 only).
 */
function _fft1d(signal: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signal.length;
  const paddedN = Math.pow(2, Math.ceil(Math.log2(N)));
  const padded = new Float64Array(paddedN);
  padded.set(signal);

  const result = _fftRecursiveReal(padded);
  return {
    re: result.slice(0, N),
    im: new Float64Array(N),
  };
}

/**
 * 1D IFFT.
 */
function _ifft1d(signalReal: Float64Array, signalImag: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signalReal.length;
  const complexSignal: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    complexSignal.push([signalReal[i], -signalImag[i]]);
  }

  const result = _fftRecursiveComplex(complexSignal);

  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    re[i] = result[i][0] / N;
    im[i] = -result[i][1] / N;
  }

  return { re, im };
}

/**
 * Recursive FFT for real signals.
 */
function _fftRecursiveReal(signal: Float64Array): Float64Array {
  const N = signal.length;
  if (N === 1) return signal;

  const even = new Float64Array(N / 2);
  const odd = new Float64Array(N / 2);
  for (let i = 0; i < N / 2; i++) {
    even[i] = signal[2 * i];
    odd[i] = signal[2 * i + 1];
  }

  const evenFFT = _fftRecursiveReal(even);
  const oddFFT = _fftRecursiveReal(odd);

  const result = new Float64Array(N);
  for (let k = 0; k < N / 2; k++) {
    const angle = (-2 * Math.PI * k) / N;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const realOdd = oddFFT[k] * cos;

    result[k] = evenFFT[k] + realOdd;
    result[k + N / 2] = evenFFT[k] - realOdd;
  }

  return result;
}

/**
 * Recursive FFT for complex signals.
 */
function _fftRecursiveComplex(signal: [number, number][]): [number, number][] {
  const N = signal.length;
  if (N === 1) return signal;

  const even: [number, number][] = [];
  const odd: [number, number][] = [];
  for (let i = 0; i < N / 2; i++) {
    even.push(signal[2 * i]);
    odd.push(signal[2 * i + 1]);
  }

  const evenFFT = _fftRecursiveComplex(even);
  const oddFFT = _fftRecursiveComplex(odd);

  const result: [number, number][] = new Array(N);
  for (let k = 0; k < N / 2; k++) {
    const angle = (-2 * Math.PI * k) / N;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    const [oddRe, oddIm] = oddFFT[k];
    const [evenRe, evenIm] = evenFFT[k];

    const twiddleRe = oddRe * cos - oddIm * sin;
    const twiddleIm = oddRe * sin + oddIm * cos;

    result[k] = [evenRe + twiddleRe, evenIm + twiddleIm];
    result[k + N / 2] = [evenRe - twiddleRe, evenIm - twiddleIm];
  }

  return result;
}

// STFT/ISTFT
export { stft, istft, hfft, ihfft } from './stft';
