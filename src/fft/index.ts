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
  const shape = input.shape;
  const isComplexInput = shape[ndim - 1] === 2;
  
  // For complex tensors (last dim = 2), operate on dim=-2, not dim=-1
  const resolvedDim = dim < 0 ? ndim + dim : dim;
  const effectiveDim = (isComplexInput && resolvedDim === ndim - 1) ? ndim - 2 : resolvedDim;
  const d = effectiveDim;
  
  const dimSize = shape[d];
  const N = n ?? dimSize;

  const inputData = await input.toArray();
  const outerDimsSize = shape.slice(0, d).reduce((a, b) => a * b, 1);
  // Don't count complex dimension as inner dimension
  const innerEnd = (isComplexInput && d !== ndim - 1) ? ndim - 1 : ndim;
  const innerDimsSize = shape.slice(d + 1, innerEnd).reduce((a, b) => a * b, 1);
  const batchCount = outerDimsSize * innerDimsSize;

  const outputReal: number[] = [];
  const outputImag: number[] = [];

  for (let batch = 0; batch < batchCount; batch++) {
    if (isComplexInput) {
      const offset = batch * dimSize * 2;
      const signalReal = new Float64Array(N);
      const signalImag = new Float64Array(N);
      const copyLen = Math.min(N, dimSize);
      for (let i = 0; i < copyLen; i++) {
        signalReal[i] = inputData[offset + i * 2];
        signalImag[i] = inputData[offset + i * 2 + 1];
      }
      const { re, im } = _fft1dComplex(signalReal, signalImag);
      outputReal.push(...re);
      outputImag.push(...im);
    } else {
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
  const shape = input.shape;
  
  // For complex tensors with last dim=2 (real/imag), default to dim=-2
  const resolvedDim = dim < 0 ? ndim + dim : dim;
  const effectiveDim = (shape[ndim - 1] === 2 && resolvedDim === ndim - 1) ? ndim - 2 : resolvedDim;
  const d = effectiveDim;
  
  const dimSize = shape[d];
  const N = n ?? dimSize;

  const inputData = await input.toArray();
  const outerDimsSize = shape.slice(0, d).reduce((a, b) => a * b, 1);
  // Don't count the complex dimension (last dim=2) as an inner dimension
  const innerEnd = (shape[ndim - 1] === 2 && d !== ndim - 1) ? ndim - 1 : ndim;
  const innerDimsSize = shape.slice(d + 1, innerEnd).reduce((a, b) => a * b, 1);
  const batchCount = outerDimsSize * innerDimsSize;

  const outputReal: number[] = [];
  const outputImag: number[] = [];

  for (let batch = 0; batch < batchCount; batch++) {
    const offset = batch * dimSize * 2; // Complex: each element has real+imag
    const signalReal = new Float64Array(N);
    const signalImag = new Float64Array(N);

    const copyLen = Math.min(N, dimSize);
    for (let i = 0; i < copyLen; i++) {
      // Complex tensor stored as [real0, imag0, real1, imag1, ...]
      signalReal[i] = inputData[offset + i * 2];
      signalImag[i] = inputData[offset + i * 2 + 1];
    }

    const { re, im } = _ifft1d(signalReal, signalImag);
    outputReal.push(...re);
    outputImag.push(...im);
  }

  const outShape = [...shape.slice(0, d), N, ...shape.slice(d + 1, -1)];
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
/**
 * Computes the inverse of rfft.
 * @pytorch torch.fft.irfft
 */
export async function irfft(
  input: Tensor,
  n?: number,
  dim: number = -1,
  norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  const ndim = input.dim();
  const shape = input.shape;

  // For complex tensors (last dim = 2), operate on dim=-2, not dim=-1
  const resolvedDim = dim < 0 ? ndim + dim : dim;
  const effectiveDim = (shape[ndim - 1] === 2 && resolvedDim === ndim - 1) ? ndim - 2 : resolvedDim;
  const d = effectiveDim;
  const dimSize = shape[d];

  // Output length for real signal
  const outputLen = n ?? (dimSize - 1) * 2;

  // Reconstruct full conjugate-symmetric spectrum from half spectrum
  const inputData = await input.toArray();
  const halfSize = dimSize; // N/2 + 1 bins
  
  // Build full spectrum with Hermitian symmetry
  const fullReal: number[] = [];
  const fullImag: number[] = [];
  
  for (let batch = 0; batch < inputData.length / (halfSize * 2); batch++) {
    const baseOffset = batch * halfSize * 2;
    
    // Copy the half spectrum
    for (let i = 0; i < halfSize; i++) {
      fullReal.push(inputData[baseOffset + i * 2]);
      fullImag.push(inputData[baseOffset + i * 2 + 1]);
    }
    
    // Fill in conjugate symmetry: X[k] = conj(X[N-k])
    for (let i = 1; i < outputLen - halfSize + 1; i++) {
      const conjIdx = halfSize - i;
      if (conjIdx >= 0 && conjIdx < halfSize) {
        fullReal.push(inputData[baseOffset + conjIdx * 2]);
        fullImag.push(-inputData[baseOffset + conjIdx * 2 + 1]);
      } else {
        fullReal.push(0);
        fullImag.push(0);
      }
    }
  }

  // Create full complex tensor
  const fullComplexShape = [...shape.slice(0, d), outputLen, ...shape.slice(d + 1)];
  const complexData = [] as number[];
  for (let i = 0; i < fullReal.length; i++) {
    complexData.push(fullReal[i], fullImag[i]);
  }
  const fullTensor = tensor(complexData, { dtype: input.dtype as any }).reshape(fullComplexShape as any);

  // Do IFFT on the full complex input
  const complexResult = await ifft(fullTensor, outputLen, d, norm);

  // Extract just the real part
  const data = await complexResult.toArray();
  const realData: number[] = [];

  // Complex stored as [real0, imag0, real1, imag1, ...]
  for (let i = 0; i < data.length; i += 2) {
    realData.push(data[i]);
  }

  // Build output shape: replace dim d with outputLen, remove complex dim
  const outShape = [...shape];
  outShape[d] = outputLen;
  const finalShape = outShape.filter((_, i) => i !== outShape.length - 1);

  return tensor(realData, { dtype: input.dtype as any }).reshape(finalShape as any);
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
  const freqs: number[] = [];
  
  // Standard FFT frequency ordering:
  // [0, 1, ..., N/2-1, -N/2, ..., -1] / (N * d) for even N
  // [0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1] / (N * d) for odd N
  for (let i = 0; i < n; i++) {
    if (i < Math.ceil(n / 2)) {
      freqs.push(i * scale / n);
    } else {
      freqs.push((i - n) * scale / n);
    }
  }
  
  return tensor(freqs);
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
 * 1D FFT for real signals using Cooley-Tukey algorithm.
 */
function _fft1d(signal: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signal.length;
  const isPowerOf2 = (N > 0) && ((N & (N - 1)) === 0);
  
  if (isPowerOf2 && N > 1) {
    // Use Cooley-Tukey FFT (O(N log N))
    const complexSignal: [number, number][] = [];
    for (let i = 0; i < N; i++) {
      complexSignal.push([signal[i], 0]);
    }
    const result = _fftRecursiveComplex(complexSignal, true);

    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      re[i] = result[i][0];
      im[i] = result[i][1];
    }
    return { re, im };
  } else {
    // Use direct DFT for non-power-of-2 (O(N²))
    const imag = new Float64Array(N);
    return _dftDirect(signal, imag, true);
  }
}

/**
 * Direct DFT for arbitrary sizes (O(N²) but numerically correct).
 * Used as fallback for non-power-of-2 sizes.
 */
function _dftDirect(signalReal: Float64Array, signalImag: Float64Array, forward: boolean): { re: Float64Array; im: Float64Array } {
  const N = signalReal.length;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  
  for (let k = 0; k < N; k++) {
    let sumRe = 0, sumIm = 0;
    for (let n = 0; n < N; n++) {
      // FFT: exp(-2πikn/N), IFFT: exp(+2πikn/N)
      const angle = (forward ? -1 : 1) * 2 * Math.PI * k * n / N;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
      sumRe += signalReal[n] * cos - signalImag[n] * sin;
      sumIm += signalReal[n] * sin + signalImag[n] * cos;
    }
    re[k] = sumRe;
    im[k] = sumIm;
  }
  
  return { re, im };
}

/**
 * 1D FFT for complex signals.
 */
function _fft1dComplex(signalReal: Float64Array, signalImag: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signalReal.length;
  const isPowerOf2 = (N > 0) && ((N & (N - 1)) === 0);
  
  if (isPowerOf2 && N > 1) {
    // Use Cooley-Tukey FFT (O(N log N))
    const complexSignal: [number, number][] = [];
    for (let i = 0; i < N; i++) {
      complexSignal.push([signalReal[i], signalImag[i]]);
    }
    
    const result = _fftRecursiveComplex(complexSignal, true); // forward = true

    const re = new Float64Array(N);
    const im = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      re[i] = result[i][0];
      im[i] = result[i][1];
    }
    return { re, im };
  } else {
    // Use direct DFT for non-power-of-2 (O(N²))
    return _dftDirect(signalReal, signalImag, true);
  }
}
function _ifft1d(signalReal: Float64Array, signalImag: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signalReal.length;
  const isPowerOf2 = (N > 0) && ((N & (N - 1)) === 0);
  
  if (isPowerOf2 && N > 1) {
    // Use direct DFT for IFFT (numerically correct)
    const result = _dftDirect(signalReal, signalImag, false);
    // Scale by 1/N
    for (let i = 0; i < N; i++) {
      result.re[i] /= N;
      result.im[i] /= N;
    }
    return result;
  } else {
    // Use direct DFT for non-power-of-2 (O(N²))
    const result = _dftDirect(signalReal, signalImag, false);
    // Scale by 1/N
    for (let i = 0; i < N; i++) {
      result.re[i] /= N;
      result.im[i] /= N;
    }
    return result;
  }
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
 * @param forward - true for FFT (exp negative), false for IFFT (exp positive)
 */
function _fftRecursiveComplex(signal: [number, number][], forward: boolean = true): [number, number][] {
  const N = signal.length;
  if (N === 1) return [[signal[0][0], signal[0][1]]];

  const even: [number, number][] = [];
  const odd: [number, number][] = [];
  for (let i = 0; i < N / 2; i++) {
    even.push(signal[2 * i]);
    odd.push(signal[2 * i + 1]);
  }

  const evenFFT = _fftRecursiveComplex(even, forward);
  const oddFFT = _fftRecursiveComplex(odd, forward);

  const result: [number, number][] = new Array(N);
  for (let k = 0; k < N / 2; k++) {
    // Twiddle factor: exp(-2πik/N) for FFT, exp(+2πik/N) for IFFT
    const angle = (forward ? -1 : 1) * 2 * Math.PI * k / N;
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
