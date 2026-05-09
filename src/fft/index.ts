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
  const isComplexInput = (input as any).is_complex === true;

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

  const outShape = [...shape.slice(0, d), N, ...shape.slice(d + 1, innerEnd)];
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
  // ifft input is always complex. Treat last dim of size 2 as (real, imag) even
  // when is_complex flag wasn't propagated through reshape().
  const isComplexInput =
    (input as any).is_complex === true ||
    (ndim >= 1 && shape[ndim - 1] === 2);

  // For complex tensors with last dim=2 (real/imag), default to dim=-2
  const resolvedDim = dim < 0 ? ndim + dim : dim;
  const effectiveDim = (isComplexInput && resolvedDim === ndim - 1) ? ndim - 2 : resolvedDim;
  const d = effectiveDim;
  
  const dimSize = shape[d];
  const N = n ?? dimSize;

  const inputData = await input.toArray();
  const outerDimsSize = shape.slice(0, d).reduce((a, b) => a * b, 1);
  // Don't count the complex dimension (last dim=2) as an inner dimension
  const innerEnd = (isComplexInput && d !== ndim - 1) ? ndim - 1 : ndim;
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
  _norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  const ndim = input.dim();
  const shape = input.shape;
  const d = dim < 0 ? ndim + dim : dim;
  const dimSize = shape[d];
  const N = n ?? dimSize;
  const nOut = (N >> 1) + 1;

  const inputData = await input.toArray();
  const outerDimsSize = shape.slice(0, d).reduce((a, b) => a * b, 1);
  const innerDimsSize = shape.slice(d + 1).reduce((a, b) => a * b, 1);
  const batchCount = outerDimsSize * innerDimsSize;

  const outReal: number[] = [];
  const outImag: number[] = [];

  for (let batch = 0; batch < batchCount; batch++) {
    const offset = batch * dimSize;
    const signal = new Float64Array(N);
    const copyLen = Math.min(N, dimSize);
    for (let i = 0; i < copyLen; i++) {
      signal[i] = inputData[offset + i];
    }
    // Packed real FFT: ~2× faster than running a full N-point complex FFT and
    // discarding the negative half of the spectrum.
    const { re, im } = _rfft1dPacked(signal);
    for (let i = 0; i < nOut; i++) {
      outReal.push(re[i]);
      outImag.push(im[i]);
    }
  }

  const outShape = [...shape.slice(0, d), nOut, ...shape.slice(d + 1)];
  return _createComplexTensor(outReal, outImag, outShape, input.dtype);
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
  _norm: 'backward' | 'forward' | 'ortho' | null = null
): Promise<Tensor> {
  const ndim = input.dim();
  const shape = input.shape;
  // irfft input is always complex. Treat last dim of size 2 as (real, imag) even
  // when is_complex flag wasn't propagated through reshape().
  const isComplexInput =
    (input as any).is_complex === true ||
    (ndim >= 1 && shape[ndim - 1] === 2);

  // For complex tensors (last dim = 2), operate on dim=-2, not dim=-1
  const resolvedDim = dim < 0 ? ndim + dim : dim;
  const effectiveDim = (isComplexInput && resolvedDim === ndim - 1) ? ndim - 2 : resolvedDim;
  const d = effectiveDim;
  const halfSize = shape[d]; // N/2 + 1 bins

  // Output length for real signal
  const outputLen = n ?? (halfSize - 1) * 2;

  const inputData = await input.toArray();
  // For a complex tensor [..., halfSize, 2] the per-batch element count is
  // halfSize * 2; for a non-complex view (rare path) treat the same.
  const elemsPerBatch = halfSize * 2;
  const batchCount = inputData.length / elemsPerBatch;

  const outData: number[] = [];
  const halfRe = new Float64Array(halfSize);
  const halfIm = new Float64Array(halfSize);

  for (let batch = 0; batch < batchCount; batch++) {
    const baseOffset = batch * elemsPerBatch;
    for (let i = 0; i < halfSize; i++) {
      halfRe[i] = inputData[baseOffset + i * 2];
      halfIm[i] = inputData[baseOffset + i * 2 + 1];
    }
    // Packed inverse real FFT: a single half-size complex IFFT instead of
    // reconstructing a full N-point Hermitian spectrum and running a full IFFT.
    const real = _irfft1dPacked(halfRe, halfIm, outputLen);
    for (let i = 0; i < outputLen; i++) {
      outData.push(real[i]);
    }
  }

  // Output shape: drop the trailing complex pair and replace dim d with outputLen.
  const finalShape = [...shape.slice(0, d), outputLen, ...shape.slice(d + 1, isComplexInput ? -1 : undefined)];
  return tensor(outData, { dtype: input.dtype as any }).reshape(finalShape as any);
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
  return tensor(complexData, { dtype: dtype as any, is_complex: true }).reshape(finalShape as any);
}

/**
 * Iterative in-place Cooley-Tukey FFT (O(N log N)) using a single Float64Array
 * pair. Replaces the previous recursive implementation that allocated
 * `[number, number][]` at every level (heavy on GC).
 *
 * @param re - real part (modified in place)
 * @param im - imag part (modified in place)
 * @param inverse - false: forward FFT (twiddle = e^{-2πi/N}); true: IFFT,
 *                  result is also scaled by 1/N at the end.
 */
function _fftIterative(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const N = re.length;
  if (N <= 1) return;

  // 1) Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; (j & bit) !== 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }

  // 2) Cooley-Tukey butterflies
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const angle = sign * 2 * Math.PI / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let i = 0; i < N; i += len) {
      let curRe = 1.0;
      let curIm = 0.0;
      for (let k = 0; k < half; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + half];
        const bIm = im[i + k + half];
        // t = cur * b
        const tRe = curRe * bRe - curIm * bIm;
        const tIm = curRe * bIm + curIm * bRe;
        re[i + k] = aRe + tRe;
        im[i + k] = aIm + tIm;
        re[i + k + half] = aRe - tRe;
        im[i + k + half] = aIm - tIm;
        // cur *= w
        const ncRe = curRe * wRe - curIm * wIm;
        const ncIm = curRe * wIm + curIm * wRe;
        curRe = ncRe;
        curIm = ncIm;
      }
    }
  }

  // 3) IFFT scaling
  if (inverse) {
    const inv = 1.0 / N;
    for (let i = 0; i < N; i++) {
      re[i] *= inv;
      im[i] *= inv;
    }
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

const _isPow2 = (n: number): boolean => n > 0 && (n & (n - 1)) === 0;

/**
 * 1D FFT for real signals.
 */
function _fft1d(signal: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signal.length;
  if (_isPow2(N) && N > 1) {
    const re = new Float64Array(signal); // copy real
    const im = new Float64Array(N);      // zero imag
    _fftIterative(re, im, false);
    return { re, im };
  }
  const imag = new Float64Array(N);
  return _dftDirect(signal, imag, true);
}

/**
 * 1D FFT for complex signals.
 */
function _fft1dComplex(signalReal: Float64Array, signalImag: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signalReal.length;
  if (_isPow2(N) && N > 1) {
    const re = new Float64Array(signalReal);
    const im = new Float64Array(signalImag);
    _fftIterative(re, im, false);
    return { re, im };
  }
  return _dftDirect(signalReal, signalImag, true);
}

/**
 * 1D IFFT for complex signals.
 *
 * Previously this used the O(N²) direct DFT even for power-of-2 sizes; now it
 * uses the same iterative Cooley-Tukey kernel (O(N log N)) with the inverse
 * twiddle sign and 1/N normalization.
 */
function _ifft1d(signalReal: Float64Array, signalImag: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signalReal.length;
  if (_isPow2(N) && N > 1) {
    const re = new Float64Array(signalReal);
    const im = new Float64Array(signalImag);
    _fftIterative(re, im, true);
    return { re, im };
  }
  const result = _dftDirect(signalReal, signalImag, false);
  const inv = 1.0 / N;
  for (let i = 0; i < N; i++) {
    result.re[i] *= inv;
    result.im[i] *= inv;
  }
  return result;
}

/**
 * Real-to-complex FFT producing only the N/2+1 non-redundant bins, using the
 * "packed" trick: pack the N real samples into N/2 complex samples and run a
 * half-size complex FFT, then untangle using the symmetry
 *
 *   X[k] = E[k] + W_N^k · O[k]
 *
 * where E and O are the even/odd halves of the spectrum. ~2× faster than
 * doing a full N-point complex FFT and discarding the negative half.
 */
function _rfft1dPacked(signal: Float64Array): { re: Float64Array; im: Float64Array } {
  const N = signal.length;
  // Edge cases
  if (N === 1) {
    return { re: new Float64Array([signal[0]]), im: new Float64Array([0]) };
  }
  if (N === 2) {
    return {
      re: new Float64Array([signal[0] + signal[1], signal[0] - signal[1]]),
      im: new Float64Array([0, 0]),
    };
  }
  // Fall back to full FFT when N is not even or its half is not a power of 2.
  const halfN = N >> 1;
  if ((N & 1) !== 0 || !_isPow2(halfN)) {
    const full = _fft1d(signal);
    const outLen = halfN + 1;
    return { re: full.re.slice(0, outLen), im: full.im.slice(0, outLen) };
  }

  // Pack: z[k] = x[2k] + i·x[2k+1]
  const zRe = new Float64Array(halfN);
  const zIm = new Float64Array(halfN);
  for (let k = 0; k < halfN; k++) {
    zRe[k] = signal[2 * k];
    zIm[k] = signal[2 * k + 1];
  }
  _fftIterative(zRe, zIm, false);

  const outRe = new Float64Array(halfN + 1);
  const outIm = new Float64Array(halfN + 1);

  // DC and Nyquist (both purely real for real input)
  outRe[0]      = zRe[0] + zIm[0];
  outIm[0]      = 0;
  outRe[halfN]  = zRe[0] - zIm[0];
  outIm[halfN]  = 0;

  for (let k = 1; k < halfN; k++) {
    const zkRe = zRe[k],          zkIm = zIm[k];
    const zmRe = zRe[halfN - k],  zmIm = zIm[halfN - k];
    // E[k] = 0.5 · (Z[k] + conj(Z[N/2-k]))
    const eRe = 0.5 * (zkRe + zmRe);
    const eIm = 0.5 * (zkIm - zmIm);
    // O[k] = -0.5j · (Z[k] - conj(Z[N/2-k]))
    //      = -0.5j · ((zkRe - zmRe) + (zkIm + zmIm) i)
    //      = 0.5·(zkIm + zmIm) - 0.5·(zkRe - zmRe) · i
    const oRe =  0.5 * (zkIm + zmIm);
    const oIm = -0.5 * (zkRe - zmRe);
    // W = exp(-2πik/N)
    const ang = -2 * Math.PI * k / N;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    // X[k] = E + W·O
    const woRe = wRe * oRe - wIm * oIm;
    const woIm = wRe * oIm + wIm * oRe;
    outRe[k] = eRe + woRe;
    outIm[k] = eIm + woIm;
  }

  return { re: outRe, im: outIm };
}

/**
 * Inverse real FFT: takes N/2+1 Hermitian-symmetric bins and produces N real
 * samples. Uses the inverse packed trick: a single half-size complex IFFT.
 */
function _irfft1dPacked(halfRe: Float64Array, halfIm: Float64Array, N: number): Float64Array {
  if (N === 1) {
    return new Float64Array([halfRe[0]]);
  }
  if (N === 2) {
    // x[0] = (X[0] + X[1]) / 2, x[1] = (X[0] - X[1]) / 2  (X[1] is real Nyquist)
    return new Float64Array([0.5 * (halfRe[0] + halfRe[1]), 0.5 * (halfRe[0] - halfRe[1])]);
  }
  const halfN = N >> 1;
  if ((N & 1) !== 0 || !_isPow2(halfN)) {
    // Fallback: reconstruct full Hermitian spectrum and run full IFFT.
    const re = new Float64Array(N);
    const im = new Float64Array(N);
    const inLen = halfRe.length;
    for (let k = 0; k < inLen && k < N; k++) {
      re[k] = halfRe[k];
      im[k] = halfIm[k];
    }
    for (let k = inLen; k < N; k++) {
      const conjIdx = N - k;
      if (conjIdx > 0 && conjIdx < inLen) {
        re[k] =  halfRe[conjIdx];
        im[k] = -halfIm[conjIdx];
      }
    }
    const out = _ifft1d(re, im);
    // Real part only
    return out.re;
  }

  // Build Z[k] for k=0..halfN-1 from the half spectrum, then IFFT of size N/2.
  const zRe = new Float64Array(halfN);
  const zIm = new Float64Array(halfN);

  // For k in [0, halfN):
  //   Let A = X[k], B = X[halfN - k] (with B[0] = X[halfN], the Nyquist)
  //   E[k] = 0.5 · (A + conj(B))
  //   O[k] = 0.5j · (A - conj(B))      (inverse of the forward packing)
  //   Z[k] = E[k] + conj(W_N^k) · O[k]
  for (let k = 0; k < halfN; k++) {
    const aRe = halfRe[k];
    const aIm = halfIm[k];
    const bIdx = halfN - k; // 0 <= bIdx <= halfN
    const bRe = halfRe[bIdx];
    const bIm = halfIm[bIdx];
    // E = 0.5 · (A + conj(B))
    const eRe = 0.5 * (aRe + bRe);
    const eIm = 0.5 * (aIm - bIm);
    // O = 0.5j · (A - conj(B)) = 0.5j · ((aRe - bRe) + (aIm + bIm) i)
    //   = -0.5·(aIm + bIm) + 0.5·(aRe - bRe) i
    const oRe = -0.5 * (aIm + bIm);
    const oIm =  0.5 * (aRe - bRe);
    // conj(W_N^k) = exp(+2πik/N)
    const ang = 2 * Math.PI * k / N;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    const woRe = wRe * oRe - wIm * oIm;
    const woIm = wRe * oIm + wIm * oRe;
    zRe[k] = eRe + woRe;
    zIm[k] = eIm + woIm;
  }

  _fftIterative(zRe, zIm, true); // half-size IFFT, scaled by 1/halfN

  // Unpack: x[2k] = Re(z[k]), x[2k+1] = Im(z[k])
  const out = new Float64Array(N);
  for (let k = 0; k < halfN; k++) {
    out[2 * k]     = zRe[k];
    out[2 * k + 1] = zIm[k];
  }
  return out;
}

// STFT/ISTFT
export { stft, istft, hfft, ihfft } from './stft';
