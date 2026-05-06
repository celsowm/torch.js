/**
 * STFT and ISTFT - Short-time Fourier transform utilities.
 * @pytorch torch.fft.stft, torch.fft.istft
 */

import { Tensor } from '../tensor';
import { tensor } from '../ops/creation';
import { fft, ifft } from './index';

/**
 * Short-time Fourier transform (STFT).
 * @pytorch torch.fft.stft
 */
export async function stft(
  input: Tensor,
  n_fft: number,
  hop_length?: number,
  win_length?: number,
  window?: Tensor,
  center: boolean = true,
  pad_mode: string = 'reflect',
  normalized: boolean = false,
  onesided: boolean = true,
): Promise<Tensor> {
  const ndim = input.dim();
  if (ndim < 1) throw new Error('stft requires at least 1D input');

  const shape = input.shape;
  const signalLen = shape[ndim - 1];
  const batchShape = shape.slice(0, -1);

  const hl = hop_length ?? Math.floor(n_fft / 4);
  const wl = win_length ?? n_fft;

  // Create Hann window
  const winData = window ? await _get_window_data(window, wl) : _hann_window(wl);

  // Pad if center=true
  let paddedData: Float64Array;
  if (center) {
    const padLen = Math.floor(n_fft / 2);
    const original = await input.toArray();
    paddedData = new Float64Array(signalLen + 2 * padLen);
    for (let i = 0; i < signalLen; i++) {
      paddedData[padLen + i] = original[i];
    }
  } else {
    const d = await input.toArray();
    paddedData = new Float64Array(d);
  }

  const effectiveLen = paddedData.length;
  const numFrames = Math.floor((effectiveLen - wl) / hl) + 1;
  const fftSize = onesided ? Math.floor(n_fft / 2) + 1 : n_fft;

  // STFT computation
  const batchCount = batchShape.reduce((a, b) => a * b, 1);
  const outputReal: number[] = [];
  const outputImag: number[] = [];

  for (let b = 0; b < batchCount; b++) {
    for (let frame = 0; frame < numFrames; frame++) {
      const start = b * (center ? signalLen + 2 * Math.floor(n_fft / 2) : signalLen) + frame * hl;
      const frameData = new Float64Array(n_fft);
      for (let i = 0; i < Math.min(wl, n_fft) && start + i < paddedData.length; i++) {
        frameData[i] = paddedData[start + i] * winData[i];
      }
      const { re, im } = _fft1d(frameData);
      const scale = normalized ? 1.0 / Math.sqrt(wl) : 1.0;
      for (let k = 0; k < fftSize; k++) {
        outputReal.push(re[k] * scale);
        outputImag.push(im[k] * scale);
      }
    }
  }

  const outShape = [...batchShape, fftSize, numFrames];
  return _interleave_to_complex(outputReal, outputImag, outShape);
}

/**
 * Inverse STFT.
 * @pytorch torch.fft.istft
 */
export async function istft(
  input: Tensor,
  n_fft: number,
  hop_length?: number,
  win_length?: number,
  window?: Tensor,
  center: boolean = true,
  normalized: boolean = false,
  onesided: boolean = true,
  length?: number,
): Promise<Tensor> {
  const ndim = input.dim();
  if (ndim < 3) throw new Error('istft requires at least 3D complex input [..., freq, time]');

  const shape = input.shape;
  const fftSize = shape[ndim - 2];
  const numFrames = shape[ndim - 1];
  const batchShape = shape.slice(0, -2);

  const hl = hop_length ?? Math.floor(n_fft / 4);
  const wl = win_length ?? n_fft;
  const winData = window ? await _get_window_data(window, wl) : _hann_window(wl);

  const signalLen = length ?? ((numFrames - 1) * hl + wl - (center ? n_fft : 0));
  const batchCount = batchShape.reduce((a, b) => a * b, 1);

  // Overlap-add
  const paddedLen = signalLen + (center ? n_fft : 0);
  const output = new Float64Array(batchCount * paddedLen);
  const windowSum = new Float64Array(batchCount * paddedLen);

  for (let b = 0; b < batchCount; b++) {
    for (let frame = 0; frame < numFrames; frame++) {
      // Extract complex frame
      const complexData = input.select(ndim - 1, frame); // [..., fftSize]
      const frameData = await _deinterleave_complex(complexData);
      const reconstructed = _ifft1d(frameData.real, frameData.imag);

      // Overlap-add
      const start = b * paddedLen + frame * hl;
      for (let i = 0; i < wl && start + i < paddedLen; i++) {
        const w = winData[i];
        output[start + i] += reconstructed[i] * w;
        windowSum[start + i] += w * w;
      }
    }
  }

  // Normalize
  const finalOutput: number[] = [];
  for (let i = 0; i < batchCount * paddedLen; i++) {
    finalOutput.push(windowSum[i] > 1e-8 ? output[i] / windowSum[i] : 0);
  }

  const outShape = [...batchShape, signalLen];
  return tensor(finalOutput, { dtype: 'float32' });
}

// Helper functions

function _hann_window(size: number): Float64Array {
  const w = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return w;
}

async function _get_window_data(window: Tensor, size: number): Promise<Float64Array> {
  const d = await window.toArray();
  const result = new Float64Array(size);
  for (let i = 0; i < Math.min(size, d.length); i++) {
    result[i] = d[i];
  }
  return result;
}

function _fft1d(signal: Float64Array): { re: number[]; im: number[] } {
  const N = signal.length;
  if (N <= 1) {
    return { re: [signal[0] || 0], im: [0] };
  }
  if ((N & (N - 1)) !== 0) {
    // Not power of 2, use DFT
    return _dft(signal);
  }
  // Cooley-Tukey FFT
  const even = new Float64Array(N / 2);
  const odd = new Float64Array(N / 2);
  for (let i = 0; i < N / 2; i++) {
    even[i] = signal[2 * i];
    odd[i] = signal[2 * i + 1];
  }
  const evenFFT = _fft1d(even);
  const oddFFT = _fft1d(odd);
  const re: number[] = new Array(N);
  const im: number[] = new Array(N);
  for (let k = 0; k < N / 2; k++) {
    const angle = (-2 * Math.PI * k) / N;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const twRe = oddFFT.re[k] * cos - oddFFT.im[k] * sin;
    const twIm = oddFFT.re[k] * sin + oddFFT.im[k] * cos;
    re[k] = evenFFT.re[k] + twRe;
    im[k] = evenFFT.im[k] + twIm;
    re[k + N / 2] = evenFFT.re[k] - twRe;
    im[k + N / 2] = evenFFT.im[k] - twIm;
  }
  return { re, im };
}

function _ifft1d(real: number[], imag: number[]): number[] {
  const N = real.length;
  const conjReal = real.slice();
  const conjImag = imag.map(x => -x);
  const result = _fft1d_from_complex(conjReal, conjImag);
  return result.re.map(x => x / N);
}

function _dft(signal: Float64Array): { re: number[]; im: number[] } {
  const N = signal.length;
  const re: number[] = new Array(N);
  const im: number[] = new Array(N);
  for (let k = 0; k < N; k++) {
    let sumRe = 0, sumIm = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      sumRe += signal[n] * Math.cos(angle);
      sumIm += signal[n] * Math.sin(angle);
    }
    re[k] = sumRe;
    im[k] = sumIm;
  }
  return { re, im };
}

function _fft1d_from_complex(re: number[], im: number[]): { re: number[]; im: number[] } {
  const N = re.length;
  if (N <= 1) return { re: [re[0]], im: [im[0]] };
  // Simplified: assume power of 2
  const evenRe: number[] = [], evenIm: number[] = [], oddRe: number[] = [], oddIm: number[] = [];
  for (let i = 0; i < N / 2; i++) {
    evenRe.push(re[2 * i]); evenIm.push(im[2 * i]);
    oddRe.push(re[2 * i + 1]); oddIm.push(im[2 * i + 1]);
  }
  const evenFFT = _fft1d_from_complex(evenRe, evenIm);
  const oddFFT = _fft1d_from_complex(oddRe, oddIm);
  const outRe: number[] = new Array(N);
  const outIm: number[] = new Array(N);
  for (let k = 0; k < N / 2; k++) {
    const angle = (-2 * Math.PI * k) / N;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const twRe = oddFFT.re[k] * cos - oddFFT.im[k] * sin;
    const twIm = oddFFT.re[k] * sin + oddFFT.im[k] * cos;
    outRe[k] = evenFFT.re[k] + twRe;
    outIm[k] = evenFFT.im[k] + twIm;
    outRe[k + N / 2] = evenFFT.re[k] - twRe;
    outIm[k + N / 2] = evenFFT.im[k] - twIm;
  }
  return { re: outRe, im: outIm };
}

async function _deinterleave_complex(tensor: Tensor): Promise<{ real: number[]; imag: number[] }> {
  const data = await tensor.toArray();
  const real: number[] = [];
  const imag: number[] = [];
  for (let i = 0; i < data.length; i += 2) {
    real.push(data[i]);
    imag.push(data[i + 1]);
  }
  return { real, imag };
}

function _interleave_to_complex(real: number[], imag: number[], shape: number[]): Tensor {
  const interleaved: number[] = [];
  for (let i = 0; i < real.length; i++) {
    interleaved.push(real[i], imag[i]);
  }
  return tensor(interleaved, { dtype: 'float32' });
}

/**
 * Half-complex to half FFT (HFFT).
 * @pytorch torch.fft.hfft
 * Returns the FFT of a Hermitian-symmetric signal, output is real.
 */
export async function hfft(
  input: Tensor,
  n?: number,
  dim: number = -1,
  norm: 'forward' | 'backward' | 'ortho' = 'backward',
): Promise<Tensor> {
  const data = await input.toArray();
  const shape = input.shape;
  const freqSize = shape[dim < 0 ? shape.length + dim : dim];
  const timeLen = n ?? 2 * (freqSize - 1);

  // Build conjugate-symmetric spectrum
  const fullSpec = new Float64Array(2 * timeLen);
  for (let i = 0; i < freqSize; i++) {
    fullSpec[2 * i] = data[2 * i];       // real
    fullSpec[2 * i + 1] = data[2 * i + 1]; // imag
  }
  // Mirror for negative frequencies
  for (let i = 1; i < timeLen - 1; i++) {
    const srcIdx = freqSize - i - 1;
    if (srcIdx > 0 && srcIdx < freqSize) {
      fullSpec[2 * (timeLen - i)] = data[2 * srcIdx];
      fullSpec[2 * (timeLen - i) + 1] = -data[2 * srcIdx + 1]; // conjugate
    }
  }

  // IFFT
  const result = _ifft1d(
    Array.from(fullSpec).filter((_, i) => i % 2 === 0),
    Array.from(fullSpec).filter((_, i) => i % 2 === 1),
  );

  // Scale
  const scale = norm === 'ortho' ? 1 / Math.sqrt(timeLen) : norm === 'forward' ? 1 / timeLen : 1;
  const outData = result.map(v => v * scale);

  return tensor(outData, { dtype: 'float32' });
}

/**
 * Inverse HFFT.
 * @pytorch torch.fft.ihfft
 */
export async function ihfft(
  input: Tensor,
  n?: number,
  dim: number = -1,
  norm: 'forward' | 'backward' | 'ortho' = 'backward',
): Promise<Tensor> {
  const data = await input.toArray();
  const timeLen = data.length;
  const shape = input.shape;
  const freqSize = n ? Math.floor(n / 2) + 1 : Math.floor(timeLen / 2) + 1;

  // FFT
  const padded = new Float64Array(timeLen);
  for (let i = 0; i < timeLen; i++) padded[i] = data[i];
  const fftResult = _fft1d(padded);

  // Scale
  const scale = norm === 'ortho' ? Math.sqrt(timeLen) : norm === 'forward' ? timeLen : 1;
  const outData: number[] = [];
  for (let i = 0; i < freqSize; i++) {
    outData.push(fftResult.re[i] * scale, fftResult.im[i] * scale);
  }

  return tensor(outData, { dtype: 'float32' });
}
