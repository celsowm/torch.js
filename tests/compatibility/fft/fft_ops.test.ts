import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

/**
 * Helper: compute max absolute difference between two float arrays.
 */
function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let max = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

/**
 * Helper: extract real and imaginary parts from complex tensor (shape: [..., 2]).
 */
async function getComplexParts(t: any): Promise<{ real: number[]; imag: number[] }> {
  const data = Array.from(await t.toArray()) as number[];
  const real: number[] = [];
  const imag: number[] = [];
  for (let i = 0; i < data.length; i += 2) {
    real.push(data[i]);
    imag.push(data[i + 1]);
  }
  return { real, imag };
}

describe('torch.fft', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('torch.fft.fft', () => {
    it('computes basic 1D FFT', async () => {
      const x = torch.tensor([1.0, 1.0, 1.0, 1.0]);
      const result = await torch.fft.fft(x);
      const { real, imag } = await getComplexParts(result);
      // FFT of constant = impulse at DC: [4, 0, 0, 0]
      expect(real[0]).toBeCloseTo(4.0, 3);
      expect(Math.abs(real[1])).toBeLessThan(1e-4);
      expect(Math.abs(real[2])).toBeLessThan(1e-4);
      expect(Math.abs(real[3])).toBeLessThan(1e-4);
    });

    it('computes FFT with n parameter (zero-padded or truncated)', async () => {
      const x = torch.tensor([1.0, 1.0, 1.0]);
      const result = await torch.fft.fft(x, 4);
      const shape = result.shape;
      // Complex tensor stored as [N, 2] for (real, imag)
      expect(shape).toEqual([4, 2]);
    });

    it('computes FFT along specified dimension', async () => {
      const x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      const result = await torch.fft.fft(x, undefined, 1);
      const shape = result.shape;
      // FFT along dim 1: [2, 3] → [2, 3, 2] for complex
      expect(shape).toEqual([2, 3, 2]);
    });

    it('reconstructs signal via ifft(fft(x)) = x', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const X = await torch.fft.fft(x);
      const xRec = await torch.fft.ifft(X);
      const { real } = await getComplexParts(xRec);
      const original = Array.from(await x.toArray());
      expect(maxAbsDiff(original, real)).toBeLessThan(1e-4);
    });

    it('Parseval theorem: energy preserved in time and frequency domain', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const X = await torch.fft.fft(x);
      const { real, imag } = await getComplexParts(X);
      // Time-domain energy: sum(|x|^2)
      const xData = Array.from(await x.toArray());
      const timeEnergy = xData.reduce((s, v) => s + v * v, 0);
      // Frequency-domain energy: sum(|X|^2) / N
      const N = xData.length;
      const freqEnergy = real.reduce((s, v, i) => s + v * v + imag[i] * imag[i], 0) / N;
      expect(timeEnergy).toBeCloseTo(freqEnergy, 3);
    });
  });

  describe('torch.fft.ifft', () => {
    it('computes basic 1D IFFT', async () => {
      const X = torch.tensor([4.0, 0.0, 0.0, 0.0]).reshape([4, 2]);
      // Set up proper complex tensor
      const complexData = [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      const Xcomplex = torch.tensor(complexData).reshape([4, 2]);
      const result = await torch.fft.ifft(Xcomplex);
      const { real } = await getComplexParts(result);
      // IFFT of [4,0,0,0] = [1,1,1,1]
      expect(real[0]).toBeCloseTo(1.0, 3);
      expect(real[1]).toBeCloseTo(1.0, 3);
      expect(real[2]).toBeCloseTo(1.0, 3);
      expect(real[3]).toBeCloseTo(1.0, 3);
    });

    it('computes IFFT with n parameter', async () => {
      const complexData = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      const X = torch.tensor(complexData).reshape([3, 2]);
      const result = await torch.fft.ifft(X, 4);
      const shape = result.shape;
      // IFFT with n=4: complex output stored as [4, 2]
      expect(shape).toEqual([4, 2]);
    });

    it('reconstructs via FFT of IFFT result', async () => {
      const complexData = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
      const X = torch.tensor(complexData).reshape([4, 2]);
      const x = await torch.fft.ifft(X);
      const Xrec = await torch.fft.fft(x);
      const { real: realOrig, imag: imagOrig } = await getComplexParts(X);
      const { real: realRec, imag: imagRec } = await getComplexParts(Xrec);
      expect(maxAbsDiff(realOrig, realRec)).toBeLessThan(1e-4);
      expect(maxAbsDiff(imagOrig, imagRec)).toBeLessThan(1e-4);
    });
  });

  describe('torch.fft.rfft', () => {
    it('computes RFFT for real input', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const result = await torch.fft.rfft(x);
      // N=4 => output has N//2+1 = 3 frequency bins
      const shape = result.shape;
      expect(shape).toEqual([3, 2]);
    });

    it('RFFT output is half spectrum of full FFT', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const fullFft = await torch.fft.fft(x);
      const rfftResult = await torch.fft.rfft(x);
      // Compare first N//2+1 entries
      const { real: fullReal, imag: fullImag } = await getComplexParts(fullFft);
      const { real: rfftReal, imag: rfftImag } = await getComplexParts(rfftResult);
      expect(maxAbsDiff(fullReal.slice(0, 3), rfftReal)).toBeLessThan(1e-4);
      expect(maxAbsDiff(fullImag.slice(0, 3), rfftImag)).toBeLessThan(1e-4);
    });

    it('RFFT with n parameter', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0]);
      const result = await torch.fft.rfft(x, 6);
      // N=6 => output has 6//2+1 = 4 frequency bins
      const shape = result.shape;
      expect(shape).toEqual([4, 2]);
    });
  });

  describe('torch.fft.irfft', () => {
    it('computes IRFFT to produce real output', async () => {
      // RFFT of [1,2,3,4] then IRFFT should give back [1,2,3,4]
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const X = await torch.fft.rfft(x);
      const xRec = await torch.fft.irfft(X, 4);
      const original = Array.from(await x.toArray());
      const recon = Array.from(await xRec.toArray());
      expect(maxAbsDiff(original, recon)).toBeLessThan(1e-4);
    });

    it('IRFFT with n parameter for output length', async () => {
      const complexData = [5.0, 0.0, -1.0, 1.0, 0.0, 0.0];
      const X = torch.tensor(complexData).reshape([3, 2]);
      const result = await torch.fft.irfft(X, 4);
      const shape = result.shape;
      expect(shape).toEqual([4]);
    });

    it('IRFFT produces real-valued output', async () => {
      const x = torch.tensor([1.0, 0.0, -1.0, 0.0]);
      const X = await torch.fft.rfft(x);
      const result = await torch.fft.irfft(X, 4);
      // Should be real (no imaginary part in tensor representation)
      const data = Array.from(await result.toArray());
      // All values should be finite real numbers
      for (const v of data) {
        expect(Number.isFinite(v)).toBe(true);
      }
    });
  });

  describe('torch.fft.fftshift', () => {
    it('shifts zero-frequency component to center (1D)', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const result = await torch.fft.fftshift(x);
      const data = Array.from(await result.toArray());
      // [1,2,3,4] -> shift by 2 -> [3,4,1,2]
      expect(data[0]).toBeCloseTo(3.0, 4);
      expect(data[1]).toBeCloseTo(4.0, 4);
      expect(data[2]).toBeCloseTo(1.0, 4);
      expect(data[3]).toBeCloseTo(2.0, 4);
    });

    it('fftshift and ifftshift are inverses', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]);
      const shifted = await torch.fft.fftshift(x);
      const unshifted = await torch.fft.ifftshift(shifted);
      const original = Array.from(await x.toArray());
      const result = Array.from(await unshifted.toArray());
      expect(maxAbsDiff(original, result)).toBeLessThan(1e-6);
    });

    it('fftshift along specific dimension', async () => {
      const x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      const result = await torch.fft.fftshift(x, 1);
      const data = Array.from(await result.toArray());
      // Each row shifted: [1,2,3] -> [3,1,2]
      expect(data[0]).toBeCloseTo(3.0, 4);
      expect(data[1]).toBeCloseTo(1.0, 4);
      expect(data[2]).toBeCloseTo(2.0, 4);
    });
  });

  describe('torch.fft.ifftshift', () => {
    it('inverse shift for even-length array', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const result = await torch.fft.ifftshift(x);
      const data = Array.from(await result.toArray());
      // [1,2,3,4] -> ifftshift by 2 -> [3,4,1,2]
      expect(data[0]).toBeCloseTo(3.0, 4);
      expect(data[1]).toBeCloseTo(4.0, 4);
      expect(data[2]).toBeCloseTo(1.0, 4);
      expect(data[3]).toBeCloseTo(2.0, 4);
    });

    it('inverse shift for odd-length array (differs from fftshift)', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]);
      const result = await torch.fft.ifftshift(x);
      const data = Array.from(await result.toArray());
      // ifftshift shifts by ceil(5/2)=3: [1,2,3,4,5] -> [3,4,5,1,2]
      expect(data[0]).toBeCloseTo(3.0, 4);
      expect(data[1]).toBeCloseTo(4.0, 4);
      expect(data[2]).toBeCloseTo(5.0, 4);
    });
  });

  describe('torch.fft.fftfreq', () => {
    it('computes frequency bins for N=4', async () => {
      const freqs = torch.fft.fftfreq(4);
      const data = Array.from(await freqs.toArray());
      expect(data.length).toBe(4);
      // [0, 0.25, -0.5, -0.25]
      expect(data[0]).toBeCloseTo(0.0, 4);
      expect(data[1]).toBeCloseTo(0.25, 4);
    });

    it('computes frequency bins with sample spacing d', async () => {
      const freqs = torch.fft.fftfreq(4, 0.5);
      const data = Array.from(await freqs.toArray());
      // scale = 1/0.5 = 2, so frequencies are doubled
      expect(data[0]).toBeCloseTo(0.0, 4);
      expect(data[1]).toBeCloseTo(0.5, 4);
    });

    it('frequency bins for even N have correct structure', async () => {
      const freqs = torch.fft.fftfreq(8);
      const data = Array.from(await freqs.toArray());
      // First element is 0, symmetric positive/negative frequencies
      expect(data[0]).toBeCloseTo(0.0, 4);
      expect(data[data.length - 1]).toBeCloseTo(-1/8, 4); // Last is -1/N
    });
  });

  describe('torch.fft.rfftfreq', () => {
    it('computes real FFT frequency bins for N=4', async () => {
      const freqs = torch.fft.rfftfreq(4);
      const data = Array.from(await freqs.toArray());
      // N//2+1 = 3 bins: [0, 0.25, 0.5]
      expect(data.length).toBe(3);
      expect(data[0]).toBeCloseTo(0.0, 4);
      expect(data[1]).toBeCloseTo(0.25, 4);
      expect(data[2]).toBeCloseTo(0.5, 4);
    });

    it('computes frequency bins with sample spacing d', async () => {
      const freqs = torch.fft.rfftfreq(4, 0.5);
      const data = Array.from(await freqs.toArray());
      // Frequencies doubled
      expect(data[0]).toBeCloseTo(0.0, 4);
      expect(data[1]).toBeCloseTo(0.5, 4);
      expect(data[2]).toBeCloseTo(1.0, 4);
    });

    it('max frequency is 0.5 (Nyquist)', async () => {
      const freqs = torch.fft.rfftfreq(8);
      const data = Array.from(await freqs.toArray());
      const maxFreq = Math.max(...data);
      expect(maxFreq).toBeCloseTo(0.5, 4);
    });
  });
});
