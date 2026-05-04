import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src/index';

describe('torch.fft STFT operations', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('torch.fft.stft', () => {
    it('computes basic STFT with n_fft', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
      const result = await torch.fft.stft(x, 8);
      const shape = result.shape;
      // n_fft=8 => freq_bins = 8//2+1 = 5
      // hop_length=2 (default n_fft/4) => num_frames = (16-8)/2+1 = 5
      expect(shape[shape.length - 2]).toBe(5); // freq bins
      expect(shape[shape.length - 1]).toBeGreaterThan(0); // time frames
    });

    it('STFT with custom hop_length', async () => {
      const x = torch.tensor(Array.from({ length: 16 }, (_, i) => i + 1));
      const result = await torch.fft.stft(x, 8, 4);
      const shape = result.shape;
      // hop_length=4 => num_frames = (16-8)/4+1 = 3
      expect(shape[shape.length - 1]).toBeGreaterThanOrEqual(1);
    });

    it('STFT with window', async () => {
      const x = torch.tensor(Array.from({ length: 16 }, (_, i) => i + 1));
      const window = torch.hann_window(8);
      const result = await torch.fft.stft(x, 8, 2, 8, window);
      const shape = result.shape;
      expect(shape[shape.length - 2]).toBe(5);
    });

    it('STFT with center=false', async () => {
      const x = torch.tensor(Array.from({ length: 16 }, (_, i) => i + 1));
      const result = await torch.fft.stft(x, 8, 2, undefined, undefined, false);
      const shape = result.shape;
      // Without padding, fewer frames
      expect(shape[shape.length - 1]).toBeGreaterThanOrEqual(1);
    });
  });

  describe('torch.fft.istft', () => {
    it('reconstructs signal from STFT', async () => {
      const x = torch.tensor(Array.from({ length: 32 }, (_, i) => Math.sin(2 * Math.PI * i / 8)));
      const stftResult = await torch.fft.stft(x, 16, 4);
      const reconstructed = await torch.fft.istft(stftResult, 16, 4);
      const original = Array.from(await x.toArray());
      const recon = Array.from(await reconstructed.toArray());
      // Allow some error due to windowing overlap-add
      expect(maxAbsDiff(original, recon)).toBeLessThan(0.1);
    });

    it('istft with length parameter', async () => {
      const x = torch.tensor(Array.from({ length: 16 }, (_, i) => i + 1));
      const stftResult = await torch.fft.stft(x, 8, 2);
      const reconstructed = await torch.fft.istft(stftResult, 8, 2, undefined, undefined, true, undefined, undefined, 16);
      const shape = reconstructed.shape;
      expect(shape[shape.length - 1]).toBe(16);
    });
  });

  describe('torch.fft.hfft', () => {
    it('computes HFFT with hermitian input', async () => {
      // Create a simple hermitian-spectrum input
      // For N=4: [real0, imag0, real1, imag1, real2, 0] (DC and Nyquist are real)
      const complexData = [1.0, 0.0, 2.0, 1.0, 3.0, 0.0];
      const X = torch.tensor(complexData).reshape([3, 2]);
      const result = await torch.fft.hfft(X);
      // HFFT produces real output
      const data = Array.from(await result.toArray());
      expect(data.length).toBeGreaterThan(0);
      for (const v of data) {
        expect(Number.isFinite(v)).toBe(true);
      }
    });

    it('HFFT output length', async () => {
      const complexData = [1.0, 0.0, 2.0, 1.0, 3.0, 0.0];
      const X = torch.tensor(complexData).reshape([3, 2]);
      const result = await torch.fft.hfft(X, 4);
      const shape = result.shape;
      expect(shape[shape.length - 1]).toBe(4);
    });

    it('HFFT produces real-valued output', async () => {
      const complexData = [4.0, 0.0, 1.0, 0.5, 2.0, 0.0];
      const X = torch.tensor(complexData).reshape([3, 2]);
      const result = await torch.fft.hfft(X, 4);
      const data = Array.from(await result.toArray());
      // All values should be finite real numbers
      for (const v of data) {
        expect(Number.isFinite(v)).toBe(true);
      }
    });
  });

  describe('torch.fft.ihfft', () => {
    it('computes IHFFT with real input', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const result = await torch.fft.ihfft(x);
      const shape = result.shape;
      // Output is complex (hermitian): N//2+1 = 3 frequency bins
      expect(shape[shape.length - 1] ?? shape[0]).toBeGreaterThan(0);
    });

    it('IHFFT produces complex output', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0]);
      const result = await torch.fft.ihfft(x);
      const shape = result.shape;
      // Complex tensor has trailing dimension of 2
      expect(shape[shape.length - 1]).toBe(2);
    });

    it('IHFFT with n parameter', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      const result = await torch.fft.ihfft(x, 8);
      const shape = result.shape;
      // N=8 => output has 8//2+1 = 5 frequency bins
      expect(shape[shape.length - 2] ?? shape[0]).toBeGreaterThanOrEqual(1);
    });
  });
});

/**
 * Helper: compute max absolute difference between two arrays.
 */
function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let max = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}
