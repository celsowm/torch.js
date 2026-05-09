import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from './src/index';

describe('debug', () => {
  beforeAll(async () => {
    await torch.init();
  });
  it('debug fft', async () => {
    const complexData = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    const X = torch.tensor(complexData).reshape([4, 2]);
    const x = await torch.fft.ifft(X);
    console.log('ifft:', Array.from(await x.toArray()));
    const Xrec = await torch.fft.fft(x);
    console.log('fft(ifft):', Array.from(await Xrec.toArray()));
  });
});
