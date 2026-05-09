// Test FFT/IFFT roundtrip numerically

function _fftRecursiveComplex(signal, forward = true) {
  const N = signal.length;
  if (N === 1) return [[signal[0][0], signal[0][1]]];

  const even = [];
  const odd = [];
  for (let i = 0; i < N / 2; i++) {
    even.push(signal[2 * i]);
    odd.push(signal[2 * i + 1]);
  }

  const e = _fftRecursiveComplex(even, forward);
  const o = _fftRecursiveComplex(odd, forward);

  const result = [];
  for (let k = 0; k < N / 2; k++) {
    const angle = (forward ? -1 : 1) * 2 * Math.PI * k / N;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const wr = o[k][0] * cos - o[k][1] * sin;
    const wi = o[k][0] * sin + o[k][1] * cos;
    
    result.push([e[k][0] + wr, e[k][1] + wi]);
    result.push([e[k][0] - wr, e[k][1] - wi]);
  }
  return result;
}

function _ifft1d(signalReal, signalImag) {
  const N = signalReal.length;
  const complexSignal = [];
  for (let i = 0; i < N; i++) {
    complexSignal.push([signalReal[i], signalImag[i]]);
  }
  
  const result = _fftRecursiveComplex(complexSignal, false);
  
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    re[i] = result[i][0] / N;
    im[i] = result[i][1] / N;
  }
  return { re, im };
}

function _fft1d(signalReal, signalImag) {
  const N = signalReal.length;
  const complexSignal = [];
  for (let i = 0; i < N; i++) {
    complexSignal.push([signalReal[i], signalImag[i]]);
  }
  
  const result = _fftRecursiveComplex(complexSignal, true);
  
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    re[i] = result[i][0];
    im[i] = result[i][1];
  }
  return { re, im };
}

// Test: FFT of IFFT should equal original
const X_real = new Float64Array([1, 2, 3, 4]);
const X_imag = new Float64Array([0, 0, 0, 0]);

console.log('Original X:', Array.from(X_real));

// IFFT
const x = _ifft1d(X_real, X_imag);
console.log('IFFT result (x):');
console.log('  real:', Array.from(x.re));
console.log('  imag:', Array.from(x.im));

// FFT of IFFT result
const Xrec = _fft1d(x.re, x.im);
console.log('FFT of IFFT (Xrec):');
console.log('  real:', Array.from(Xrec.re));
console.log('  imag:', Array.from(Xrec.im));

// Compare
let maxDiff = 0;
for (let i = 0; i < 4; i++) {
  maxDiff = Math.max(maxDiff, Math.abs(X_real[i] - Xrec.re[i]));
}
console.log('\nmaxAbsDiff (real):', maxDiff);

maxDiff = 0;
for (let i = 0; i < 4; i++) {
  maxDiff = Math.max(maxDiff, Math.abs(X_imag[i] - Xrec.im[i]));
}
console.log('maxAbsDiff (imag):', maxDiff);
