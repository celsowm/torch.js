// Test FFT/IFFT with DFT direct to find bug

function _dftDirect(signalReal, signalImag, forward) {
  const N = signalReal.length;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  
  for (let k = 0; k < N; k++) {
    let sumRe = 0, sumIm = 0;
    for (let n = 0; n < N; n++) {
      const angle = (forward ? -1 : 1) * 2 * Math.PI * k * n / N;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      sumRe += signalReal[n] * cos - signalImag[n] * sin;
      sumIm += signalReal[n] * sin + signalImag[n] * cos;
    }
    re[k] = sumRe;
    im[k] = sumIm;
  }
  
  return { re, im };
}

function _ifft1dDirect(signalReal, signalImag) {
  const N = signalReal.length;
  const result = _dftDirect(signalReal, signalImag, false);
  
  for (let i = 0; i < N; i++) {
    result.re[i] /= N;
    result.im[i] /= N;
  }
  return result;
}

// Test 1: Known IFFT -> FFT roundtrip
const X_real = new Float64Array([1, 2, 3, 4]);
const X_imag = new Float64Array([0, 0, 0, 0]);

console.log('Test 1: IFFT -> FFT roundtrip with direct DFT');
console.log('Original X:', Array.from(X_real));

// IFFT
const x = _ifft1dDirect(X_real, X_imag);
console.log('IFFT result (x):');
console.log('  real:', Array.from(x.re));
console.log('  imag:', Array.from(x.im));

// FFT using direct DFT
const Xrec = _dftDirect(x.re, x.im, true);
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

// Test 2: FFT -> IFFT roundtrip
console.log('\n\nTest 2: FFT -> IFFT roundtrip with direct DFT');
const x_orig = new Float64Array([1, 2, 3, 4]);
const x_imag_orig = new Float64Array([0, 0, 0, 0]);

console.log('Original x:', Array.from(x_orig));

// FFT
const X = _dftDirect(x_orig, x_imag_orig, true);
console.log('FFT result (X):');
console.log('  real:', Array.from(X.re));
console.log('  imag:', Array.from(X.im));

// IFFT
const xRec = _ifft1dDirect(X.re, X.im);
console.log('IFFT of FFT (xRec):');
console.log('  real:', Array.from(xRec.re));
console.log('  imag:', Array.from(xRec.im));

maxDiff = 0;
for (let i = 0; i < 4; i++) {
  maxDiff = Math.max(maxDiff, Math.abs(x_orig[i] - xRec.re[i]));
}
console.log('\nmaxAbsDiff (real):', maxDiff);
