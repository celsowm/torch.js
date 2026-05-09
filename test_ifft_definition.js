// Test IFFT with mathematical definition: x[n] = (1/N) * sum(X[k] * exp(2*pi*i*k*n/N))

function _ifftMath(signalReal, signalImag) {
  const N = signalReal.length;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  
  for (let n = 0; n < N; n++) {
    let sumRe = 0, sumIm = 0;
    for (let k = 0; k < N; k++) {
      // exp(2*pi*i*k*n/N) = cos(2*pi*k*n/N) + i*sin(2*pi*k*n/N)
      const angle = 2 * Math.PI * k * n / N;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      // X[k] * exp(...) = (Xr[k] + i*Xi[k]) * (cos + i*sin)
      // = (Xr[k]*cos - Xi[k]*sin) + i*(Xr[k]*sin + Xi[k]*cos)
      sumRe += signalReal[k] * cos - signalImag[k] * sin;
      sumIm += signalReal[k] * sin + signalImag[k] * cos;
    }
    re[n] = sumRe / N;
    im[n] = sumIm / N;
  }
  
  return { re, im };
}

// Test: IFFT of [10, -2, -2, -2] + i[0, 0, 2, -2]
const X_real = new Float64Array([10, -2, -2, -2]);
const X_imag = new Float64Array([0, 0, 2, -2]);

console.log('IFFT input:');
console.log('  real:', Array.from(X_real));
console.log('  imag:', Array.from(X_imag));

const result = _ifftMath(X_real, X_imag);
console.log('\nIFFT result (math definition):');
console.log('  real:', Array.from(result.re));
console.log('  imag:', Array.from(result.im));

console.log('\nExpected: [1, 2, 3, 4], imag=[0,0,0,0]');

// Also test FFT -> IFFT roundtrip
function _fftMath(signalReal, signalImag) {
  const N = signalReal.length;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  
  for (let k = 0; k < N; k++) {
    let sumRe = 0, sumIm = 0;
    for (let n = 0; n < N; n++) {
      // exp(-2*pi*i*k*n/N)
      const angle = -2 * Math.PI * k * n / N;
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

console.log('\n\n=== FFT -> IFFT roundtrip ===');
const x_orig = new Float64Array([1, 2, 3, 4]);
const x_imag_orig = new Float64Array([0, 0, 0, 0]);

console.log('Original:', Array.from(x_orig));

const X = _fftMath(x_orig, x_imag_orig);
console.log('FFT result:');
console.log('  real:', Array.from(X.re));
console.log('  imag:', Array.from(X.im));

const xRec = _ifftMath(X.re, X.im);
console.log('IFFT of FFT:');
console.log('  real:', Array.from(xRec.re));
console.log('  imag:', Array.from(xRec.im));

let maxDiff = 0;
for (let i = 0; i < 4; i++) {
  maxDiff = Math.max(maxDiff, Math.abs(x_orig[i] - xRec.re[i]));
}
console.log('\nmaxAbsDiff:', maxDiff);
