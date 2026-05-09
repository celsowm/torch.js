// Test IFFT specifically

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

// Test IFFT: forward=false
const X_real = new Float64Array([10, -2, -2, -2]);
const X_imag = new Float64Array([0, 0, 2, -2]);

console.log('IFFT input:');
console.log('  real:', Array.from(X_real));
console.log('  imag:', Array.from(X_imag));

// Cooley-Tukey IFFT
const complexInput = [];
for (let i = 0; i < 4; i++) {
  complexInput.push([X_real[i], X_imag[i]]);
}
const ifftResult = _fftRecursiveComplex(complexInput, false);
console.log('\nCooley-Tukey IFFT (before scaling):');
console.log('  real:', ifftResult.map(v => v[0]));
console.log('  imag:', ifftResult.map(v => v[1]));

// Scale by 1/N
const N = 4;
console.log('\nAfter scaling by 1/N:');
const scaledReal = ifftResult.map(v => v[0] / N);
const scaledImag = ifftResult.map(v => v[1] / N);
console.log('  real:', scaledReal);
console.log('  imag:', scaledImag);

// Direct DFT IFFT
const directIfft = _dftDirect(X_real, X_imag, false);
console.log('\nDirect DFT IFFT (before scaling):');
console.log('  real:', Array.from(directIfft.re));
console.log('  imag:', Array.from(directIfft.im));

console.log('\nAfter scaling by 1/N:');
const directReal = Array.from(directIfft.re).map(v => v / N);
const directImag = Array.from(directIfft.im).map(v => v / N);
console.log('  real:', directReal);
console.log('  imag:', directImag);

// Expected: [1, 2, 3, 4]
console.log('\nExpected: [1, 2, 3, 4], imag=[0,0,0,0]');
console.log('Cooley-Tukey correct?', JSON.stringify(scaledReal.map((v,i) => Math.round(v*100)/100)) === JSON.stringify([1,2,3,4]));
