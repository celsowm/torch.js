// Compare Cooley-Tukey vs Direct DFT

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

// Test: Compare Cooley-Tukey vs Direct DFT
const x_real = new Float64Array([1, 2, 3, 4]);
const x_imag = new Float64Array([0, 0, 0, 0]);

console.log('Input:', Array.from(x_real));

// Direct DFT
const X_direct = _dftDirect(x_real, x_imag, true);
console.log('\nDirect DFT result:');
console.log('  real:', Array.from(X_direct.re));
console.log('  imag:', Array.from(X_direct.im));

// Cooley-Tukey FFT
const complexSignal = [];
for (let i = 0; i < 4; i++) {
  complexSignal.push([x_real[i], x_imag[i]]);
}
const X_ct = _fftRecursiveComplex(complexSignal, true);
console.log('\nCooley-Tukey FFT result:');
console.log('  real:', X_ct.map(v => v[0]));
console.log('  imag:', X_ct.map(v => v[1]));

// Compare
console.log('\nDifferences:');
for (let i = 0; i < 4; i++) {
  const diffReal = Math.abs(X_direct.re[i] - X_ct[i][0]);
  const diffImag = Math.abs(X_direct.im[i] - X_ct[i][1]);
  console.log(`  [${i}] real: ${diffReal}, imag: ${diffImag}`);
}
