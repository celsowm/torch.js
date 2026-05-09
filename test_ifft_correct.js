// Test correct IFFT formula: IFFT(X) = conj(FFT(conj(X))) / N

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

// Correct IFFT: conj(FFT(conj(X))) / N
function _ifftCorrect(signalReal, signalImag) {
  const N = signalReal.length;
  
  // Step 1: Conjugate input
  const conjInput = [];
  for (let i = 0; i < N; i++) {
    conjInput.push([signalReal[i], -signalImag[i]]);
  }
  
  // Step 2: FFT
  const fftResult = _fftRecursiveComplex(conjInput, true);
  
  // Step 3: Conjugate and scale
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    re[i] = fftResult[i][0] / N;
    im[i] = -fftResult[i][1] / N;  // Conjugate
  }
  
  return { re, im };
}

// Test
const X_real = new Float64Array([10, -2, -2, -2]);
const X_imag = new Float64Array([0, 0, 2, -2]);

console.log('IFFT input:');
console.log('  real:', Array.from(X_real));
console.log('  imag:', Array.from(X_imag));

const result = _ifftCorrect(X_real, X_imag);
console.log('\nIFFT result (correct formula):');
console.log('  real:', Array.from(result.re));
console.log('  imag:', Array.from(result.im));

console.log('\nExpected: [1, 2, 3, 4], imag=[0,0,0,0]');
const correct = JSON.stringify(Array.from(result.re)) === JSON.stringify([1,2,3,4]) &&
                JSON.stringify(Array.from(result.im)) === JSON.stringify([0,0,0,0]);
console.log('Correct?', correct);
