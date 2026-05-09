// Debug Cooley-Tukey FFT step by step

function _fftRecursiveComplexDebug(signal, forward = true, depth = 0) {
  const N = signal.length;
  const indent = '  '.repeat(depth);
  console.log(`${indent}FFT input:`, signal, 'forward=', forward);
  
  if (N === 1) {
    console.log(`${indent}Base case:`, [[signal[0][0], signal[0][1]]]);
    return [[signal[0][0], signal[0][1]]];
  }

  const even = [];
  const odd = [];
  for (let i = 0; i < N / 2; i++) {
    even.push(signal[2 * i]);
    odd.push(signal[2 * i + 1]);
  }

  console.log(`${indent}even:`, even);
  console.log(`${indent}odd:`, odd);

  const e = _fftRecursiveComplexDebug(even, forward, depth + 1);
  const o = _fftRecursiveComplexDebug(odd, forward, depth + 1);

  console.log(`${indent}FFT(even):`, e);
  console.log(`${indent}FFT(odd):`, o);

  const result = [];
  for (let k = 0; k < N / 2; k++) {
    const angle = (forward ? -1 : 1) * 2 * Math.PI * k / N;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    
    console.log(`${indent}k=${k}: angle=${angle}, cos=${cos}, sin=${sin}`);
    console.log(`${indent}  odd[${k}]=`, o[k]);
    
    const wr = o[k][0] * cos - o[k][1] * sin;
    const wi = o[k][0] * sin + o[k][1] * cos;
    
    console.log(`${indent}  twiddle: [${wr}, ${wi}]`);
    console.log(`${indent}  even[${k}]=`, e[k]);
    
    const top = [e[k][0] + wr, e[k][1] + wi];
    const bottom = [e[k][0] - wr, e[k][1] - wi];
    
    console.log(`${indent}  top:`, top);
    console.log(`${indent}  bottom:`, bottom);
    
    result.push(top);
    result.push(bottom);
  }
  
  console.log(`${indent}result:`, result);
  return result;
}

// Test
const input = [[1, 0], [2, 0], [3, 0], [4, 0]];
console.log('Input:', input);
console.log('\n=== FFT Forward ===\n');
const result = _fftRecursiveComplexDebug(input, true);
console.log('\nFinal result:', result);
