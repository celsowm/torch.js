import { Tensor } from '../tensor';
import * as ops from '../ops';
import { choleskyDecomp } from './advanced';
import {
  getDevice,
  getOrCreatePipeline,
    calculateWorkgroups,
      BufferUsage,
      TRIANGULAR_SOLVE_SHADER,
      LU_SHADER,
    } from '../backend';
    
    /**
     * Computes the LU factorization with partial pivoting of a matrix A.
     * @pytorch torch.linalg.lu_factor
     */
    export const lu_factor = (A: Tensor, pivot: boolean = true): [Tensor, Tensor] => {
      if (!pivot) throw new Error('linalg.lu_factor: pivot=false not yet implemented');
      if (A.dim() < 2) throw new Error('linalg.lu_factor: expected tensor with at least 2 dimensions');
      
      const n = A.shape[A.dim() - 1];
      const m = A.shape[A.dim() - 2];
      if (n !== m) throw new Error('linalg.lu_factor: expected square matrix');
    
          const batch = A.dim() > 2 ? A.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
          const device = getDevice();
          
          // Result matrix (packed L and U)
          const LU = A.clone();
          // Permutation vector (1-indexed in PyTorch, but we'll use 0-indexed internally and maybe convert later)
              // Actually PyTorch pivots are row swaps.
              const P = ops.arange(n).unsqueeze(0).expand([batch, n]).to('uint32').clone();
              
              console.log('lu_factor P details:', {
                  numel: P.numel(),
                  dtype: P.dtype,
                  shape: P.shape,
                  buffer_size: (P.buffer as any).size
              });
          
            const pipelinePivot = getOrCreatePipeline(LU_SHADER, 'lu_pivot');
      const pipelineUpdate = getOrCreatePipeline(LU_SHADER, 'lu_update');
    
      for (let k = 0; k < n; k++) {
        const paramsData = new Uint32Array([n, batch, k, 0]);
        const paramsBuffer = device.createBuffer({
          size: 16,
          usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    
        const bindGroup = device.createBindGroup({
          layout: pipelinePivot.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: LU.buffer, offset: 0, size: LU.buffer.size } },
            { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
            { binding: 2, resource: { buffer: P.buffer, offset: 0, size: P.buffer.size } },
          ],
        });
    
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipelinePivot);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(batch / 256), 1, 1);
        passEncoder.end();
        
        if (k < n - 1) {
            const remaining = n - 1 - k;
            const bindGroupUpdate = device.createBindGroup({
                layout: pipelineUpdate.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: LU.buffer, offset: 0, size: LU.buffer.size } },
                    { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
                    { binding: 2, resource: { buffer: P.buffer, offset: 0, size: P.buffer.size } },
                ],
            });
            const passEncoderUpdate = commandEncoder.beginComputePass();
            passEncoderUpdate.setPipeline(pipelineUpdate);
            passEncoderUpdate.setBindGroup(0, bindGroupUpdate);
            passEncoderUpdate.dispatchWorkgroups(Math.ceil(remaining / 16), Math.ceil(remaining / 16), batch);
            passEncoderUpdate.end();
        }
        
        device.queue.submit([commandEncoder.finish()]);
      }
    
      return [LU, P];
    };
    
    /**
     * Computes the solution
     to a system of linear equations with a triangular coefficient matrix A and multiple right-hand sides B.
   * @pytorch torch.linalg.solve_triangular
   */
  export const solve_triangular = (A: Tensor, B: Tensor, upper: boolean, left: boolean = true, unitriangular: boolean = false): Tensor => {
    if (!left) throw new Error('linalg.solve_triangular: right-side solve (left=false) not yet implemented');
    if (unitriangular) throw new Error('linalg.solve_triangular: unitriangular=true not yet implemented');
    if (A.dim() < 2 || B.dim() < 2) throw new Error('linalg.solve_triangular: expected tensors with at least 2 dimensions');
    
    const n = A.shape[A.dim() - 1];
    const m = B.shape[B.dim() - 1]; // num RHS columns
    const batch = A.dim() > 2 ? A.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
    const device = getDevice();
  
    // Work on a copy of B
    const X = B.clone();
    const pipeline = getOrCreatePipeline(TRIANGULAR_SOLVE_SHADER, upper ? 'backward_sub_step' : 'forward_sub_step');
  
    for (let step = 0; step < n; step++) {
      const k = upper ? (n - 1 - step) : step;
      const paramsData = new Uint32Array([n, m, batch, k]);
      const paramsBuffer = device.createBuffer({
        size: 16,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
  
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: A.buffer, offset: 0, size: A.buffer.size } },
          { binding: 1, resource: { buffer: X.buffer, offset: 0, size: X.buffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        ],
      });
  
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      // Workgroup x covers columns of B, y covers batch
      passEncoder.dispatchWorkgroups(Math.ceil(m / 256), batch, 1);
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
    }
  
    return X;
  };

/**
 * Computes the Cholesky decomposition
 of a complex Hermitian or real symmetric positive-definite matrix.
 * @pytorch torch.linalg.cholesky
 */
export const cholesky = async (A: Tensor, upper: boolean = false): Promise<Tensor> => {
  return choleskyDecomp(A, upper);
};

/**
 * Computes a vector or matrix norm.
 * @pytorch torch.linalg.norm
 */
export const norm = (input: Tensor, ord?: number | string, dim?: number | number[], keepdim: boolean = false): Tensor => {
  if (typeof dim === 'number') dim = [dim];

  // Matrix norm
  if ((dim === undefined && input.dim() === 2) || (dim !== undefined && dim.length === 2)) {
    const matrixDim = dim ?? [-2, -1];
    return matrix_norm(input, ord, matrixDim as [number, number], keepdim);
  }

  // Vector norm
  return vector_norm(input, ord as any, dim, keepdim);
};

/**
 * Computes a vector norm.
 * @pytorch torch.linalg.vector_norm
 */
export const vector_norm = (input: Tensor, ord: number | string = 2, dim?: number | number[], keepdim: boolean = false): Tensor => {
  if (typeof dim === 'number') dim = [dim];
  
  if (ord === 2 || ord === undefined) {
    return input.pow(2).sum(dim, keepdim).sqrt();
  }
  if (ord === 1) {
    return input.abs().sum(dim, keepdim);
  }
  if (ord === Infinity || ord === 'inf') {
    return input.abs().amax(dim, keepdim);
  }
  if (ord === -Infinity || ord === '-inf') {
    return input.abs().amin(dim, keepdim);
  }
  if (ord === 0) {
    // Number of non-zero elements
    return input.ne(0).sum(dim, keepdim);
  }
  if (typeof ord === 'number') {
    return input.abs().pow(ord).sum(dim, keepdim).pow(1 / ord);
  }
  throw new Error(`linalg.vector_norm: ord=${ord} not supported`);
};

/**
 * Computes a matrix norm.
 * @pytorch torch.linalg.matrix_norm
 */
export const matrix_norm = (input: Tensor, ord: number | string = 'fro', dim: [number, number] = [-2, -1], keepdim: boolean = false): Tensor => {
  if (ord === 'fro' || ord === 2) {
    return input.pow(2).sum(dim, keepdim).sqrt();
  }
  if (ord === 'nuc') {
    // Requires SVD
    throw new Error('linalg.matrix_norm: ord="nuc" (nuclear norm) not yet implemented');
  }
  if (ord === 1) {
    // Max absolute column sum
    const absSum = input.abs().sum([dim[0]], keepdim);
    const reduceDim = dim[1] < 0 ? dim[1] : (keepdim ? dim[1] : dim[1] - 1);
    return absSum.amax([reduceDim], keepdim);
  }
  if (ord === Infinity || ord === 'inf') {
    // Max absolute row sum
    const absSum = input.abs().sum([dim[1]], keepdim);
    const reduceDim = dim[0] < 0 ? dim[0] : (keepdim ? dim[0] : dim[0] - 1);
    return absSum.amax([reduceDim], keepdim);
  }
  throw new Error(`linalg.matrix_norm: ord=${ord} not supported`);
};

/**
 * Alias for torch.diagonal() with defaults dim1= -2, dim2= -1.
 * @pytorch torch.linalg.diagonal
 */
export const diagonal = (input: Tensor, offset: number = 0, dim1: number = -2, dim2: number = -1): Tensor => {
    // For 2D tensors, we can use ops.diag
    if (input.dim() === 2) {
        return ops.diag(input, offset);
    }
    // TODO: support N-D tensors by permuting and then taking diag
    throw new Error('linalg.diagonal: only 2D tensors supported currently');
};

/**
 * Alias for torch.matmul()
 * @pytorch torch.linalg.matmul
 */
export const matmul = ops.matmul;

/**
 * Computes the dot product of two batches of vectors along a dimension.
 * @pytorch torch.linalg.vecdot
 */
export const vecdot = (x: Tensor, y: Tensor, dim: number = -1): Tensor => {
    return x.mul(y).sum(dim);
};

/**
 * Generates a Vandermonde matrix.
 * @pytorch torch.linalg.vander
 */
export const vander = (x: Tensor, N?: number, increasing: boolean = false): Tensor => {
    const n = x.shape[0];
    const m = N ?? n;
    
    const xCol = x.unsqueeze(1); // [n, 1]
    
    if (!increasing) {
        // [m-1, m-2, ..., 0]
        const exponents = ops.arange(m - 1, -1, -1).to(x.dtype).unsqueeze(0);
        return xCol.pow(exponents);
    } else {
        // [0, 1, ..., m-1]
        const exponents = ops.arange(m).to(x.dtype).unsqueeze(0);
        return xCol.pow(exponents);
    }
};

/**
 * Computes the n-th power of a square matrix for an integer n.
 * @pytorch torch.linalg.matrix_power
 */
export const matrix_power = async (A: Tensor, n: number): Promise<Tensor> => {
    if (A.dim() < 2) throw new Error('linalg.matrix_power: expected tensor with at least 2 dimensions');
    const h = A.shape[A.dim() - 2];
    const w = A.shape[A.dim() - 1];
    if (h !== w) throw new Error('linalg.matrix_power: expected square matrix');
    
    if (n === 0) {
        return ops.eye(h).expand(A.shape);
    }
    if (n < 0) {
        return matrix_power(await inv(A), -n);
    }
    
    let res = A;
    for (let i = 1; i < n; i++) {
        res = ops.matmul(res, A);
    }
    return res;
};

/**
 * Applies row permutation P to matrix B in-place.
 * P is the swap-sequence from lu_factor: for each step k, if P[k] !== k, rows k and P[k] were swapped.
 * Swaps are replayed in reverse to get the inverse permutation (i.e., P^T).
 * B shape: [batch, n, m]
 */
function applyPermutation(B: Tensor, P: Tensor, n: number): void {
  const batch = B.shape[0];
  const m = B.shape[2];
  const device = getDevice();

  // Read P from GPU
  const pSize = P.numel() * 4;
  const pStaging = device.createBuffer({ size: pSize, usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST });
  const pCmd = device.createCommandEncoder();
  pCmd.copyBufferToBuffer(P.buffer, 0, pStaging, 0, pSize);
  device.queue.submit([pCmd.finish()]);
  // We need the buffer mapped; use a synchronous read approach
  // Read via the global buffer utility if available
  const pData = new Uint32Array(P.numel());
  // Use the same pattern as the shader test - read buffer directly
  // Actually let's use getDevice() readBuffer or create a readBuffer approach
  // For now, use a simpler approach: dispatch a shader to apply permutation

  // Read P data via a readBuffer utility
  const readStaging = device.createBuffer({
    size: pSize,
    usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
  });
  const cmd = device.createCommandEncoder();
  cmd.copyBufferToBuffer(P.buffer, 0, readStaging, 0, pSize);
  device.queue.submit([cmd.finish()]);
  
  // Map the staging buffer
  const mapped = readStaging.mapAsync(0, 0, pSize).then(() => {
    const arr = new Uint32Array(readStaging.getMappedRange(0, pSize));
    pData.set(arr);
    readStaging.unmap();
  });
  // Synchronous wait isn't possible; we need a GPU-only permutation approach
}

/**
 * Computes the inverse of a square matrix if it exists.
 * @pytorch torch.linalg.inv
 */
export const inv = async (A: Tensor): Promise<Tensor> => {
  const [m, n] = [A.shape[A.dim() - 2], A.shape[A.dim() - 1]];
  if (m !== n) throw new Error('linalg.inv: expected square matrix');
  
  const data = await A.toArray();
  
  // For 1x1 matrix, just invert the single element
  if (n === 1) {
    return ops.tensor([1 / data[0]], { dtype: 'float32' }).reshape([1, 1]);
  }
  
  // For 2x2 matrix, use direct formula
  if (n === 2) {
    const [a, b, c, d] = [data[0], data[1], data[2], data[3]];
    const det = a * d - b * c;
    if (Math.abs(det) < 1e-12) throw new Error('linalg.inv: singular matrix');
    const invData = [d / det, -b / det, -c / det, a / det];
    return ops.tensor(invData, { dtype: 'float32' }).reshape([2, 2]);
  }
  
  // For 3x3 matrix, use adjugate method
  if (n === 3) {
    const [a, b, c, d, e, f, g, h, i] = [data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]];
    
    const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    if (Math.abs(det) < 1e-12) throw new Error('linalg.inv: singular matrix');
    
    const invData = [
      (e*i - f*h) / det, (c*h - b*i) / det, (b*f - c*e) / det,
      (f*g - d*i) / det, (a*i - c*g) / det, (c*d - a*f) / det,
      (d*h - e*g) / det, (b*g - a*h) / det, (a*e - b*d) / det
    ];
    return ops.tensor(invData, { dtype: 'float32' }).reshape([3, 3]);
  }
  
  // For larger matrices, use Gaussian elimination
  // Create augmented matrix [A | I]
  const aug: number[][] = [];
  for (let r = 0; r < n; r++) {
    const row: number[] = [];
    for (let c = 0; c < n; c++) {
      row.push(data[r * n + c]);
    }
    for (let c = 0; c < n; c++) {
      row.push(r === c ? 1 : 0);
    }
    aug.push(row);
  }
  
  // Forward elimination
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }
    }
    if (maxVal < 1e-12) throw new Error('linalg.inv: singular matrix');
    
    // Swap rows
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    
    // Scale pivot row
    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) {
      aug[col][j] /= pivot;
    }
    
    // Eliminate other rows
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }
  
  // Extract inverse from augmented matrix
  const invData: number[] = [];
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      invData.push(aug[r][n + c]);
    }
  }
  
  return ops.tensor(invData, { dtype: 'float32' }).reshape([n, n]);
};

/**
 * Computes the cross product of two 3-dimensional vectors.
 * @pytorch torch.linalg.cross
 */
export const cross = (input: Tensor, other: Tensor, dim: number = -1): Tensor => {
    const d = dim < 0 ? dim + input.dim() : dim;
    if (input.shape[d] !== 3 || other.shape[d] !== 3) {
        throw new Error('linalg.cross: expected dimension of size 3');
    }
    const ux = input.narrow(d, 0, 1).squeeze(d);
    const uy = input.narrow(d, 1, 1).squeeze(d);
    const uz = input.narrow(d, 2, 1).squeeze(d);
    const vx = other.narrow(d, 0, 1).squeeze(d);
    const vy = other.narrow(d, 1, 1).squeeze(d);
    const vz = other.narrow(d, 2, 1).squeeze(d);
    const cx = uy.mul(vz).sub(uz.mul(vy)).unsqueeze(d);
    const cy = uz.mul(vx).sub(ux.mul(vz)).unsqueeze(d);
    const cz = ux.mul(vy).sub(uy.mul(vx)).unsqueeze(d);
    return ops.cat([cx, cy, cz], d);
};

export { svd, eigh, qr, solve, pinv } from './advanced';