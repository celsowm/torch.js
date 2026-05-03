import { Tensor } from '../tensor';
import * as ops from '../ops';
import {
  getDevice,
  getOrCreatePipeline,
    calculateWorkgroups,
      BufferUsage,
      CHOLESKY_SHADER,
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
export const cholesky = (A: Tensor, upper: boolean = false): Tensor => {
  if (A.dim() < 2) throw new Error('linalg.cholesky: expected tensor with at least 2 dimensions');
  const n = A.shape[A.dim() - 1];
  const m = A.shape[A.dim() - 2];
  if (n !== m) throw new Error('linalg.cholesky: expected square matrix');

  const batch = A.dim() > 2 ? A.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
  const device = getDevice();

  // Work on a copy
  const res = A.clone();
  const buffer = res.buffer;

  if (n <= 16) {
    // For small matrices, use a single pass shader
    const pipeline = getOrCreatePipeline(CHOLESKY_SHADER, 'cholesky_small');
    const paramsData = new Uint32Array([n, batch, 0, 0]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer, offset: 0, size: buffer.size } },
        { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(batch));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
  } else {
    // For larger matrices, use multi-pass approach
    const pipeline1 = getOrCreatePipeline(CHOLESKY_SHADER, 'cholesky_step1');
    const pipeline2 = getOrCreatePipeline(CHOLESKY_SHADER, 'cholesky_step2');

    for (let k = 0; k < n; k++) {
      const paramsData = new Uint32Array([n, batch, k, 0]);
      const paramsBuffer = device.createBuffer({
        size: 16,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);

      const bindGroup = device.createBindGroup({
        layout: pipeline1.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer, offset: 0, size: buffer.size } },
          { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        ],
      });

      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline1);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(batch));
      passEncoder.end();
      
      // If not the last step, update remaining submatrix
      if (k < n - 1) {
          const remaining = n - 1 - k;
          const pipeline2 = getOrCreatePipeline(CHOLESKY_SHADER, 'cholesky_step2');
          const bindGroup2 = device.createBindGroup({
              layout: pipeline2.getBindGroupLayout(0),
              entries: [
                  { binding: 0, resource: { buffer, offset: 0, size: buffer.size } },
                  { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
              ],
          });
          const passEncoder2 = commandEncoder.beginComputePass();
          passEncoder2.setPipeline(pipeline2);
          passEncoder2.setBindGroup(0, bindGroup2);
          // dispatchWorkgroups(x, y, z) where x,y cover the submatrix and z covers the batch
          passEncoder2.dispatchWorkgroups(
              Math.ceil(remaining / 16),
              Math.ceil(remaining / 16),
              batch
          );
          passEncoder2.end();
      }
      
      device.queue.submit([commandEncoder.finish()]);
    }
    
    // Zero out upper triangle (if not already done by step2 or if we want to be sure)
    // Actually our step2 only updates j <= i, so upper triangle stays as is.
    // We should zero it out at the end.
    return res.tril();
  }

  if (upper) {
    return res.transpose(-2, -1);
  }
  return res;
};

/**
 * Computes a vector or matrix norm.
 * @pytorch torch.linalg.norm
 */
export const norm = (input: Tensor, ord?: number | string, dim?: number | number[], keepdim: boolean = false): Tensor => {
  if (typeof dim === 'number') dim = [dim];
  
  // Matrix norm
  if ((dim === undefined && input.dim() === 2) || (dim !== undefined && dim.length === 2)) {
    return matrix_norm(input, ord, dim as [number, number], keepdim);
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
export const matrix_power = (A: Tensor, n: number): Tensor => {
    if (A.dim() < 2) throw new Error('linalg.matrix_power: expected tensor with at least 2 dimensions');
    const h = A.shape[A.dim() - 2];
    const w = A.shape[A.dim() - 1];
    if (h !== w) throw new Error('linalg.matrix_power: expected square matrix');
    
    if (n === 0) {
        return ops.eye(h, A.device as any).expand(A.shape);
    }
    if (n < 0) {
        return matrix_power(inv(A), -n);
    }
    
    let res = A;
    for (let i = 1; i < n; i++) {
        res = ops.matmul(res, A);
    }
    return res;
};

/**
 * Computes the inverse of a square matrix if it exists.
 * @pytorch torch.linalg.inv
 */
export const inv = (A: Tensor): Tensor => {
  const [LU, P] = lu_factor(A);
  const n = A.shape[A.dim() - 1];
  const batch = A.dim() > 2 ? A.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
  
  // Solve LUX = P I
  // We can create permuted identity matrix
  const I = ops.eye(n).expand(A.shape).clone();
  // TODO: apply permutation P to I. 
  // For now, let's assume we can use a simpler approach or implement row permutation.
  
  // Actually, PyTorch's solve_triangular doesn't handle permutations.
  // We need to implement a shader to apply the permutation.
  
  throw new Error('linalg.inv: permutation handling not yet implemented');
};

/**
 * Computes the determinant of a square matrix.
 * @pytorch torch.linalg.det
 */
export const det = (A: Tensor): Tensor => {
  const [LU, P] = lu_factor(A);
  const n = A.shape[A.dim() - 1];
  
  // det(A) = det(P^-1) * det(L) * det(U)
  // det(L) = 1 (unit triangular)
  // det(U) = product of diagonal elements
  // det(P^-1) = (-1)^num_swaps
  
  const diagU = ops.diag(LU); // Wait, ops.diag(matrix) returns diagonal as vector
  let d = diagU.prod(-1);
  
  // TODO: adjust sign based on P
  return d;
};

/**
 * Computes the cross product of two 3-dimensional vectors.
 * @pytorch torch.linalg.cross
 */
export const cross = (input: Tensor, other: Tensor, dim: number = -1): Tensor => {
    if (input.shape[dim] !== 3 || other.shape[dim] !== 3) {
        throw new Error('linalg.cross: expected dimension of size 3');
    }
    // TODO: implement cross product
    throw new Error('linalg.cross: not yet implemented');
};