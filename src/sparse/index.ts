/**
 * Sparse tensor operations - Sparse COO tensors.
 * @status implemented
 * @pytorch torch.sparse
 */

import { Tensor } from '../tensor/index';
import { tensor, zeros, arange, cat, stack } from '../ops/creation';
import { DType, getTypedArrayConstructor, TypedArray } from '../dtype';

// ---------------------------------------------------------------------------
// SparseTensor class
// ---------------------------------------------------------------------------

export class SparseTensor {
  private _indices: Tensor;
  private _values: Tensor;
  private _size: readonly number[];
  private _dtype: DType;
  private _coalesced: boolean | null = null;

  /**
   * Create a sparse COO tensor.
   * @param indices - Tensor of shape [nnz, ndim] containing sparse indices
   * @param values - Tensor of shape [nnz] containing non-zero values
   * @param size - Dense shape of the sparse tensor
   * @param dtype - Data type
   */
  constructor(
    indices: Tensor,
    values: Tensor,
    size?: readonly number[],
    dtype?: DType
  ) {
    if (indices.dim() === 0 && values.dim() === 0) {
      // Empty sparse tensor
      this._indices = indices;
      this._values = values;
      this._size = size ?? [];
      this._dtype = dtype ?? 'float32';
      return;
    }

    if (indices.dim() !== 2) {
      throw new Error(
        `indices must be 2-D, got ${indices.dim()} dimensions`
      );
    }
    if (values.dim() !== 1) {
      throw new Error(
        `values must be 1-D, got ${values.dim()} dimensions`
      );
    }

    const nnz = indices.shape[0];
    const ndim = indices.shape[1];

    if (values.shape[0] !== nnz) {
      throw new Error(
        `mismatch: indices has ${nnz} non-zero elements but values has ${values.shape[0]}`
      );
    }

    this._indices = indices;
    this._values = values;
    this._size = size ?? [];
    this._dtype = dtype ?? 'float32';

    // If size was not provided, infer from indices and values
    if (this._size.length === 0) {
      const inferred: number[] = new Array(ndim).fill(0);
      this._size = inferred;
    }
  }

  /** Number of non-zero elements. */
  get nnz(): number {
    if (this._indices.dim() === 0) return 0;
    return this._indices.shape[0];
  }

  /** Number of dimensions. */
  get ndim(): number {
    if (this._indices.dim() === 0) return 0;
    return this._indices.shape[1];
  }

  /** Dense shape. */
  get size(): readonly number[] {
    return this._size;
  }

  /** Data type. */
  get dtype(): DType {
    return this._dtype;
  }

  /** Indices tensor of shape [nnz, ndim]. */
  get indices(): Tensor {
    return this._indices;
  }

  /** Values tensor of shape [nnz]. */
  get values(): Tensor {
    return this._values;
  }

  /**
   * Check if the sparse tensor is coalesced (indices sorted and unique).
   */
  is_coalesced(): boolean {
    if (this._coalesced !== null) {
      return this._coalesced;
    }
    if (this.nnz <= 1) {
      this._coalesced = true;
      return true;
    }
    // Conservative default - would need to read from GPU to verify
    return false;
  }

  /**
   * Coalesce the sparse tensor: sort indices lexicographically and merge duplicates.
   */
  async coalesce(): Promise<SparseTensor> {
    if (this.nnz === 0) {
      return this.clone();
    }

    // Read indices and values to CPU for sorting
    const indicesArr = await this._indices.toArray();
    const valuesArr = await this._values.toArray();
    const nnz = this.nnz;
    const ndim = this.ndim;

    // Build array of {indices: number[], value: number}
    type Entry = { indices: number[]; value: number };
    const entries: Entry[] = [];
    for (let i = 0; i < nnz; i++) {
      const idx: number[] = [];
      for (let d = 0; d < ndim; d++) {
        idx.push(indicesArr[i * ndim + d]);
      }
      entries.push({ indices: idx, value: valuesArr[i] });
    }

    // Sort lexicographically
    entries.sort((a, b) => {
      for (let d = 0; d < ndim; d++) {
        if (a.indices[d] !== b.indices[d]) {
          return a.indices[d] - b.indices[d];
        }
      }
      return 0;
    });

    // Merge duplicates by summing values
    const mergedIndices: number[] = [];
    const mergedValues: number[] = [];

    let i = 0;
    while (i < entries.length) {
      const currentIndices = entries[i].indices;
      let sumValue = entries[i].value;
      let j = i + 1;
      while (j < entries.length) {
        let isDuplicate = true;
        for (let d = 0; d < ndim; d++) {
          if (entries[j].indices[d] !== currentIndices[d]) {
            isDuplicate = false;
            break;
          }
        }
        if (!isDuplicate) break;
        sumValue += entries[j].value;
        j++;
      }
      mergedIndices.push(...currentIndices);
      mergedValues.push(sumValue);
      i = j;
    }

    // Build result tensors
    const newIndices = tensor(mergedIndices, { dtype: 'int32' }).reshape([mergedValues.length, ndim]);
    const newValues = tensor(mergedValues, { dtype: this._dtype });

    const result = new SparseTensor(newIndices, newValues, this._size, this._dtype);
    result._coalesced = true;
    return result;
  }

  /**
   * Convert to a dense tensor.
   */
  async to_dense(): Promise<Tensor> {
    if (this._size.length === 0) {
      // Scalar or empty
      if (this.nnz === 0) {
        return tensor(0, { dtype: this._dtype });
      }
      return this._values;
    }

    const result = zeros([...this._size], { dtype: this._dtype });

    if (this.nnz === 0) {
      return result;
    }

    const indicesArr = await this._indices.toArray();
    const valuesArr = await this._values.toArray();
    const ndim = this.ndim;

    // Build flat index and scatter values
    const strides: number[] = new Array(ndim).fill(1);
    for (let d = ndim - 2; d >= 0; d--) {
      strides[d] = strides[d + 1] * this._size[d + 1];
    }

    // Read dense tensor to modify (convert to mutable number array)
    const denseDataRaw: number[] = Array.from(await result.toArray());
    for (let i = 0; i < this.nnz; i++) {
      let flatIdx = 0;
      for (let d = 0; d < ndim; d++) {
        flatIdx += indicesArr[i * ndim + d] * strides[d];
      }
      denseDataRaw[flatIdx] += valuesArr[i] as number;
    }

    // Write back
    const resultBuffer = await _tensorFromArray(denseDataRaw, this._dtype, this._size);
    return new Tensor({
      buffer: resultBuffer,
      shape: [...this._size],
      dtype: this._dtype,
      device: 'webgpu',
      requires_grad: false,
    });
  }

  /**
   * Clone the sparse tensor.
   */
  clone(): SparseTensor {
    return new SparseTensor(
      this._indices.clone(),
      this._values.clone(),
      [...this._size],
      this._dtype
    );
  }
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/**
 * Create a sparse COO tensor from indices and values.
 * @param indices - Tensor of shape [nnz, ndim]
 * @param values - Tensor of shape [nnz]
 * @param size - Dense shape
 * @param dtype - Data type
 * @pytorch torch.sparse_coo_tensor
 */
export function sparse_coo_tensor(
  indices: Tensor,
  values: Tensor,
  size?: readonly number[],
  dtype?: DType
): SparseTensor {
  return new SparseTensor(indices, values, size, dtype);
}

/**
 * Convert a dense tensor to a sparse COO tensor.
 * @param dense_tensor - Dense tensor to convert
 * @param threshold - Values with absolute value below this are treated as zero (default: 0)
 * @pytorch Tensor.to_sparse
 */
export async function to_sparse(
  dense_tensor: Tensor,
  threshold: number = 0
): Promise<SparseTensor> {
  const shape = [...dense_tensor.shape];
  const ndim = shape.length;

  // Find non-zero elements
  const mask = dense_tensor.abs();
  let nonzeroMask: Tensor;
  if (threshold > 0) {
    nonzeroMask = mask.gt(threshold);
  } else {
    nonzeroMask = mask.ne(0);
  }

  // Get indices of non-zero elements
  const sparseIndices = await nonzeroMask.nonzero();

  if (sparseIndices.shape[0] === 0) {
    // All zeros - return empty sparse tensor
    const emptyIndices = tensor([], { dtype: 'int32' }).reshape([0, ndim]);
    const emptyValues = tensor([], { dtype: dense_tensor.dtype });
    return new SparseTensor(emptyIndices, emptyValues, shape, dense_tensor.dtype);
  }

  // Gather values at those indices
  // Flatten dense tensor and use advanced indexing
  const flatDense = dense_tensor.reshape([-1]);
  const nnz = sparseIndices.shape[0];

  // Compute flat indices from multi-dimensional indices
  const strides: number[] = new Array(ndim).fill(1);
  for (let d = ndim - 2; d >= 0; d--) {
    strides[d] = strides[d + 1] * shape[d + 1];
  }

  const indicesArr = await sparseIndices.toArray();
  const flatIndices: number[] = [];
  for (let i = 0; i < nnz; i++) {
    let flatIdx = 0;
    for (let d = 0; d < ndim; d++) {
      flatIdx += indicesArr[i * ndim + d] * strides[d];
    }
    flatIndices.push(flatIdx);
  }

  const flatIndexTensor = tensor(flatIndices, { dtype: 'int32' });
  const sparseValues = flatDense.index_select(0, flatIndexTensor);

  return new SparseTensor(sparseIndices, sparseValues, shape, dense_tensor.dtype);
}

// ---------------------------------------------------------------------------
// Sparse operations
// ---------------------------------------------------------------------------

/**
 * Add two sparse tensors by merging indices and summing values.
 * Both tensors must have the same shape.
 * @pytorch torch.sparse.add
 */
export async function sparse_add(
  a: SparseTensor,
  b: SparseTensor
): Promise<SparseTensor> {
  // Check shape compatibility
  if (a.size.length !== b.size.length) {
    throw new Error(
      `sparse_add: shape mismatch - a has ${a.size.length} dims, b has ${b.size.length} dims`
    );
  }
  for (let i = 0; i < a.size.length; i++) {
    if (a.size[i] !== b.size[i]) {
      throw new Error(
        `sparse_add: shape mismatch at dim ${i}: ${a.size[i]} vs ${b.size[i]}`
      );
    }
  }

  // Concatenate indices and values
  const combinedIndices = cat([a.indices, b.indices], 0);
  const combinedValues = cat([a.values, b.values], 0);

  const combined = new SparseTensor(combinedIndices, combinedValues, a.size, a.dtype);
  // Coalesce will sort and merge duplicates
  return combined.coalesce();
}

/**
 * Multiply sparse values by a scalar.
 * @pytorch torch.sparse.mul (scalar variant)
 */
export function sparse_mul(
  sparse: SparseTensor,
  scalar: number
): SparseTensor {
  const newValues = sparse.values.mul(scalar);
  return new SparseTensor(sparse.indices, newValues, sparse.size, sparse.dtype);
}

/**
 * Sparse-dense matrix multiplication.
 * Computes sparse @ dense where sparse is [M, K] and dense is [K, N].
 * Returns a dense tensor of shape [M, N].
 *
 * For each non-zero entry sparse[i][k], multiply values[k] * dense[k][j].
 * @pytorch torch.sparse.mm
 */
export async function sparse_matmul(
  sparse: SparseTensor,
  dense: Tensor
): Promise<Tensor> {
  if (sparse.ndim !== 2) {
    throw new Error(
      `sparse_matmul: sparse tensor must be 2D, got ${sparse.ndim} dimensions`
    );
  }

  const [M, K] = sparse.size;
  const denseShape = [...dense.shape];

  if (denseShape.length === 1) {
    // dense is a vector [K]
    if (denseShape[0] !== K) {
      throw new Error(
        `sparse_matmul: size mismatch - sparse has K=${K}, dense has ${denseShape[0]}`
      );
    }
    const result = new Float32Array(M);
    const indicesArr = await sparse.indices.toArray();
    const valuesArr = await sparse.values.toArray();
    const denseArr = await dense.toArray();

    for (let i = 0; i < sparse.nnz; i++) {
      const row = indicesArr[i * 2];
      const col = indicesArr[i * 2 + 1];
      result[row] += valuesArr[i] * denseArr[col];
    }

    return tensor(Array.from(result), { dtype: sparse.dtype });
  }

  if (denseShape.length !== 2) {
    throw new Error(
      `sparse_matmul: dense tensor must be 1D or 2D, got ${denseShape.length} dimensions`
    );
  }

  const N = denseShape[1];
  if (denseShape[0] !== K) {
    throw new Error(
      `sparse_matmul: size mismatch - sparse has K=${K}, dense has ${denseShape[0]}`
    );
  }

  if (sparse.nnz === 0) {
    const emptyResult = zeros([M, N], { dtype: sparse.dtype });
    return emptyResult;
  }

  const indicesArr = await sparse.indices.toArray();
  const valuesArr = await sparse.values.toArray();
  const denseArr = await dense.toArray();
  const resultArr: number[] = new Array(M * N).fill(0);

  // For each non-zero entry, multiply value by entire row of dense
  for (let i = 0; i < sparse.nnz; i++) {
    const row = indicesArr[i * 2];
    const col = indicesArr[i * 2 + 1];
    const val = valuesArr[i] as number;
    for (let j = 0; j < N; j++) {
      resultArr[row * N + j] += val * denseArr[col * N + j];
    }
  }

  return new Tensor({
    buffer: await _tensorFromArray(resultArr, sparse.dtype, [M, N]),
    shape: [M, N],
    dtype: sparse.dtype,
    device: 'webgpu',
    requires_grad: false,
  });
}

/**
 * Create a sparse identity matrix.
 * @param n - Size of the matrix
 * @pytorch torch.sparse.eye
 */
export function sparse_eye(n: number): SparseTensor {
  // Indices: diagonal positions [0,0], [1,1], ..., [n-1,n-1]
  const rowIndices = arange(0, n, 1, { dtype: 'int32' });
  const colIndices = arange(0, n, 1, { dtype: 'int32' });

  // Stack to form [n, 2] indices
  const indices = stack([rowIndices, colIndices], 1);

  // Values: all ones
  const values = tensor(new Array(n).fill(1), { dtype: 'float32' });

  return new SparseTensor(indices, values, [n, n], 'float32');
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Create a GPU tensor buffer from a numeric array.
 */
async function _tensorFromArray(
  data: number[] | TypedArray,
  dtype: DType,
  shape: readonly number[]
): Promise<GPUBuffer> {
  const { createBufferWithData } = await import('../backend');
  const TypedArrayCtor = getTypedArrayConstructor(dtype);
  const typedData = new TypedArrayCtor(data) as TypedArray;
  return createBufferWithData(typedData, dtype);
}

// ---------------------------------------------------------------------------
// Module exports
// ---------------------------------------------------------------------------

export const sparse = {
  SparseTensor,
  sparse_coo_tensor,
  to_sparse,
  sparse_add,
  sparse_mul,
  sparse_matmul,
  sparse_eye,
};

export default sparse;
