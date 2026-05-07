/**
 * Dataset and DataLoader utilities.
 * @pytorch torch.utils.data
 */

import { Tensor } from '../tensor';
import { tensor } from '../ops/creation';

/**
 * Abstract base class for all datasets.
 * @pytorch torch.utils.data.Dataset
 */
export abstract class Dataset<T = any> {
  /**
   * Returns the size of the dataset.
   */
  abstract length: number;

  /**
   * Returns a sample from the dataset at the given index.
   */
  abstract __getitem__(index: number): T;
}

/**
 * TensorDataset wraps tensors as a dataset.
 * @pytorch torch.utils.data.TensorDataset
 */
export class TensorDataset extends Dataset<Tensor[]> {
  readonly tensors: Tensor[];
  private _length: number;

  constructor(...tensors: Tensor[]) {
    super();
    if (tensors.length === 0 || tensors.some(t => !t || t.shape.length === 0)) {
      throw new Error('TensorDataset: at least one non-empty tensor is required');
    }
    this.tensors = tensors;
    this._length = tensors[0].shape[0];
    
    // Validate all tensors have the same first dimension
    for (let i = 1; i < tensors.length; i++) {
      if (tensors[i].shape[0] !== this._length) {
        throw new Error(
          `TensorDataset: all tensors must have the same first dimension. ` +
          `Got ${tensors[0].shape[0]} and ${tensors[i].shape[0]}`
        );
      }
    }
  }

  /**
   * Returns the size of the dataset.
   */
  len(): number {
    return this._length;
  }

  /**
   * Returns a sample from the dataset at the given index.
   * Returns an array of tensors [t1[idx], t2[idx], ...]
   */
  get(index: number): Tensor[] {
    return this.tensors.map(t => t.select(0, index));
  }

  /**
   * Returns the size of the dataset (alias for len).
   */
  get length(): number {
    return this._length;
  }

  __getitem__(index: number): Tensor[] {
    return this.get(index);
  }
}

/**
 * DataLoader options.
 */
export interface DataLoaderOptions {
  batchSize?: number;
  shuffle?: boolean;
  dropLast?: boolean;
}

/**
 * DataLoader wraps a dataset and provides an iterable over it.
 * @pytorch torch.utils.data.DataLoader
 */
export class DataLoader<T = any> {
  private dataset: Dataset<T>;
  private batchSize: number;
  private shuffle: boolean;
  private dropLast: boolean;

  constructor(dataset: Dataset<T>, batchSizeOrOptions: number | DataLoaderOptions = 1) {
    this.dataset = dataset;
    if (typeof batchSizeOrOptions === 'number') {
      this.batchSize = batchSizeOrOptions;
      this.shuffle = false;
      this.dropLast = false;
    } else {
      const opts = batchSizeOrOptions as DataLoaderOptions;
      this.batchSize = opts.batchSize ?? 1;
      this.shuffle = opts.shuffle ?? false;
      this.dropLast = opts.dropLast ?? false;
    }
  }

  async *[Symbol.asyncIterator](): AsyncIterableIterator<T> {
    const indices = Array.from({ length: this.dataset.length }, (_, i) => i);
    if (this.shuffle) {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }

    const numBatches = this.dropLast
      ? Math.floor(indices.length / this.batchSize)
      : Math.ceil(indices.length / this.batchSize);

    for (let b = 0; b < numBatches; b++) {
      const start = b * this.batchSize;
      const end = Math.min(start + this.batchSize, indices.length);
      const batchIndices = indices.slice(start, end);

      if (this.dropLast && batchIndices.length < this.batchSize) {
        continue;
      }

      // For TensorDataset, batch the tensors
      if (this.dataset instanceof TensorDataset) {
        const batchedTensors = (this.dataset as TensorDataset).tensors.map(t => {
          const slices = batchIndices.map(i => {
            const slice = t.select(0, i);
            // For 0D tensors (from 1D input), we need to unsqueeze before stacking
            if (slice.shape.length === 0) {
              return slice.unsqueeze(0);
            }
            return slice;
          });
          return Tensor.cat(slices, 0);
        });
        yield batchedTensors as unknown as T;
      } else {
        const batch = batchIndices.map(i => this.dataset.__getitem__(i));
        yield batch as unknown as T;
      }
    }
  }

  *[Symbol.iterator](): Iterator<T> {
    const indices = Array.from({ length: this.dataset.length }, (_, i) => i);
    if (this.shuffle) {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }

    const numBatches = this.dropLast
      ? Math.floor(indices.length / this.batchSize)
      : Math.ceil(indices.length / this.batchSize);

    for (let b = 0; b < numBatches; b++) {
      const start = b * this.batchSize;
      const end = Math.min(start + this.batchSize, indices.length);
      const batchIndices = indices.slice(start, end);

      if (this.dropLast && batchIndices.length < this.batchSize) {
        continue;
      }

      // For TensorDataset, batch the tensors
      if (this.dataset instanceof TensorDataset) {
        const batchedTensors = (this.dataset as TensorDataset).tensors.map(t => {
          const slices = batchIndices.map(i => {
            const slice = t.select(0, i);
            // For 0D tensors (from 1D input), we need to unsqueeze before stacking
            if (slice.shape.length === 0) {
              return slice.unsqueeze(0);
            }
            return slice;
          });
          return Tensor.cat(slices, 0);
        });
        yield batchedTensors as unknown as T;
      } else {
        const batch = batchIndices.map(i => this.dataset.__getitem__(i));
        yield batch as unknown as T;
      }
    }
  }
}

/**
 * Randomly split a dataset into non-overlapping new datasets.
 * @pytorch torch.utils.data.random_split
 */
export function randomSplit<T>(dataset: Dataset<T>, lengths: number[]): Dataset<T>[] {
  const total = lengths.reduce((a, b) => a + b, 0);
  if (total !== dataset.length) {
    throw new Error(`Sum of split lengths (${total}) does not equal dataset length (${dataset.length})`);
  }

  const indices = Array.from({ length: dataset.length }, (_, i) => i);
  // Shuffle
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const subsets: Dataset<T>[] = [];
  let offset = 0;
  for (const len of lengths) {
    const subsetIndices = indices.slice(offset, offset + len);
    offset += len;

    subsets.push(new _SubsetDataset(dataset, subsetIndices));
  }
  return subsets;
}

class _SubsetDataset<T> extends Dataset<T> {
  private parent: Dataset<T>;
  private indices: number[];

  constructor(parent: Dataset<T>, indices: number[]) {
    super();
    this.parent = parent;
    this.indices = indices;
  }

  get length(): number {
    return this.indices.length;
  }

  __getitem__(index: number): T {
    return this.parent.__getitem__(this.indices[index]);
  }
}

export default {
  Dataset,
  TensorDataset,
  DataLoader,
  randomSplit,
};
