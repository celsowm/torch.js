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
export class TensorDataset extends Dataset<{ [key: string]: Tensor }> {
  readonly tensors: Tensor[];
  private keys: string[];

  constructor(...tensors: Tensor[]) {
    super();
    this.tensors = tensors;
    this.keys = tensors.map((_, i) => `tensor_${i}`);
  }

  get length(): number {
    return this.tensors[0].shape[0];
  }

  __getitem__(index: number): { [key: string]: Tensor } {
    const result: { [key: string]: Tensor } = {};
    for (let i = 0; i < this.tensors.length; i++) {
      result[this.keys[i]] = this.tensors[i].select(0, index);
    }
    return result;
  }

  /**
   * Set names for the tensors.
   */
  named(...names: string[]): this {
    this.keys = names;
    return this;
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

  constructor(dataset: Dataset<T>, options: DataLoaderOptions = {}) {
    this.dataset = dataset;
    this.batchSize = options.batchSize ?? 1;
    this.shuffle = options.shuffle ?? false;
    this.dropLast = options.dropLast ?? false;
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
          const slices = batchIndices.map(i => t.select(0, i));
          return Tensor.cat(slices, 0);
        });
        const result: any = {};
        const keys = (this.dataset as TensorDataset)['keys'] || batchedTensors.map((_, i) => `tensor_${i}`);
        for (let i = 0; i < batchedTensors.length; i++) {
          result[keys[i]] = batchedTensors[i];
        }
        yield result as T;
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
