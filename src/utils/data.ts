import { Tensor } from '../tensor';
import { stack } from '../ops/creation';

export abstract class Dataset<T = [Tensor, Tensor]> {
  abstract len(): number;
  abstract get(index: number): T;
}

export class TensorDataset extends Dataset<[Tensor, Tensor]> {
  private tensors: Tensor[];

  constructor(...tensors: Tensor[]) {
    super();
    if (tensors.length === 0) throw new Error('TensorDataset: at least one tensor required');
    const n = tensors[0].shape[0];
    for (let i = 1; i < tensors.length; i++) {
      if (tensors[i].shape[0] !== n) {
        throw new Error('TensorDataset: all tensors must have the same size in dim 0');
      }
    }
    this.tensors = tensors;
  }

  len(): number {
    return this.tensors[0].shape[0];
  }

  get(index: number): [Tensor, Tensor] {
    const selected = this.tensors.map(t => t.narrow(0, index, 1).squeeze(0));
    return [selected[0], selected.length > 1 ? selected[1] : selected[0]];
  }
}

export class DataLoader<T extends [Tensor, Tensor] = [Tensor, Tensor]> {
  private dataset: Dataset<T>;
  private batchSize: number;
  private shuffle: boolean;
  private dropLast: boolean;

  constructor(
    dataset: Dataset<T>,
    batchSize: number = 1,
    shuffle: boolean = false,
    dropLast: boolean = false,
  ) {
    this.dataset = dataset;
    this.batchSize = batchSize;
    this.shuffle = shuffle;
    this.dropLast = dropLast;
  }

  getDataset(): Dataset<T> {
    return this.dataset;
  }

  getBatchSize(): number {
    return this.batchSize;
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<T> {
    const n = this.dataset.len();
    let indices = Array.from({ length: n }, (_, i) => i);
    if (this.shuffle) {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }

    const limit = this.dropLast ? n - (n % this.batchSize) : n;
    for (let start = 0; start < limit; start += this.batchSize) {
      const batchIndices = indices.slice(start, start + this.batchSize);
      const batch: T[] = batchIndices.map(i => this.dataset.get(i));
      const numComponents = batch[0].length;
      const batched = Array.from({ length: numComponents }, (_, comp) => {
        const tensors = batch.map(item => (item as any)[comp] as Tensor);
        return stack(tensors, 0);
      }) as unknown as T;
      yield batched;
    }
  }
}

export const data = {
  Dataset,
  TensorDataset,
  DataLoader,
};
