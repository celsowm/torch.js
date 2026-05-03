/**
 * Flatten module.
 * @status implemented
 * @pytorch torch.nn.Flatten
 */

import { Tensor } from '../tensor';
import { Module } from './module';

/**
 * Flattens a contiguous range of dims into a tensor.
 */
export class Flatten extends Module {
  private start_dim: number;
  private end_dim: number;

  constructor(start_dim: number = 1, end_dim: number = -1) {
    super();
    this.start_dim = start_dim;
    this.end_dim = end_dim;
  }

  forward(input: Tensor): Tensor {
    return input.flatten(this.start_dim, this.end_dim);
  }
}
