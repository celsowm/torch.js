/**
 * Dropout module.
 * @status partial
 * @pytorch torch.nn.Dropout
 */

import { Tensor } from '../tensor';
import { Module } from './module';
import * as F from './functional';

/**
 * During training, randomly zeroes some elements of the input tensor.
 */
export class Dropout extends Module {
  private p: number;
  private _inplace: boolean;

  constructor(p: number = 0.5, inplace: boolean = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    return F.dropout(input, this.p, this.training);
  }
}

/**
 * 2D Dropout for channel-wise dropout on 4D tensors.
 * @pytorch torch.nn.Dropout2d
 */
export class Dropout2d extends Module {
  private p: number;
  private _inplace: boolean;

  constructor(p: number = 0.5, inplace: boolean = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    // For now, use same behavior as 1D dropout
    return F.dropout(input, this.p, this.training);
  }
}
