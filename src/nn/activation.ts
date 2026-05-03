/**
 * Activation function modules.
 * @status implemented
 * @pytorch torch.nn
 */

import { Tensor } from '../tensor';
import { Module } from './module';
import * as F from './functional';

/**
 * ReLU activation.
 * @pytorch torch.nn.ReLU
 */
export class ReLU extends Module {
  private inplace: boolean;

  constructor(inplace: boolean = false) {
    super();
    this.inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    return F.relu(input, this.inplace);
  }
}

/**
 * Sigmoid activation.
 * @pytorch torch.nn.Sigmoid
 */
export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    return input.sigmoid();
  }
}

/**
 * Tanh activation.
 * @pytorch torch.nn.Tanh
 */
export class Tanh extends Module {
  forward(input: Tensor): Tensor {
    return input.tanh();
  }
}

/**
 * LogSoftmax activation.
 * @pytorch torch.nn.LogSoftmax
 */
export class LogSoftmax extends Module {
  private dim: number;

  constructor(dim: number = -1) {
    super();
    this.dim = dim;
  }

  forward(input: Tensor): Tensor {
    return F.log_softmax(input, this.dim);
  }
}

/**
 * GELU activation (Gaussian Error Linear Unit).
 * Uses the tanh approximation as in GPT-2/BERT.
 * @pytorch torch.nn.GELU
 */
export class GELU extends Module {
  forward(input: Tensor): Tensor {
    return F.gelu(input);
  }
}

/**
 * Softmax activation.
 * @pytorch torch.nn.Softmax
 */
export class Softmax extends Module {
  private dim: number;

  constructor(dim: number = -1) {
    super();
    this.dim = dim;
  }

  forward(input: Tensor): Tensor {
    return F.softmax(input, this.dim);
  }
}
