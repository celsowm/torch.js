/**
 * Parameter class - a Tensor that is automatically registered as a parameter when assigned.
 * @status implemented
 * @pytorch torch.nn.Parameter
 */

import { Tensor, TensorOptions } from '../tensor';
import { tensor } from '../ops/creation';

/**
 * A Parameter is a Tensor that is registered as a module parameter.
 */
export class Parameter extends Tensor {
  /**
   * Create a Parameter from data or another tensor.
   */
  static create(
    data: Tensor | number[] | number[][] | number[][][] | number[][][][],
    options: TensorOptions = {}
  ): Parameter {
    // Default requires_grad to true for parameters
    const opts = { requires_grad: true, ...options };

    if (data instanceof Tensor) {
      // Wrap existing tensor - enable requires_grad
      data.requires_grad_(opts.requires_grad ?? true);
      return data as unknown as Parameter;
    }

    // Create new tensor with requires_grad
    const t = tensor(data as number[], opts);
    return t as unknown as Parameter;
  }
}
