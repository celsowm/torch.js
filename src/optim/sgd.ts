/**
 * Stochastic Gradient Descent optimizer.
 * @status implemented
 * @pytorch torch.optim.SGD
 */

import { Tensor } from '../tensor';
import { getDevice } from '../backend';
import { Optimizer } from './optimizer';

export interface SGDOptions {
  lr: number;
  momentum?: number;
  weight_decay?: number;
  dampening?: number;
  nesterov?: boolean;
}

/**
 * Implements stochastic gradient descent (optionally with momentum).
 */
export class SGD extends Optimizer {
  private momentum_buffers: Map<Tensor, Tensor>;

  constructor(params: Tensor[] | Iterable<Tensor>, options: SGDOptions) {
    super(params, {
      lr: options.lr,
      momentum: options.momentum ?? 0,
      weight_decay: options.weight_decay ?? 0,
      dampening: options.dampening ?? 0,
      nesterov: options.nesterov ?? false,
    });

    this.momentum_buffers = new Map();

    // Validate nesterov
    if (options.nesterov && (!options.momentum || options.dampening !== 0)) {
      throw new Error('Nesterov momentum requires a momentum and zero dampening');
    }
  }

  /**
   * Performs a single optimization step.
   */
  async step(): Promise<void> {
    for (const group of this.param_groups) {
      const lr = group.lr as number;
      const momentum = (group.momentum as number) ?? 0;
      const weight_decay = (group.weight_decay as number) ?? 0;
      const dampening = (group.dampening as number) ?? 0;
      const nesterov = (group.nesterov as boolean) ?? false;

      for (const param of group.params) {
        if (!param.grad) {
          continue;
        }

        let grad = param.grad;

        // Weight decay (L2 regularization)
        if (weight_decay !== 0) {
          grad = grad.add(param.mul(weight_decay));
        }

        // Momentum
        if (momentum !== 0) {
          let buf = this.momentum_buffers.get(param);
          if (!buf) {
            // First step: initialize momentum buffer with gradient
            buf = grad.clone();
            this.momentum_buffers.set(param, buf);
          } else {
            // Update momentum buffer: buf = momentum * buf + (1 - dampening) * grad
            buf = buf.mul(momentum).add(grad.mul(1 - dampening));
            this.momentum_buffers.set(param, buf);
          }

          if (nesterov) {
            // Nesterov momentum: grad = grad + momentum * buf
            grad = grad.add(buf.mul(momentum));
          } else {
            grad = buf;
          }
        }

        // Update parameter: param = param - lr * grad
        // We need to update the buffer in-place
        // For now, we'll read the values, compute, and write back
        const paramData = Array.from(await param.toArray());
        const gradData = Array.from(await grad.toArray());

        const newParamData = paramData.map((p: number, i: number) => p - lr * gradData[i]);

        // Write updated values back to the parameter's buffer
        await this._updateParamBuffer(param, newParamData);
      }
    }
  }

  /**
   * Update a parameter's buffer with new values.
   * @internal
   */
  private async _updateParamBuffer(param: Tensor, newData: number[]): Promise<void> {
    const device = getDevice();
    const data = new Float32Array(newData);
    device.queue.writeBuffer(param.buffer, 0, data);
  }
}
