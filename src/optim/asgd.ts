/**
 * ASGD optimizer (Averaged Stochastic Gradient Descent).
 * @status implemented
 * @pytorch torch.optim.ASGD
 */

import { Tensor } from '../tensor';
import { getDevice } from '../backend';
import { Optimizer } from './optimizer';
import { zeros_like } from '../ops/creation';

export interface ASGDOptions {
  lr?: number;
  lambd?: number;
  alpha?: number;
  t0?: number;
  weight_decay?: number;
}

/**
 * Implements Averaged SGD algorithm.
 *
 * Reference: "Acceleration of Stochastic Approximation by Averaging" (Polyak & Juditsky, 1992)
 */
export class ASGD extends Optimizer {
  private state: Map<Tensor, {
    step: number;
    ax: Tensor;
    eta: number;
  }>;

  constructor(params: Tensor[] | Iterable<Tensor>, options: ASGDOptions = {}) {
    super(params, {
      lr: options.lr ?? 1e-2,
      lambd: options.lambd ?? 1e-4,
      alpha: options.alpha ?? 0.75,
      t0: options.t0 ?? 1e6,
      weight_decay: options.weight_decay ?? 0,
    });

    this.state = new Map();
  }

  /**
   * Performs a single optimization step.
   */
  async step(): Promise<void> {
    for (const group of this.param_groups) {
      const lr = group.lr as number;
      const lambd = group.lambd as number;
      const alpha = group.alpha as number;
      const t0 = group.t0 as number;
      const weight_decay = group.weight_decay as number;

      for (const param of group.params) {
        if (!param.grad) {
          continue;
        }

        let grad = param.grad;

        // Weight decay (L2 penalty)
        if (weight_decay !== 0) {
          grad = grad.add(param.mul(weight_decay));
        }

        // Initialize state if needed
        let state = this.state.get(param);
        if (!state) {
          state = {
            step: 0,
            ax: zeros_like(param),
            eta: lr,
          };
          this.state.set(param, state);
        }

        state.step += 1;

        // Compute learning rate with decay
        // eta = lr0 / (1 + lambd * t)^alpha
        state.eta = lr / Math.pow(1 + lambd * state.step, alpha);

        // SGD step: param = param - eta * grad
        const pData = await param.toArray();
        const gData = await grad.toArray();

        const newParamData = new Float32Array(pData.length);
        for (let i = 0; i < pData.length; i++) {
          newParamData[i] = pData[i] - state.eta * gData[i];
        }

        // Write updated values back to the parameter's buffer
        await this._updateParamBuffer(param, newParamData);

        // Update running average
        // ax = (ax * n + param) / (n + 1)
        const aData = await state.ax.toArray();
        const n = state.step;
        const newAxData = new Float32Array(aData.length);

        for (let i = 0; i < aData.length; i++) {
          newAxData[i] = (aData[i] * (n - 1) + newParamData[i]) / n;
        }

        // Write updated average back to ax buffer
        const device = getDevice();
        device.queue.writeBuffer(state.ax.buffer, 0, newAxData);

        // Compute mu for future use (when to switch to averaged parameters)
        // This is used by some implementations to decide when to use ax instead of param
      }
    }
  }

  /**
   * Update a parameter's buffer with new values.
   * @internal
   */
  private async _updateParamBuffer(param: Tensor, newData: Float32Array): Promise<void> {
    const device = getDevice();
    device.queue.writeBuffer(param.buffer, 0, newData);
  }
}
