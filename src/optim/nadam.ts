/**
 * NAdam optimizer (Nesterov-accelerated Adam).
 * @status implemented
 * @pytorch torch.optim.NAdam
 */

import { Tensor } from '../tensor';
import { getDevice } from '../backend';
import { Optimizer } from './optimizer';
import { zeros_like } from '../ops/creation';

export interface NAdamOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weight_decay?: number;
}

/**
 * Implements NAdam algorithm (Nesterov-accelerated Adam).
 *
 * Reference: "Incorporating Nesterov Momentum into Adam" (Dozat, 2016)
 */
export class NAdam extends Optimizer {
  private state: Map<Tensor, {
    step: number;
    exp_avg: Tensor;
    exp_avg_sq: Tensor;
  }>;

  constructor(params: Tensor[] | Iterable<Tensor>, options: NAdamOptions = {}) {
    super(params, {
      lr: options.lr ?? 2e-3,
      betas: options.betas ?? [0.9, 0.999],
      eps: options.eps ?? 1e-8,
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
      const [beta1, beta2] = (group.betas as [number, number]);
      const eps = group.eps as number;
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
            exp_avg: zeros_like(param),
            exp_avg_sq: zeros_like(param),
          };
          this.state.set(param, state);
        }

        state.step += 1;

        // Bias-corrected decay rates
        const beta1_prod = beta1 * (1 - 0.5 * Math.pow(0.96, state.step));
        const bias_correction1 = 1 - Math.pow(beta1, state.step);
        const bias_correction2 = 1 - Math.pow(beta2, state.step);

        // Nesterov momentum: g_hat = beta1 * exp_avg + (1 - beta1) * grad
        const g_hat = state.exp_avg.mul(beta1_prod).add(grad.mul(1 - beta1));

        // Update exp_avg (for next step)
        state.exp_avg = state.exp_avg.mul(beta1).add(grad.mul(1 - beta1));

        // Update exp_avg_sq
        state.exp_avg_sq = state.exp_avg_sq.mul(beta2).add(grad.pow(2).mul(1 - beta2));

        // Bias-corrected second moment
        const denom = state.exp_avg_sq.div(bias_correction2).sqrt().add(eps);

        const step_size = lr / bias_correction1;

        // Read back data to update buffer
        const pData = await param.toArray();
        const gHatData = await g_hat.toArray();
        const denomData = await denom.toArray();

        const newParamData = new Float32Array(pData.length);

        for (let i = 0; i < pData.length; i++) {
          newParamData[i] = pData[i] - step_size * (gHatData[i] / denomData[i]);
        }

        // Write updated values back to the parameter's buffer
        await this._updateParamBuffer(param, newParamData);
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
