/**
 * RAdam optimizer (Rectified Adam).
 * @status implemented
 * @pytorch None
 */

import { Tensor } from '../tensor';
import { getDevice } from '../backend';
import { Optimizer } from './optimizer';
import { zeros_like } from '../ops/creation';

export interface RAdamOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weight_decay?: number;
}

/**
 * Implements RAdam algorithm (Rectified Adam with variance rectification).
 *
 * Reference: "On the Variance of the Adaptive Learning Rate and Beyond" (Liu et al., 2019)
 */
export class RAdam extends Optimizer {
  private state: Map<Tensor, {
    step: number;
    exp_avg: Tensor;
    exp_avg_sq: Tensor;
  }>;

  constructor(params: Tensor[] | Iterable<Tensor>, options: RAdamOptions = {}) {
    super(params, {
      lr: options.lr ?? 1e-3,
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

      // Compute the variance rectification term
      const beta2_t = Math.pow(beta2, 1);
      const var_inf = beta2 / (1 - beta2) * Math.pow(1 - beta1, 2);
      const rect_threshold = 4;

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

        // Update exp_avg and exp_avg_sq (same as Adam)
        state.exp_avg = state.exp_avg.mul(beta1).add(grad.mul(1 - beta1));
        state.exp_avg_sq = state.exp_avg_sq.mul(beta2).add(grad.pow(2).mul(1 - beta2));

        // Bias correction
        const bias_correction1 = 1 - Math.pow(beta1, state.step);
        const bias_correction2 = 1 - Math.pow(beta2, state.step);

        // Variance rectification
        const denom = state.exp_avg_sq.div(bias_correction2).sqrt().add(eps);

        const step_size = lr / bias_correction1;

        let rect_term = 1;
        const use_rect = state.step > rect_threshold;

        if (use_rect) {
          const r_inf = (2 / (1 - beta2)) - 1;
          const r_t = r_inf - (2 * state.step * Math.pow(beta2, state.step)) / (1 - Math.pow(beta2, state.step));
          const v_t = r_t / r_inf;
          rect_term = Math.sqrt(Math.max(1 - var_inf * v_t, 0));
        }

        const corrected_step_size = use_rect ? step_size * rect_term : step_size;

        // Read back data to update buffer
        const pData = await param.toArray();
        const mData = await state.exp_avg.toArray();
        const denomData = await denom.toArray();

        const newParamData = new Float32Array(pData.length);

        for (let i = 0; i < pData.length; i++) {
          newParamData[i] = pData[i] - corrected_step_size * (mData[i] / denomData[i]);
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
