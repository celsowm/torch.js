/**
 * AdamW optimizer.
 * @status implemented
 * @pytorch torch.optim.AdamW
 */

import { Tensor } from '../tensor';
import { getDevice } from '../backend';
import { Optimizer } from './optimizer';
import { zeros_like } from '../ops/creation';

export interface AdamWOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weight_decay?: number;
  amsgrad?: boolean;
}

/**
 * Implements AdamW algorithm.
 */
export class AdamW extends Optimizer {
  private state: Map<Tensor, {
    step: number;
    exp_avg: Tensor;
    exp_avg_sq: Tensor;
    max_exp_avg_sq?: Tensor; // For amsgrad
  }>;

  constructor(params: Tensor[] | Iterable<Tensor>, options: AdamWOptions = {}) {
    super(params, {
      lr: options.lr ?? 1e-3,
      betas: options.betas ?? [0.9, 0.999],
      eps: options.eps ?? 1e-8,
      weight_decay: options.weight_decay ?? 1e-2,
      amsgrad: options.amsgrad ?? false,
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
      const amsgrad = group.amsgrad as boolean;

      for (const param of group.params) {
        if (!param.grad) {
          continue;
        }

        const grad = param.grad;

        // Initialize state if needed
        let state = this.state.get(param);
        if (!state) {
          state = {
            step: 0,
            exp_avg: zeros_like(param),
            exp_avg_sq: zeros_like(param),
            max_exp_avg_sq: amsgrad ? zeros_like(param) : undefined,
          };
          this.state.set(param, state);
        }

        state.step += 1;

        // Update exp_avg and exp_avg_sq
        // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        state.exp_avg = state.exp_avg.mul(beta1).add(grad.mul(1 - beta1));
        
        // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
        state.exp_avg_sq = state.exp_avg_sq.mul(beta2).add(grad.pow(2).mul(1 - beta2));

        // Bias correction
        const bias_correction1 = 1 - Math.pow(beta1, state.step);
        const bias_correction2 = 1 - Math.pow(beta2, state.step);

        let denom: Tensor;
        if (amsgrad && state.max_exp_avg_sq) {
          // Maintains the maximum of all 2nd moment running avg. till now
          state.max_exp_avg_sq = state.max_exp_avg_sq.fmax(state.exp_avg_sq);
          throw new Error("AMSGrad not yet fully implemented");
        }
        
        denom = state.exp_avg_sq.div(bias_correction2).sqrt().add(eps);

        const step_size = lr / bias_correction1;

        // Update parameters
        // param = param - step_size * (exp_avg / denom) - lr * weight_decay * param
        // param = param * (1 - lr * weight_decay) - step_size * (exp_avg / denom)
        
        // Read back data to update buffer
        // Note: This is extremely inefficient (readback 4 tensors!). 
        // We really need a kernel for this.
        // But for correctness first:
        
        const pData = await param.toArray();
        const mData = await state.exp_avg.toArray();
        const vData = await state.exp_avg_sq.toArray(); // we need v for denom calculation locally if we do it here
        // actually we calculated denom tensor.
        const denomData = await denom.toArray();
        
        const newParamData = new Float32Array(pData.length);
        
        for (let i = 0; i < pData.length; i++) {
          let p = pData[i];
          const m = mData[i]; // bias corrected implicitly by step_size calc? No.
          // Wait, logic above:
          // step_size = lr / bias_correction1
          // term1 = exp_avg / bias_correction1 (if we divide denom by sqrt(bias_corr2))?
          
          // Let's follow PyTorch exactly:
          // denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
          // step_size = group['lr'] / bias_correction1
          // p.addcdiv_(exp_avg, denom, value=-step_size)
          
          // Here I calculated denom tensor correctly above.
          // So new_p = p * (1 - lr * wd) - step_size * (exp_avg / denom)
          
          p = p * (1 - lr * weight_decay);
          p = p - step_size * (m / denomData[i]); // m is exp_avg[i]
          
          newParamData[i] = p;
        }

        // Write updated values back to the parameter's buffer
        await this._updateParamBuffer(param, newParamData);
        
        // We also need to keep state tensors on GPU, so we assign the new tensors to state
        // But we computed state.exp_avg and state.exp_avg_sq as new tensors above via operations.
        // So they are already fresh tensors on GPU.
        // We just need to dispose old ones? GC should handle it.
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
