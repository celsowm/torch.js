/**
 * Adamax optimizer (Adam with L-infinity norm).
 * @status implemented
 * @pytorch torch.optim.Adamax
 */

import { Tensor } from '../tensor';
import { getDevice } from '../backend';
import { Optimizer } from './optimizer';
import { zeros_like } from '../ops/creation';

export interface AdamaxOptions {
  lr?: number;
  betas?: [number, number];
  eps?: number;
  weight_decay?: number;
}

/**
 * Implements Adamax algorithm (Adam with L-infinity norm).
 *
 * Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
 */
export class Adamax extends Optimizer {
  private state: Map<Tensor, {
    step: number;
    exp_avg: Tensor;
    exp_inf: Tensor;
  }>;

  constructor(params: Tensor[] | Iterable<Tensor>, options: AdamaxOptions = {}) {
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
            exp_inf: zeros_like(param),
          };
          this.state.set(param, state);
        }

        state.step += 1;

        // Update exp_avg (exponential moving average of gradient)
        // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        state.exp_avg = state.exp_avg.mul(beta1).add(grad.mul(1 - beta1));

        // Update exp_inf (exponential moving average of absolute gradient)
        // exp_inf = max(beta2 * exp_inf, |grad|)
        const grad_abs = grad.abs();
        const exp_inf_scaled = state.exp_inf.mul(beta2);
        state.exp_inf = await this.elementwiseMax(exp_inf_scaled, grad_abs);

        // Bias correction (only for beta1)
        const bias_correction1 = 1 - Math.pow(beta1, state.step);
        const step_size = lr / bias_correction1;

        // Read back data to update buffer
        const pData = await param.toArray();
        const mData = await state.exp_avg.toArray();
        const infData = await state.exp_inf.toArray();

        const newParamData = new Float32Array(pData.length);

        for (let i = 0; i < pData.length; i++) {
          newParamData[i] = pData[i] - step_size * (mData[i] / (infData[i] + eps));
        }

        // Write updated values back to the parameter's buffer
        await this._updateParamBuffer(param, newParamData);
      }
    }
  }

  /**
   * Element-wise maximum of two tensors.
   * @internal
   */
  private async elementwiseMax(a: Tensor, b: Tensor): Promise<Tensor> {
    const aData = await a.toArray();
    const bData = await b.toArray();
    const result = new Float32Array(aData.length);

    for (let i = 0; i < aData.length; i++) {
      result[i] = Math.max(aData[i], bData[i]);
    }

    // Create a new tensor with the result data
    const device = getDevice();
    const buffer = device.createBuffer({
      size: result.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, result);

    // Return a new tensor with the same shape as a
    return new Tensor({
      buffer,
      shape: a.shape,
      dtype: a.dtype,
      device: 'webgpu',
      requires_grad: false,
    });
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
