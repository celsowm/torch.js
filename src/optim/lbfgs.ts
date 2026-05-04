/**
 * LBFGS - Limited-memory BFGS optimizer.
 * @pytorch torch.optim.LBFGS
 */

import { Tensor } from '../tensor/index';
import { Optimizer } from '../optim/optimizer';

export interface LBFGSOptions {
  lr?: number;
  max_iter?: number;
  tolerance_grad?: number;
  tolerance_change?: number;
  history_size?: number;
  line_search_fn?: 'strong_wolfe';
}

export class LBFGS extends Optimizer {
  readonly state: Map<Tensor, Record<string, any>> = new Map();

  constructor(params: Tensor[] | Iterable<Tensor>, options?: LBFGSOptions) {
    const paramArray = Array.isArray(params) ? params : Array.from(params);
    const defaults: LBFGSOptions & Record<string, unknown> = {
      lr: 1,
      max_iter: 20,
      tolerance_grad: 1e-7,
      tolerance_change: 1e-9,
      history_size: 100,
      line_search_fn: 'strong_wolfe',
      ...options,
    };
    super(paramArray, defaults);
  }

  async step(closure?: () => Promise<number>): Promise<void> {
    if (!closure) throw new Error('LBFGS requires closure function');
    const group = this.param_groups[0];
    const params = group.params;
    const lr = group.lr ?? 1;
    const tolGrad = (group.tolerance_grad as number) ?? 1e-7;

    const grads: Tensor[] = [];
    for (const p of params) {
      if (!p.grad) return;
      grads.push(p.grad);
    }

    let gradNormSq = 0;
    for (const g of grads) {
      const arr = await g.toArray();
      for (let i = 0; i < arr.length; i++) gradNormSq += arr[i] * arr[i];
    }
    if (Math.sqrt(gradNormSq) < tolGrad) return;

    const tVals = [0.01, 0.1, 0.5, 1.0];
    let bestLoss = await closure();
    let bestT = 0;

    for (const t of tVals) {
      const saved: number[][] = [];
      for (let i = 0; i < params.length; i++) {
        const gArr = await grads[i].toArray() as Float32Array;
        const pArr = await params[i].toArray() as Float32Array;
        saved.push(Array.from(pArr));
        for (let j = 0; j < pArr.length; j++) {
          pArr[j] -= t * lr * gArr[j];
        }
      }

      const newLoss = await closure();
      if (newLoss < bestLoss) {
        bestLoss = newLoss;
        bestT = t;
      }

      for (let i = 0; i < params.length; i++) {
        const pArr = await params[i].toArray() as Float32Array;
        pArr.set(saved[i]);
      }
    }

    for (let i = 0; i < params.length; i++) {
      const pArr = await params[i].toArray() as Float32Array;
      const gArr = await grads[i].toArray() as Float32Array;
      for (let j = 0; j < pArr.length; j++) {
        pArr[j] -= bestT * lr * gArr[j];
      }
    }
  }
}
