/**
 * Adagrad optimizer.
 * @status implemented
 * @pytorch torch.optim.Adagrad
 */

import { Tensor } from '../tensor';
import { Optimizer } from './optimizer';

export interface AdagradOptions {
  lr?: number;
  lr_decay?: number;
  weight_decay?: number;
  initial_accumulator_value?: number;
  eps?: number;
}

export class Adagrad extends Optimizer {
  private sumGradSq: Tensor[];

  constructor(params: Tensor[], options: AdagradOptions = {}) {
    const {
      lr = 0.01,
      lr_decay = 0,
      weight_decay = 0,
      initial_accumulator_value = 0,
      eps = 1e-10,
    } = options;

    super(params, {
      lr,
      lr_decay,
      weight_decay,
      initial_accumulator_value,
      eps,
    });

    this.sumGradSq = [];
  }

  async step(): Promise<void> {
    let paramIdx = 0;
    
    for (const group of this.param_groups) {
      const lr = group.lr as number;
      const weight_decay = group.weight_decay as number;
      const eps = group.eps as number;

      for (const param of group.params) {
        if (param.grad === null) continue;

        const grad = param.grad;

        // Weight decay
        let g = grad;
        if (weight_decay !== 0) {
          g = grad.add(param.mul(weight_decay));
        }

        // Get or create sum of squares
        if (!this.sumGradSq[paramIdx]) {
          this.sumGradSq[paramIdx] = g.zeros_like();
        }
        const sum = this.sumGradSq[paramIdx];

        // Update sum of squares: sum += g^2
        const newSum = sum.add(g.pow(2));
        this.sumGradSq[paramIdx] = newSum;

        // Update parameter: p -= lr * g / (sqrt(sum) + eps)
        const std = newSum.sqrt().add(eps);
        param.copy_(param.sub(g.div(std).mul(lr)));
        
        paramIdx++;
      }
    }
  }
}
