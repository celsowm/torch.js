/**
 * RMSprop optimizer.
 * @status implemented
 * @pytorch torch.optim.RMSprop
 */

import { Tensor } from '../tensor';
import { Optimizer } from './optimizer';

export interface RMSpropOptions {
  lr?: number;
  alpha?: number;
  eps?: number;
  weight_decay?: number;
  momentum?: number;
  centered?: boolean;
}

export class RMSprop extends Optimizer {
  private squareAvg: Tensor[];
  private momentumBuffer: Tensor[];

  constructor(params: Tensor[], options: RMSpropOptions = {}) {
    const {
      lr = 0.01,
      alpha = 0.99,
      eps = 1e-8,
      weight_decay = 0,
      momentum = 0,
      centered = false,
    } = options;

    super(params, {
      lr,
      alpha,
      eps,
      weight_decay,
      momentum,
      centered,
    });

    this.squareAvg = [];
    this.momentumBuffer = [];
  }

  async step(): Promise<void> {
    let paramIdx = 0;
    
    for (const group of this.param_groups) {
      const lr = group.lr as number;
      const alpha = group.alpha as number;
      const eps = group.eps as number;
      const weight_decay = group.weight_decay as number;
      const momentum = group.momentum as number;

      for (const param of group.params) {
        if (param.grad === null) continue;

        const grad = param.grad;
        
        // Weight decay
        let g = grad;
        if (weight_decay !== 0) {
          g = grad.add(param.mul(weight_decay));
        }

        // Get or create square average
        if (!this.squareAvg[paramIdx]) {
          this.squareAvg[paramIdx] = g.zeros_like();
        }
        const squareAvg = this.squareAvg[paramIdx];

        // Update square average: E[g^2] = alpha * E[g^2] + (1 - alpha) * g^2
        const newSquareAvg = squareAvg.mul(alpha).add(g.pow(2).mul(1 - alpha));
        this.squareAvg[paramIdx] = newSquareAvg;

        let update = g.div(newSquareAvg.sqrt().add(eps));

        // Momentum
        if (momentum > 0) {
          if (!this.momentumBuffer[paramIdx]) {
            this.momentumBuffer[paramIdx] = g.zeros_like();
          }
          const buf = this.momentumBuffer[paramIdx];
          const newBuf = buf.mul(momentum).add(update);
          this.momentumBuffer[paramIdx] = newBuf;
          update = newBuf;
        }

        // Update parameter
        param.copy_(param.sub(update.mul(lr)));
        
        paramIdx++;
      }
    }
  }
}
