/**
 * Base Optimizer class.
 * @status implemented
 * @pytorch torch.optim.Optimizer
 */

import { Tensor } from '../tensor';

export interface ParamGroup {
  params: Tensor[];
  lr: number;
  initial_lr?: number;
  [key: string]: unknown;
}

/**
 * Base class for all optimizers.
 */
export abstract class Optimizer {
  public param_groups: ParamGroup[];
  protected defaults: Record<string, unknown>;

  constructor(params: Tensor[] | Iterable<Tensor>, defaults: Record<string, unknown>) {
    this.defaults = defaults;
    this.param_groups = [];

    // Convert params to array if needed
    const paramArray = Array.isArray(params) ? params : Array.from(params);

    // Create initial param group
    const lr = (defaults.lr as number) ?? 0.01;
    this.param_groups.push({
      params: paramArray,
      lr,
      initial_lr: lr,
      ...defaults,
    });
  }

  /**
   * Clears the gradients of all optimized parameters.
   * @pytorch optimizer.zero_grad()
   */
  zero_grad(): void {
    for (const group of this.param_groups) {
      for (const param of group.params) {
        param.grad = null;
      }
    }
  }

  /**
   * Performs a single optimization step.
   * @pytorch optimizer.step()
   */
  abstract step(): Promise<void>;
}
