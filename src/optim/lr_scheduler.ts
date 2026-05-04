/**
 * Learning rate schedulers.
 * @pytorch torch.optim.lr_scheduler
 */

import type { Optimizer } from './optimizer';

/**
 * Base class for all learning rate schedulers.
 */
abstract class _LRScheduler {
  protected optimizer: Optimizer;
  protected base_lrs: number[];
  protected last_epoch: number;
  protected verbose: boolean;

  constructor(optimizer: Optimizer, last_epoch: number = -1, verbose: boolean = false) {
    this.optimizer = optimizer;
    this.base_lrs = optimizer.param_groups.map(g => g.lr);
    this.last_epoch = last_epoch;
    this.verbose = verbose;

    // Set initial learning rates
    if (last_epoch === -1) {
      for (const group of this.optimizer.param_groups) {
        group.lr = group.initial_lr ?? group.lr;
      }
    }
  }

  /** Return the computed learning rate. */
  abstract get_lr(): number[];

  /** Step the scheduler (called after each epoch). */
  step(epoch?: number): void {
    if (epoch === undefined || epoch === null) {
      epoch = this.last_epoch + 1;
    }
    this.last_epoch = epoch;

    const lrs = this.get_lr();
    for (let i = 0; i < this.optimizer.param_groups.length; i++) {
      this.optimizer.param_groups[i].lr = lrs[i];
      if (this.verbose) {
        console.log(`Adjusting learning rate of group ${i} to ${lrs[i].toFixed(6)}`);
      }
    }
  }

  /** Get the last computed learning rate. */
  get_last_lr(): number[] {
    return this.optimizer.param_groups.map(g => g.lr);
  }

  /** Print the current learning rate. */
  print_lr(): void {
    console.log(this.get_last_lr());
  }

  /** Reset the scheduler to initial state. */
  state_dict(): Record<string, any> {
    return { last_epoch: this.last_epoch };
  }

  load_state_dict(state_dict: Record<string, any>): void {
    this.last_epoch = state_dict.last_epoch ?? -1;
  }
}

/**
 * Multiplicative factor scheduler.
 * @pytorch torch.optim.lr_scheduler.MultiplicativeLR
 */
export class MultiplicativeLR extends _LRScheduler {
  private lr_lambda: (epoch: number) => number;

  constructor(optimizer: Optimizer, lr_lambda: (epoch: number) => number, last_epoch: number = -1, verbose: boolean = false) {
    super(optimizer, last_epoch, verbose);
    this.lr_lambda = lr_lambda;
  }

  get_lr(): number[] {
    if (this.last_epoch <= 0) {
      return this.base_lrs;
    }
    const factor = this.lr_lambda(this.last_epoch);
    return this.optimizer.param_groups.map((group, i) => group.lr * factor);
  }
}

/**
 * Multiply learning rate by gamma every step_size epochs.
 * @pytorch torch.optim.lr_scheduler.StepLR
 *
 * @example
 * ```ts
 * const scheduler = new StepLR(optimizer, step_size=30, gamma=0.1);
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   train(model);
 *   scheduler.step();
 * }
 * ```
 */
export class StepLR extends _LRScheduler {
  private step_size: number;
  private gamma: number;

  constructor(optimizer: Optimizer, step_size: number, gamma: number = 0.1, last_epoch: number = -1, verbose: boolean = false) {
    super(optimizer, last_epoch, verbose);
    this.step_size = step_size;
    this.gamma = gamma;
  }

  get_lr(): number[] {
    // Decay every step_size epochs
    const num_steps = Math.floor(this.last_epoch / this.step_size);
    return this.base_lrs.map(lr => lr * Math.pow(this.gamma, num_steps));
  }
}

/**
 * Multiply learning rate by gamma at each epoch where epoch is in milestones.
 * @pytorch torch.optim.lr_scheduler.MultiStepLR
 *
 * @example
 * ```ts
 * const scheduler = new MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1);
 * ```
 */
export class MultiStepLR extends _LRScheduler {
  private milestones: number[];
  private gamma: number;

  constructor(optimizer: Optimizer, milestones: number[], gamma: number = 0.1, last_epoch: number = -1, verbose: boolean = false) {
    super(optimizer, last_epoch, verbose);
    this.milestones = milestones.sort((a, b) => a - b);
    this.gamma = gamma;
  }

  get_lr(): number[] {
    // Count how many milestones have been passed
    let num_milestones = 0;
    for (const m of this.milestones) {
      if (m <= this.last_epoch) num_milestones++;
    }
    return this.base_lrs.map(lr => lr * Math.pow(this.gamma, num_milestones));
  }
}

/**
 * Exponential decay scheduler.
 * lr = base_lr * gamma^epoch
 * @pytorch torch.optim.lr_scheduler.ExponentialLR
 */
export class ExponentialLR extends _LRScheduler {
  private gamma: number;

  constructor(optimizer: Optimizer, gamma: number, last_epoch: number = -1, verbose: boolean = false) {
    super(optimizer, last_epoch, verbose);
    this.gamma = gamma;
  }

  get_lr(): number[] {
    return this.base_lrs.map(lr => lr * Math.pow(this.gamma, this.last_epoch));
  }
}

/**
 * Cosine annealing scheduler.
 * lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / T_max))
 * @pytorch torch.optim.lr_scheduler.CosineAnnealingLR
 */
export class CosineAnnealingLR extends _LRScheduler {
  private T_max: number;
  private eta_min: number;

  constructor(optimizer: Optimizer, T_max: number, eta_min: number = 0, last_epoch: number = -1, verbose: boolean = false) {
    super(optimizer, last_epoch, verbose);
    this.T_max = T_max;
    this.eta_min = eta_min;
  }

  get_lr(): number[] {
    if (this.last_epoch === 0) {
      return this.base_lrs;
    }
    if ((this.last_epoch - 1 - this.T_max) % (2 * this.T_max) === 0) {
      return this.base_lrs.map(lr =>
        lr + (this.eta_min - lr) * (1 - Math.cos(Math.PI / this.T_max)) / 2
      );
    }
    return this.base_lrs.map(lr =>
      (1 + Math.cos(Math.PI * this.last_epoch / this.T_max)) /
      (1 + Math.cos(Math.PI * (this.last_epoch - 1) / this.T_max)) *
      (lr - this.eta_min) + this.eta_min
    );
  }
}

/**
 * Reduce learning rate on plateau.
 * @pytorch torch.optim.lr_scheduler.ReduceLROnPlateau
 *
 * @example
 * ```ts
 * const scheduler = new ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10);
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   const val_loss = evaluate(model);
 *   scheduler.step(val_loss);
 * }
 * ```
 */
export class ReduceLROnPlateau {
  private optimizer: Optimizer;
  private mode: 'min' | 'max';
  private factor: number;
  private patience: number;
  private threshold: number;
  private threshold_mode: 'rel' | 'abs';
  private cooldown: number;
  private cooldown_counter: number;
  private best: number;
  private num_bad_epochs: number;
  private min_lrs: number[];
  private last_epoch: number;
  private verbose: boolean;

  constructor(
    optimizer: Optimizer,
    mode: 'min' | 'max' = 'min',
    factor: number = 0.1,
    patience: number = 10,
    threshold: number = 1e-4,
    threshold_mode: 'rel' | 'abs' = 'rel',
    cooldown: number = 0,
    min_lr: number | number[] = 0,
    verbose: boolean = false,
  ) {
    this.optimizer = optimizer;
    this.mode = mode;
    this.factor = factor;
    this.patience = patience;
    this.threshold = threshold;
    this.threshold_mode = threshold_mode;
    this.cooldown = cooldown;
    this.cooldown_counter = 0;
    this.best = mode === 'min' ? Infinity : -Infinity;
    this.num_bad_epochs = 0;
    this.min_lrs = Array.isArray(min_lr) ? min_lr : optimizer.param_groups.map(() => min_lr);
    this.last_epoch = 0;
    this.verbose = verbose;
  }

  step(metrics: number): void {
    this.last_epoch++;

    if (this.cooldown_counter > 0) {
      this.cooldown_counter--;
      this.num_bad_epochs = 0; // reset patience during cooldown
      return;
    }

    const is_better = this.mode === 'min'
      ? metrics < this.best - this.threshold
      : metrics > this.best + this.threshold;

    const is_equal = this.mode === 'min'
      ? Math.abs(metrics - this.best) <= this.threshold
      : Math.abs(metrics - this.best) <= this.threshold;

    if (is_better) {
      this.best = metrics;
      this.num_bad_epochs = 0;
    } else if (is_equal) {
      this.num_bad_epochs = 0;
    } else {
      this.num_bad_epochs++;
    }

    if (this.num_bad_epochs > this.patience) {
      this._reduce_lr();
      this.cooldown_counter = this.cooldown;
      this.num_bad_epochs = 0;
    }
  }

  private _reduce_lr(): void {
    for (let i = 0; i < this.optimizer.param_groups.length; i++) {
      const old_lr = this.optimizer.param_groups[i].lr;
      const new_lr = Math.max(old_lr * this.factor, this.min_lrs[i] ?? 0);
      if (Math.abs(old_lr - new_lr) > 1e-8) {
        this.optimizer.param_groups[i].lr = new_lr;
        if (this.verbose) {
          console.log(`Reducing learning rate of group ${i} to ${new_lr.toFixed(6)}`);
        }
      }
    }
  }

  get_last_lr(): number[] {
    return this.optimizer.param_groups.map(g => g.lr);
  }

  state_dict(): Record<string, any> {
    return {
      best: this.best,
      cooldown_counter: this.cooldown_counter,
      num_bad_epochs: this.num_bad_epochs,
      last_epoch: this.last_epoch,
    };
  }

  load_state_dict(state_dict: Record<string, any>): void {
    this.best = state_dict.best ?? (this.mode === 'min' ? Infinity : -Infinity);
    this.cooldown_counter = state_dict.cooldown_counter ?? 0;
    this.num_bad_epochs = state_dict.num_bad_epochs ?? 0;
    this.last_epoch = state_dict.last_epoch ?? 0;
  }
}

/**
 * Constant learning rate scheduler (just keeps the base lr).
 * @pytorch torch.optim.lr_scheduler.ConstantLR
 */
export class ConstantLR extends _LRScheduler {
  private factor: number;

  constructor(optimizer: Optimizer, factor: number = 1.0, last_epoch: number = -1, verbose: boolean = false) {
    super(optimizer, last_epoch, verbose);
    this.factor = factor;
  }

  get_lr(): number[] {
    return this.base_lrs.map(lr => lr * this.factor);
  }
}

/**
 * Lambda scheduler - user provides a function for each group.
 * @pytorch torch.optim.lr_scheduler.LambdaLR
 */
export class LambdaLR extends _LRScheduler {
  private lr_lambdas: ((epoch: number) => number)[];

  constructor(
    optimizer: Optimizer,
    lr_lambda: ((epoch: number) => number) | ((epoch: number) => number)[],
    last_epoch: number = -1,
    verbose: boolean = false,
  ) {
    super(optimizer, last_epoch, verbose);
    if (typeof lr_lambda === 'function') {
      this.lr_lambdas = optimizer.param_groups.map(() => lr_lambda);
    } else {
      this.lr_lambdas = lr_lambda;
    }
  }

  get_lr(): number[] {
    if (this.last_epoch < 0) {
      return this.optimizer.param_groups.map(g => g.initial_lr ?? g.lr);
    }
    return this.optimizer.param_groups.map((group, i) =>
      (group.initial_lr ?? group.lr) * this.lr_lambdas[i](this.last_epoch)
    );
  }
}

/**
 * Linear warmup scheduler.
 * lr = base_lr * (epoch / warmup_epochs) for epoch < warmup_epochs
 * lr = base_lr for epoch >= warmup_epochs
 */
export class LinearWarmupLR extends _LRScheduler {
  private warmup_epochs: number;

  constructor(optimizer: Optimizer, warmup_epochs: number, last_epoch: number = -1, verbose: boolean = false) {
    super(optimizer, last_epoch, verbose);
    this.warmup_epochs = warmup_epochs;
  }

  get_lr(): number[] {
    if (this.last_epoch < this.warmup_epochs) {
      const scale = (this.last_epoch + 1) / this.warmup_epochs;
      return this.base_lrs.map(lr => lr * scale);
    }
    return this.base_lrs;
  }
}

export default {
  _LRScheduler,
  MultiplicativeLR,
  StepLR,
  MultiStepLR,
  ExponentialLR,
  CosineAnnealingLR,
  ReduceLROnPlateau,
  ConstantLR,
  LambdaLR,
  LinearWarmupLR,
};
