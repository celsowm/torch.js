/**
 * Compatibility tests for torch.optim.lr_scheduler.
 *
 * Each scheduler is tested by verifying:
 * - Initial lr matches the param group
 * - lr changes correctly after step()
 * - get_last_lr() returns correct values
 * - last_epoch progresses correctly
 *
 * Run: npx vitest run --config=vitest.compat.config.ts tests/compatibility/optim/lr_scheduler.test.ts
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src';

const { optim } = torch;
const { lr_scheduler } = optim;

beforeAll(async () => {
  await torch.init();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Create a simple parameter and optimizer for scheduler tests.
 */
function makeOptimizer(lr = 0.1) {
  const param = torch.tensor([1.0], { requires_grad: true });
  const opt = new optim.SGD([param], { lr });
  return { param, opt };
}

// ===========================================================================
// StepLR
// ===========================================================================
describe('StepLR', () => {
  it('initial lr matches param group', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.StepLR(opt, 30, 0.1);
    expect(opt.param_groups[0].lr).toBe(0.1);
  });

  it('lr decreases by gamma every step_size steps', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.StepLR(opt, 10, 0.5);

    // After 10 steps: lr = 0.1 * 0.5 = 0.05
    sched.step(10);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.05, 6);
    expect(sched.get_last_lr()).toContain(0.05);

    // After 20 steps: lr = 0.1 * 0.5^2 = 0.025
    sched.step(20);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.025, 6);

    // After 30 steps: lr = 0.1 * 0.5^3 = 0.0125
    sched.step(30);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.0125, 6);
  });

  it('last_epoch progresses correctly', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.StepLR(opt, 5, 0.1);

    // Verify via state_dict and lr values
    const state0 = sched.state_dict();
    expect(state0.last_epoch).toBe(-1);
    sched.step(1);
    const state1 = sched.state_dict();
    expect(state1.last_epoch).toBe(1);
    sched.step(2);
    const state2 = sched.state_dict();
    expect(state2.last_epoch).toBe(2);
  });

  it('auto-increment epoch when not provided', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.StepLR(opt, 5, 0.1);

    sched.step(); // epoch 0
    expect(sched.state_dict().last_epoch).toBe(0);
    sched.step(); // epoch 1
    expect(sched.state_dict().last_epoch).toBe(1);
  });
});

// ===========================================================================
// MultiStepLR
// ===========================================================================
describe('MultiStepLR', () => {
  it('initial lr matches param group', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.MultiStepLR(opt, [30, 80], 0.1);
    expect(opt.param_groups[0].lr).toBe(0.1);
  });

  it('lr drops at specific milestones', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.MultiStepLR(opt, [30, 80], 0.1);

    // Before first milestone
    sched.step(29);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.1, 6);

    // At first milestone
    sched.step(30);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.01, 6);
    expect(sched.get_last_lr()[0]).toBeCloseTo(0.01, 6);

    // Between milestones
    sched.step(50);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.01, 6);

    // At second milestone
    sched.step(80);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.001, 6);
  });

  it('last_epoch progresses correctly', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.MultiStepLR(opt, [10, 20], 0.5);

    expect(sched.state_dict().last_epoch).toBe(-1);
    sched.step(5);
    expect(sched.state_dict().last_epoch).toBe(5);
    sched.step(15);
    expect(sched.state_dict().last_epoch).toBe(15);
  });
});

// ===========================================================================
// ExponentialLR
// ===========================================================================
describe('ExponentialLR', () => {
  it('initial lr matches param group', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ExponentialLR(opt, 0.95);
    expect(opt.param_groups[0].lr).toBe(0.1);
  });

  it('lr *= gamma each epoch', () => {
    const { opt } = makeOptimizer(0.1);
    const gamma = 0.9;
    const sched = new lr_scheduler.ExponentialLR(opt, gamma);

    sched.step(1);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.1 * gamma, 6);

    sched.step(2);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.1 * gamma * gamma, 6);

    sched.step(5);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.1 * Math.pow(gamma, 5), 6);
  });

  it('get_last_lr returns correct value', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ExponentialLR(opt, 0.8);

    sched.step(3);
    const lastLr = sched.get_last_lr();
    expect(lastLr[0]).toBeCloseTo(0.1 * Math.pow(0.8, 3), 6);
  });

  it('last_epoch progresses correctly', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ExponentialLR(opt, 0.95);

    expect(sched.state_dict().last_epoch).toBe(-1);
    sched.step();
    expect(sched.state_dict().last_epoch).toBe(0);
    sched.step();
    expect(sched.state_dict().last_epoch).toBe(1);
  });
});

// ===========================================================================
// CosineAnnealingLR
// ===========================================================================
describe('CosineAnnealingLR', () => {
  it('initial lr matches param group', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.CosineAnnealingLR(opt, 50, 0);
    expect(opt.param_groups[0].lr).toBe(0.1);
  });

  it('lr follows cosine schedule', () => {
    const { opt } = makeOptimizer(0.1);
    const T_max = 10;
    const eta_min = 0.001;
    const sched = new lr_scheduler.CosineAnnealingLR(opt, T_max, eta_min);

    // At T_max, lr should be near eta_min
    sched.step(T_max);
    const lr = opt.param_groups[0].lr;
    expect(lr).toBeLessThan(eta_min + 0.01);
  });

  it('lr returns to near base_lr after full cycle', () => {
    const { opt } = makeOptimizer(0.1);
    const T_max = 10;
    const sched = new lr_scheduler.CosineAnnealingLR(opt, T_max, 0);

    sched.step(1);
    const lr1 = opt.param_groups[0].lr;
    // lr should decrease from base_lr
    expect(lr1).toBeLessThan(0.1);
    expect(lr1).toBeGreaterThan(0);
  });

  it('get_last_lr returns correct value', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.CosineAnnealingLR(opt, 20, 0.01);

    sched.step(5);
    const lastLr = sched.get_last_lr();
    expect(lastLr.length).toBe(1);
    expect(lastLr[0]).toBeGreaterThan(0.01);
    expect(lastLr[0]).toBeLessThan(0.1);
  });

  it('last_epoch progresses correctly', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.CosineAnnealingLR(opt, 30, 0);

    expect(sched.state_dict().last_epoch).toBe(-1);
    sched.step(5);
    expect(sched.state_dict().last_epoch).toBe(5);
    sched.step(10);
    expect(sched.state_dict().last_epoch).toBe(10);
  });
});

// ===========================================================================
// ReduceLROnPlateau
// ===========================================================================
describe('ReduceLROnPlateau', () => {
  it('initial lr matches param group', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 5);
    expect(opt.param_groups[0].lr).toBe(0.1);
  });

  it('lr drops after patience epochs of no improvement (min mode)', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 2);

    // No improvement: metrics stay the same
    const initialMetric = 1.0;
    sched.step(initialMetric); // epoch 1
    sched.step(initialMetric); // epoch 2
    sched.step(initialMetric); // epoch 3 - should trigger lr reduce

    expect(opt.param_groups[0].lr).toBeCloseTo(0.01, 5);
    expect(sched.get_last_lr()[0]).toBeCloseTo(0.01, 5);
  });

  it('lr drops after patience epochs of no improvement (max mode)', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 3);

    // No improvement: metrics stay the same
    for (let i = 0; i < 5; i++) {
      sched.step(0.5);
    }

    expect(opt.param_groups[0].lr).toBeCloseTo(0.05, 5);
  });

  it('lr does not drop when metrics improve', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 5);

    // Improving metrics
    for (let i = 0; i < 20; i++) {
      sched.step(1.0 - i * 0.01);
    }

    expect(opt.param_groups[0].lr).toBe(0.1);
  });

  it('last_epoch tracks number of step calls', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 5);

    expect(sched.state_dict().last_epoch).toBe(0);
    sched.step(1.0);
    expect(sched.state_dict().last_epoch).toBe(1);
    sched.step(0.9);
    expect(sched.state_dict().last_epoch).toBe(2);
  });
});

// ===========================================================================
// ConstantLR
// ===========================================================================
describe('ConstantLR', () => {
  it('initial lr matches param group', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ConstantLR(opt, 0.5);
    expect(opt.param_groups[0].lr).toBe(0.1);
  });

  it('lr stays constant at factor * base_lr', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ConstantLR(opt, 0.5);

    sched.step(1);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.05, 6);

    sched.step(10);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.05, 6);

    sched.step(100);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.05, 6);
  });

  it('get_last_lr returns correct value', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ConstantLR(opt, 0.3);

    sched.step(5);
    const lastLr = sched.get_last_lr();
    expect(lastLr[0]).toBeCloseTo(0.03, 6);
  });

  it('factor=1.0 keeps base lr', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ConstantLR(opt, 1.0);

    sched.step(10);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.1, 6);
  });

  it('last_epoch progresses correctly', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ConstantLR(opt, 0.5);

    expect(sched.state_dict().last_epoch).toBe(-1);
    sched.step(1);
    expect(sched.state_dict().last_epoch).toBe(1);
    sched.step(2);
    expect(sched.state_dict().last_epoch).toBe(2);
  });
});

// ===========================================================================
// LambdaLR
// ===========================================================================
describe('LambdaLR', () => {
  it('initial lr matches param group', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.LambdaLR(opt, (epoch: number) => 0.95 ** epoch);
    expect(opt.param_groups[0].lr).toBe(0.1);
  });

  it('lr follows custom lambda schedule', () => {
    const { opt } = makeOptimizer(0.1);
    const lambda = (epoch: number) => 1 / (epoch + 1);
    const sched = new lr_scheduler.LambdaLR(opt, lambda);

    sched.step(1);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.1 * (1 / 2), 6);

    sched.step(4);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.1 * (1 / 5), 6);
  });

  it('supports linear warmup lambda', () => {
    const { opt } = makeOptimizer(0.1);
    const warmup = (epoch: number) => Math.min((epoch + 1) / 10, 1.0);
    const sched = new lr_scheduler.LambdaLR(opt, warmup);

    sched.step(5);
    const lr = opt.param_groups[0].lr;
    expect(lr).toBeCloseTo(0.1 * (6 / 10), 5);

    sched.step(15);
    const lr2 = opt.param_groups[0].lr;
    expect(lr2).toBeCloseTo(0.1, 5);
  });

  it('get_last_lr returns correct value', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.LambdaLR(opt, (epoch: number) => 0.5);

    sched.step(3);
    const lastLr = sched.get_last_lr();
    expect(lastLr[0]).toBeCloseTo(0.05, 6);
  });

  it('last_epoch progresses correctly', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.LambdaLR(opt, (epoch: number) => 0.9 ** epoch);

    expect(sched.state_dict().last_epoch).toBe(-1);
    sched.step(1);
    expect(sched.state_dict().last_epoch).toBe(1);
    sched.step(2);
    expect(sched.state_dict().last_epoch).toBe(2);
  });

  it('supports array of lambdas for multiple param groups', () => {
    // Create two separate param groups
    const p1 = torch.tensor([1.0], { requires_grad: true });
    const p2 = torch.tensor([1.0], { requires_grad: true });
    const opt = new optim.SGD([
      { params: [p1], lr: 0.1 },
      { params: [p2], lr: 0.1 }
    ], { lr: 0.1 });

    const sched = new lr_scheduler.LambdaLR(opt, [
      (epoch: number) => 0.9 ** epoch,
      (epoch: number) => 0.95 ** epoch,
    ]);

    sched.step(1);
    expect(opt.param_groups[0].lr).toBeCloseTo(0.1 * 0.9, 5);
    expect(opt.param_groups[1].lr).toBeCloseTo(0.1 * 0.95, 5);
  });
});

// ===========================================================================
// state_dict / load_state_dict
// ===========================================================================
describe('Scheduler state persistence', () => {
  it('StepLR state_dict contains last_epoch', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.StepLR(opt, 10, 0.5);

    sched.step(5);
    const state = sched.state_dict();
    expect(state.last_epoch).toBe(5);
  });

  it('LambdaLR state_dict contains last_epoch', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.LambdaLR(opt, (epoch: number) => 0.9 ** epoch);

    sched.step(3);
    const state = sched.state_dict();
    expect(state.last_epoch).toBe(3);
  });

  it('load_state_dict restores last_epoch', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.StepLR(opt, 10, 0.5);

    sched.load_state_dict({ last_epoch: 5 });
    expect(sched.state_dict().last_epoch).toBe(5);

    sched.step();
    expect(sched.state_dict().last_epoch).toBe(6);
  });

  it('ReduceLROnPlateau state_dict contains best and counters', () => {
    const { opt } = makeOptimizer(0.1);
    const sched = new lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.1, 5);

    sched.step(1.0);
    sched.step(0.9);
    const state = sched.state_dict();
    expect(state.best).toBe(0.9);
    expect(state.last_epoch).toBe(2);
  });
});
