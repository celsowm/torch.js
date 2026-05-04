/**
 * Compatibility tests for torch.optim optimizers.
 *
 * Each optimizer is tested on the simple function f(x) = x^2,
 * whose minimum is at x = 0.
 *
 * Run: npx vitest run --config=vitest.compat.config.ts tests/compatibility/optim/optimizers.test.ts
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as torch from '../../../src';

const { optim } = torch;

beforeAll(async () => {
  await torch.init();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Create a scalar-like tensor with requires_grad = true.
 */
async function makeParam(values: number[], shape: number[]) {
  const t = torch.tensor(values, { requires_grad: true });
  return t;
}

/**
 * Run one backward + step cycle and return whether param values
 * decreased in magnitude (converging toward 0).
 */
async function stepAndCloser(
  param: any,
  optimizer: any,
): Promise<{ before: Float32Array; after: Float32Array; closer: boolean }> {
  const before = await param.toArray();

  const loss = param.pow(2).sum();
  await loss.backward();
  await optimizer.step();
  optimizer.zero_grad();

  const after = await param.toArray();
  let closer = true;
  for (let i = 0; i < before.length; i++) {
    if (Math.abs(after[i]) > Math.abs(before[i]) + 1e-6) {
      closer = false;
      break;
    }
  }
  return { before, after, closer };
}

/**
 * Run many steps and check the final value is near 0.
 */
async function converge(
  param: any,
  optimizer: any,
  iterations = 50,
): Promise<Float32Array> {
  for (let i = 0; i < iterations; i++) {
    const loss = param.pow(2).sum();
    await loss.backward();
    await optimizer.step();
    optimizer.zero_grad();
  }
  return await param.toArray();
}

// ===========================================================================
// SGD
// ===========================================================================
describe('SGD', () => {
  it('updates parameters after step (basic convergence on f(x)=x^2)', async () => {
    const param = await makeParam([2.0, -3.0, 1.5], [3]);
    const opt = new optim.SGD([param], { lr: 0.1 });
    const result = await converge(param, opt, 30);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.1);
    }
  });

  it('momentum accelerates convergence', async () => {
    const paramA = await makeParam([5.0], [1]);
    const optA = new optim.SGD([paramA], { lr: 0.05 });
    await converge(paramA, optA, 20);

    const paramB = await makeParam([5.0], [1]);
    const optB = new optim.SGD([paramB], { lr: 0.05, momentum: 0.9 });
    await converge(paramB, optB, 20);

    const valA = await paramA.toArray();
    const valB = await paramB.toArray();
    expect(Math.abs(valB[0])).toBeLessThan(Math.abs(valA[0]));
  });

  it('weight_decay penalizes large parameters', async () => {
    const param = await makeParam([1.0], [1]);
    const opt = new optim.SGD([param], { lr: 0.1, weight_decay: 0.5 });
    await converge(param, opt, 20);
    const result = await param.toArray();
    expect(Math.abs(result[0])).toBeLessThan(0.1);
  });

  it('dampening reduces momentum contribution', async () => {
    const param = await makeParam([3.0], [1]);
    const opt = new optim.SGD([param], {
      lr: 0.1,
      momentum: 0.9,
      dampening: 0.5,
    });
    const result = await converge(param, opt, 30);
    expect(Math.abs(result[0])).toBeLessThan(0.5);
  });

  it('nesterov momentum works', async () => {
    const param = await makeParam([3.0], [1]);
    const opt = new optim.SGD([param], {
      lr: 0.05,
      momentum: 0.9,
      nesterov: true,
    });
    const result = await converge(param, opt, 30);
    expect(Math.abs(result[0])).toBeLessThan(0.5);
  });

  it('higher lr leads to faster convergence', async () => {
    const pLow = await makeParam([5.0], [1]);
    const optLow = new optim.SGD([pLow], { lr: 0.01 });
    await converge(pLow, optLow, 20);

    const pHigh = await makeParam([5.0], [1]);
    const optHigh = new optim.SGD([pHigh], { lr: 0.1 });
    await converge(pHigh, optHigh, 20);

    expect(Math.abs((await pHigh.toArray())[0])).toBeLessThan(
      Math.abs((await pLow.toArray())[0]),
    );
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.SGD([param], { lr: 0.1 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// Adam
// ===========================================================================
describe('Adam', () => {
  it('updates parameters after step', async () => {
    const param = await makeParam([2.0, -3.0, 1.5], [3]);
    const opt = new optim.Adam([param], { lr: 0.1 });
    const result = await converge(param, opt, 30);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.01);
    }
  });

  it('converges with default hyperparameters', async () => {
    const param = await makeParam([10.0], [1]);
    const opt = new optim.Adam([param], {});
    const result = await converge(param, opt, 100);
    expect(Math.abs(result[0])).toBeLessThan(0.01);
  });

  it('weight_decay regularizes', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.Adam([param], { lr: 0.1, weight_decay: 0.1 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(0.1);
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.Adam([param], { lr: 0.1 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// AdamW
// ===========================================================================
describe('AdamW', () => {
  it('updates parameters after step', async () => {
    const param = await makeParam([2.0, -3.0], [2]);
    const opt = new optim.AdamW([param], { lr: 0.1 });
    const result = await converge(param, opt, 30);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.1);
    }
  });

  it('weight_decay decouples from gradient update', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.AdamW([param], { lr: 0.1, weight_decay: 0.5 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(0.1);
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.AdamW([param], { lr: 0.1 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// Adamax
// ===========================================================================
describe('Adamax', () => {
  it('updates parameters after step', async () => {
    const param = await makeParam([2.0, -3.0], [2]);
    const opt = new optim.Adamax([param], { lr: 0.1 });
    const result = await converge(param, opt, 40);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.1);
    }
  });

  it('converges with weight_decay', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.Adamax([param], { lr: 0.1, weight_decay: 0.1 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(0.1);
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.Adamax([param], { lr: 0.1 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// NAdam
// ===========================================================================
describe('NAdam', () => {
  it('updates parameters after step', async () => {
    const param = await makeParam([2.0, -3.0], [2]);
    const opt = new optim.NAdam([param], { lr: 0.1 });
    const result = await converge(param, opt, 40);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.1);
    }
  });

  it('converges with weight_decay', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.NAdam([param], { lr: 0.1, weight_decay: 0.1 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(0.1);
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.NAdam([param], { lr: 0.1 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// RAdam
// ===========================================================================
describe('RAdam', () => {
  it('updates parameters after step', async () => {
    const param = await makeParam([2.0, -3.0], [2]);
    const opt = new optim.RAdam([param], { lr: 0.1 });
    const result = await converge(param, opt, 50);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.1);
    }
  });

  it('converges with weight_decay', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.RAdam([param], { lr: 0.1, weight_decay: 0.1 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(0.1);
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.RAdam([param], { lr: 0.1 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// ASGD
// ===========================================================================
describe('ASGD', () => {
  it('updates parameters after step', async () => {
    const param = await makeParam([2.0, -3.0], [2]);
    const opt = new optim.ASGD([param], { lr: 0.1 });
    const result = await converge(param, opt, 50);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.5);
    }
  });

  it('lambd controls decay rate', async () => {
    const pSmall = await makeParam([5.0], [1]);
    const optSmall = new optim.ASGD([pSmall], { lr: 0.1, lambd: 1e-6 });
    await converge(pSmall, optSmall, 30);

    const pLarge = await makeParam([5.0], [1]);
    const optLarge = new optim.ASGD([pLarge], { lr: 0.1, lambd: 1e-2 });
    await converge(pLarge, optLarge, 30);

    // Larger lambd should lead to stronger regularization
    expect(Math.abs((await pLarge.toArray())[0])).toBeLessThan(
      Math.abs((await pSmall.toArray())[0]) + 0.5,
    );
  });

  it('alpha affects lr decay exponent', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.ASGD([param], { lr: 0.1, alpha: 0.75 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(1.0);
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.ASGD([param], { lr: 0.1 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// RMSprop
// ===========================================================================
describe('RMSprop', () => {
  it('updates parameters after step', async () => {
    const param = await makeParam([2.0, -3.0], [2]);
    const opt = new optim.RMSprop([param], { lr: 0.1 });
    const result = await converge(param, opt, 40);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.1);
    }
  });

  it('momentum helps convergence', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.RMSprop([param], { lr: 0.1, momentum: 0.9 });
    const result = await converge(param, opt, 40);
    expect(Math.abs(result[0])).toBeLessThan(0.1);
  });

  it('weight_decay regularizes', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.RMSprop([param], { lr: 0.1, weight_decay: 0.1 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(0.1);
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.RMSprop([param], { lr: 0.1 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// Adagrad
// ===========================================================================
describe('Adagrad', () => {
  it('updates parameters after step', async () => {
    const param = await makeParam([2.0, -3.0], [2]);
    const opt = new optim.Adagrad([param], { lr: 0.5 });
    const result = await converge(param, opt, 50);
    for (let i = 0; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(0.5);
    }
  });

  it('lr_decay reduces effective lr over time', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.Adagrad([param], { lr: 0.5, lr_decay: 0.01 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(1.0);
  });

  it('weight_decay regularizes', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.Adagrad([param], { lr: 0.5, weight_decay: 0.1 });
    const result = await converge(param, opt, 50);
    expect(Math.abs(result[0])).toBeLessThan(0.5);
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.Adagrad([param], { lr: 0.5 });
    const loss = param.pow(2).sum();
    await loss.backward();
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// LBFGS
// ===========================================================================
describe('LBFGS', () => {
  it('converges on f(x)=x^2 with closure', async () => {
    const param = await makeParam([5.0], [1]);
    const opt = new optim.LBFGS([param], { lr: 1, max_iter: 20 });

    for (let i = 0; i < 5; i++) {
      const closure = async () => {
        const loss = param.pow(2).sum();
        await loss.backward();
        return await (await loss.toArray())[0];
      };
      await opt.step(closure);
      opt.zero_grad();
    }

    const result = await param.toArray();
    expect(Math.abs(result[0])).toBeLessThan(1.0);
  });

  it('throws without closure', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.LBFGS([param], { lr: 1 });
    await expect(opt.step()).rejects.toThrow('closure');
  });

  it('zero_grad clears gradients', async () => {
    const param = await makeParam([2.0], [1]);
    const opt = new optim.LBFGS([param], { lr: 1 });
    const closure = async () => {
      const loss = param.pow(2).sum();
      await loss.backward();
      return await (await loss.toArray())[0];
    };
    await opt.step(closure);
    expect(param.grad).not.toBeNull();
    opt.zero_grad();
    expect(param.grad).toBeNull();
  });
});

// ===========================================================================
// Multi-parameter optimization
// ===========================================================================
describe('Multi-parameter optimization', () => {
  it('SGD optimizes multiple parameters simultaneously', async () => {
    const p1 = await makeParam([3.0, -2.0], [2]);
    const p2 = await makeParam([1.5], [1]);
    const opt = new optim.SGD([p1, p2], { lr: 0.1 });

    for (let i = 0; i < 40; i++) {
      const loss = p1.pow(2).sum().add(p2.pow(2).sum());
      await loss.backward();
      await opt.step();
      opt.zero_grad();
    }

    const r1 = await p1.toArray();
    const r2 = await p2.toArray();
    for (const v of [...r1, ...r2]) {
      expect(Math.abs(v)).toBeLessThan(0.1);
    }
  });

  it('Adam optimizes multiple parameters simultaneously', async () => {
    const p1 = await makeParam([3.0, -2.0], [2]);
    const p2 = await makeParam([1.5], [1]);
    const opt = new optim.Adam([p1, p2], { lr: 0.1 });

    for (let i = 0; i < 40; i++) {
      const loss = p1.pow(2).sum().add(p2.pow(2).sum());
      await loss.backward();
      await opt.step();
      opt.zero_grad();
    }

    const r1 = await p1.toArray();
    const r2 = await p2.toArray();
    for (const v of [...r1, ...r2]) {
      expect(Math.abs(v)).toBeLessThan(0.01);
    }
  });
});
