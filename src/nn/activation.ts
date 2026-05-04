/**
 * Activation function modules.
 * @status implemented
 * @pytorch torch.nn
 */

import { Tensor } from '../tensor';
import { Module } from './module';
import { Parameter } from './parameter';
import * as F from './functional';
import { tensor, zeros, rand } from '../ops/creation';

/**
 * ReLU activation.
 * @pytorch torch.nn.ReLU
 */
export class ReLU extends Module {
  private inplace: boolean;

  constructor(inplace: boolean = false) {
    super();
    this.inplace = inplace;
  }

  forward(input: Tensor): Tensor {
    return F.relu(input, this.inplace);
  }
}

/**
 * Sigmoid activation.
 * @pytorch torch.nn.Sigmoid
 */
export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    return input.sigmoid();
  }
}

/**
 * Tanh activation.
 * @pytorch torch.nn.Tanh
 */
export class Tanh extends Module {
  forward(input: Tensor): Tensor {
    return input.tanh();
  }
}

/**
 * LogSoftmax activation.
 * @pytorch torch.nn.LogSoftmax
 */
export class LogSoftmax extends Module {
  private dim: number;

  constructor(dim: number = -1) {
    super();
    this.dim = dim;
  }

  forward(input: Tensor): Tensor {
    return F.log_softmax(input, this.dim);
  }
}

/**
 * GELU activation (Gaussian Error Linear Unit).
 * Uses the tanh approximation as in GPT-2/BERT.
 * @pytorch torch.nn.GELU
 */
export class GELU extends Module {
  forward(input: Tensor): Tensor {
    return F.gelu(input);
  }
}

/**
 * Softmax activation.
 * @pytorch torch.nn.Softmax
 */
export class Softmax extends Module {
  private dim: number;

  constructor(dim: number = -1) {
    super();
    this.dim = dim;
  }

  forward(input: Tensor): Tensor {
    return F.softmax(input, this.dim);
  }
}

/**
 * Parametric ReLU activation.
 * y = x if x > 0, else num_parameters * x
 * The negative slope is a learnable parameter.
 * @pytorch torch.nn.PReLU
 */
export class PReLU extends Module {
  readonly num_parameters: number;
  readonly init: number;
  public weight: Parameter;

  constructor(num_parameters: number = 1, init: number = 0.25) {
    super();
    this.num_parameters = num_parameters;
    this.init = init;

    // Create a tensor with the init value
    const w = tensor(Array(num_parameters).fill(init), { requires_grad: true });
    this.weight = Parameter.create(w);
    this.register_parameter('weight', this.weight);
  }

  forward(input: Tensor): Tensor {
    // PReLU: x if x > 0, else weight * x
    const positive = input.relu();
    const negative = input.neg().relu().mul(this.weight.reshape([1, this.num_parameters, 1, 1]));
    return positive.sub(negative);
  }
}

/**
 * Continuously Differentiable ELU.
 * y = x if x > 0, else alpha * (exp(x / alpha) - 1)
 * @pytorch torch.nn.CELU
 */
export class CELU extends Module {
  readonly alpha: number;

  constructor(alpha: number = 1.0) {
    super();
    this.alpha = alpha;
  }

  forward(input: Tensor): Tensor {
    // CELU: x if x > 0, else alpha * (exp(x / alpha) - 1)
    const positive = input.relu();
    const negative = input.div(this.alpha).exp().sub(1).mul(this.alpha);
    return positive.add(negative);
  }
}

/**
 * Randomized Leaky ReLU.
 * During training, uses a random slope from [lower, upper] for negative values.
 * During evaluation, uses the average of lower and upper.
 * @pytorch torch.nn.RReLU
 */
export class RReLU extends Module {
  readonly lower: number;
  readonly upper: number;
  private _noise: Tensor | null = null;

  constructor(lower: number = 0.125, upper: number = 0.33333333333) {
    super();
    this.lower = lower;
    this.upper = upper;
  }

  forward(input: Tensor): Tensor {
    if (this.training) {
      // Generate random noise for negative slope
      const noise = rand([...input.shape]).mul(this.upper - this.lower).add(this.lower);
      this._noise = noise;
      const positive = input.relu();
      const negative = input.neg().relu().mul(noise);
      return positive.sub(negative);
    } else {
      // Use average of lower and upper
      const avgSlope = (this.lower + this.upper) / 2;
      const positive = input.relu();
      const negative = input.neg().relu().mul(avgSlope);
      return positive.sub(negative);
    }
  }
}

/**
 * HardTanh activation.
 * y = max(min(x, max_val), min_val)
 * @pytorch torch.nn.Hardtanh
 */
export class Hardtanh extends Module {
  readonly min_val: number;
  readonly max_val: number;

  constructor(min_val: number = -1.0, max_val: number = 1.0) {
    super();
    this.min_val = min_val;
    this.max_val = max_val;
  }

  forward(input: Tensor): Tensor {
    return input.clamp(this.min_val, this.max_val);
  }
}

/**
 * Hardshrink activation.
 * y = x if x > lambda, else 0; y = x if x < -lambda, else 0
 * @pytorch torch.nn.Hardshrink
 */
export class Hardshrink extends Module {
  readonly lambd: number;

  constructor(lambd: number = 0.5) {
    super();
    this.lambd = lambd;
  }

  forward(input: Tensor): Tensor {
    // Hardshrink: x if |x| > lambd, else 0
    const condition = input.abs().gt(this.lambd);
    return condition.where(input, zeros([...input.shape], { dtype: input.dtype }));
  }
}

/**
 * Softshrink activation.
 * y = x - lambda if x > lambda, else 0; y = x + lambda if x < -lambda, else 0
 * @pytorch torch.nn.Softshrink
 */
export class Softshrink extends Module {
  readonly lambd: number;

  constructor(lambd: number = 0.5) {
    super();
    this.lambd = lambd;
  }

  forward(input: Tensor): Tensor {
    // Softshrink: x - lambd if x > lambd, x + lambd if x < -lambd, else 0
    const positive = input.sub(this.lambd).relu();
    const negative = input.neg().sub(this.lambd).relu().neg();
    return positive.add(negative);
  }
}

/**
 * LogSigmoid activation.
 * y = log(1 / (1 + exp(-x)))
 * @pytorch torch.nn.LogSigmoid
 */
export class LogSigmoid extends Module {
  forward(input: Tensor): Tensor {
    return input.neg().exp().add(1).log().neg();
  }
}

/**
 * Softmin activation.
 * Softmin(x) = softmax(-x) = exp(-x) / sum(exp(-x))
 * @pytorch torch.nn.Softmin
 */
export class Softmin extends Module {
  private dim: number;

  constructor(dim: number = -1) {
    super();
    this.dim = dim;
  }

  forward(input: Tensor): Tensor {
    return F.softmax(input.neg(), this.dim);
  }
}
