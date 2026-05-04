/**
 * Probability distributions module.
 * Provides classes for sampling and computing log-probability for common distributions.
 * @status implemented
 * @pytorch torch.distributions
 */

import { Tensor } from '../tensor';
import { randn, rand, tensor } from '../ops/creation';
import { lgamma } from '../ops';

const PI = Math.PI;
const LOG_2PI = Math.log(2 * PI);

// ---------------------------------------------------------------------------
// Abstract base class
// ---------------------------------------------------------------------------

/**
 * Abstract base class for all probability distributions.
 * @pytorch torch.distributions.Distribution
 */
export abstract class Distribution {
  /**
   * Returns a sample from the distribution.
   * @param sample_shape Shape of the samples to draw
   * @pytorch Distribution.sample()
   */
  abstract sample(sample_shape?: number[]): Tensor | Promise<Tensor>;

  /**
   * Computes the log probability of the given value(s).
   * @param value Tensor of values to compute log probability for
   * @pytorch Distribution.log_prob()
   */
  abstract log_prob(value: Tensor): Tensor;

  /**
   * Computes the entropy of the distribution.
   * @pytorch Distribution.entropy()
   */
  abstract entropy(): Tensor;

  /**
   * Returns the mean of the distribution.
   * @pytorch Distribution.mean
   */
  abstract get mean(): Tensor;

  /**
   * Returns the variance of the distribution.
   * @pytorch Distribution.variance
   */
  abstract get variance(): Tensor;

  /**
   * Returns the perplexity of the distribution (exp(entropy)).
   * @pytorch Distribution.perplexity()
   */
  perplexity(): Tensor {
    return this.entropy().exp();
  }
}

// ---------------------------------------------------------------------------
// Normal (Gaussian) distribution
// ---------------------------------------------------------------------------

/**
 * Normal (Gaussian) distribution.
 * @pytorch torch.distributions.Normal
 */
export class Normal extends Distribution {
  private _loc: Tensor;
  private _scale: Tensor;

  constructor(loc: Tensor, scale: Tensor) {
    super();
    this._loc = loc;
    this._scale = scale;
  }

  get loc(): Tensor {
    return this._loc;
  }

  get scale(): Tensor {
    return this._scale;
  }

  sample(sample_shape: number[] = []): Tensor {
    const shape = [...sample_shape, ...this._loc.shape];
    const eps = randn(shape);
    return eps.mul(this._scale).add(this._loc);
  }

  log_prob(value: Tensor): Tensor {
    const var_ = this._scale.pow(2);
    const log_prob = value
      .sub(this._loc)
      .pow(2)
      .div(var_.mul(2))
      .neg()
      .sub(0.5 * LOG_2PI)
      .sub(this._scale.log());
    return log_prob;
  }

  entropy(): Tensor {
    return this._scale.log().add(0.5 * (LOG_2PI + 1));
  }

  get mean(): Tensor {
    return this._loc;
  }

  get variance(): Tensor {
    return this._scale.pow(2);
  }
}

// ---------------------------------------------------------------------------
// Bernoulli distribution
// ---------------------------------------------------------------------------

/**
 * Bernoulli distribution.
 * @pytorch torch.distributions.Bernoulli
 */
export class Bernoulli extends Distribution {
  private _probs: Tensor;

  constructor(probs: Tensor) {
    super();
    this._probs = probs;
  }

  get probs(): Tensor {
    return this._probs;
  }

  sample(sample_shape: number[] = []): Tensor {
    const shape = [...sample_shape, ...this._probs.shape];
    const u = rand(shape);
    return u.lt(this._probs).to('float32');
  }

  log_prob(value: Tensor): Tensor {
    const log_p = this._probs.log();
    const log_1mp = this._probs.mul(-1).add(1).log();
    return value.mul(log_p).add(value.mul(-1).add(1).mul(log_1mp));
  }

  entropy(): Tensor {
    const p = this._probs;
    const one_mp = p.mul(-1).add(1);
    return p.mul(p.log()).neg().add(one_mp.mul(one_mp.log()).neg());
  }

  get mean(): Tensor {
    return this._probs;
  }

  get variance(): Tensor {
    return this._probs.mul(this._probs.mul(-1).add(1));
  }
}

// ---------------------------------------------------------------------------
// Categorical distribution
// ---------------------------------------------------------------------------

/**
 * Categorical distribution.
 * @pytorch torch.distributions.Categorical
 */
export class Categorical extends Distribution {
  private _probs: Tensor;

  constructor(probs: Tensor) {
    super();
    this._probs = probs;
  }

  get probs(): Tensor {
    return this._probs;
  }

  async sample(sample_shape: number[] = []): Promise<Tensor> {
    const originalShape = this._probs.shape;
    const numCategories = originalShape[originalShape.length - 1];

    // Flatten all but last dimension for sampling
    const flatProbs = this._probs.reshape([-1, numCategories]);
    const numBatches = flatProbs.shape[0];
    const numSamples = sample_shape.length > 0 ? sample_shape.reduce((a, b) => a * b, 1) : 1;

    const samples: Tensor[] = [];
    for (let i = 0; i < numBatches; i++) {
      for (let j = 0; j < numSamples; j++) {
        const prob = flatProbs.select(0, i);
        const s = await prob.multinomial(1, true);
        samples.push(s);
      }
    }

    // Stack and reshape
    let result: Tensor;
    if (samples.length === 1) {
      result = samples[0];
    } else {
      result = Tensor.cat(samples.map((s) => s.reshape([1])), 0);
    }

    const finalShape = [...sample_shape, ...originalShape.slice(0, -1)];
    result = result.reshape(finalShape);

    return result;
  }

  log_prob(value: Tensor): Tensor {
    const numCategories = this._probs.shape[this._probs.shape.length - 1];
    const flatProbs = this._probs.reshape([-1, numCategories]);
    const flatValue = value.reshape([-1]);
    const batchSize = flatProbs.shape[0];

    const logP = flatProbs.log();
    const results: Tensor[] = [];

    for (let i = 0; i < batchSize; i++) {
      const logRow = logP.select(0, i);
      const idx = flatValue.select(0, i);
      // Use gather-like selection via indexing
      results.push(logRow.select(0, idx as unknown as number));
    }

    const finalShape = value.shape;
    return Tensor.cat(results.map((r) => r.reshape([1])), 0).reshape(finalShape);
  }

  entropy(): Tensor {
    const logP = this._probs.log();
    return this._probs.mul(logP).neg().sum(-1);
  }

  get mean(): Tensor {
    throw new Error('mean is not defined for Categorical distribution');
  }

  get variance(): Tensor {
    throw new Error('variance is not defined for Categorical distribution');
  }
}

// ---------------------------------------------------------------------------
// Uniform distribution
// ---------------------------------------------------------------------------

/**
 * Uniform distribution.
 * @pytorch torch.distributions.Uniform
 */
export class Uniform extends Distribution {
  private _low: Tensor;
  private _high: Tensor;

  constructor(low: Tensor, high: Tensor) {
    super();
    this._low = low;
    this._high = high;
  }

  get low(): Tensor {
    return this._low;
  }

  get high(): Tensor {
    return this._high;
  }

  sample(sample_shape: number[] = []): Tensor {
    const shape = [...sample_shape, ...this._low.shape];
    const u = rand(shape);
    return u.mul(this._high.sub(this._low)).add(this._low);
  }

  log_prob(value: Tensor): Tensor {
    const range = this._high.sub(this._low);
    const log_prob = range.log().neg();
    const below = value.ge(this._low);
    const above = value.lt(this._high);
    const mask = below.mul(above);
    const negInf = tensor(-Infinity).to('float32');
    return mask.mul(log_prob).add(mask.mul(-1).add(1).mul(negInf));
  }

  entropy(): Tensor {
    return this._high.sub(this._low).log();
  }

  get mean(): Tensor {
    return this._low.add(this._high).div(2);
  }

  get variance(): Tensor {
    const range = this._high.sub(this._low);
    return range.pow(2).div(12);
  }
}

// ---------------------------------------------------------------------------
// Exponential distribution
// ---------------------------------------------------------------------------

/**
 * Exponential distribution.
 * @pytorch torch.distributions.Exponential
 */
export class Exponential extends Distribution {
  private _rate: Tensor;

  constructor(rate: Tensor) {
    super();
    this._rate = rate;
  }

  get rate(): Tensor {
    return this._rate;
  }

  sample(sample_shape: number[] = []): Tensor {
    const shape = [...sample_shape, ...this._rate.shape];
    const u = rand(shape);
    return u.mul(-1).add(1).log().neg().div(this._rate);
  }

  log_prob(value: Tensor): Tensor {
    return this._rate.log().sub(this._rate.mul(value));
  }

  entropy(): Tensor {
    return this._rate.log().neg().add(1);
  }

  get mean(): Tensor {
    return this._rate.reciprocal();
  }

  get variance(): Tensor {
    return this._rate.pow(2).reciprocal();
  }
}

// ---------------------------------------------------------------------------
// Poisson distribution
// ---------------------------------------------------------------------------

/**
 * Poisson distribution.
 * @pytorch torch.distributions.Poisson
 */
export class Poisson extends Distribution {
  private _rate: Tensor;

  constructor(rate: Tensor) {
    super();
    this._rate = rate;
  }

  get rate(): Tensor {
    return this._rate;
  }

  sample(sample_shape: number[] = []): Tensor {
    const shape = [...sample_shape, ...this._rate.shape];
    const eps = randn(shape);
    const approx = this._rate.add(this._rate.sqrt().mul(eps));
    return approx.floor().clamp(0);
  }

  log_prob(value: Tensor): Tensor {
    return value.mul(this._rate.log()).sub(this._rate).sub(value.add(1).lgamma());
  }

  entropy(): Tensor {
    return this._rate.log().mul(0.5).add(0.5 * (LOG_2PI + 1));
  }

  get mean(): Tensor {
    return this._rate;
  }

  get variance(): Tensor {
    return this._rate;
  }
}

// ---------------------------------------------------------------------------
// Gamma distribution
// ---------------------------------------------------------------------------

/**
 * Gamma distribution.
 * @pytorch torch.distributions.Gamma
 */
export class Gamma extends Distribution {
  private _concentration: Tensor;
  private _rate: Tensor;

  constructor(concentration: Tensor, rate: Tensor) {
    super();
    this._concentration = concentration;
    this._rate = rate;
  }

  get concentration(): Tensor {
    return this._concentration;
  }

  get rate(): Tensor {
    return this._rate;
  }

  sample(sample_shape: number[] = []): Tensor {
    const shape = [...sample_shape, ...this._concentration.shape];
    const conc = this._concentration;
    const mean = conc.div(this._rate);
    const std = conc.sqrt().div(this._rate);
    const sample = randn(shape).mul(std).add(mean);
    return sample.clamp(1e-10);
  }

  log_prob(value: Tensor): Tensor {
    const a = this._concentration;
    const b = this._rate;
    return a
      .mul(b.log())
      .sub(a.lgamma())
      .add(a.sub(1).mul(value.log()))
      .sub(b.mul(value));
  }

  entropy(): Tensor {
    const a = this._concentration;
    return a
      .sub(this._rate.log())
      .add(a.lgamma())
      .add(a.mul(-1).add(1).mul(a.digamma()));
  }

  get mean(): Tensor {
    return this._concentration.div(this._rate);
  }

  get variance(): Tensor {
    return this._concentration.div(this._rate.pow(2));
  }
}

// ---------------------------------------------------------------------------
// Beta distribution
// ---------------------------------------------------------------------------

/**
 * Beta distribution.
 * @pytorch torch.distributions.Beta
 */
export class Beta extends Distribution {
  private _concentration1: Tensor;
  private _concentration0: Tensor;

  constructor(concentration1: Tensor, concentration0: Tensor) {
    super();
    this._concentration1 = concentration1;
    this._concentration0 = concentration0;
  }

  get concentration1(): Tensor {
    return this._concentration1;
  }

  get concentration0(): Tensor {
    return this._concentration0;
  }

  sample(sample_shape: number[] = []): Tensor {
    const onesTensor = tensor(1).to('float32');
    const gamma1 = new Gamma(this._concentration1, onesTensor);
    const gamma0 = new Gamma(this._concentration0, onesTensor);

    const x = gamma1.sample(sample_shape);
    const y = gamma0.sample(sample_shape);

    return x.div(x.add(y));
  }

  log_prob(value: Tensor): Tensor {
    const a = this._concentration1;
    const b = this._concentration0;
    return a
      .sub(1)
      .mul(value.log())
      .add(b.sub(1).mul(value.mul(-1).add(1).log()))
      .sub(a.lgamma())
      .sub(b.lgamma())
      .add(a.add(b).lgamma());
  }

  entropy(): Tensor {
    const a = this._concentration1;
    const b = this._concentration0;
    const lgab = a.add(b).lgamma();
    return lgab
      .sub(a.lgamma())
      .sub(b.lgamma())
      .sub(a.sub(1).mul(a.digamma()))
      .sub(b.sub(1).mul(b.digamma()))
      .add(a.add(b).sub(2).mul(a.add(b).digamma()));
  }

  get mean(): Tensor {
    const a = this._concentration1;
    const b = this._concentration0;
    return a.div(a.add(b));
  }

  get variance(): Tensor {
    const a = this._concentration1;
    const b = this._concentration0;
    const ab = a.add(b);
    return a.mul(b).div(ab.pow(2).mul(ab.add(1)));
  }
}

// ---------------------------------------------------------------------------
// Dirichlet distribution
// ---------------------------------------------------------------------------

/**
 * Dirichlet distribution.
 * @pytorch torch.distributions.Dirichlet
 */
export class Dirichlet extends Distribution {
  private _concentration: Tensor;

  constructor(concentration: Tensor) {
    super();
    this._concentration = concentration;
  }

  get concentration(): Tensor {
    return this._concentration;
  }

  sample(sample_shape: number[] = []): Tensor {
    const numCategories = this._concentration.shape[this._concentration.shape.length - 1];
    const concExpanded = this._concentration.expand([...sample_shape, ...this._concentration.shape]);
    const rateExpanded = tensor(1).to('float32').expand([...sample_shape, ...this._concentration.shape]);

    const gamma = new Gamma(concExpanded, rateExpanded);
    const samples = gamma.sample([]);

    const total = samples.sum(-1, true);
    return samples.div(total);
  }

  log_prob(value: Tensor): Tensor {
    const a = this._concentration;
    const sumConcentration = a.sum(-1);
    const logValue = value.log();

    return a
      .sub(1)
      .mul(logValue)
      .sum(-1)
      .sub(a.lgamma().sum(-1))
      .add(sumConcentration.lgamma());
  }

  entropy(): Tensor {
    const a = this._concentration;
    const sumA = a.sum(-1);
    const lgSumA = sumA.lgamma();
    const lgA = a.lgamma().sum(-1);

    return lgA
      .sub(lgSumA)
      .sub(a.sub(1).mul(a.digamma()).sum(-1))
      .add(sumA.sub(a.shape[a.shape.length - 1]).mul(sumA.digamma()));
  }

  get mean(): Tensor {
    return this._concentration.div(this._concentration.sum(-1, true));
  }

  get variance(): Tensor {
    const a = this._concentration;
    const sumA = a.sum(-1, true);
    const num = a.mul(sumA.sub(a));
    return num.div(sumA.pow(2).mul(sumA.add(1)));
  }
}

// ---------------------------------------------------------------------------
// Namespace and default export
// ---------------------------------------------------------------------------

export const distributions = {
  Distribution,
  Normal,
  Bernoulli,
  Categorical,
  Uniform,
  Exponential,
  Poisson,
  Beta,
  Gamma,
  Dirichlet,
};

export default distributions;
