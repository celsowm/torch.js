import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import { save, load } from '../../../src/serialization/browser';
import type { Distribution } from '../../../src/distributions';

const torch = createTorch(save, load);

const RTOL = 1e-4;
const ATOL = 1e-6;

function approxEqual(actual: number, expected: number, msg?: string) {
  expect(actual).toBeGreaterThanOrEqual(expected * (1 - RTOL) - ATOL);
  expect(actual).toBeLessThanOrEqual(expected * (1 + RTOL) + ATOL);
}

describe('torch.distributions', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ---------------------------------------------------------------------------
  // Normal (Gaussian)
  // ---------------------------------------------------------------------------
  describe('Normal', () => {
    it('constructs with loc and scale', async () => {
      const loc = torch.tensor([0.0, 1.0]);
      const scale = torch.tensor([1.0, 2.0]);
      const dist = new torch.distributions.Normal(loc, scale);
      const locOut = Array.from(await dist.loc.toArray());
      const scaleOut = Array.from(await dist.scale.toArray());
      expect(locOut).toEqual([0.0, 1.0]);
      expect(scaleOut).toEqual([1.0, 2.0]);
    });

    it('sample returns correct shape', async () => {
      const loc = torch.tensor([0.0, 1.0, 2.0]);
      const scale = torch.tensor([1.0, 1.0, 1.0]);
      const dist = new torch.distributions.Normal(loc, scale);
      const sample = dist.sample([5]);
      expect(sample.shape).toEqual([5, 3]);
    });

    it('sample returns correct shape with no sample_shape', async () => {
      const loc = torch.tensor([0.0, 1.0]);
      const scale = torch.tensor([1.0, 1.0]);
      const dist = new torch.distributions.Normal(loc, scale);
      const sample = dist.sample();
      expect(sample.shape).toEqual([2]);
    });

    it('log_prob returns finite values', async () => {
      const loc = torch.tensor([0.0]);
      const scale = torch.tensor([1.0]);
      const dist = new torch.distributions.Normal(loc, scale);
      const sample = dist.sample([100]);
      const lp = dist.log_prob(sample);
      const lpArr = Array.from(await lp.toArray());
      lpArr.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    });

    it('mean approximates loc and variance approximates scale^2', async () => {
      const loc = torch.tensor([5.0]);
      const scale = torch.tensor([2.0]);
      const dist = new torch.distributions.Normal(loc, scale);

      const samples = dist.sample([10000]);
      const sampleMean = samples.mean(0);
      const sampleVar = samples.sub(sampleMean).pow(2).mean(0);

      const meanVal = (Array.from(await sampleMean.toArray()))[0];
      const varVal = (Array.from(await sampleVar.toArray()))[0];

      expect(meanVal).toBeCloseTo(5.0, 1);
      expect(varVal).toBeCloseTo(4.0, 1);
    });

    it('entropy has correct sign for unit normal', async () => {
      const loc = torch.tensor([0.0]);
      const scale = torch.tensor([1.0]);
      const dist = new torch.distributions.Normal(loc, scale);
      const entropy = dist.entropy();
      const h = (Array.from(await entropy.toArray()))[0];
      // entropy of N(0,1) = 0.5*log(2*pi*e) â‰ˆ 1.4189
      expect(h).toBeGreaterThan(0);
      expect(h).toBeCloseTo(1.4189, 2);
    });
  });

  // ---------------------------------------------------------------------------
  // Bernoulli
  // ---------------------------------------------------------------------------
  describe('Bernoulli', () => {
    it('constructs with probs', async () => {
      const probs = torch.tensor([0.3, 0.7]);
      const dist = new torch.distributions.Bernoulli(probs);
      const probsOut = Array.from(await dist.probs.toArray());
      expect(probsOut).toEqual([0.3, 0.7]);
    });

    it('sample returns values in {0, 1}', async () => {
      const probs = torch.tensor([0.5, 0.5, 0.5]);
      const dist = new torch.distributions.Bernoulli(probs);
      const sample = dist.sample([100]);
      expect(sample.shape).toEqual([100, 3]);
      const sampleArr = Array.from(await sample.toArray());
      sampleArr.forEach((v) => {
        expect(v === 0.0 || v === 1.0).toBe(true);
      });
    });

    it('sample shape without sample_shape', async () => {
      const probs = torch.tensor([0.5]);
      const dist = new torch.distributions.Bernoulli(probs);
      const sample = dist.sample();
      expect(sample.shape).toEqual([1]);
    });

    it('log_prob returns finite values', async () => {
      const probs = torch.tensor([0.5]);
      const dist = new torch.distributions.Bernoulli(probs);
      const value = torch.tensor([1.0]);
      const lp = dist.log_prob(value);
      const lpVal = (Array.from(await lp.toArray()))[0];
      expect(Number.isFinite(lpVal)).toBe(true);
      // log(0.5) = -0.6931
      expect(lpVal).toBeCloseTo(-0.6931, 3);
    });

    it('mean equals probs', async () => {
      const probs = torch.tensor([0.3, 0.7]);
      const dist = new torch.distributions.Bernoulli(probs);
      const mean = await dist.mean.toArray();
      expect(Array.from(mean)).toEqual([0.3, 0.7]);
    });

    it('variance = p*(1-p)', async () => {
      const probs = torch.tensor([0.5]);
      const dist = new torch.distributions.Bernoulli(probs);
      const var_ = await dist.variance.toArray();
      expect(var_[0]).toBeCloseTo(0.25, 6);
    });
  });

  // ---------------------------------------------------------------------------
  // Categorical
  // ---------------------------------------------------------------------------
  describe('Categorical', () => {
    it('constructs with probs that sum to 1', async () => {
      const probs = torch.tensor([0.2, 0.3, 0.5]);
      const dist = new torch.distributions.Categorical(probs);
      const probsOut = Array.from(await dist.probs.toArray());
      expect(probsOut.reduce((a, b) => a + b, 0)).toBeCloseTo(1.0, 5);
    });

    it('sample returns valid category indices', async () => {
      const probs = torch.tensor([0.2, 0.3, 0.5]);
      const dist = new torch.distributions.Categorical(probs);
      const sample = await dist.sample([200]);
      expect(sample.shape).toEqual([200]);
      const sampleArr = Array.from(await sample.toArray());
      sampleArr.forEach((v) => {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(3);
        expect(Number.isInteger(v)).toBe(true);
      });
    });

    it('log_prob returns finite values', async () => {
      const probs = torch.tensor([0.25, 0.25, 0.25, 0.25]);
      const dist = new torch.distributions.Categorical(probs);
      const value = torch.tensor([0]);
      const lp = dist.log_prob(value);
      const lpVal = Array.from(await lp.toArray());
      lpVal.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    });

    it('entropy is positive for uniform probs', async () => {
      const probs = torch.tensor([0.25, 0.25, 0.25, 0.25]);
      const dist = new torch.distributions.Categorical(probs);
      const entropy = dist.entropy();
      const h = (Array.from(await entropy.toArray()))[0];
      expect(h).toBeGreaterThan(0);
      // entropy of uniform categorical with 4 categories â‰ˆ 1.3863
      expect(h).toBeCloseTo(1.3863, 2);
    });
  });

  // ---------------------------------------------------------------------------
  // Uniform
  // ---------------------------------------------------------------------------
  describe('Uniform', () => {
    it('constructs with low and high', async () => {
      const low = torch.tensor([0.0]);
      const high = torch.tensor([10.0]);
      const dist = new torch.distributions.Uniform(low, high);
      expect(Array.from(await dist.low.toArray())).toEqual([0.0]);
      expect(Array.from(await dist.high.toArray())).toEqual([10.0]);
    });

    it('sample returns values in [low, high]', async () => {
      const low = torch.tensor([0.0, 5.0]);
      const high = torch.tensor([1.0, 10.0]);
      const dist = new torch.distributions.Uniform(low, high);
      const sample = dist.sample([500]);
      expect(sample.shape).toEqual([500, 2]);
      const sampleArr = Array.from(await sample.toArray());
      for (let i = 0; i < sampleArr.length; i++) {
        const dim = i % 2;
        const lo = dim === 0 ? 0.0 : 5.0;
        const hi = dim === 0 ? 1.0 : 10.0;
        expect(sampleArr[i]).toBeGreaterThanOrEqual(lo - 1e-6);
        expect(sampleArr[i]).toBeLessThanOrEqual(hi + 1e-6);
      }
    });

    it('log_prob returns finite for in-range values', async () => {
      const low = torch.tensor([0.0]);
      const high = torch.tensor([1.0]);
      const dist = new torch.distributions.Uniform(low, high);
      const value = torch.tensor([0.5]);
      const lp = dist.log_prob(value);
      const lpVal = (Array.from(await lp.toArray()))[0];
      // log(1/(1-0)) = 0
      expect(lpVal).toBeCloseTo(0.0, 4);
    });

    it('entropy = log(high - low)', async () => {
      const low = torch.tensor([0.0]);
      const high = torch.tensor([1.0]);
      const dist = new torch.distributions.Uniform(low, high);
      const entropy = dist.entropy();
      const h = (Array.from(await entropy.toArray()))[0];
      expect(h).toBeCloseTo(0.0, 6); // log(1) = 0
    });

    it('mean = (low + high) / 2', async () => {
      const low = torch.tensor([0.0]);
      const high = torch.tensor([10.0]);
      const dist = new torch.distributions.Uniform(low, high);
      const mean = await dist.mean.toArray();
      expect(mean[0]).toBeCloseTo(5.0, 6);
    });

    it('variance = (high - low)^2 / 12', async () => {
      const low = torch.tensor([0.0]);
      const high = torch.tensor([1.0]);
      const dist = new torch.distributions.Uniform(low, high);
      const var_ = await dist.variance.toArray();
      // 1/12 â‰ˆ 0.08333
      expect(var_[0]).toBeCloseTo(0.08333, 3);
    });

    it('entropy is positive for non-degenerate range', async () => {
      const low = torch.tensor([0.0]);
      const high = torch.tensor([5.0]);
      const dist = new torch.distributions.Uniform(low, high);
      const entropy = dist.entropy();
      const h = (Array.from(await entropy.toArray()))[0];
      expect(h).toBeGreaterThan(0);
    });
  });

  // ---------------------------------------------------------------------------
  // Exponential
  // ---------------------------------------------------------------------------
  describe('Exponential', () => {
    it('constructs with rate', async () => {
      const rate = torch.tensor([2.0]);
      const dist = new torch.distributions.Exponential(rate);
      expect(Array.from(await dist.rate.toArray())).toEqual([2.0]);
    });

    it('sample returns values > 0', async () => {
      const rate = torch.tensor([1.0, 2.0, 3.0]);
      const dist = new torch.distributions.Exponential(rate);
      const sample = dist.sample([500]);
      expect(sample.shape).toEqual([500, 3]);
      const sampleArr = Array.from(await sample.toArray());
      sampleArr.forEach((v) => {
        expect(v).toBeGreaterThanOrEqual(0);
      });
    });

    it('log_prob returns finite values', async () => {
      const rate = torch.tensor([1.0]);
      const dist = new torch.distributions.Exponential(rate);
      const value = torch.tensor([0.5]);
      const lp = dist.log_prob(value);
      const lpVal = (Array.from(await lp.toArray()))[0];
      expect(Number.isFinite(lpVal)).toBe(true);
    });

    it('mean = 1/rate', async () => {
      const rate = torch.tensor([2.0]);
      const dist = new torch.distributions.Exponential(rate);
      const mean = await dist.mean.toArray();
      expect(mean[0]).toBeCloseTo(0.5, 6);
    });

    it('variance = 1/rate^2', async () => {
      const rate = torch.tensor([2.0]);
      const dist = new torch.distributions.Exponential(rate);
      const var_ = await dist.variance.toArray();
      expect(var_[0]).toBeCloseTo(0.25, 6);
    });

    it('entropy is positive', async () => {
      const rate = torch.tensor([1.0]);
      const dist = new torch.distributions.Exponential(rate);
      const entropy = dist.entropy();
      const h = (Array.from(await entropy.toArray()))[0];
      expect(h).toBeGreaterThan(0);
      // entropy of Exp(1) = 1
      expect(h).toBeCloseTo(1.0, 2);
    });
  });

  // ---------------------------------------------------------------------------
  // Gamma
  // ---------------------------------------------------------------------------
  describe('Gamma', () => {
    it('constructs with concentration and rate', async () => {
      const conc = torch.tensor([2.0]);
      const rate = torch.tensor([1.0]);
      const dist = new torch.distributions.Gamma(conc, rate);
      expect(Array.from(await dist.concentration.toArray())).toEqual([2.0]);
      expect(Array.from(await dist.rate.toArray())).toEqual([1.0]);
    });

    it('sample returns values > 0', async () => {
      const conc = torch.tensor([2.0, 3.0]);
      const rate = torch.tensor([1.0, 1.0]);
      const dist = new torch.distributions.Gamma(conc, rate);
      const sample = dist.sample([500]);
      expect(sample.shape).toEqual([500, 2]);
      const sampleArr = Array.from(await sample.toArray());
      sampleArr.forEach((v) => {
        expect(v).toBeGreaterThan(0);
      });
    });

    it('log_prob returns finite values', async () => {
      const conc = torch.tensor([2.0]);
      const rate = torch.tensor([1.0]);
      const dist = new torch.distributions.Gamma(conc, rate);
      const value = torch.tensor([1.5]);
      const lp = dist.log_prob(value);
      const lpVal = (Array.from(await lp.toArray()))[0];
      expect(Number.isFinite(lpVal)).toBe(true);
    });

    it('mean = concentration / rate', async () => {
      const conc = torch.tensor([3.0]);
      const rate = torch.tensor([2.0]);
      const dist = new torch.distributions.Gamma(conc, rate);
      const mean = await dist.mean.toArray();
      expect(mean[0]).toBeCloseTo(1.5, 6);
    });

    it('variance = concentration / rate^2', async () => {
      const conc = torch.tensor([3.0]);
      const rate = torch.tensor([2.0]);
      const dist = new torch.distributions.Gamma(conc, rate);
      const var_ = await dist.variance.toArray();
      expect(var_[0]).toBeCloseTo(0.75, 5);
    });
  });

  // ---------------------------------------------------------------------------
  // Beta
  // ---------------------------------------------------------------------------
  describe('Beta', () => {
    it('constructs with concentration1 and concentration0', async () => {
      const c1 = torch.tensor([2.0]);
      const c0 = torch.tensor([5.0]);
      const dist = new torch.distributions.Beta(c1, c0);
      expect(Array.from(await dist.concentration1.toArray())).toEqual([2.0]);
      expect(Array.from(await dist.concentration0.toArray())).toEqual([5.0]);
    });

    it('sample returns values in (0, 1)', async () => {
      const c1 = torch.tensor([2.0, 3.0]);
      const c0 = torch.tensor([5.0, 1.0]);
      const dist = new torch.distributions.Beta(c1, c0);
      const sample = dist.sample([500]);
      expect(sample.shape).toEqual([500, 2]);
      const sampleArr = Array.from(await sample.toArray());
      sampleArr.forEach((v) => {
        expect(v).toBeGreaterThan(0);
        expect(v).toBeLessThan(1);
      });
    });

    it('log_prob returns finite values for valid samples', async () => {
      const c1 = torch.tensor([2.0]);
      const c0 = torch.tensor([5.0]);
      const dist = new torch.distributions.Beta(c1, c0);
      const value = torch.tensor([0.5]);
      const lp = dist.log_prob(value);
      const lpVal = (Array.from(await lp.toArray()))[0];
      expect(Number.isFinite(lpVal)).toBe(true);
    });

    it('mean = c1 / (c1 + c0)', async () => {
      const c1 = torch.tensor([2.0]);
      const c0 = torch.tensor([5.0]);
      const dist = new torch.distributions.Beta(c1, c0);
      const mean = await dist.mean.toArray();
      expect(mean[0]).toBeCloseTo(2.0 / 7.0, 5);
    });

    it('variance matches formula', async () => {
      const c1 = torch.tensor([2.0]);
      const c0 = torch.tensor([5.0]);
      const dist = new torch.distributions.Beta(c1, c0);
      const var_ = await dist.variance.toArray();
      const expected = (2 * 5) / (7 * 7 * 8); // a*b / ((a+b)^2 * (a+b+1))
      expect(var_[0]).toBeCloseTo(expected, 5);
    });

    it('entropy is positive for uniform-like params', async () => {
      const c1 = torch.tensor([1.0]);
      const c0 = torch.tensor([1.0]);
      const dist = new torch.distributions.Beta(c1, c0);
      const entropy = dist.entropy();
      const h = (Array.from(await entropy.toArray()))[0];
      expect(h).toBeGreaterThan(0);
      // entropy of Beta(1,1) = 0 (uniform), actually exactly 0
    });
  });

  // ---------------------------------------------------------------------------
  // Dirichlet
  // ---------------------------------------------------------------------------
  describe('Dirichlet', () => {
    it('constructs with concentration vector', async () => {
      const conc = torch.tensor([1.0, 2.0, 3.0]);
      const dist = new torch.distributions.Dirichlet(conc);
      expect(Array.from(await dist.concentration.toArray())).toEqual([1.0, 2.0, 3.0]);
    });

    it('sample returns simplex (values sum to 1)', async () => {
      const conc = torch.tensor([1.0, 2.0, 3.0]);
      const dist = new torch.distributions.Dirichlet(conc);
      const sample = dist.sample([100]);
      expect(sample.shape).toEqual([100, 3]);
      const sampleArr = Array.from(await sample.toArray());
      for (let i = 0; i < 100; i++) {
        const rowSum = sampleArr[i * 3] + sampleArr[i * 3 + 1] + sampleArr[i * 3 + 2];
        expect(rowSum).toBeCloseTo(1.0, 3);
      }
    });

    it('sample values are positive', async () => {
      const conc = torch.tensor([1.0, 2.0, 3.0]);
      const dist = new torch.distributions.Dirichlet(conc);
      const sample = dist.sample([50]);
      const sampleArr = Array.from(await sample.toArray());
      sampleArr.forEach((v) => {
        expect(v).toBeGreaterThanOrEqual(0);
      });
    });

    it('log_prob returns finite values', async () => {
      const conc = torch.tensor([1.0, 2.0, 3.0]);
      const dist = new torch.distributions.Dirichlet(conc);
      const value = torch.tensor([0.2, 0.3, 0.5]);
      const lp = dist.log_prob(value);
      const lpVal = (Array.from(await lp.toArray()))[0];
      expect(Number.isFinite(lpVal)).toBe(true);
    });

    it('mean = concentration / sum(concentration)', async () => {
      const conc = torch.tensor([1.0, 2.0, 3.0]);
      const dist = new torch.distributions.Dirichlet(conc);
      const mean = await dist.mean.toArray();
      expect(mean[0]).toBeCloseTo(1.0 / 6.0, 5);
      expect(mean[1]).toBeCloseTo(2.0 / 6.0, 5);
      expect(mean[2]).toBeCloseTo(3.0 / 6.0, 5);
    });

    it('entropy is positive for symmetric params', async () => {
      const conc = torch.tensor([1.0, 1.0, 1.0]);
      const dist = new torch.distributions.Dirichlet(conc);
      const entropy = dist.entropy();
      const h = (Array.from(await entropy.toArray()))[0];
      expect(h).toBeGreaterThan(0);
    });
  });

  // ---------------------------------------------------------------------------
  // Poisson
  // ---------------------------------------------------------------------------
  describe('Poisson', () => {
    it('constructs with rate', async () => {
      const rate = torch.tensor([5.0]);
      const dist = new torch.distributions.Poisson(rate);
      expect(Array.from(await dist.rate.toArray())).toEqual([5.0]);
    });

    it('sample returns integer values >= 0', async () => {
      const rate = torch.tensor([3.0, 5.0, 7.0]);
      const dist = new torch.distributions.Poisson(rate);
      const sample = dist.sample([200]);
      expect(sample.shape).toEqual([200, 3]);
      const sampleArr = Array.from(await sample.toArray());
      sampleArr.forEach((v) => {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(Number.isInteger(v)).toBe(true);
      });
    });

    it('log_prob returns finite values for integer samples', async () => {
      const rate = torch.tensor([3.0]);
      const dist = new torch.distributions.Poisson(rate);
      const value = torch.tensor([2.0]);
      const lp = dist.log_prob(value);
      const lpVal = (Array.from(await lp.toArray()))[0];
      expect(Number.isFinite(lpVal)).toBe(true);
    });

    it('mean approximates rate', async () => {
      const rate = torch.tensor([5.0]);
      const dist = new torch.distributions.Poisson(rate);
      const samples = dist.sample([10000]);
      const sampleMean = samples.mean(0);
      const meanVal = (Array.from(await sampleMean.toArray()))[0];
      expect(meanVal).toBeCloseTo(5.0, 1);
    });

    it('variance approximates rate', async () => {
      const rate = torch.tensor([5.0]);
      const dist = new torch.distributions.Poisson(rate);
      const samples = dist.sample([10000]);
      const sampleMean = samples.mean(0);
      const sampleVar = samples.sub(sampleMean).pow(2).mean(0);
      const varVal = (Array.from(await sampleVar.toArray()))[0];
      expect(varVal).toBeCloseTo(5.0, 1);
    });

    it('entropy is positive', async () => {
      const rate = torch.tensor([5.0]);
      const dist = new torch.distributions.Poisson(rate);
      const entropy = dist.entropy();
      const h = (Array.from(await entropy.toArray()))[0];
      expect(h).toBeGreaterThan(0);
    });
  });
});
