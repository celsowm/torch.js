import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import { save, load } from '../../../src/serialization/browser';

const torch = createTorch(save, load);

const RTOL = 1e-4;
const ATOL = 1e-6;

function expectClose(actual: number, expected: number, msg?: string) {
  expect(actual).toBeGreaterThanOrEqual(expected * (1 - RTOL) - ATOL);
  expect(actual).toBeLessThanOrEqual(expected * (1 + RTOL) + ATOL);
}

describe('torch.special', () => {
  beforeAll(async () => {
    await torch.init();
  });

  // ---------------------------------------------------------------------------
  // erfinv
  // ---------------------------------------------------------------------------
  describe('erfinv', () => {
    it('erfinv is inverse of erf: erfinv(erf(x)) â‰ˆ x', async () => {
      const x = torch.tensor([0.0, 0.5, -0.5]);
      const erfX = x.erf();
      const erfinvResult = await torch.special.erfinv(erfX);
      const resultArr = Array.from(await erfinvResult.toArray());
      const inputArr = Array.from(await x.toArray());

      for (let i = 0; i < resultArr.length; i++) {
        expect(resultArr[i]).toBeCloseTo(inputArr[i], 2);
      }
    });

    it('erfinv(0) = 0', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.erfinv(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 4);
    });

    it('erfinv handles positive and negative values', async () => {
      const x = torch.tensor([0.5, -0.3]);
      const result = await torch.special.erfinv(x);
      const resultArr = Array.from(await result.toArray());
      // erfinv(0.5) > 0, erfinv(-0.3) < 0
      expect(resultArr[0]).toBeGreaterThan(0);
      expect(resultArr[1]).toBeLessThan(0);
    });
  });

  // ---------------------------------------------------------------------------
  // logit
  // ---------------------------------------------------------------------------
  describe('logit', () => {
    it('logit(0.5) = 0', async () => {
      const x = torch.tensor([0.5]);
      const result = torch.special.logit(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 5);
    });

    it('logit(p) = log(p / (1 - p))', async () => {
      const x = torch.tensor([0.25]);
      const result = torch.special.logit(x);
      const val = (Array.from(await result.toArray()))[0];
      const expected = Math.log(0.25 / 0.75);
      expect(val).toBeCloseTo(expected, 5);
    });

    it('logit with eps clips values', () => {
      const x = torch.tensor([0.0, 1.0]);
      const result = torch.special.logit(x, 0.01);
      const resultArr = Array.from(result.toArray());
      resultArr.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    });

    it('logit of sigmoid returns original', () => {
      const x = torch.tensor([-2.0, 0.0, 2.0]);
      const sigmoidX = x.sigmoid();
      const logitResult = torch.special.logit(sigmoidX);
      const resultArr = Array.from(logitResult.toArray());
      const inputArr = Array.from(x.toArray());
      for (let i = 0; i < resultArr.length; i++) {
        expect(resultArr[i]).toBeCloseTo(inputArr[i], 3);
      }
    });
  });

  // ---------------------------------------------------------------------------
  // sinc
  // ---------------------------------------------------------------------------
  describe('sinc', () => {
    it('sinc(0) = 1', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.sinc(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(1.0, 4);
    });

    it('sinc(1) = 0 (sin(pi) = 0)', async () => {
      const x = torch.tensor([1.0]);
      const result = await torch.special.sinc(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0.0, 4);
    });

    it('sinc(-1) = 0', async () => {
      const x = torch.tensor([-1.0]);
      const result = await torch.special.sinc(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0.0, 4);
    });

    it('sinc(0.5) = sin(pi/2) / (pi/2) = 2/pi', async () => {
      const x = torch.tensor([0.5]);
      const result = await torch.special.sinc(x);
      const val = (Array.from(await result.toArray()))[0];
      const expected = 2.0 / Math.PI;
      expect(val).toBeCloseTo(expected, 3);
    });
  });

  // ---------------------------------------------------------------------------
  // entr
  // ---------------------------------------------------------------------------
  describe('entr', () => {
    it('entr(x) = -x*log(x) for x > 0', async () => {
      const x = torch.tensor([0.5]);
      const result = await torch.special.entr(x);
      const val = (Array.from(await result.toArray()))[0];
      const expected = -0.5 * Math.log(0.5);
      expect(val).toBeCloseTo(expected, 4);
    });

    it('entr(1) = 0', async () => {
      const x = torch.tensor([1.0]);
      const result = await torch.special.entr(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 4);
    });

    it('entr(0) = 0', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.entr(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 4);
    });

    it('entr returns positive for 0 < x < 1', async () => {
      const x = torch.tensor([0.1, 0.3, 0.7, 0.9]);
      const result = await torch.special.entr(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => expect(v).toBeGreaterThan(0));
    });

    it('entr is -inf for negative x', async () => {
      const x = torch.tensor([-1.0]);
      const result = await torch.special.entr(x);
      const val = (Array.from(await result.toArray()))[0];
      expect(val).toBe(-Infinity);
    });
  });

  // ---------------------------------------------------------------------------
  // i1, i1e
  // ---------------------------------------------------------------------------
  describe('i1', () => {
    it('i1(0) â‰ˆ 0', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.i1(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 3);
    });

    it('i1 returns positive for positive input', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0]);
      const result = await torch.special.i1(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => expect(v).toBeGreaterThan(0));
    });

    it('i1 is odd function: i1(-x) = -i1(x)', async () => {
      const x = torch.tensor([1.0]);
      const resultPos = await torch.special.i1(x);
      const resultNeg = await torch.special.i1(x.neg());
      const valPos = (Array.from(await resultPos.toArray()))[0];
      const valNeg = (Array.from(await resultNeg.toArray()))[0];
      expect(valNeg).toBeCloseTo(-valPos, 3);
    });
  });

  describe('i1e', () => {
    it('i1e(0) â‰ˆ 0', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.i1e(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 3);
    });

    it('i1e returns positive values', async () => {
      const x = torch.tensor([0.5, 1.0, 2.0]);
      const result = await torch.special.i1e(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => expect(v).toBeGreaterThan(0));
    });

    it('i1e is exponentially scaled i1', async () => {
      const x = torch.tensor([1.0]);
      const i1Result = await torch.special.i1(x);
      const i1eResult = await torch.special.i1e(x);
      const i1Val = (Array.from(await i1Result.toArray()))[0];
      const i1eVal = (Array.from(await i1eResult.toArray()))[0];
      // i1e(x) â‰ˆ i1(x) * exp(-|x|)
      const expected = i1Val * Math.exp(-1.0);
      expect(i1eVal).toBeCloseTo(expected, 2);
    });
  });

  // ---------------------------------------------------------------------------
  // xlogy
  // ---------------------------------------------------------------------------
  describe('xlogy', () => {
    it('xlogy(x, y) = x * log(y)', async () => {
      const x = torch.tensor([2.0]);
      const y = torch.tensor([10.0]);
      const result = await torch.special.xlogy(x, y);
      const val = (Array.from(await result.toArray()))[0];
      expect(val).toBeCloseTo(2.0 * Math.log(10.0), 5);
    });

    it('xlogy(0, y) should handle 0 case', async () => {
      const x = torch.tensor([0.0]);
      const y = torch.tensor([5.0]);
      const result = await torch.special.xlogy(x, y);
      const val = (Array.from(await result.toArray()))[0];
      expect(val).toBeCloseTo(0, 5);
    });

    it('xlogy with scalar other', async () => {
      const x = torch.tensor([3.0]);
      const result = await torch.special.xlogy(x, 2.0);
      const val = (Array.from(await result.toArray()))[0];
      expect(val).toBeCloseTo(3.0 * Math.log(2.0), 5);
    });
  });

  // ---------------------------------------------------------------------------
  // xlog1py
  // ---------------------------------------------------------------------------
  describe('xlog1py', () => {
    it('xlog1py(x, y) = x * log1p(y)', async () => {
      const x = torch.tensor([2.0]);
      const y = torch.tensor([1.0]);
      const result = await torch.special.xlog1py(x, y);
      const val = (Array.from(await result.toArray()))[0];
      expect(val).toBeCloseTo(2.0 * Math.log1p(1.0), 5);
    });

    it('xlog1py(0, y) = 0', async () => {
      const x = torch.tensor([0.0]);
      const y = torch.tensor([5.0]);
      const result = await torch.special.xlog1py(x, y);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 5);
    });

    it('xlog1py with scalar other', async () => {
      const x = torch.tensor([3.0]);
      const result = await torch.special.xlog1py(x, 0.5);
      const val = (Array.from(await result.toArray()))[0];
      expect(val).toBeCloseTo(3.0 * Math.log1p(0.5), 5);
    });
  });

  // ---------------------------------------------------------------------------
  // erfcx
  // ---------------------------------------------------------------------------
  describe('erfcx', () => {
    it('erfcx(0) = erfc(0) = 1', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.erfcx(x);
      // erfc(0) = 1, exp(0) = 1, so erfcx(0) = 1
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(1.0, 3);
    });

    it('erfcx returns positive for positive input', async () => {
      const x = torch.tensor([0.5, 1.0, 2.0]);
      const result = await torch.special.erfcx(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => expect(v).toBeGreaterThan(0));
    });

    it('erfcx(x) = exp(x^2) * erfc(x)', async () => {
      const x = torch.tensor([0.5]);
      const result = await torch.special.erfcx(x);
      const manual = (await x.square()).exp().mul(await x.erfc());
      const valResult = (Array.from(await result.toArray()))[0];
      const valManual = (Array.from(await manual.toArray()))[0];
      expect(valResult).toBeCloseTo(valManual, 5);
    });
  });

  // ---------------------------------------------------------------------------
  // expit (sigmoid)
  // ---------------------------------------------------------------------------
  describe('expit', () => {
    it('expit(0) = 0.5', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.expit(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0.5, 5);
    });

    it('expit(x) = 1/(1+exp(-x))', async () => {
      const x = torch.tensor([1.0]);
      const result = await torch.special.expit(x);
      const val = (Array.from(await result.toArray()))[0];
      const expected = 1.0 / (1.0 + Math.exp(-1.0));
      expect(val).toBeCloseTo(expected, 5);
    });

    it('expit(-x) = 1 - expit(x)', async () => {
      const x = torch.tensor([2.0]);
      const resultPos = await torch.special.expit(x);
      const resultNeg = await torch.special.expit(x.neg());
      const valPos = (Array.from(await resultPos.toArray()))[0];
      const valNeg = (Array.from(await resultNeg.toArray()))[0];
      expect(valPos + valNeg).toBeCloseTo(1.0, 4);
    });

    it('expit returns values in (0, 1)', async () => {
      const x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0]);
      const result = await torch.special.expit(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => {
        expect(v).toBeGreaterThan(0);
        expect(v).toBeLessThan(1);
      });
    });
  });

  // ---------------------------------------------------------------------------
  // ndtr (normal CDF)
  // ---------------------------------------------------------------------------
  describe('ndtr', () => {
    it('ndtr(0) = 0.5', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.ndtr(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0.5, 4);
    });

    it('ndtr returns values in (0, 1)', async () => {
      const x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]);
      const result = await torch.special.ndtr(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => {
        expect(v).toBeGreaterThan(0);
        expect(v).toBeLessThan(1);
      });
    });

    it('ndtr(x) + ndtr(-x) = 1 (symmetry)', async () => {
      const x = torch.tensor([1.5]);
      const resultPos = await torch.special.ndtr(x);
      const resultNeg = await torch.special.ndtr(x.neg());
      const sum = (Array.from(await resultPos.toArray()))[0] + (Array.from(await resultNeg.toArray()))[0];
      expect(sum).toBeCloseTo(1.0, 4);
    });
  });

  // ---------------------------------------------------------------------------
  // log_ndtr
  // ---------------------------------------------------------------------------
  describe('log_ndtr', () => {
    it('log_ndtr(0) = log(0.5) â‰ˆ -0.6931', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.log_ndtr(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(-0.6931, 3);
    });

    it('log_ndtr returns negative values', async () => {
      const x = torch.tensor([-1.0, 0.0, 1.0]);
      const result = await torch.special.log_ndtr(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => expect(v).toBeLessThan(0));
    });

    it('log_ndtr(x) = log(ndtr(x))', async () => {
      const x = torch.tensor([0.5]);
      const logNdtr = await torch.special.log_ndtr(x);
      const ndtr = await torch.special.ndtr(x);
      const logOfNdtr = ndtr.log();
      const val1 = (Array.from(await logNdtr.toArray()))[0];
      const val2 = (Array.from(await logOfNdtr.toArray()))[0];
      expect(val1).toBeCloseTo(val2, 3);
    });
  });

  // ---------------------------------------------------------------------------
  // bessel_j0
  // ---------------------------------------------------------------------------
  describe('bessel_j0', () => {
    it('bessel_j0(0) â‰ˆ 1', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.bessel_j0(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(1.0, 2);
    });

    it('bessel_j0 returns values in reasonable range', async () => {
      const x = torch.tensor([1.0, 2.0, 5.0]);
      const result = await torch.special.bessel_j0(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => {
        expect(v).toBeGreaterThan(-1.5);
        expect(v).toBeLessThan(1.5);
      });
    });

    it('bessel_j0 is even: J0(-x) â‰ˆ J0(x)', async () => {
      const x = torch.tensor([2.0]);
      const resultPos = await torch.special.bessel_j0(x);
      const resultNeg = await torch.special.bessel_j0(x.neg());
      const valPos = (Array.from(await resultPos.toArray()))[0];
      const valNeg = (Array.from(await resultNeg.toArray()))[0];
      expect(valNeg).toBeCloseTo(valPos, 2);
    });
  });

  // ---------------------------------------------------------------------------
  // bessel_j1
  // ---------------------------------------------------------------------------
  describe('bessel_j1', () => {
    it('bessel_j1(0) â‰ˆ 0', async () => {
      const x = torch.tensor([0.0]);
      const result = await torch.special.bessel_j1(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 2);
    });

    it('bessel_j1 is odd: J1(-x) â‰ˆ -J1(x)', async () => {
      const x = torch.tensor([1.5]);
      const resultPos = await torch.special.bessel_j1(x);
      const resultNeg = await torch.special.bessel_j1(x.neg());
      const valPos = (Array.from(await resultPos.toArray()))[0];
      const valNeg = (Array.from(await resultNeg.toArray()))[0];
      expect(valNeg).toBeCloseTo(-valPos, 2);
    });

    it('bessel_j1 returns finite values', async () => {
      const x = torch.tensor([0.1, 1.0, 3.0, 5.0]);
      const result = await torch.special.bessel_j1(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    });
  });

  // ---------------------------------------------------------------------------
  // spherical_bessel_j0
  // ---------------------------------------------------------------------------
  describe('spherical_bessel_j0', () => {
    it('spherical_bessel_j0(x) = sin(x) / x', async () => {
      const x = torch.tensor([1.0]);
      const result = await torch.special.spherical_bessel_j0(x);
      const val = (Array.from(await result.toArray()))[0];
      const expected = Math.sin(1.0) / 1.0;
      expect(val).toBeCloseTo(expected, 5);
    });

    it('spherical_bessel_j0(pi) â‰ˆ 0', async () => {
      const x = torch.tensor([Math.PI]);
      const result = await torch.special.spherical_bessel_j0(x);
      expect((Array.from(await result.toArray()))[0]).toBeCloseTo(0, 4);
    });

    it('spherical_bessel_j0 returns finite values for positive x', async () => {
      const x = torch.tensor([0.1, 0.5, 1.0, 2.0]);
      const result = await torch.special.spherical_bessel_j0(x);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    });
  });

  // ---------------------------------------------------------------------------
  // zeta (Riemann zeta)
  // ---------------------------------------------------------------------------
  describe('zeta', () => {
    it('zeta(s, 1) converges for s > 1', async () => {
      const s = torch.tensor([2.0]);
      const result = await torch.special.zeta(s, 1);
      const val = (Array.from(await result.toArray()))[0];
      // zeta(2) = pi^2/6 â‰ˆ 1.6449
      expect(val).toBeCloseTo(1.6449, 2);
    });

    it('zeta returns finite for valid inputs', async () => {
      const s = torch.tensor([1.5, 2.0, 3.0]);
      const result = await torch.special.zeta(s, 1);
      const resultArr = Array.from(await result.toArray());
      resultArr.forEach((v) => expect(Number.isFinite(v)).toBe(true));
    });

    it('zeta(s) decreases as s increases', async () => {
      const s2 = torch.tensor([2.0]);
      const s3 = torch.tensor([3.0]);
      const result2 = await torch.special.zeta(s2, 1);
      const result3 = await torch.special.zeta(s3, 1);
      const val2 = (Array.from(await result2.toArray()))[0];
      const val3 = (Array.from(await result3.toArray()))[0];
      expect(val2).toBeGreaterThan(val3);
    });
  });
});
