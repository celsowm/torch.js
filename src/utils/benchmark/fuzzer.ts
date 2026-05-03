import { Tensor } from '../../tensor';
import { randn } from '../../ops/creation';

/**
 * A parameter that can be fuzzed for benchmarks.
 */
export class FuzzedParameter {
  constructor(
    public readonly name: string,
    public readonly values: any[],
    public readonly distribution?: 'uniform' | 'loguniform'
  ) {}
}

/**
 * A tensor with randomized shape and values for fuzzing.
 */
export class FuzzedTensor {
  constructor(
    public readonly name: string,
    public readonly min_shape: number[],
    public readonly max_shape: number[],
    public readonly probability_contiguous: number = 1.0,
    public readonly dtype: string = 'float32'
  ) {}

  generate(): Tensor {
    const shape = this.min_shape.map((min, i) => {
      const max = this.max_shape[i];
      return Math.floor(Math.random() * (max - min + 1)) + min;
    });
    
    // For now, always returns normal distribution
    return randn(shape);
  }
}

/**
 * Generator for randomized benchmark inputs.
 */
export class Fuzzer {
  constructor(
    private readonly parameters: (FuzzedParameter | FuzzedTensor)[],
    private readonly seed?: number
  ) {
    if (seed !== undefined) {
      // TODO: Implement seeded random generator if needed
    }
  }

  /**
   * Returns a generator that yields randomized inputs.
   */
  *take(n: number): Generator<Record<string, any>> {
    for (let i = 0; i < n; i++) {
      const result: Record<string, any> = {};
      for (const p of this.parameters) {
        if (p instanceof FuzzedParameter) {
          result[p.name] = p.values[Math.floor(Math.random() * p.values.length)];
        } else if (p instanceof FuzzedTensor) {
          result[p.name] = p.generate();
        }
      }
      yield result;
    }
  }
}
