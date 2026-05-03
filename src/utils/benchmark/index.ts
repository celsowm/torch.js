export * from './timer';
export * from './fuzzer';
export * from './compare';

/**
 * An alias for a fuzzed parameter.
 */
export class ParameterAlias {
  constructor(public readonly name: string, public readonly alias: string) {}
}

/**
 * Operation-specific fuzzers for standard benchmarks.
 */
export const op_fuzzers = {
  // Add common op fuzzers here as needed
  unary: (name: string) => {
    // Returns a function that generates a fuzzer for a unary op
  }
};
