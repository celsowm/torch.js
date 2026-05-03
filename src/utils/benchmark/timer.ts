import { syncDevice } from '../../backend';

/**
 * Result of a benchmark measurement.
 */
export class Measurement {
  constructor(
    public readonly name: string,
    public readonly times: number[],
    public readonly task_spec: string = '',
    public readonly label: string = '',
    public readonly sub_label: string = '',
    public readonly description: string = '',
    public readonly env: string = ''
  ) {}

  get mean(): number {
    return this.times.reduce((a, b) => a + b, 0) / this.times.length;
  }

  get median(): number {
    const sorted = [...this.times].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  get iqr(): number {
    if (this.times.length < 4) return 0;
    const sorted = [...this.times].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    return q3 - q1;
  }

  toString(): string {
    const median = (this.median * 1000 * 1000).toFixed(2); // Convert seconds to microseconds
    return `${this.label}${this.sub_label ? ': ' + this.sub_label : ''}  ${median} us (${this.times.length} iterations)`;
  }
}

/**
 * Timer class for benchmarking code snippets.
 * Generic T represents the return type of the statement being timed.
 */
export class Timer<T = any> {
  constructor(
    private readonly stmt: () => T | Promise<T>,
    private readonly setup: () => void | Promise<void> = () => {},
    public readonly label: string = '',
    public readonly sub_label: string = '',
    public readonly description: string = '',
    public readonly env: string = ''
  ) {}

  /**
   * Run the statement multiple times and return a Measurement.
   */
  async timeit(number: number = 100): Promise<Measurement> {
    await this.setup();
    await syncDevice();

    const times: number[] = [];
    for (let i = 0; i < number; i++) {
      const start = performance.now();
      const res = this.stmt();
      if (res instanceof Promise) await res;
      await syncDevice();
      const end = performance.now();
      times.push((end - start) / 1000); // Store in seconds to match PyTorch conventions
    }

    return new Measurement(
      '',
      times,
      '',
      this.label,
      this.sub_label,
      this.description,
      this.env
    );
  }

  /**
   * Automatically determine number of iterations to get stable results.
   */
  async blocked_autorange(min_run_time: number = 0.2): Promise<Measurement> {
    await this.setup();
    await syncDevice();

    let number = 1;
    let totalTime = 0;

    // Phase 1: Determine 'number'
    while (totalTime < 0.01) { 
      const start = performance.now();
      for (let i = 0; i < number; i++) {
        const res = this.stmt();
        if (res instanceof Promise) await res;
      }
      await syncDevice();
      totalTime = (performance.now() - start) / 1000;
      if (totalTime < 0.01) number *= 10;
    }

    // Phase 2: Run until min_run_time is met
    const actual_number = Math.max(number, Math.ceil(min_run_time / (totalTime / number)));
    return this.timeit(actual_number);
  }
}