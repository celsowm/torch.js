import { Measurement } from './timer';

/**
 * Utility for comparing multiple benchmark measurements.
 */
export class Compare {
  constructor(private readonly measurements: Measurement[]) {}

  /**
   * Print a formatted comparison table.
   */
  print(): void {
    console.log('\n[Benchmark Comparison]');
    console.log('--------------------------------------------------------------------------------');
    console.log(`${'Label'.padEnd(20)} | ${'Sub-label'.padEnd(20)} | ${'Median (us)'.padStart(12)} | ${'IQR'.padStart(10)}`);
    console.log('--------------------------------------------------------------------------------');
    
    for (const m of this.measurements) {
      const median = (m.median * 1000).toFixed(2);
      const iqr = (m.iqr * 1000).toFixed(2);
      console.log(
        `${m.label.padEnd(20)} | ` +
        `${m.sub_label.padEnd(20)} | ` +
        `${median.padStart(12)} | ` +
        `${iqr.padStart(10)}`
      );
    }
    console.log('--------------------------------------------------------------------------------\n');
  }
}
