import { syncDevice } from '../backend';
import { bufferPool } from '../backend/webgpu/buffer';

export enum ProfilerActivity {
  CPU = 'cpu',
  CUDA = 'cuda', // Alias for compatibility
  WEBGPU = 'webgpu',
}

export interface ProfilerOptions {
  activities?: ProfilerActivity[];
  record_shapes?: boolean;
  profile_memory?: boolean;
  with_stack?: boolean;
}

export interface ProfilerEvent {
  name: string;
  activity: ProfilerActivity;
  start: number;
  end?: number;
  duration?: number;
  shapes?: number[][];
  memory_usage?: number;
}

/**
 * Collection of averages grouped by name/shape.
 */
export class KinetoStep {
  constructor(private events: ProfilerEvent[]) {}

  key_averages(group_by_input_shape: boolean = false): KinetoStep {
    return this;
  }

  table(options: { sort_by?: string; row_limit?: number } = {}): string {
    const { sort_by = 'duration', row_limit = 50 } = options;
    
    const stats = new Map<string, { count: number; total_duration: number; total_memory: number }>();
    
    for (const e of this.events) {
      const key = e.name;
      const entry = stats.get(key) || { count: 0, total_duration: 0, total_memory: 0 };
      entry.count++;
      entry.total_duration += e.duration || 0;
      entry.total_memory += e.memory_usage || 0;
      stats.set(key, entry);
    }

    const sorted = Array.from(stats.entries())
      .sort((a, b) => b[1].total_duration - a[1].total_duration)
      .slice(0, row_limit);

    const lines: string[] = ['\n-------------------------------------------------------'];
    lines.push(`${'Name'.padEnd(30)} | ${'Count'.padStart(8)} | ${'Total Time'.padStart(12)} | ${'CPU Mem'.padStart(10)}`);
    lines.push('-------------------------------------------------------');
    
    for (const [name, data] of sorted) {
      lines.push(
        `${name.padEnd(30)} | ` +
        `${data.count.toString().padStart(8)} | ` +
        `${(data.total_duration).toFixed(3).padStart(9)} ms | ` +
        `${(data.total_memory / 1024).toFixed(0).padStart(7)} KB`
      );
    }
    lines.push('------------------------------------------------------- \n');
    return lines.join('\n');
  }
}

/**
 * Global profiler state to allow record_function to work without explicit passing.
 */
let activeProfiler: Profiler | null = null;

/**
 * Profiler for tracking operation execution.
 */
export class Profiler {
  private events: ProfilerEvent[] = [];
  private active: boolean = false;
  private options: ProfilerOptions;

  constructor(options: ProfilerOptions = {}) {
    this.options = {
      activities: [ProfilerActivity.CPU, ProfilerActivity.WEBGPU],
      record_shapes: false,
      profile_memory: false,
      ...options
    };
  }

  start(): void {
    this.events = [];
    this.active = true;
    activeProfiler = this;
  }

  async stop(): Promise<void> {
    await syncDevice();
    this.active = false;
    activeProfiler = null;
  }

  _record_event(event: ProfilerEvent) {
    if (this.active) {
      this.events.push(event);
    }
  }

  get_events(): ProfilerEvent[] {
    return this.events;
  }

  key_averages(): KinetoStep {
    return new KinetoStep(this.events);
  }
}

/**
 * High-level profiling context.
 */
export async function profile(
  options: ProfilerOptions | ((p: Profiler) => Promise<void> | void),
  fn?: (p: Profiler) => Promise<void> | void
): Promise<Profiler> {
  const actualOptions = typeof options === 'function' ? {} : options;
  const actualFn = typeof options === 'function' ? options : fn;

  if (!actualFn) throw new Error("Profile function is required");

  const p = new Profiler(actualOptions);
  p.start();
  const res = actualFn(p);
  if (res instanceof Promise) await res;
  await p.stop();
  return p;
}

/**
 * Synchronous utility to record a function block.
 * For WebGPU, this measures CPU time up to the submission.
 */
export function record_function<T>(name: string, fn: () => T): T {
  if (!activeProfiler) return fn();

  const start = performance.now();
  const startMem = bufferPool.getStats().activeBytes;
  
  const result = fn();
  
  const end = performance.now();
  const endMem = bufferPool.getStats().activeBytes;

  activeProfiler._record_event({
    name,
    activity: ProfilerActivity.CPU,
    start,
    end,
    duration: end - start,
    memory_usage: endMem - startMem
  });

  return result;
}

/**
 * Asynchronous version of record_function that includes synchronization.
 */
export async function record_function_async<T>(name: string, fn: () => Promise<T>): Promise<T> {
  if (!activeProfiler) return fn();

  const start = performance.now();
  const startMem = bufferPool.getStats().activeBytes;
  
  const result = await fn();
  await syncDevice();
  
  const end = performance.now();
  const endMem = bufferPool.getStats().activeBytes;

  activeProfiler._record_event({
    name,
    activity: ProfilerActivity.WEBGPU,
    start,
    end,
    duration: end - start,
    memory_usage: endMem - startMem
  });

  return result;
}
