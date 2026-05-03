/**
 * PyTorch-compatible WebGPU-style memory management.
 * Since we use WebGPU, this maps to GPU memory stats.
 */

import { bufferPool } from './backend/webgpu/buffer';

/**
 * Returns a dictionary of WebGPU memory allocator statistics.
 */
export function memory_stats() {
  const stats = bufferPool.getStats();
  return {
    active_bytes: stats.activeBytes,
    pooled_bytes: stats.pooledBytes,
    peak_bytes: stats.peakBytes,
    total_allocations: stats.allocationCount,
  };
}

/**
 * Returns a human-readable summary of the memory allocator statistics.
 */
export function memory_summary() {
  const stats = memory_stats();
  const toMB = (bytes: number) => (bytes / 1024 / 1024).toFixed(2);
  
  return `
GPU Memory Summary
-------------------------------------------------------
Active Memory:   ${toMB(stats.active_bytes)} MB
Pooled Memory:   ${toMB(stats.pooled_bytes)} MB
Peak Memory:     ${toMB(stats.peak_bytes)} MB
Total Allocs:    ${stats.total_allocations}
-------------------------------------------------------
`;
}

/**
 * Releases all unoccupied cached memory.
 */
export function empty_cache() {
  bufferPool.clear();
}

/**
 * Resets the peak memory stats.
 */
export function reset_peak_memory_stats() {
  bufferPool.resetPeak();
}
