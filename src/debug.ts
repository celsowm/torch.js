/**
 * Debug utilities for torch.js
 *
 * Usage:
 *   DEBUG(() => {
 *     console.log('expensive debug computation', someValue);
 *     // This entire block is stripped in production
 *   });
 *
 * In production builds (NODE_ENV=production), the DEBUG function becomes
 * a no-op and esbuild eliminates the callback entirely via dead code elimination.
 *
 * @status implemented
 */

// Replaced by tsup/esbuild at build time
declare const __DEBUG__: boolean;

/**
 * Execute a debug callback. The entire callback is stripped in production builds.
 *
 * @example
 * DEBUG(() => {
 *   const data = await readBuffer(buffer, dtype, size);
 *   console.log('Buffer contents:', data);
 * });
 */
export function DEBUG(fn: () => void): void {
  if (__DEBUG__) {
    fn();
  }
}

/**
 * Async version of DEBUG for callbacks that need await.
 */
export function DEBUG_ASYNC(fn: () => Promise<void>): void {
  if (__DEBUG__) {
    fn();
  }
}

/**
 * Debug log helper - use inside DEBUG() blocks.
 */
export function debugLog(message: string, ...args: unknown[]): void {
  console.log(`[torch.js] ${message}`, ...args);
}

/**
 * Debug warn helper - use inside DEBUG() blocks.
 */
export function warn(message: string, ...args: unknown[]): void {
  console.warn(`[torch.js] ${message}`, ...args);
}

/**
 * Debug error helper - use inside DEBUG() blocks.
 */
export function error(message: string, ...args: unknown[]): void {
  console.error(`[torch.js] ${message}`, ...args);
}

/**
 * Debug assert - use inside DEBUG() blocks.
 */
export function assert(condition: boolean, message: string): void {
  if (!condition) {
    throw new Error(`[torch.js] Assertion failed: ${message}`);
  }
}
