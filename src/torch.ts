/**
 * Default torch object for browser-like environments or general imports.
 * Uses browser-specific serialization.
 */
import { createTorch } from './index';
import { save, load } from './serialization/browser';

export const torch = createTorch(save, load);
export * from './index';

export default torch;
