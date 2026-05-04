/**
 * Global setup for compatibility tests.
 * Runs before each test file to initialize WebGPU.
 */

export async function setup() {
  const { createTorch } = await import('../../src/create_torch');
  const torch = createTorch();
  await torch.init();
}
