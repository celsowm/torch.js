/**
 * GPU capability detection for runtime feature selection.
 * Tests actual GPU behavior since browser/driver bugs vary by platform.
 */

import { getDevice, getAdapter } from './device.js';
import { BufferUsage, MapMode, WebGPUDevice } from './types.js';
import { REDUCE_SUM_SHADER } from './shaders/index.js';

export interface GPULimits {
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupsPerDimension: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

export interface GPUCapabilities {
  /** Whether workgroup shared memory works correctly */
  workgroupSharedMemory: boolean;
  /** Whether high-precision timestamp queries are supported */
  timestampQuery: boolean;
  /** Whether subgroups are supported (future) */
  subgroups: boolean;
  /** GPU limits from adapter */
  limits: GPULimits;
  /** Detected platform info for debugging */
  platform: {
    browser: string;
    gpu: string;
  };
}

let capabilities: GPUCapabilities | null = null;

/**
 * Detect GPU capabilities by running actual tests.
 * Called automatically during torch.init().
 */
export async function detectCapabilities(): Promise<GPUCapabilities> {
  if (capabilities) return capabilities;

  const device = getDevice();
  const adapter = getAdapter();

  // Get GPU limits
  const adapterLimits = adapter.limits;
  const limits: GPULimits = {
    maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapterLimits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapterLimits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupsPerDimension: adapterLimits.maxComputeWorkgroupsPerDimension,
    maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
    maxBufferSize: adapterLimits.maxBufferSize,
  };

  // Get platform info for debugging
  const platform = {
    browser: detectBrowser(),
    gpu: 'unknown',
  };

  // Test workgroup shared memory with a simple reduction
  const workgroupSharedMemory = await testWorkgroupSharedMemory(device);

  capabilities = {
    workgroupSharedMemory,
    timestampQuery: !!(adapter.features && adapter.features.has('timestamp-query')),
    subgroups: false, // Future: test subgroup operations
    limits,
    platform,
  };

  // Always log capabilities for debugging
  console.log('[torch.js] GPU capabilities:', capabilities);

  return capabilities;
}

/**
 * Get cached capabilities. Throws if not yet detected.
 */
export function getCapabilities(): GPUCapabilities {
  if (!capabilities) {
    throw new Error('GPU capabilities not detected. Call torch.init() first.');
  }
  return capabilities;
}

/**
 * Test if workgroup shared memory works correctly.
 * Some browsers (Chrome on Mac) have bugs with var<workgroup>.
 */
async function testWorkgroupSharedMemory(device: WebGPUDevice): Promise<boolean> {
  try {
    // Create test input - 5 elements that should sum to 15
    // Use same size as actual reduce shader (256 workgroup) to detect Chrome bugs
    const inputBuffer = device.createBuffer({
      size: 20, // 5 floats
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(inputBuffer.getMappedRange()).set([1, 2, 3, 4, 5]);
    inputBuffer.unmap();

    // Output buffer
    const outputBuffer = device.createBuffer({
      size: 4,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC,
    });

    // Params buffer with length
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM,
      mappedAtCreation: true,
    });
    new Uint32Array(paramsBuffer.getMappedRange())[0] = 5;
    paramsBuffer.unmap();

    // Use the actual reduce shader to test with exact same code
    const shaderCode = REDUCE_SUM_SHADER;

    const shaderModule = device.createShaderModule({ code: shaderCode });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer, offset: 0, size: inputBuffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    // Read result
    const stagingBuffer = device.createBuffer({
      size: 4,
      usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST,
    });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4);
    device.queue.submit([copyEncoder.finish()]);

    await stagingBuffer.mapAsync(MapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange())[0];
    stagingBuffer.unmap();

    // Clean up
    inputBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    stagingBuffer.destroy();

    // Check if result is correct (should be 15: 1+2+3+4+5)
    const works = Math.abs(result - 15) < 0.001;
    console.log('[torch.js] Shared memory test result:', result, 'expected: 15, works:', works);
    return works;
  } catch (e) {
    // If test fails, assume shared memory doesn't work
    console.warn('[torch.js] Workgroup shared memory test failed:', e);
    return false;
  }
}

function detectBrowser(): string {
  if (typeof navigator === 'undefined') return 'node';
  const ua = navigator.userAgent;
  if (ua.includes('Firefox')) return 'firefox';
  if (ua.includes('Chrome')) return 'chrome';
  if (ua.includes('Safari')) return 'safari';
  return 'unknown';
}