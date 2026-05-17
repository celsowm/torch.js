import { describe, it, expect } from 'vitest';

/**
 * Tests for GPU capabilities detection.
 * Note: These tests verify the logic used to extract GPU name from WebGPU adapter info.
 * Full integration tests require a real WebGPU environment (browser).
 */
describe('GPU Capabilities - Info Detection', () => {
  it('extracts GPU name from adapter.info.description', () => {
    const adapter = {
      info: {
        description: 'NVIDIA GeForce RTX 4090',
        vendor: 'NVIDIA',
        device: 'GeForce RTX 4090',
        architecture: 'Ada',
      },
    };

    const adapterInfo = (adapter as any).info;
    let gpuName = 'unknown';
    if (adapterInfo) {
      gpuName = adapterInfo.description ||
        [adapterInfo.vendor, adapterInfo.device, adapterInfo.architecture]
          .filter(Boolean)
          .join(' ') ||
        'unknown';
    }

    expect(gpuName).toBe('NVIDIA GeForce RTX 4090');
  });

  it('falls back to vendor+device when description is missing', () => {
    const adapter = {
      info: {
        vendor: 'AMD',
        device: 'Radeon RX 7900 XTX',
      },
    };

    const adapterInfo = (adapter as any).info;
    let gpuName = 'unknown';
    if (adapterInfo) {
      gpuName = adapterInfo.description ||
        [adapterInfo.vendor, adapterInfo.device, adapterInfo.architecture]
          .filter(Boolean)
          .join(' ') ||
        'unknown';
    }

    expect(gpuName).toBe('AMD Radeon RX 7900 XTX');
  });

  it('uses "unknown" when adapter.info is not available', () => {
    const adapter = { info: undefined };

    const adapterInfo = (adapter as any).info;
    let gpuName = 'unknown';
    if (adapterInfo) {
      gpuName = adapterInfo.description ||
        [adapterInfo.vendor, adapterInfo.device, adapterInfo.architecture]
          .filter(Boolean)
          .join(' ') ||
        'unknown';
    }

    expect(gpuName).toBe('unknown');
  });

  it('uses "unknown" when adapter.info has no useful fields', () => {
    const adapter = { info: {} };

    const adapterInfo = (adapter as any).info;
    let gpuName = 'unknown';
    if (adapterInfo) {
      gpuName = adapterInfo.description ||
        [adapterInfo.vendor, adapterInfo.device, adapterInfo.architecture]
          .filter(Boolean)
          .join(' ') ||
        'unknown';
    }

    expect(gpuName).toBe('unknown');
  });

  it('includes architecture in fallback chain', () => {
    const adapter = {
      info: {
        architecture: 'RDNA3',
      },
    };

    const adapterInfo = (adapter as any).info;
    let gpuName = 'unknown';
    if (adapterInfo) {
      gpuName = adapterInfo.description ||
        [adapterInfo.vendor, adapterInfo.device, adapterInfo.architecture]
          .filter(Boolean)
          .join(' ') ||
        'unknown';
    }

    expect(gpuName).toBe('RDNA3');
  });
});
