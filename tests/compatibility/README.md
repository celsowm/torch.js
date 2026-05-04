# Compatibility Tests for torch.js

This directory contains comprehensive compatibility tests that verify torch.js implements PyTorch APIs correctly.

## Purpose

These tests ensure:
1. **API Coverage**: Every public API in torch.js has at least one test
2. **Behavioral Compatibility**: torch.js behaves like PyTorch for the same inputs
3. **Edge Cases**: Boundary conditions match PyTorch (empty tensors, NaN, inf, etc.)
4. **Regression Prevention**: Future changes don't break existing behavior

## Structure

```
tests/compatibility/
├── _registry.json        # Registry of all APIs with coverage status (auto-generated)
├── gen_registry.mjs      # Script to generate registry from source
├── report.mjs            # Coverage report script
├── README.md             # This file
├── core/                 # torch.* functions (tensor creation, ops, math, etc.)
├── tensor/               # Tensor.prototype methods
├── nn/                   # torch.nn modules
├── optim/                # torch.optim modules
├── autograd/             # torch.autograd module
├── linalg/               # torch.linalg module
├── fft/                  # torch.fft module
├── distributions/        # torch.distributions module
├── sparse/               # torch.sparse module
└── special/              # torch.special module
```

## Running Tests

```bash
# Run all compatibility tests
npx vitest run --config=vitest.compat.config.ts

# Run a specific category
npx vitest run tests/compatibility/core --config=vitest.compat.config.ts

# Run a specific test file
npx vitest run tests/compatibility/core/tensor_creation.test.ts --config=vitest.compat.config.ts
```

## Coverage Report

```bash
# Summary report
node tests/compatibility/report.mjs --summary

# Full API listing
node tests/compatibility/report.mjs --verbose
```

## Writing Tests

Each test file should:
1. Test the API with standard inputs
2. Test edge cases (empty, NaN, inf, negative, zero, one)
3. Test dtype handling
4. Test shape preservation
5. Compare expected behavior with PyTorch documentation

Example:
```typescript
import { describe, it, expect } from 'vitest';
import { createTorch } from '../../../src/create_torch';

const torch = createTorch();

describe('torch.zeros', () => {
  it('creates tensor of zeros with given shape', async () => {
    const t = torch.zeros([2, 3]);
    expect(t.shape).toEqual([2, 3]);
    const data = await t.toArray();
    for (let i = 0; i < data.length; i++) {
      expect(data[i]).toBe(0);
    }
  });

  it('supports dtype option', async () => {
    const t = torch.zeros([2], { dtype: 'int32' });
    expect(t.dtype).toBe('int32');
  });

  it('supports device option', async () => {
    const t = torch.zeros([2], { device: 'webgpu' });
    expect(t.device).toBe('webgpu');
  });
});
```

## Coverage Status

| Metric | Value |
|--------|-------|
| Total APIs | 588 |
| Covered | See `_registry.json` |
| Percentage | Run `node tests/compatibility/report.mjs --summary` |
