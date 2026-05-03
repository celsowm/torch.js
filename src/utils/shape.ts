/**
 * Shape utilities for tensor operations.
 * @status implemented
 */

/**
 * Calculate the total number of elements from a shape.
 */
export function numel(shape: readonly number[]): number {
  if (shape.length === 0) return 1; // Scalar
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Calculate strides for a contiguous tensor with the given shape.
 * Strides are in row-major (C) order.
 */
export function computeStrides(shape: readonly number[]): number[] {
  if (shape.length === 0) return [];

  const strides = new Array(shape.length);
  let stride = 1;

  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }

  return strides;
}

/**
 * Check if two shapes are equal.
 */
export function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Format a shape for display.
 */
export function formatShape(shape: readonly number[]): string {
  return `[${shape.join(', ')}]`;
}

/**
 * Validate a shape array.
 */
export function validateShape(shape: readonly number[]): void {
  for (let i = 0; i < shape.length; i++) {
    if (!Number.isInteger(shape[i]) || shape[i] < 0) {
      throw new Error(`Invalid shape dimension at index ${i}: ${shape[i]}`);
    }
  }
}

/**
 * Infer -1 dimension in reshape.
 */
export function inferShape(shape: readonly number[], totalElements: number): number[] {
  const negOneIndex = shape.indexOf(-1);
  if (negOneIndex === -1) {
    return [...shape];
  }

  // Check only one -1
  if (shape.lastIndexOf(-1) !== negOneIndex) {
    throw new Error('Only one dimension can be -1 in reshape');
  }

  // Calculate the inferred dimension
  let knownProduct = 1;
  for (let i = 0; i < shape.length; i++) {
    if (i !== negOneIndex) {
      knownProduct *= shape[i];
    }
  }

  if (totalElements % knownProduct !== 0) {
    throw new Error(`Cannot reshape tensor of ${totalElements} elements to shape [${shape.join(', ')}]`);
  }

  const result = [...shape];
  result[negOneIndex] = totalElements / knownProduct;
  return result;
}
