/**
 * Broadcasting utilities for tensor operations.
 * Follows NumPy/PyTorch broadcasting semantics.
 * @status implemented
 */

/**
 * Check if two shapes are broadcastable and return the result shape.
 * Throws if shapes are not compatible.
 */
export function broadcastShapes(
  shapeA: readonly number[],
  shapeB: readonly number[]
): number[] {
  const maxDim = Math.max(shapeA.length, shapeB.length);
  const result: number[] = new Array(maxDim);

  for (let i = 0; i < maxDim; i++) {
    // Get dimensions from right to left
    const dimA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1;
    const dimB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;

    if (dimA === dimB) {
      result[maxDim - 1 - i] = dimA;
    } else if (dimA === 1) {
      result[maxDim - 1 - i] = dimB;
    } else if (dimB === 1) {
      result[maxDim - 1 - i] = dimA;
    } else {
      throw new Error(
        `Shapes [${shapeA.join(', ')}] and [${shapeB.join(', ')}] are not broadcastable`
      );
    }
  }

  return result;
}

/**
 * Check if shapeA needs broadcasting to match shapeB.
 */
export function needsBroadcast(
  shapeA: readonly number[],
  shapeB: readonly number[]
): boolean {
  if (shapeA.length !== shapeB.length) return true;

  for (let i = 0; i < shapeA.length; i++) {
    if (shapeA[i] !== shapeB[i]) return true;
  }

  return false;
}

/**
 * Check if a shape is broadcastable to a target shape.
 */
export function canBroadcastTo(
  shape: readonly number[],
  targetShape: readonly number[]
): boolean {
  if (shape.length > targetShape.length) return false;

  for (let i = 0; i < shape.length; i++) {
    const dim = shape[shape.length - 1 - i];
    const targetDim = targetShape[targetShape.length - 1 - i];
    if (dim !== 1 && dim !== targetDim) return false;
  }

  return true;
}
