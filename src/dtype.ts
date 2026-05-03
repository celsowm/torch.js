/**
 * Supported data types for tensors.
 * @status implemented
 */
export type DType = 'float32' | 'float16' | 'int32' | 'uint32' | 'int8' | 'uint8' | 'bool';

export const DTypeInfo: Record<DType, { bytes: number; arrayType: Float32ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor | Int8ArrayConstructor | Uint8ArrayConstructor }> = {
  float32: { bytes: 4, arrayType: Float32Array },
  float16: { bytes: 2, arrayType: Float32Array }, // Use Float32Array, convert at GPU boundary
  int32: { bytes: 4, arrayType: Int32Array },
  uint32: { bytes: 4, arrayType: Uint32Array },
  int8: { bytes: 1, arrayType: Int8Array },
  uint8: { bytes: 1, arrayType: Uint8Array },
  bool: { bytes: 1, arrayType: Uint8Array },
};

export function getDTypeBytes(dtype: DType): number {
  return DTypeInfo[dtype].bytes;
}

export function getTypedArrayConstructor(dtype: DType) {
  return DTypeInfo[dtype].arrayType;
}

export type TypedArray = Float32Array | Int32Array | Uint32Array | Int8Array | Uint8Array;
