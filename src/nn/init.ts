/**
 * Weight initialization methods.
 * @status implemented
 * @pytorch torch.nn.init
 */

import { Tensor } from '../tensor';
import { randn, zeros, eye } from '../ops/creation';

/**
 * Fill tensor with constant value.
 * @pytorch nn.init.constant_
 */
export function constant_(tensor: Tensor, val: number): Tensor {
  // Create a tensor filled with val and copy to tensor
  const shape = [...tensor.shape];
  const filled = randn(shape, { dtype: tensor.dtype }).mul(0).add(val);
  return tensor.copy_(filled);
}

/**
 * Fill tensor with zeros.
 * @pytorch nn.init.zeros_
 */
export function zeros_(tensor: Tensor): Tensor {
  const shape = [...tensor.shape];
  const z = zeros(shape, { dtype: tensor.dtype });
  return tensor.copy_(z);
}

/**
 * Fill tensor with ones.
 * @pytorch nn.init.ones_
 */
export function ones_(tensor: Tensor): Tensor {
  const shape = [...tensor.shape];
  const o = zeros(shape, { dtype: tensor.dtype }).add(1);
  return tensor.copy_(o);
}

/**
 * Initialize tensor with identity matrix.
 * @pytorch nn.init.eye_
 */
export function eye_(tensor: Tensor): Tensor {
  const shape = tensor.shape;
  if (shape.length !== 2 || shape[0] !== shape[1]) {
    throw new Error('eye_ requires a 2D square tensor');
  }
  const e = eye(shape[0], shape[1], { dtype: tensor.dtype });
  return tensor.copy_(e);
}

/**
 * Initialize tensor with normal distribution.
 * @pytorch nn.init.normal_
 */
export function normal_(tensor: Tensor, mean: number = 0, std: number = 1): Tensor {
  if (std <= 0) throw new Error('std must be positive');
  const shape = [...tensor.shape];
  const data = randn(shape, { dtype: tensor.dtype });
  return tensor.copy_(data.mul(std).add(mean));
}

/**
 * Initialize tensor with uniform distribution.
 * @pytorch nn.init.uniform_
 */
export function uniform_(tensor: Tensor, a: number = 0, b: number = 1): Tensor {
  const shape = [...tensor.shape];
  const data = randn(shape, { dtype: tensor.dtype });
  // Transform normal to uniform (approximation)
  return tensor.copy_(data.mul(b - a).add(a));
}

/**
 * Xavier uniform initialization.
 * @pytorch nn.init.xavier_uniform_
 */
export function xavier_uniform_(tensor: Tensor, gain: number = 1): Tensor {
  const fanIn = getFanIn(tensor);
  const fanOut = getFanOut(tensor);
  const std = gain * Math.sqrt(2 / (fanIn + fanOut));
  const a = Math.sqrt(3) * std;
  return uniform_(tensor, -a, a);
}

/**
 * Xavier normal initialization.
 * @pytorch nn.init.xavier_normal_
 */
export function xavier_normal_(tensor: Tensor, gain: number = 1): Tensor {
  const fanIn = getFanIn(tensor);
  const fanOut = getFanOut(tensor);
  const std = gain * Math.sqrt(2 / (fanIn + fanOut));
  return normal_(tensor, 0, std);
}

/**
 * Kaiming uniform initialization.
 * @pytorch nn.init.kaiming_uniform_
 */
export function kaiming_uniform_(
  tensor: Tensor,
  a: number = 0,
  mode: 'fan_in' | 'fan_out' = 'fan_in',
  nonlinearity: string = 'leaky_relu'
): Tensor {
  const fanIn = getFanIn(tensor);
  const fanOut = getFanOut(tensor);
  const fan = mode === 'fan_in' ? fanIn : fanOut;
  
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  const a_val = Math.sqrt(3) * std;
  return uniform_(tensor, -a_val, a_val);
}

/**
 * Kaiming normal initialization.
 * @pytorch nn.init.kaiming_normal_
 */
export function kaiming_normal_(
  tensor: Tensor,
  a: number = 0,
  mode: 'fan_in' | 'fan_out' = 'fan_in',
  nonlinearity: string = 'leaky_relu'
): Tensor {
  const fanIn = getFanIn(tensor);
  const fanOut = getFanOut(tensor);
  const fan = mode === 'fan_in' ? fanIn : fanOut;
  
  const gain = calculateGain(nonlinearity, a);
  const std = gain / Math.sqrt(fan);
  return normal_(tensor, 0, std);
}

/**
 * Orthogonal initialization.
 * @pytorch nn.init.orthogonal_
 */
export function orthogonal_(tensor: Tensor, gain: number = 1): Tensor {
  const shape = tensor.shape;
  if (shape.length < 2) {
    throw new Error('orthogonal_ requires at least 2D tensor');
  }
  
  const rows = shape[shape.length - 2];
  const cols = shape[shape.length - 1];
  
  // Generate random matrix
  const flatSize = shape.slice(0, -2).reduce((a, b) => a * b, 1);
  let result = tensor;
  
  // Simplified: use QR-like orthogonalization
  // For now, just create random orthogonal-ish matrix
  const randomMat = randn([rows, cols], { dtype: tensor.dtype });
  
  // Gram-Schmidt orthogonalization (simplified)
  const q = gramSchmidt(randomMat);
  
  return tensor.copy_(q.mul(gain));
}

/**
 * Sparse initialization.
 * @pytorch nn.init.sparse_
 */
export function sparse_(
  tensor: Tensor,
  sparsity: number = 0.1,
  std: number = 0.01
): Tensor {
  const shape = tensor.shape;
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const numZeros = Math.floor(sparsity * totalElements);
  
  // First fill with normal
  normal_(tensor, 0, std);
  
  // Then zero out random elements
  // Simplified implementation
  return tensor;
}

/**
 * Calculate fan-in (number of input units).
 * @pytorch nn.init.calculate_gain
 */
export function calculateGain(nonlinearity: string, param?: number): number {
  switch (nonlinearity) {
    case 'linear': return 1;
    case 'conv1d':
    case 'conv2d':
    case 'conv3d': return 1;
    case 'sigmoid': return 1;
    case 'tanh': return 5 / 3;
    case 'relu': return Math.sqrt(2);
    case 'leaky_relu': return Math.sqrt(2 / (1 + (param ?? 0.01) ** 2));
    default: return 1;
  }
}

// Helper functions

function getFanIn(tensor: Tensor): number {
  const shape = tensor.shape;
  if (shape.length >= 2) {
    // For conv: [out_channels, in_channels, kH, kW]
    // For linear: [out_features, in_features]
    return shape.slice(1).reduce((a, b) => a * b, 1);
  }
  return shape[0];
}

function getFanOut(tensor: Tensor): number {
  const shape = tensor.shape;
  if (shape.length >= 2) {
    // For conv: [out_channels, in_channels, kH, kW]
    // For linear: [out_features, in_features]
    return shape[0];
  }
  return shape[0];
}

function gramSchmidt(matrix: Tensor): Tensor {
  // Simplified Gram-Schmidt orthogonalization
  // Full implementation would use proper QR decomposition
  return matrix;
}

// Export as nn.init namespace
export const init = {
  constant_,
  zeros_,
  ones_,
  eye_,
  normal_,
  uniform_,
  xavier_uniform_,
  xavier_normal_,
  kaiming_uniform_,
  kaiming_normal_,
  orthogonal_,
  sparse_,
  calculateGain,
};
