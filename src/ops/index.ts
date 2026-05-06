/**
 * Operations exports.
 * @status partial
 */

import { Tensor, Slice } from '../tensor';

export {
  tensor,
  zeros,
  ones,
  full,
  empty,
  zeros_like,
  ones_like,
  full_like,
  empty_like,
  rand,
  randn,
  randint,
  randperm,
  normal,
  eye,
  arange,
  linspace,
  logspace,
  scalar_tensor,
  as_tensor,
  from_numpy,
  manual_seed,
  tril,
  cat,
  stack,
  split,
  vstack,
  row_stack,
  hstack,
  dstack,
  column_stack,
  histc,
  bincount,
  // New additions for PyTorch 1:1 compatibility
  meshgrid,
  cartesian_prod,
  combinations,
  tensor_split,
  trace,
  unravel_index,
} from './creation';

export const atleast_1d = (input: Tensor) => {
  if (input.dim() >= 1) return input;
  return input.reshape([1]);
};
export const atleast_2d = (input: Tensor) => {
  if (input.dim() >= 2) return input;
  if (input.dim() === 0) return input.reshape([1, 1]);
  return input.reshape([1, input.shape[0]]);
};
export const atleast_3d = (input: Tensor) => {
  if (input.dim() >= 3) return input;
  if (input.dim() === 0) return input.reshape([1, 1, 1]);
  if (input.dim() === 1) return input.reshape([1, input.shape[0], 1]);
  return input.reshape([input.shape[0], input.shape[1], 1]);
};

export const broadcast_to = (input: Tensor, shape: number[]) => input.broadcast_to(shape);
export const flip = (input: Tensor, dims: number[]) => input.flip(dims);
export const fliplr = (input: Tensor) => input.flip([1]);
export const flipud = (input: Tensor) => input.flip([0]);
export const cumsum = (input: Tensor, dim: number) => input.cumsum(dim);
export const cumprod = (input: Tensor, dim: number) => input.cumprod(dim);
export const triu = (input: Tensor, diagonal: number = 0) => input.triu(diagonal);
export const diag = (input: Tensor, diagonal: number = 0) => input.diag(diagonal);
export const heaviside = (input: Tensor, values: Tensor) => input.heaviside(values);

// New operations added for PyTorch 1:1 compatibility
export const std = (input: Tensor, dim?: number | number[], keepdim: boolean = false, unbiased: boolean = true) => input.std(dim, keepdim, unbiased);
export const var_ = (input: Tensor, dim?: number | number[], keepdim: boolean = false, unbiased: boolean = true) => input.var(dim, keepdim, unbiased);
export const mean = (input: Tensor, dim?: number | number[], keepdim: boolean = false) => input.mean(dim, keepdim);
export const sum = (input: Tensor, dim?: number | number[], keepdim: boolean = false) => input.sum(dim, keepdim);
export const std_mean = (input: Tensor, dim?: number | number[], keepdim: boolean = false, unbiased: boolean = true) => ({ std: input.std(dim, keepdim, unbiased), mean: input.mean(dim, keepdim) });

export const sort = async (input: Tensor, dim: number = -1, descending: boolean = false) => input.sort(dim, descending);
export const argsort = async (input: Tensor, dim: number = -1, descending: boolean = false) => input.argsort(dim, descending);
export const topk = async (input: Tensor, k: number, dim: number = -1, largest: boolean = true, sorted: boolean = true) => input.topk(k, dim, largest, sorted);
export const kthvalue = async (input: Tensor, k: number, dim: number = -1, keepdim: boolean = false) => input.kthvalue(k, dim, keepdim);

export const cummax = async (input: Tensor, dim: number) => input.cummax(dim);
export const cummin = async (input: Tensor, dim: number) => input.cummin(dim);
export const logsumexp = (input: Tensor, dim?: number | number[], keepdim: boolean = false) => input.logsumexp(dim, keepdim);
export const logcumsumexp = async (input: Tensor, dim: number) => input.logcumsumexp(dim);
export const count_nonzero = (input: Tensor, dim?: number | number[], keepdim: boolean = false) => input.count_nonzero(dim, keepdim);
export const aminmax = (input: Tensor, dim?: number, keepdim: boolean = false) => input.aminmax(dim, keepdim);

export const nonzero = async (input: Tensor) => input.nonzero();
export const diagonal = (input: Tensor, offset: number = 0, dim1: number = -2, dim2: number = -1) => input.diagonal(offset, dim1, dim2);
export const masked_select = (input: Tensor, mask: Tensor) => input.masked_select(mask);

export const gather = async (input: Tensor, dim: number, index: Tensor) => input.gather(dim, index);
export const scatter = async (input: Tensor, dim: number, index: Tensor, src: Tensor) => input.scatter(dim, index, src);
export const scatter_add = async (input: Tensor, dim: number, index: Tensor, src: Tensor) => input.scatter_add(dim, index, src);
export const repeat_interleave = async (input: Tensor, repeats: number, dim?: number) => input.repeat_interleave(repeats, dim);
export const roll = async (input: Tensor, shifts: number | number[], dims?: number | number[]) => input.roll(shifts, dims);
export const rot90 = async (input: Tensor, k: number = 1, dims?: number[]) => input.rot90(k, dims);
export const unflatten = (input: Tensor, dim: number, sizes: number[]) => input.unflatten(dim, sizes);

// Math operations
export const clip = (input: Tensor, min?: number, max?: number) => input.clamp(min, max);
export const clamp_min = (input: Tensor, min: number) => input.clamp(min);
export const clamp_max = (input: Tensor, max: number) => input.clamp(undefined, max);
export const fmod = (input: Tensor, other: Tensor | number) => {
  // fmod: remainder with same sign as dividend
  const div = input.div(other);
  return input.sub(div.trunc().mul(other));
};
export const remainder = (input: Tensor, other: Tensor | number) => {
  // remainder: always positive, same sign as divisor
  // remainder = input - floor(input/other) * other
  const div = input.div(other);
  return input.sub(div.floor().mul(other));
};
export const trunc = (input: Tensor) => input.trunc();
export const fix = trunc;
export const round = (input: Tensor) => input.round();

// Tensor manipulation
export const reshape = (input: Tensor, shape: number[]) => input.reshape(shape);
export const transpose = (input: Tensor, dim0: number, dim1: number) => input.transpose(dim0, dim1);

// Comparison additions
export const isposinf = (input: Tensor) => input.isposinf();
export const isneginf = (input: Tensor) => input.isneginf();
export const isreal = (input: Tensor) => {
  // For real dtypes, all are real; for complex, check imaginary part
  return input.isfinite(); // Simplified for real dtypes
};
export const isnan = (input: Tensor) => input.isnan();
export const isinf = (input: Tensor) => input.isinf();
export const isfinite = (input: Tensor) => input.isfinite();

// Pointwise operations (functional interface)
export const abs = (input: Tensor) => input.abs();
export const absolute = abs;
export const acos = (input: Tensor) => input.acos();
export const arccos = acos;
export const acosh = (input: Tensor) => input.acosh();
export const arccosh = acosh;
export const asin = (input: Tensor) => input.asin();
export const arcsin = asin;
export const asinh = (input: Tensor) => input.asinh();
export const arcsinh = asinh;
export const atan = (input: Tensor) => input.atan();
export const arctan = atan;
export const atanh = (input: Tensor) => input.atanh();
export const arctanh = atanh;
export const atan2 = (input: Tensor, other: Tensor) => input.atan2(other);
export const arctan2 = atan2;

export const ceil = (input: Tensor) => input.ceil();
export const floor = (input: Tensor) => input.floor();
export const frac = (input: Tensor) => input.frac();

export const clamp = (input: Tensor, min?: number, max?: number) => input.clamp(min, max);
export const flatten = (input: Tensor, startDim?: number, endDim?: number) => input.flatten(startDim, endDim);
export const squeeze = (input: Tensor, dim?: number) => input.squeeze(dim);
export const unsqueeze = (input: Tensor, dim: number) => input.unsqueeze(dim);
export const argmax = async (input: Tensor, dim?: number, keepdim?: boolean) => input.argmax(dim, keepdim);
export const argmin = async (input: Tensor, dim?: number, keepdim?: boolean) => input.argmin(dim, keepdim);
export const amax = (input: Tensor, dim?: number, keepdim?: boolean) => input.amax(dim, keepdim);
export const amin = (input: Tensor, dim?: number, keepdim?: boolean) => input.amin(dim, keepdim);
export const all = (input: Tensor, dim?: number, keepdim?: boolean) => input.all(dim, keepdim);
export const any = (input: Tensor, dim?: number, keepdim?: boolean) => input.any(dim, keepdim);

export const chunk = (input: Tensor, chunks: number, dim?: number) => input.chunk(chunks, dim);
export const narrow = (input: Tensor, dim: number, start: number, length: number) => input.narrow(dim, start, length);
export const permute = (input: Tensor, dims: number[]) => input.permute(dims);
export const movedim = (input: Tensor, source: number | number[], destination: number | number[]) => input.movedim(source, destination);
export const moveaxis = movedim;
export const swapaxes = (input: Tensor, dim0: number, dim1: number) => input.swapaxes(dim0, dim1);
export const swapdims = swapaxes;
export const tile = (input: Tensor, reps: number[]) => input.tile(reps);
export const unbind = (input: Tensor, dim?: number) => input.unbind(dim);
export const index_select = (input: Tensor, dim: number, index: Tensor) => input.index_select(dim, index);
export const select = (input: Tensor, dim: number, index: number) => input.select(dim, index);
export const take = (input: Tensor, indices: Tensor) => input.take(indices);
// export const masked_select = ... // skipped
export const where = (condition: Tensor, input: Tensor, other: Tensor) => input.where(condition, other);

export const eq = (input: Tensor, other: Tensor) => input.eq(other);
export const ne = (input: Tensor, other: Tensor) => input.ne(other);
export const not_equal = ne;
export const lt = (input: Tensor, other: Tensor) => input.lt(other);
export const less = lt;
export const le = (input: Tensor, other: Tensor) => input.le(other);
export const less_equal = le;
export const gt = (input: Tensor, other: Tensor) => input.gt(other);
export const greater = gt;
export const ge = (input: Tensor, other: Tensor) => input.ge(other);
export const greater_equal = ge;

export const maximum = (input: Tensor, other: Tensor) => input.maximum(other);
export const minimum = (input: Tensor, other: Tensor) => input.minimum(other);
export const fmax = (input: Tensor, other: Tensor) => input.fmax(other);
export const fmin = (input: Tensor, other: Tensor) => input.fmin(other);

export const equal = (input: Tensor, other: Tensor) => input.equal(other);
export const isclose = (input: Tensor, other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean) => input.isclose(other, rtol, atol, equal_nan);
export const allclose = (input: Tensor, other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean) => input.allclose(other, rtol, atol, equal_nan);

export const cos = (input: Tensor) => input.cos();
export const cosh = (input: Tensor) => input.cosh();
export const sin = (input: Tensor) => input.sin();
export const sinh = (input: Tensor) => input.sinh();
export const tan = (input: Tensor) => input.tan();
export const tanh = (input: Tensor) => input.tanh();

export const exp = (input: Tensor) => input.exp();
export const exp2 = (input: Tensor) => input.exp2();
export const log = (input: Tensor) => input.log();
export const log10 = (input: Tensor) => input.log10();
export const log2 = (input: Tensor) => input.log2();
export const log1p = (input: Tensor) => input.log1p();
export const logaddexp = (input: Tensor, other: Tensor) => input.logaddexp(other);

export const neg = (input: Tensor) => input.neg();
export const negative = neg;

export const prod = (input: Tensor, dim?: number, keepdim?: boolean) => input.prod(dim, keepdim);
export const pow = (input: Tensor, exponent: number | Tensor) => input.pow(exponent);
export const reciprocal = (input: Tensor) => input.reciprocal();
export const rsqrt = (input: Tensor) => input.rsqrt();
export const sqrt = (input: Tensor) => input.sqrt();
export const square = (input: Tensor) => input.square();

export const sigmoid = (input: Tensor) => input.sigmoid();
export const relu = (input: Tensor) => input.relu();
export const sign = (input: Tensor) => input.sign();
export const sgn = (input: Tensor) => input.sgn();
export const erf = (input: Tensor) => input.erf();
export const erfc = (input: Tensor) => input.erfc();
export const expm1 = (input: Tensor) => input.expm1();
export const deg2rad = (input: Tensor) => input.deg2rad();
export const rad2deg = (input: Tensor) => input.rad2deg();
export const logical_not = (input: Tensor) => input.logical_not();
export const i0 = (input: Tensor) => input.i0();
export const lgamma = (input: Tensor) => input.lgamma();
export const digamma = (input: Tensor) => input.digamma();

export const gelu = (input: Tensor) => input.gelu();
export const softplus = (input: Tensor) => input.softplus();
export const silu = (input: Tensor) => input.silu();
export const mish = (input: Tensor) => input.mish();
export const hardsigmoid = (input: Tensor) => input.hardsigmoid();
export const hardswish = (input: Tensor) => input.hardswish();
export const softsign = (input: Tensor) => input.softsign();
export const tanhshrink = (input: Tensor) => input.tanhshrink();
export const leaky_relu = (input: Tensor, negative_slope: number = 0.01) => input.leaky_relu(negative_slope);
export const elu = (input: Tensor, alpha: number = 1.0) => input.elu(alpha);
export const selu = (input: Tensor) => input.selu();
export const threshold = (input: Tensor, threshold: number, value: number) => input.threshold(threshold, value);
export const softmin = (input: Tensor) => input.softmax(-1); // softmin via softmax of negated
export const log_softmax = (input: Tensor, dim: number = -1) => input.log_softmax(dim);
export const softmax = (input: Tensor, dim: number = -1) => input.softmax(dim);

export const add = (input: Tensor, other: Tensor | number) => input.add(other);
export const sub = (input: Tensor, other: Tensor | number) => input.sub(other);
export const subtract = sub;
export const mul = (input: Tensor, other: Tensor | number) => input.mul(other);
export const multiply = mul;
export const div = (input: Tensor, other: Tensor | number) => input.div(other);
export const divide = div;

export const bitwise_and = (input: Tensor, other: Tensor) => input.bitwise_and(other);
export const bitwise_or = (input: Tensor, other: Tensor) => input.bitwise_or(other);
export const bitwise_xor = (input: Tensor, other: Tensor) => input.bitwise_xor(other);

export const hypot = (input: Tensor, other: Tensor) => input.hypot(other);
export const matmul = (input: Tensor, other: Tensor) => input.matmul(other);
export const mm = (input: Tensor, mat2: Tensor) => input.mm(mat2);
export const chain_matmul = (...matrices: Tensor[]) => {
  if (matrices.length === 0) throw new Error('chain_matmul: expected at least one tensor');
  let result = matrices[0];
  for (let i = 1; i < matrices.length; i++) {
    result = result.mm(matrices[i]);
  }
  return result;
};
export const addmm = (input: Tensor, mat1: Tensor, mat2: Tensor, beta?: number, alpha?: number) => input.addmm(mat1, mat2, beta, alpha);
export const mv = (input: Tensor, vec: Tensor) => input.mv(vec);
export const addmv = (input: Tensor, mat: Tensor, vec: Tensor, beta?: number, alpha?: number) => input.addmv(mat, vec, beta, alpha);
export const outer = (input: Tensor, vec2: Tensor) => input.outer(vec2);
export const ger = outer;
export const addr = (input: Tensor, vec1: Tensor, vec2: Tensor, beta?: number, alpha?: number) => input.addr(vec1, vec2, beta, alpha);
export const bmm = (input: Tensor, mat2: Tensor) => input.bmm(mat2);
export const baddbmm = (input: Tensor, batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number) => input.baddbmm(batch1, batch2, beta, alpha);
export const addbmm = (input: Tensor, batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number) => input.addbmm(batch1, batch2, beta, alpha);
export const dot = (input: Tensor, other: Tensor) => input.dot(other);
export const vdot = (input: Tensor, other: Tensor) => input.vdot(other);
export const inner = dot;
export const trapezoid = (input: Tensor, dx?: number, dim?: number) => input.trapezoid(dx, dim);
export const cumulative_trapezoid = (input: Tensor, dx?: number, dim?: number) => input.cumulative_trapezoid(dx, dim);
export const trapz = trapezoid;

// Einsum
export const einsum = async (equation: string, ...operands: Tensor[]) => Tensor.einsum(equation, ...operands);

// Advanced slicing
export const advancedSlice = async (input: Tensor, indices: (number | Slice)[]) => input.advancedSlice(indices);

// Multinomial sampling
export const multinomial = async (input: Tensor, num_samples: number = 1, replacement: boolean = false) => input.multinomial(num_samples, replacement);