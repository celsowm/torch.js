/**
 * Generate _registry.json by scanning all source files.
 * Usage: node tests/compatibility/gen_registry.mjs
 */

import { readFileSync, writeFileSync, readdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, '..', '..', '..');

// All API categories
const categories = {
  // torch.* functions
  'torch_creation': {
    prefix: 'torch',
    source: 'src/ops/creation.ts',
    apis: [
      'tensor', 'zeros', 'ones', 'full', 'empty', 'zeros_like', 'ones_like',
      'full_like', 'empty_like', 'rand', 'randn', 'randint', 'randperm',
      'normal', 'eye', 'arange', 'linspace', 'logspace', 'scalar_tensor',
      'as_tensor', 'from_numpy', 'manual_seed', 'histc', 'bincount',
    ],
  },
  'torch_manipulation': {
    prefix: 'torch',
    source: 'src/ops/index.ts',
    apis: [
      'cat', 'stack', 'split', 'vstack', 'row_stack', 'hstack', 'dstack',
      'column_stack', 'tril', 'meshgrid', 'cartesian_prod', 'combinations',
      'tensor_split', 'trace', 'unravel_index', 'atleast_1d', 'atleast_2d',
      'atleast_3d', 'broadcast_to', 'flip', 'fliplr', 'flipud', 'cumsum',
      'cumprod', 'triu', 'diag', 'heaviside', 'gather', 'scatter',
      'scatter_add', 'repeat_interleave', 'roll', 'rot90', 'unflatten',
      'nonzero', 'diagonal', 'masked_select', 'where', 'index_select',
      'select', 'take', 'chunk', 'narrow', 'permute', 'movedim', 'moveaxis',
      'swapaxes', 'swapdims', 'tile', 'unbind', 'flatten', 'squeeze', 'unsqueeze',
    ],
  },
  'torch_stats': {
    prefix: 'torch',
    source: 'src/ops/index.ts',
    apis: [
      'std', 'var', 'std_mean', 'var_mean', 'sort', 'argsort', 'topk',
      'kthvalue', 'cummax', 'cummin', 'logsumexp', 'logcumsumexp',
      'count_nonzero', 'aminmax',
    ],
  },
  'torch_math': {
    prefix: 'torch',
    source: 'src/ops/index.ts',
    apis: [
      'abs', 'absolute', 'acos', 'arccos', 'acosh', 'arccosh', 'asin',
      'arcsin', 'asinh', 'arcsinh', 'atan', 'arctan', 'atanh', 'arctanh',
      'atan2', 'arctan2', 'ceil', 'floor', 'frac', 'clip', 'clamp',
      'clamp_min', 'clamp_max', 'fmod', 'remainder', 'trunc', 'fix',
      'round', 'cos', 'cosh', 'sin', 'sinh', 'tan', 'tanh', 'exp', 'exp2',
      'log', 'log10', 'log2', 'log1p', 'logaddexp', 'neg', 'negative',
      'prod', 'pow', 'reciprocal', 'rsqrt', 'sqrt', 'square', 'sigmoid',
      'relu', 'sign', 'sgn', 'erf', 'erfc', 'expm1', 'deg2rad', 'rad2deg',
      'logical_not', 'i0', 'lgamma', 'digamma', 'isposinf', 'isneginf',
      'isreal', 'isnan', 'isinf', 'isfinite',
    ],
  },
  'torch_arithmetic': {
    prefix: 'torch',
    source: 'src/ops/index.ts',
    apis: ['add', 'sub', 'subtract', 'mul', 'multiply', 'div', 'divide'],
  },
  'torch_bitwise': {
    prefix: 'torch',
    source: 'src/ops/index.ts',
    apis: ['bitwise_and', 'bitwise_or', 'bitwise_xor'],
  },
  'torch_comparison': {
    prefix: 'torch',
    source: 'src/ops/index.ts',
    apis: [
      'eq', 'ne', 'not_equal', 'lt', 'less', 'le', 'less_equal', 'gt',
      'greater', 'ge', 'greater_equal', 'maximum', 'minimum', 'fmax',
      'fmin', 'equal', 'isclose', 'allclose',
    ],
  },
  'torch_reduction': {
    prefix: 'torch',
    source: 'src/ops/index.ts',
    apis: ['argmax', 'argmin', 'amax', 'amin', 'all', 'any'],
  },
  'torch_linalg': {
    prefix: 'torch',
    source: 'src/ops/index.ts',
    apis: [
      'hypot', 'matmul', 'mm', 'chain_matmul', 'addmm', 'mv', 'addmv',
      'outer', 'ger', 'addr', 'bmm', 'baddbmm', 'addbmm', 'dot', 'vdot',
      'inner', 'trapezoid', 'cumulative_trapezoid', 'trapz', 'einsum',
      'multinomial',
    ],
  },
  'torch_grad_mode': {
    prefix: 'torch',
    source: 'src/grad_mode.ts',
    apis: ['no_grad', 'enable_grad', 'inference_mode', 'is_grad_enabled'],
  },
  // Tensor methods
  'tensor_basic': {
    prefix: 'Tensor.prototype',
    source: 'src/tensor/Tensor.ts',
    apis: [
      'toArray', 'item', 'clone', 'detach', 'destroy', 'contiguous',
      'copy_', 'to', 'requires_grad_', 'backward',
    ],
  },
  'tensor_math': {
    prefix: 'Tensor.prototype',
    source: 'src/tensor/Tensor.ts',
    apis: [
      'add', 'sub', 'mul', 'div', 'pow', 'abs', 'acos', 'asin', 'atan',
      'ceil', 'cos', 'cosh', 'exp', 'exp2', 'floor', 'log', 'log10',
      'log2', 'log1p', 'neg', 'round', 'sin', 'sinh', 'acosh', 'asinh',
      'atanh', 'sqrt', 'tan', 'tanh', 'trunc', 'frac', 'reciprocal',
      'rsqrt', 'square', 'sigmoid', 'relu', 'gelu', 'softmax', 'log_softmax',
      'sign', 'sgn', 'bool', 'erf', 'erfc', 'expm1', 'deg2rad', 'rad2deg',
      'logical_not', 'i0', 'lgamma', 'digamma', 'softplus', 'silu', 'mish',
      'hardsigmoid', 'hardswish', 'softsign', 'tanhshrink', 'leaky_relu',
      'elu', 'selu', 'threshold', 'zeros_like', 'ones_like', 'heaviside', 'glu',
    ],
  },
  'tensor_shape': {
    prefix: 'Tensor.prototype',
    source: 'src/tensor/Tensor.ts',
    apis: [
      'numel', 'dim', 'size', 'stride', 'reshape', 'view', 'expand',
      'broadcast_to', 'squeeze', 'unsqueeze', 'flatten', 'transpose',
      'permute', 't', 'movedim', 'moveaxis', 'swapaxes', 'swapdims',
      'unbind', 'narrow', 'slice', 'select', 'take', 'split', 'chunk',
      'unflatten', 'tile', 'repeat', 'flip', 'triu', 'tril', 'diag',
      'diagonal', 'roll', 'rot90', 'get', 'at', 'advancedSlice', 'set',
    ],
  },
  'tensor_reduction': {
    prefix: 'Tensor.prototype',
    source: 'src/tensor/Tensor.ts',
    apis: [
      'sum', 'mean', 'var', 'std', 'amax', 'amin', 'prod', 'all', 'any',
      'argmax', 'argmin', 'aminmax', 'logsumexp', 'logcumsumexp',
      'count_nonzero', 'cumsum', 'cumprod',
    ],
  },
  'tensor_sort': {
    prefix: 'Tensor.prototype',
    source: 'src/tensor/Tensor.ts',
    apis: ['sort', 'argsort', 'topk', 'kthvalue', 'cummax', 'cummin'],
  },
  'tensor_linalg': {
    prefix: 'Tensor.prototype',
    source: 'src/tensor/Tensor.ts',
    apis: [
      'mm', 'matmul', 'bmm', 'addmm', 'mv', 'addmv', 'outer', 'addr',
      'baddbmm', 'addbmm', 'dot', 'vdot', 'trapezoid', 'cumulative_trapezoid',
    ],
  },
  'tensor_advanced': {
    prefix: 'Tensor.prototype',
    source: 'src/tensor/Tensor.ts',
    apis: [
      'atan2', 'hypot', 'logaddexp', 'bitwise_and', 'bitwise_or',
      'bitwise_xor', 'clamp', 'masked_fill', 'eq', 'ne', 'lt', 'gt',
      'ge', 'le', 'greater', 'greater_equal', 'less', 'less_equal',
      'not_equal', 'isnan', 'isinf', 'isfinite', 'isposinf', 'isneginf',
      'maximum', 'minimum', 'fmax', 'fmin', 'equal', 'isclose', 'allclose',
      'gather', 'scatter', 'scatter_', 'scatter_add', 'repeat_interleave',
      'masked_select', 'nonzero', 'nonzeroIndices', 'multinomial',
      'index_select', 'where',
    ],
  },
};

// Build registry
const registry = {};

for (const [category, { prefix, apis }] of Object.entries(categories)) {
  for (const api of apis) {
    const key = prefix === 'torch' ? `torch.${api}` : `Tensor.${api}`;
    registry[key] = {
      category,
      tested: false,
      testFile: null,
      pytorchTest: null,
      notes: '',
    };
  }
}

// Add nn, optim, linalg, fft, etc.
const moduleApis = {
  // nn modules
  'torch.nn.Module': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Parameter': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Linear': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Sequential': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Flatten': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Embedding': { category: 'nn', tested: false, testFile: null },
  'torch.nn.LayerNorm': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Conv1d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Conv2d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.BatchNorm1d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.BatchNorm2d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.MaxPool2d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.AvgPool2d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Dropout': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Dropout1d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Dropout2d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Dropout3d': { category: 'nn', tested: false, testFile: null },
  'torch.nn.AlphaDropout': { category: 'nn', tested: false, testFile: null },
  'torch.nn.ReLU': { category: 'nn', tested: false, testFile: null },
  'torch.nn.GELU': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Sigmoid': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Tanh': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Softmax': { category: 'nn', tested: false, testFile: null },
  'torch.nn.LogSoftmax': { category: 'nn', tested: false, testFile: null },
  'torch.nn.PReLU': { category: 'nn', tested: false, testFile: null },
  'torch.nn.CELU': { category: 'nn', tested: false, testFile: null },
  'torch.nn.RReLU': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Hardtanh': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Hardshrink': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Softshrink': { category: 'nn', tested: false, testFile: null },
  'torch.nn.LogSigmoid': { category: 'nn', tested: false, testFile: null },
  'torch.nn.Softmin': { category: 'nn', tested: false, testFile: null },
  'torch.nn.RNN': { category: 'nn', tested: false, testFile: null },
  'torch.nn.LSTM': { category: 'nn', tested: false, testFile: null },
  'torch.nn.GRU': { category: 'nn', tested: false, testFile: null },
  'torch.nn.MultiheadAttention': { category: 'nn', tested: false, testFile: null },
  'torch.nn.TransformerEncoder': { category: 'nn', tested: false, testFile: null },
  'torch.nn.TransformerDecoder': { category: 'nn', tested: false, testFile: null },
  'torch.nn.TransformerEncoderLayer': { category: 'nn', tested: false, testFile: null },
  'torch.nn.TransformerDecoderLayer': { category: 'nn', tested: false, testFile: null },
  'torch.nn.ModuleList': { category: 'nn', tested: false, testFile: null },
  'torch.nn.ModuleDict': { category: 'nn', tested: false, testFile: null },
  'torch.nn.NLLLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.CrossEntropyLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.BCELoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.BCEWithLogitsLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.HuberLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.SmoothL1Loss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.L1Loss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.MSELoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.KLDivLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.HingeEmbeddingLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.CosineEmbeddingLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.TripletMarginLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.CTCLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.MultiMarginLoss': { category: 'nn', tested: false, testFile: null },
  'torch.nn.functional.relu': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.gelu': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.sigmoid': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.tanh': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.softmax': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.log_softmax': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.dropout': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.conv2d': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.max_pool2d': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.avg_pool2d': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.interpolate': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.pad': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.normalize': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.one_hot': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.scaled_dot_product_attention': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.functional.grid_sample': { category: 'nn.functional', tested: false, testFile: null },
  'torch.nn.init.uniform_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.normal_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.xavier_uniform_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.xavier_normal_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.kaiming_uniform_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.kaiming_normal_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.orthogonal_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.sparse_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.zeros_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.ones_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.eye_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.constant_': { category: 'nn.init', tested: false, testFile: null },
  'torch.nn.init.calculate_gain': { category: 'nn.init', tested: false, testFile: null },

  // optimizers
  'torch.optim.SGD': { category: 'optim', tested: false, testFile: null },
  'torch.optim.Adam': { category: 'optim', tested: false, testFile: null },
  'torch.optim.AdamW': { category: 'optim', tested: false, testFile: null },
  'torch.optim.Adamax': { category: 'optim', tested: false, testFile: null },
  'torch.optim.NAdam': { category: 'optim', tested: false, testFile: null },
  'torch.optim.RAdam': { category: 'optim', tested: false, testFile: null },
  'torch.optim.ASGD': { category: 'optim', tested: false, testFile: null },
  'torch.optim.RMSprop': { category: 'optim', tested: false, testFile: null },
  'torch.optim.Adagrad': { category: 'optim', tested: false, testFile: null },
  'torch.optim.LBFGS': { category: 'optim', tested: false, testFile: null },
  'torch.optim.lr_scheduler.StepLR': { category: 'optim.lr_scheduler', tested: false, testFile: null },
  'torch.optim.lr_scheduler.MultiStepLR': { category: 'optim.lr_scheduler', tested: false, testFile: null },
  'torch.optim.lr_scheduler.ExponentialLR': { category: 'optim.lr_scheduler', tested: false, testFile: null },
  'torch.optim.lr_scheduler.CosineAnnealingLR': { category: 'optim.lr_scheduler', tested: false, testFile: null },
  'torch.optim.lr_scheduler.ReduceLROnPlateau': { category: 'optim.lr_scheduler', tested: false, testFile: null },
  'torch.optim.lr_scheduler.ConstantLR': { category: 'optim.lr_scheduler', tested: false, testFile: null },
  'torch.optim.lr_scheduler.LambdaLR': { category: 'optim.lr_scheduler', tested: false, testFile: null },
  'torch.optim.lr_scheduler.LinearWarmupLR': { category: 'optim.lr_scheduler', tested: false, testFile: null },
  'torch.optim.lr_scheduler.MultiplicativeLR': { category: 'optim.lr_scheduler', tested: false, testFile: null },

  // linalg
  'torch.linalg.norm': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.vector_norm': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.matrix_norm': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.cholesky': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.inv': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.lu_factor': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.solve_triangular': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.diagonal': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.vecdot': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.vander': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.matrix_power': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.cross': { category: 'linalg', tested: false, testFile: null },
  'torch.linalg.svd': { category: 'linalg.advanced', tested: false, testFile: null },
  'torch.linalg.eigh': { category: 'linalg.advanced', tested: false, testFile: null },
  'torch.linalg.qr': { category: 'linalg.advanced', tested: false, testFile: null },
  'torch.linalg.solve': { category: 'linalg.advanced', tested: false, testFile: null },
  'torch.linalg.pinv': { category: 'linalg.advanced', tested: false, testFile: null },

  // fft
  'torch.fft.fft': { category: 'fft', tested: false, testFile: null },
  'torch.fft.ifft': { category: 'fft', tested: false, testFile: null },
  'torch.fft.fft2': { category: 'fft', tested: false, testFile: null },
  'torch.fft.fftn': { category: 'fft', tested: false, testFile: null },
  'torch.fft.rfft': { category: 'fft', tested: false, testFile: null },
  'torch.fft.irfft': { category: 'fft', tested: false, testFile: null },
  'torch.fft.fftshift': { category: 'fft', tested: false, testFile: null },
  'torch.fft.ifftshift': { category: 'fft', tested: false, testFile: null },
  'torch.fft.fftfreq': { category: 'fft', tested: false, testFile: null },
  'torch.fft.rfftfreq': { category: 'fft', tested: false, testFile: null },
  'torch.fft.stft': { category: 'fft', tested: false, testFile: null },
  'torch.fft.istft': { category: 'fft', tested: false, testFile: null },
  'torch.fft.hfft': { category: 'fft', tested: false, testFile: null },
  'torch.fft.ihfft': { category: 'fft', tested: false, testFile: null },

  // autograd
  'torch.autograd.grad': { category: 'autograd', tested: false, testFile: null },
  'torch.autograd.vjp': { category: 'autograd', tested: false, testFile: null },
  'torch.autograd.jvp': { category: 'autograd', tested: false, testFile: null },
  'torch.autograd.Function': { category: 'autograd', tested: false, testFile: null },
  'torch.autograd.gradcheck': { category: 'autograd', tested: false, testFile: null },

  // distributions
  'torch.distributions.Normal': { category: 'distributions', tested: false, testFile: null },
  'torch.distributions.Bernoulli': { category: 'distributions', tested: false, testFile: null },
  'torch.distributions.Categorical': { category: 'distributions', tested: false, testFile: null },
  'torch.distributions.Uniform': { category: 'distributions', tested: false, testFile: null },
  'torch.distributions.Exponential': { category: 'distributions', tested: false, testFile: null },
  'torch.distributions.Poisson': { category: 'distributions', tested: false, testFile: null },
  'torch.distributions.Gamma': { category: 'distributions', tested: false, testFile: null },
  'torch.distributions.Beta': { category: 'distributions', tested: false, testFile: null },
  'torch.distributions.Dirichlet': { category: 'distributions', tested: false, testFile: null },

  // sparse
  'torch.sparse.SparseTensor': { category: 'sparse', tested: false, testFile: null },
  'torch.sparse.sparse_coo_tensor': { category: 'sparse', tested: false, testFile: null },
  'torch.sparse.to_sparse': { category: 'sparse', tested: false, testFile: null },
  'torch.sparse.sparse_add': { category: 'sparse', tested: false, testFile: null },
  'torch.sparse.sparse_mul': { category: 'sparse', tested: false, testFile: null },
  'torch.sparse.sparse_matmul': { category: 'sparse', tested: false, testFile: null },
  'torch.sparse.sparse_eye': { category: 'sparse', tested: false, testFile: null },

  // special
  'torch.special.erfinv': { category: 'special', tested: false, testFile: null },
  'torch.special.logit': { category: 'special', tested: false, testFile: null },
  'torch.special.sinc': { category: 'special', tested: false, testFile: null },
  'torch.special.entr': { category: 'special', tested: false, testFile: null },
  'torch.special.i1': { category: 'special', tested: false, testFile: null },
  'torch.special.i1e': { category: 'special', tested: false, testFile: null },
  'torch.special.xlogy': { category: 'special', tested: false, testFile: null },
  'torch.special.xlog1py': { category: 'special', tested: false, testFile: null },
  'torch.special.multigammaln': { category: 'special', tested: false, testFile: null },
  'torch.special.zeta': { category: 'special', tested: false, testFile: null },
  'torch.special.erfcx': { category: 'special', tested: false, testFile: null },
  'torch.special.expit': { category: 'special', tested: false, testFile: null },
  'torch.special.log_ndtr': { category: 'special', tested: false, testFile: null },
  'torch.special.ndtr': { category: 'special', tested: false, testFile: null },
  'torch.special.ndtri': { category: 'special', tested: false, testFile: null },
  'torch.special.bessel_j0': { category: 'special', tested: false, testFile: null },
  'torch.special.bessel_j1': { category: 'special', tested: false, testFile: null },
  'torch.special.spherical_bessel_j0': { category: 'special', tested: false, testFile: null },
  'torch.special.log_softmax': { category: 'special', tested: false, testFile: null },
  'torch.special.softmax': { category: 'special', tested: false, testFile: null },
};

const fullRegistry = { ...registry, ...moduleApis };

// Sort by key
const sorted = Object.fromEntries(Object.entries(fullRegistry).sort());

// Write
writeFileSync(
  join(__dirname, '_registry.json'),
  JSON.stringify(sorted, null, 2) + '\n',
  'utf8',
);

const total = Object.keys(sorted).length;
const tested = Object.values(sorted).filter((v) => v.tested).length;
console.log(`Registry generated: ${total} APIs tracked, ${tested} covered (${((tested / total) * 100).toFixed(1)}%)`);
