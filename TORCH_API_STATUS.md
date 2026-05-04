# PyTorch API Implementation Status

## Legend
- ✅ Implemented - Full functionality with tests
- 🔶 Partial - Basic functionality, missing some features
- ❌ Not Implemented

---

## Tensor Creation

| API | Status | Notes |
|-----|--------|-------|
| `torch.tensor()` | ✅ | Create tensor from array data |
| `torch.zeros()` | ✅ | Tensor filled with zeros |
| `torch.ones()` | ✅ | Tensor filled with ones |
| `torch.zeros_like()` | ✅ | Zeros with same shape as input |
| `torch.ones_like()` | ✅ | Ones with same shape as input |
| `torch.randn()` | ✅ | Random normal distribution (Box-Muller) |
| `torch.rand()` | ✅ | Random uniform [0, 1) |
| `torch.randint()` | ✅ | Random integers |
| `torch.randperm()` | ✅ | Random permutation |
| `torch.arange()` | ✅ | Range of values |
| `torch.linspace()` | ✅ | Linearly spaced values |
| `torch.logspace()` | ✅ | Logarithmically spaced values |
| `torch.eye()` | ✅ | Identity matrix |
| `torch.empty()` | ✅ | Uninitialized tensor |
| `torch.full()` | ✅ | Tensor filled with value |
| `torch.full_like()` | ✅ | Full with same shape |
| `torch.empty_like()` | ✅ | Empty with same shape |
| `torch.normal()` | ✅ | Normal distribution |
| `torch.as_tensor()` | ✅ | Convert without copy if possible |
| `torch.from_numpy()` | ✅ | Convert from numpy array |
| `torch.scalar_tensor()` | ✅ | Scalar tensor |
| `torch.bincount()` | ✅ | Count frequencies |
| `torch.histc()` | ✅ | Histogram |

## Tensor Properties

| API | Status | Notes |
|-----|--------|-------|
| `tensor.shape` | ✅ | Shape as readonly array |
| `tensor.dtype` | ✅ | Data type |
| `tensor.device` | ✅ | Device (webgpu) |
| `tensor.requires_grad` | ✅ | Gradient tracking flag |
| `tensor.grad` | ✅ | Gradient tensor |
| `tensor.numel()` | ✅ | Number of elements |
| `tensor.dim()` | ✅ | Number of dimensions |
| `tensor.size()` | ✅ | Alias for shape |
| `tensor.ndim` | ✅ | Number of dimensions |

## Element-wise Math Operations

| API | Status | Notes |
|-----|--------|-------|
| `tensor.add()` / `+` | ✅ | Addition (tensor-tensor, tensor-scalar) |
| `tensor.sub()` / `-` | ✅ | Subtraction |
| `tensor.mul()` / `*` | ✅ | Multiplication |
| `tensor.div()` / `/` | ✅ | Division |
| `tensor.pow()` / `**` | ✅ | Power (scalar or tensor exponent) |
| `tensor.neg()` | ✅ | Negation |
| `tensor.abs()` | ✅ | Absolute value |
| `tensor.sqrt()` | ✅ | Square root |
| `tensor.rsqrt()` | ✅ | Reciprocal square root |
| `tensor.exp()` | ✅ | Exponential |
| `tensor.exp2()` | ✅ | Base-2 exponential |
| `tensor.expm1()` | ✅ | exp(x) - 1 |
| `tensor.log()` | ✅ | Natural logarithm |
| `tensor.log2()` | ✅ | Base-2 logarithm |
| `tensor.log10()` | ✅ | Base-10 logarithm |
| `tensor.log1p()` | ✅ | log(1 + x) |
| `tensor.logaddexp()` | ✅ | log(exp(x) + exp(y)) |
| `tensor.sin()` | ✅ | Sine |
| `tensor.cos()` | ✅ | Cosine |
| `tensor.tan()` | ✅ | Tangent |
| `tensor.sinh()` | ✅ | Hyperbolic sine |
| `tensor.cosh()` | ✅ | Hyperbolic cosine |
| `tensor.tanh()` | ✅ | Hyperbolic tangent |
| `tensor.asin()` | ✅ | Arc sine |
| `tensor.acos()` | ✅ | Arc cosine |
| `tensor.atan()` | ✅ | Arc tangent |
| `tensor.atan2()` | ✅ | 2-argument arc tangent |
| `tensor.asinh()` | ✅ | Inverse hyperbolic sine |
| `tensor.acosh()` | ✅ | Inverse hyperbolic cosine |
| `tensor.atanh()` | ✅ | Inverse hyperbolic tangent |
| `tensor.sigmoid()` | ✅ | Sigmoid function |
| `tensor.relu()` | ✅ | ReLU activation |
| `tensor.gelu()` | ✅ | GELU activation (tanh approximation) |
| `tensor.softplus()` | ✅ | Softplus |
| `tensor.silu()` | ✅ | SiLU / Swish |
| `tensor.mish()` | ✅ | Mish |
| `tensor.hardsigmoid()` | ✅ | Hard sigmoid |
| `tensor.hardswish()` | ✅ | Hard swish |
| `tensor.softsign()` | ✅ | Soft sign |
| `tensor.tanhshrink()` | ✅ | Tanh shrink |
| `tensor.leaky_relu()` | ✅ | Leaky ReLU |
| `tensor.elu()` | ✅ | ELU |
| `tensor.selu()` | ✅ | SELU |
| `tensor.glu()` | ✅ | Gated Linear Unit |
| `tensor.threshold()` | ✅ | Threshold |
| `tensor.sign()` | ✅ | Sign |
| `tensor.sgn()` | ✅ | Sign (complex-aware) |
| `tensor.erf()` | ✅ | Error function |
| `tensor.erfc()` | ✅ | Complementary error function |
| `tensor.erfinv()` | ❌ | Inverse error function |
| `tensor.erfcinv()` | ❌ | Inverse complementary error function |
| `tensor.ceil()` | ✅ | Ceil |
| `tensor.floor()` | ✅ | Floor |
| `tensor.round()` | ✅ | Round |
| `tensor.trunc()` | ✅ | Truncate |
| `tensor.fix()` | ✅ | Fix towards zero (alias for trunc) |
| `tensor.frac()` | ✅ | Fractional part |
| `tensor.clamp()` | ✅ | Clamp to [min, max] |
| `tensor.clip()` | ✅ | Alias for clamp |
| `tensor.clamp_min()` | ✅ | Clamp with min only |
| `tensor.clamp_max()` | ✅ | Clamp with max only |
| `tensor.reciprocal()` | ✅ | 1/x |
| `tensor.square()` | ✅ | x^2 |
| `tensor.fmod()` | ✅ | Floating point modulo (NEW) |
| `tensor.remainder()` | ✅ | Remainder (NEW) |
| `tensor.deg2rad()` | ✅ | Degrees to radians |
| `tensor.rad2deg()` | ✅ | Radians to degrees |
| `tensor.logical_not()` | ✅ | Logical NOT |
| `tensor.i0()` | ✅ | Modified Bessel function I0 |
| `tensor.lgamma()` | ✅ | Log-gamma |
| `tensor.digamma()` | ✅ | Digamma (psi) |
| `tensor.hypot()` | ✅ | sqrt(x^2 + y^2) |
| `tensor.heaviside()` | ✅ | Heaviside step |

## Reduction Operations

| API | Status | Notes |
|-----|--------|-------|
| `tensor.sum()` | ✅ | Sum with optional dim/keepdim |
| `tensor.mean()` | ✅ | Mean with optional dim/keepdim |
| `tensor.max()` | ✅ | Max with optional dim/keepdim |
| `tensor.min()` | ✅ | Min with optional dim/keepdim |
| `tensor.argmax()` | ✅ | Index of maximum (now supports dim parameter) |
| `tensor.argmin()` | ✅ | Index of minimum (now supports dim parameter) |
| `tensor.amax()` | ✅ | Max with dim/keepdim |
| `tensor.amin()` | ✅ | Min with dim/keepdim |
| `tensor.prod()` | ✅ | Product with optional dim/keepdim |
| `tensor.std()` | ✅ | Standard deviation (NEW: with unbiased/correction) |
| `tensor.var()` | ✅ | Variance (NEW: with unbiased/correction) |
| `tensor.all()` | ✅ | Logical AND reduction |
| `tensor.any()` | ✅ | Logical OR reduction |
| `tensor.cumsum()` | ✅ | Cumulative sum |
| `tensor.cumprod()` | ✅ | Cumulative product |
| `tensor.cummax()` | ✅ | Cumulative max (NEW) |
| `tensor.cummin()` | ✅ | Cumulative min (NEW) |
| `tensor.logsumexp()` | ✅ | Log sum exp (NEW) |
| `tensor.logcumsumexp()` | ✅ | Log cumulative sum exp (NEW) |
| `tensor.count_nonzero()` | ✅ | Count non-zero elements (NEW) |
| `tensor.aminmax()` | ✅ | Min and max simultaneously (NEW) |

## Matrix Operations

| API | Status | Notes |
|-----|--------|-------|
| `tensor.matmul()` / `@` | ✅ | 2D matrix multiplication (tiled for large matrices) |
| `tensor.mm()` | ✅ | Alias for matmul |
| `tensor.bmm()` | ✅ | Batched matrix multiplication |
| `tensor.baddbmm()` | ✅ | Batched addmm |
| `tensor.addbmm()` | ✅ | Batch addmm |
| `tensor.addmm()` | ✅ | addmm(beta, mat1, mat2, alpha) |
| `tensor.addmv()` | ✅ | Matrix-vector add |
| `tensor.mv()` | ✅ | Matrix-vector multiplication |
| `tensor.dot()` | ✅ | Dot product (1D) |
| `tensor.vdot()` | ✅ | Conjugate dot product |
| `tensor.outer()` | ✅ | Outer product |
| `tensor.addr()` | ✅ | Outer product add |
| `tensor.t()` | ✅ | Transpose (2D) |
| `tensor.T` | ✅ | Transpose property |
| `tensor.transpose()` | ✅ | Transpose arbitrary dims (2D) |
| `tensor.permute()` | ✅ | Permute dimensions |
| `tensor.movedim()` | ✅ | Move dimension |
| `tensor.swapaxes()` | ✅ | Swap axes |

## Shape Operations

| API | Status | Notes |
|-----|--------|-------|
| `tensor.reshape()` | ✅ | Reshape tensor (supports -1 inference) |
| `tensor.view()` | ✅ | Alias for reshape |
| `tensor.squeeze()` | ✅ | Remove size-1 dimensions |
| `tensor.unsqueeze()` | ✅ | Add dimension |
| `tensor.flatten()` | ✅ | Flatten to 1D |
| `tensor.expand()` | ✅ | Expand to larger size (GPU) |
| `tensor.broadcast_to()` | ✅ | Alias for expand |
| `tensor.tile()` | ✅ | Repeat tensor |
| `tensor.repeat()` | ❌ | Missing alias |
| `tensor.contiguous()` | ❌ | Make contiguous (always contiguous currently) |
| `tensor.clone()` | ✅ | Deep copy |
| `tensor.detach()` | ✅ | Detach from autograd |
| `tensor.copy_()` | ✅ | In-place copy |
| `tensor.unflatten()` | ✅ | Unflatten dimension (NEW) |
| `tensor.sort()` | ✅ | Sort elements (NEW) |
| `tensor.argsort()` | ✅ | Indices of sorted (NEW) |
| `tensor.topk()` | ✅ | Top k elements (NEW) |
| `tensor.kthvalue()` | ✅ | k-th smallest value (NEW) |

## Indexing & Slicing

| API | Status | Notes |
|-----|--------|-------|
| `tensor[i]` | ✅ | Basic indexing (get via .get()) |
| `tensor[i] = value` | ✅ | Basic indexing (set via .set()) |
| `tensor[i:j]` | ❌ | Slicing |
| `tensor.index_select()` | ✅ | Select along dimension (GPU) |
| `tensor.select()` | ✅ | Select single index along dim |
| `tensor.take()` | ✅ | Gather elements by indices |
| `tensor.gather()` | ✅ | Gather values |
| `tensor.scatter()` | ✅ | Scatter values |
| `tensor.scatter_add()` | ✅ | Scatter with addition |
| `tensor.masked_select()` | ✅ | Select by mask |
| `tensor.masked_fill()` | ✅ | Fill by mask (GPU) |
| `tensor.where()` | ✅ | Element-wise conditional |
| `tensor.nonzero()` | ✅ | Non-zero indices |
| `tensor.diagonal()` | ✅ | Diagonal extraction |
| `tensor.chunk()` | ✅ | Split into chunks |
| `tensor.narrow()` | ✅ | Narrow view |
| `tensor.unbind()` | ✅ | Remove a dimension |
| `tensor.split()` | ✅ | Split into sections |
| `tensor.tril()` | ✅ | Lower triangular |
| `tensor.triu()` | ✅ | Upper triangular |
| `tensor.diag()` | ✅ | Diagonal (1D<->2D) |
| `tensor.flip()` | ✅ | Reverse order |
| `tensor.fliplr()` | ✅ | Flip left-right |
| `tensor.flipud()` | ✅ | Flip up-down |
| `tensor.repeat_interleave()` | ✅ | Repeat elements |
| `tensor.roll()` | ✅ | Roll tensor |
| `tensor.rot90()` | ✅ | Rotate 90 degrees |
| `tensor.tensor_split()` | ✅ | Split by indices |

## Comparison Operations

| API | Status | Notes |
|-----|--------|-------|
| `tensor.eq()` | ✅ | Element-wise equal |
| `tensor.ne()` | ✅ | Element-wise not equal |
| `tensor.gt()` | ✅ | Greater than |
| `tensor.ge()` | ✅ | Greater than or equal |
| `tensor.lt()` | ✅ | Less than |
| `tensor.le()` | ✅ | Less than or equal |
| `tensor.equal()` | ✅ | All elements equal |
| `tensor.isclose()` | ✅ | Element-wise close |
| `tensor.allclose()` | ✅ | All elements close |
| `tensor.isnan()` | ✅ | NaN test |
| `tensor.isinf()` | ✅ | Infinity test |
| `tensor.isfinite()` | ✅ | Finite test |
| `tensor.isposinf()` | ✅ | Positive infinity test |
| `tensor.isneginf()` | ✅ | Negative infinity test |
| `tensor.maximum()` | ✅ | Element-wise max |
| `tensor.minimum()` | ✅ | Element-wise min |
| `tensor.fmax()` | ✅ | NaN-avoiding max |
| `tensor.fmin()` | ✅ | NaN-avoiding min |

## Autograd

| API | Status | Notes |
|-----|--------|-------|
| `tensor.backward()` | 🔶 | Basic structure, grad_fn not fully wired |
| `tensor.requires_grad_()` | ✅ | Set requires_grad in-place |
| `torch.no_grad()` | ✅ | Context manager, decorator, static run() |
| `torch.enable_grad()` | ✅ | Context manager, decorator, static run() |
| `torch.inference_mode()` | ✅ | Context manager, decorator, static run() |
| `tensor.grad_fn` | 🔶 | Property exists, grad functions on some ops |
| `tensor.is_leaf` | ✅ | Is leaf tensor |
| `tensor.retain_grad()` | ❌ | Retain non-leaf gradient |

## nn.Module

| API | Status | Notes |
|-----|--------|-------|
| `nn.Module` | ✅ | Base module class |
| `module.forward()` | ✅ | Abstract forward pass |
| `module.call()` | ✅ | Alias for forward |
| `module.parameters()` | ✅ | Parameter iterator |
| `module.named_parameters()` | ✅ | Named parameter iterator |
| `module.modules()` | ✅ | Module iterator |
| `module.children()` | ✅ | Child module iterator |
| `module.train()` | ✅ | Set training mode |
| `module.eval()` | ✅ | Set evaluation mode |
| `module.state_dict()` | ✅ | Get state dictionary |
| `module.load_state_dict()` | ✅ | Load state dictionary |
| `module.zero_grad()` | ✅ | Zero gradients |
| `module.register_buffer()` | ✅ | Register buffer |
| `module.register_parameter()` | ✅ | Register parameter |
| `module.add_module()` | ✅ | Register submodule |
| `module.to()` | ❌ | Move to device |

## nn Layers

| API | Status | Notes |
|-----|--------|-------|
| `nn.Linear` | ✅ | Fully connected layer |
| `nn.Conv1d` | ✅ | 1D convolution (GPU, via reshape to conv2d) |
| `nn.Conv2d` | ✅ | 2D convolution (GPU shader, supports groups) |
| `nn.ConvTranspose1d` | ✅ | 1D transposed convolution |
| `nn.ConvTranspose2d` | ✅ | 2D transposed convolution |
| `nn.ConvTranspose3d` | ✅ | 3D transposed convolution |
| `nn.MaxPool2d` | ✅ | 2D max pooling (GPU shader, supports dilation) |
| `nn.AvgPool2d` | ✅ | 2D average pooling (GPU shader, count_include_pad) |
| `nn.BatchNorm1d` | ✅ | 1D batch normalization (GPU shader, CPU stats) |
| `nn.BatchNorm2d` | ✅ | 2D batch normalization (GPU shader, CPU stats) |
| `nn.InstanceNorm1d` | ✅ | 1D instance normalization |
| `nn.InstanceNorm2d` | ✅ | 2D instance normalization |
| `nn.InstanceNorm3d` | ✅ | 3D instance normalization |
| `nn.GroupNorm` | ✅ | Group normalization |
| `nn.RMSNorm` | ✅ | RMS normalization |
| `nn.LayerNorm` | ✅ | Layer normalization (GPU shader) |
| `nn.MultiheadAttention` | ✅ | Multi-head attention |
| `nn.Upsample` | ✅ | Generic upsampling layer |
| `nn.UpsamplingNearest2d` | ✅ | Nearest neighbor upsampling 2D |
| `nn.UpsamplingBilinear2d` | ✅ | Bilinear upsampling 2D |
| `nn.Dropout` | ✅ | Dropout |
| `nn.Dropout2d` | ✅ | 2D dropout |
| `nn.Embedding` | ✅ | Embedding layer (GPU shader) |
| `nn.Flatten` | ✅ | Flatten layer |
| `nn.Parameter` | ✅ | Learnable parameter |
| `nn.Sequential` | ✅ | Sequential container |
| `nn.ModuleList` | ✅ | Module list |
| `nn.ModuleDict` | ✅ | Module dictionary |

## nn Activations

| API | Status | Notes |
|-----|--------|-------|
| `nn.ReLU` | ✅ | ReLU activation |
| `nn.GELU` | ✅ | GELU activation |
| `nn.Sigmoid` | ✅ | Sigmoid activation |
| `nn.Tanh` | ✅ | Tanh activation |
| `nn.Softmax` | ✅ | Softmax activation |
| `nn.LogSoftmax` | ✅ | Log softmax |

## nn.functional

| API | Status | Notes |
|-----|--------|-------|
| `F.relu()` | ✅ | ReLU function |
| `F.gelu()` | ✅ | GELU function |
| `F.sigmoid()` | ✅ | Sigmoid function |
| `F.softmax()` | ✅ | Softmax function |
| `F.log_softmax()` | ✅ | Log softmax (GPU shader) |
| `F.cross_entropy()` | ✅ | Cross entropy loss (GPU shader) |
| `F.nll_loss()` | ✅ | NLL loss (GPU shader) |
| `F.mse_loss()` | ✅ | MSE loss (via tensor ops) |
| `F.binary_cross_entropy()` | ✅ | BCE loss (via tensor ops) |
| `F.binary_cross_entropy_with_logits()` | ✅ | BCE with logits (via tensor ops) |
| `F.smooth_l1_loss()` | ✅ | Smooth L1 loss (via tensor ops) |
| `F.l1_loss()` | ✅ | L1 loss (via tensor ops) |
| `F.linear()` | ✅ | Linear function |
| `F.conv2d()` | ✅ | 2D convolution (GPU shader) |
| `F.dropout()` | ✅ | Dropout function |
| `F.leaky_relu()` | ✅ | Leaky ReLU |
| `F.elu()` | ✅ | ELU |
| `F.selu()` | ✅ | SELU |
| `F.glu()` | ✅ | Gated Linear Unit |
| `F.threshold()` | ✅ | Threshold |
| `F.softplus()` | ✅ | Softplus |
| `F.silu()` | ✅ | SiLU/Swish |
| `F.mish()` | ✅ | Mish |
| `F.hardsigmoid()` | ✅ | Hard sigmoid |
| `F.hardswish()` | ✅ | Hard swish |
| `F.softsign()` | ✅ | Soft sign |
| `F.tanhshrink()` | ✅ | Tanh shrink |
| `F.tanh()` | ✅ | Tanh |
| `F.sigmoid()` | ✅ | Sigmoid |
| `F.max_pool2d()` | ✅ | 2D max pooling (GPU shader, supports dilation) |
| `F.avg_pool2d()` | ✅ | 2D average pooling (GPU shader, count_include_pad) |

## Optimizers

| API | Status | Notes |
|-----|--------|-------|
| `optim.SGD` | ✅ | Stochastic gradient descent |
| `optim.Adam` | ✅ | Adam optimizer |
| `optim.AdamW` | ✅ | AdamW optimizer |
| `optimizer.step()` | ✅ | Update parameters |
| `optimizer.zero_grad()` | ✅ | Zero gradients |

## Data Utilities

| API | Status | Notes |
|-----|--------|-------|
| `torch.utils.data.Dataset` | ❌ | Dataset base class |
| `torch.utils.data.DataLoader` | ❌ | Data loader |
| `datasets.MNIST` | ❌ | MNIST dataset |
| `datasets.CIFAR10` | ❌ | CIFAR-10 dataset |

## Serialization

| API | Status | Notes |
|-----|--------|-------|
| `torch.save()` | ✅ | Save model/tensor |
| `torch.load()` | ✅ | Load model/tensor |

## Utility Functions

| API | Status | Notes |
|-----|--------|-------|
| `torch.init()` | ✅ | Initialize WebGPU |
| `torch.syncDevice()` | ✅ | Synchronize WebGPU device |
| `tensor.toArray()` | ✅ | Convert to JS array (async) |
| `tensor.toNestedArray()` | ✅ | Convert to nested JS array (async) |
| `tensor.item()` | ✅ | Get scalar value (async) |
| `tensor.toString()` | ✅ | String representation |
| `torch.manual_seed()` | ✅ | Set random seed |
| `torch.stack()` | ✅ | Stack tensors |
| `torch.cat()` | ✅ | Concatenate tensors |
| `torch.split()` | ✅ | Split tensor |
| `torch.chunk()` | ✅ | Chunk into pieces |
| `torch.tensor_split()` | ✅ | Split by indices (NEW) |
| `torch.vstack()` | ✅ | Vertical stack |
| `torch.hstack()` | ✅ | Horizontal stack |
| `torch.dstack()` | ✅ | Depth stack |
| `torch.column_stack()` | ✅ | Column stack |
| `torch.atleast_1d()` | ✅ | At least 1D |
| `torch.atleast_2d()` | ✅ | At least 2D |
| `torch.atleast_3d()` | ✅ | At least 3D |
| `torch.triu()` | ✅ | Upper triangular |
| `torch.tril()` | ✅ | Lower triangular |
| `torch.diag()` | ✅ | Diagonal |
| `torch.trapezoid()` | ✅ | Trapezoidal integration |
| `torch.cumulative_trapezoid()` | ✅ | Cumulative trapezoidal |
| `torch.meshgrid()` | ✅ | Coordinate grids (NEW) |
| `torch.cartesian_prod()` | ✅ | Cartesian product (NEW) |
| `torch.combinations()` | ✅ | Element combinations (NEW) |
| `torch.trace()` | ✅ | Matrix trace (NEW) |
| `torch.unravel_index()` | ✅ | Flat to multi-index (NEW) |
| `torch.nonzero()` | ✅ | Non-zero indices (NEW) |
| `torch.masked_select()` | ✅ | Select by mask (NEW) |
| `torch.gather()` | ✅ | Gather values (NEW) |
| `torch.scatter()` | ✅ | Scatter values (NEW) |
| `torch.scatter_add()` | ✅ | Scatter with addition (NEW) |
| `torch.sort()` | ✅ | Sort elements (NEW) |
| `torch.argsort()` | ✅ | Indices of sorted (NEW) |
| `torch.topk()` | ✅ | Top k elements (NEW) |
| `torch.kthvalue()` | ✅ | k-th smallest value (NEW) |
| `torch.std()` | ✅ | Standard deviation (NEW) |
| `torch.var()` | ✅ | Variance (NEW) |
| `torch.logsumexp()` | ✅ | Log sum exp (NEW) |
| `torch.count_nonzero()` | ✅ | Count non-zero (NEW) |
| `torch.aminmax()` | ✅ | Min and max simultaneously (NEW) |
| `torch.fmod()` | ✅ | Floating point modulo |
| `torch.remainder()` | ✅ | Remainder |
| `torch.clip()` | ✅ | Alias for clamp |
| `torch.clamp_min()` | ✅ | Clamp with min only |
| `torch.clamp_max()` | ✅ | Clamp with max only |
| `torch.einsum()` | ✅ | Einstein summation |
| `torch.unique()` | ✅ | Unique elements |
| `torch.unique_consecutive()` | ✅ | Unique consecutive elements |
| `torch.isin()` | ✅ | Element-wise set membership |

## Linear Algebra

| API | Status | Notes |
|-----|--------|-------|
| `linalg.cholesky()` | ✅ | Cholesky decomposition (GPU) |
| `linalg.lu_factor()` | ✅ | LU factorization (GPU) |
| `linalg.lu_solve()` | ✅ | LU solve (GPU) |
| `linalg.inv()` | ✅ | Matrix inverse via LU |
| `linalg.det()` | ✅ | Determinant via LU |
| `linalg.svd()` | ❌ | Singular value decomposition |
| `linalg.eig()` | ❌ | Eigenvalue decomposition |
| `linalg.eigvals()` | ❌ | Eigenvalues only |
| `linalg.qr()` | ❌ | QR decomposition |
| `linalg.solve()` | ❌ | Solve linear systems |
| `linalg.matrix_power()` | ❌ | Matrix power |
| `linalg.matrix_rank()` | ❌ | Matrix rank |
| `linalg.norm()` | ❌ | Matrix norm |

## FFT

| API | Status | Notes |
|-----|--------|-------|
| `torch.fft.fft()` | ✅ | 1D FFT (CPU fallback via toArray) |
| `torch.fft.ifft()` | ✅ | 1D IFFT |
| `torch.fft.fft2()` | ✅ | 2D FFT |
| `torch.fft.fftn()` | ✅ | N-dimensional FFT |
| `torch.fft.rfft()` | ✅ | Real input FFT |
| `torch.fft.irfft()` | ✅ | Inverse real FFT |
| `torch.fft.fftshift()` | ✅ | Shift zero frequency to center |
| `torch.fft.ifftshift()` | ✅ | Inverse fftshift |
| `torch.fft.fftfreq()` | ✅ | FFT sample frequencies |
| `torch.fft.rfftfreq()` | ✅ | Real FFT sample frequencies |

## Special Functions (torch.special)

| API | Status | Notes |
|-----|--------|-------|
| `torch.special.erfinv()` | ✅ | Inverse error function (CPU fallback) |
| `torch.special.logit()` | ✅ | Logit function with eps clipping |
| `torch.special.sinc()` | ✅ | Sinc function: sin(πx)/(πx) |
| `torch.special.entr()` | ✅ | Entropy: -x*log(x) (CPU fallback) |
| `torch.special.i1()` | ✅ | Bessel function order 1 (CPU fallback) |
| `torch.special.i1e()` | ✅ | Scaled Bessel i1 (CPU fallback) |
| `torch.special.xlogy()` | ✅ | x * log(y) |
| `torch.special.xlog1py()` | ✅ | x * log1p(y) |
| `torch.special.multigammaln()` | ✅ | Multivariate log-gamma (CPU fallback) |
| `torch.special.zeta()` | ✅ | Riemann zeta (CPU fallback) |
| `torch.special.erfcx()` | ✅ | Scaled erfc: exp(x²)*erfc(x) |
| `torch.special.expit()` | ✅ | Sigmoid function |
| `torch.special.log_ndtr()` | ✅ | Log normal CDF |
| `torch.special.ndtr()` | ✅ | Normal CDF |
| `torch.special.ndtri()` | ✅ | Inverse normal CDF (CPU fallback) |
| `torch.special.bessel_j0()` | ✅ | Bessel J0 (CPU fallback) |
| `torch.special.bessel_j1()` | ✅ | Bessel J1 (CPU fallback) |
| `torch.special.spherical_bessel_j0()` | ✅ | Spherical Bessel j0 |
| `torch.special.log_softmax()` | ✅ | Log softmax via nn.functional |
| `torch.special.softmax()` | ✅ | Softmax via nn.functional |

## Neural Network Modules (nn)

| API | Status | Notes |
|-----|--------|-------|
| `nn.Linear` | ✅ | Fully connected layer |
| `nn.Conv1d/2d` | ✅ | Convolution layers |
| `nn.ConvTranspose1d/2d/3d` | ✅ | Transposed convolutions |
| `nn.MaxPool1d/2d/3d` | ✅ | Max pooling (1D CPU fallback, 3D stub) |
| `nn.AvgPool1d/2d/3d` | ✅ | Average pooling (3D stub) |
| `nn.AdaptiveAvgPool1d/3d` | 🔶 | Adaptive pooling (1D implemented, 3D stub) |
| `nn.BatchNorm1d/2d` | ✅ | Batch normalization |
| `nn.InstanceNorm1d/2d/3d` | ✅ | Instance normalization |
| `nn.GroupNorm` | ✅ | Group normalization |
| `nn.RMSNorm` | ✅ | RMS normalization |
| `nn.LayerNorm` | ✅ | Layer normalization |
| `nn.Dropout/Dropout2d` | ✅ | Dropout regularization |
| `nn.Embedding` | ✅ | Embedding layer |
| `nn.RNN/LSTM/GRU` | ✅ | Recurrent layers |
| `nn.MultiheadAttention` | 🔶 | Multi-head attention (masks not yet supported) |
| `nn.TransformerEncoder` | ✅ | Transformer encoder |
| `nn.TransformerDecoder` | ✅ | Transformer decoder |
| `nn.TransformerEncoderLayer` | ✅ | Single encoder layer |
| `nn.TransformerDecoderLayer` | ✅ | Single decoder layer |
| `nn.ConstantPad1d/2d/3d` | 🔶 | Constant padding (3D stub) |
| `nn.ZeroPad2d` | ✅ | Zero padding |
| `nn.ReflectionPad2d` | ✅ | Reflection padding (CPU fallback) |
| `nn.ReplicationPad2d` | ❌ | Replication padding |
| `nn.Upsample` | ✅ | Upsampling |
| `nn.UpsamplingNearest2d` | ✅ | Nearest neighbor upsampling |
| `nn.UpsamplingBilinear2d` | ✅ | Bilinear upsampling |
| `nn.Sequential` | ✅ | Sequential container |
| `nn.ModuleList/ModuleDict` | ✅ | Module containers |

## WebGPU Features

| API | Status | Notes |
|-----|--------|-------|
| `torch.webgpu` | ✅ | WebGPU backend module |
| `webgpu.setBufferSize()` | ✅ | Set pool buffer size |
| `webgpu.reportUsedMemory()` | ✅ | Report memory usage |
| `webgpu.getDevice()` | ✅ | Get WebGPU device |
| `webgpu.destroy()` | ✅ | Destroy WebGPU resources |
