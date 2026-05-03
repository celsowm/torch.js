# torch.js vs PyTorch — API Comparison

> This document compares the torch.js library (a browser-first, WebGPU-accelerated tensor library) against the official PyTorch Python library.

## Overview

| Aspect | PyTorch (Official) | torch.js |
|--------|-------------------|----------|
| Language | Python + C++ (ATen) | TypeScript |
| Backend | CPU, CUDA, MPS, XPU, MTIA | WebGPU |
| Target | Servers, workstations, GPUs | Browser, Node.js |
| Size | ~1000+ modules | ~53 WGSL shaders, ~40 source files |
| Tensor storage | C++ native arrays | GPU buffers via TypeGPU |

---

## Implemented Features

### Core Tensor Operations

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| Tensor creation | ✅ `torch.tensor()` | ✅ `torch.tensor()` | Nested array support |
| Zeros / Ones / Full | ✅ | ✅ | `zeros`, `ones`, `full` + `_like` variants |
| Random | ✅ `rand`, `randn`, `randint`, `randperm` | ✅ `rand`, `randn` | Box-Muller for normal |
| Arange / Linspace / Logspace | ✅ | ✅ | |
| Eye / Diag | ✅ | ✅ | |
| Tril / Triu | ✅ | ✅ | |
| Manual seed | ✅ | ✅ | |
| Clone / Detach / Copy | ✅ | ✅ | |
| to(device/dtype) | ✅ | ⚠️ Partial | dtype conversion limited |
| destroy() | N/A (GC) | ✅ | Explicit GPU buffer release |

### Element-wise Math (Pointwise)

| Category | PyTorch | torch.js | Notes |
|----------|---------|----------|-------|
| Basic arithmetic | ✅ add, sub, mul, div | ✅ add, sub, mul, div | + aliases (subtract, multiply, divide) |
| Trigonometric | ✅ sin, cos, tan, sinh, cosh, tanh | ✅ All 6 | |
| Inverse trig | ✅ asin, acos, atan, asinh, acosh, atanh | ✅ All 6 | + aliases (arcsin, arccos, etc.) |
| atan2 | ✅ | ✅ | |
| Exponential | ✅ exp, exp2 | ✅ | |
| Logarithmic | ✅ log, log10, log2, log1p | ✅ | + logaddexp |
| Rounding | ✅ ceil, floor, round, trunc, frac | ✅ | |
| Power/Root | ✅ sqrt, rsqrt, square, pow, reciprocal | ✅ | |
| Absolute / Negation | ✅ abs, neg | ✅ | + aliases (absolute, negative) |
| Activations | ✅ sigmoid, relu, gelu, elu, selu, silu, mish, hardsigmoid, hardswish, softplus, softsign, tanhshrink, glu | ✅ All | Tensor methods + nn.functional |
| Comparison | ✅ eq, ne, lt, le, gt, ge | ✅ | + aliases (equal, not_equal, less, etc.) |
| Min/Max (element-wise) | ✅ minimum, maximum, fmin, fmax | ✅ | |
| Bitwise | ✅ and, or, xor | ✅ | |
| Special checks | ✅ isnan, isinf, isfinite, isposinf, isneginf | ✅ | |
| where | ✅ | ✅ | |
| clamp / clip | ✅ | ✅ | |
| masked_fill | ✅ | ✅ | |

### Reduction Operations

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| sum, mean, prod | ✅ | ✅ | dim + keepdim support |
| max, min, amax, amin | ✅ | ✅ | Returns values (+ indices for max/min) |
| argmax, argmin | ✅ | ✅ | dim + keepdim support |
| all, any | ✅ | ✅ | |
| var, std | ✅ | ✅ | Bessel correction support |
| equal, isclose, allclose | ✅ | ✅ | Async in torch.js |

### Shape / Indexing Operations

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| reshape, view | ✅ | ✅ | |
| flatten, squeeze, unsqueeze | ✅ | ✅ | |
| permute, transpose | ✅ | ✅ | + `T` property |
| transpose(dim0, dim1) | ✅ | ✅ | |
| swapaxes, swapdims | ✅ | ✅ | |
| movedim, moveaxis | ✅ | ✅ | |
| narrow, slice, select | ✅ | ✅ | Python-style slicing supported |
| split, chunk, unbind | ✅ | ✅ | |
| tile, repeat, expand, broadcast_to | ✅ | ✅ | |
| index_select, take | ✅ | ✅ | |
| flip, fliplr, flipud | ✅ | ✅ | |
| cumsum, cumprod | ✅ | ✅ | |
| diag (vec↔matrix) | ✅ | ✅ | |

### Concatenation / Stacking

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| cat | ✅ | ✅ | |
| stack | ✅ | ✅ | |
| vstack, hstack, dstack, column_stack | ✅ | ✅ | |
| atleast_1d, atleast_2d, atleast_3d | ✅ | ✅ | |

### Matrix / Linear Algebra

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| matmul, mm, bmm | ✅ | ✅ | 1D, 2D, 3D batch support |
| mv (matrix-vector) | ✅ | ✅ | |
| dot, vdot, inner, outer | ✅ | ✅ | + aliases (ger, vdot) |
| addmm, addmv, addr, baddbmm, addbmm | ✅ | ✅ | With alpha/beta |
| t() | ✅ | ✅ | 2D transpose |
| **linalg.lu_factor** | ✅ | ✅ | With pivoting |
| **linalg.solve_triangular** | ✅ | ✅ | |
| **linalg.cholesky** | ✅ | ✅ | |
| **linalg.norm** | ✅ | ✅ | vector + matrix norms |
| **linalg.vector_norm, matrix_norm** | ✅ | ✅ | |
| **linalg.diagonal** | ✅ | ✅ | |
| **linalg.matmul** | ✅ | ✅ | |
| **linalg.vecdot** | ✅ | ✅ | |
| **linalg.vander** | ✅ | ✅ | Vandermonde matrix |
| **linalg.matrix_power** | ✅ | ✅ | |
| **linalg.det** | ✅ | ✅ | Via LU decomposition |
| **linalg.inv** | ✅ | ⚠️ Partial | Pending permutation fix |
| linalg.cross | ✅ | ❌ | Not implemented |
| linalg.svd, eig, eigh, qr | ✅ | ❌ | |
| linalg.solve, lstsq, pinverse | ✅ | ❌ | |
| linalg.slogdet, matrix_rank | ✅ | ❌ | |

### Neural Network — `torch.nn`

#### Modules

| Module | PyTorch | torch.js | Notes |
|--------|---------|----------|-------|
| **Module** (base class) | ✅ | ✅ | `forward()`, `parameters()`, `state_dict()`, `train()`, `eval()`, `zero_grad()`, `register_parameter`, `register_buffer`, `add_module`, `get_submodule`, `get_parameter`, `get_buffer`, `children()`, `modules()`, `named_parameters()`, `named_buffers()`, `load_state_dict()` |
| **Parameter** | ✅ | ✅ | Auto `requires_grad=true`, `create()` factory |
| **Linear** | ✅ | ✅ | Kaiming initialization |
| **Sequential** | ✅ | ✅ | `get(i)`, `length` |
| **ModuleList** | ✅ | ✅ | `append`, `get`, `length`, iterator |
| **ModuleDict** | ✅ | ✅ | `get`, `set`, `has`, `keys`, `values`, `entries`, `size` |
| **Flatten** | ✅ | ✅ | start_dim, end_dim |
| **Embedding** | ✅ | ✅ | GPU-accelerated lookup |
| **LayerNorm** | ✅ | ✅ | elementwise_affine support |
| Conv1d/2d/3d | ✅ | ❌ | |
| ConvTranspose* | ✅ | ❌ | |
| MaxPool/AvgPool/LPPool | ✅ | ❌ | `F.max_pool2d` stub exists |
| AdaptiveAvgPool/MaxPool | ✅ | ❌ | |
| BatchNorm1d/2d/3d | ✅ | ❌ | |
| InstanceNorm* | ✅ | ❌ | |
| GroupNorm | ✅ | ❌ | |
| RNN/LSTM/GRU + Cells | ✅ | ❌ | |
| Transformer + MultiheadAttention | ✅ | ❌ | |
| Upsample/Interpolate | ✅ | ❌ | |
| Pad (Constant/Reflection/Replication) | ✅ | ❌ | |
| Fold/Unfold | ✅ | ❌ | |
| PixelShuffle/PixelUnshuffle | ✅ | ❌ | |
| CosineSimilarity/PairwiseDistance | ✅ | ❌ | |

#### Activations

| Activation | PyTorch | torch.js | Notes |
|------------|---------|----------|-------|
| ReLU | ✅ | ✅ | Class + functional |
| Sigmoid | ✅ | ✅ | Class + functional |
| Tanh | ✅ | ✅ | Class + functional |
| Softmax | ✅ | ✅ | Class + functional (with gradient) |
| LogSoftmax | ✅ | ✅ | Class + functional (with gradient) |
| GELU | ✅ | ✅ | Class + functional |
| LeakyReLU | ✅ | ✅ | Functional |
| ELU | ✅ | ✅ | Functional |
| SELU | ✅ | ✅ | Functional |
| Softplus | ✅ | ✅ | Functional |
| SiLU | ✅ | ✅ | Functional |
| Mish | ✅ | ✅ | Functional |
| HardSigmoid | ✅ | ✅ | Functional |
| HardSwish | ✅ | ✅ | Functional |
| Softsign | ✅ | ✅ | Functional |
| TanhShrink | ✅ | ✅ | Functional |
| GLU | ✅ | ✅ | Functional + tensor method |
| Dropout/Dropout2d | ✅ | ✅ | Functional |
| PReLU, CELU, RReLU, Hardshrink, Softshrink, Softmin, LogSigmoid, Threshold, MultiheadAttention | ✅ | ❌ | |

#### Loss Functions

| Loss | PyTorch | torch.js | Notes |
|------|---------|----------|-------|
| NLLLoss | ✅ | ✅ | Functional + backward GPU |
| CrossEntropyLoss | ✅ | ✅ | Functional (log_softmax + nll_loss) |
| MSELoss, BCELoss, CTCLoss, SmoothL1Loss, HuberLoss, KLDivLoss, CosineEmbeddingLoss, TripletMarginLoss, HingeEmbeddingLoss, MarginRankingLoss, MultiLabelMarginLoss, PoissonNLLLoss, SoftMarginLoss, MultiMarginLoss, GaussianNLLLoss, L1Loss | ✅ | ❌ | Only NLL + CrossEntropy implemented |

#### `torch.nn.functional`

| Function | PyTorch | torch.js | Notes |
|----------|---------|----------|-------|
| F.relu, F.gelu, F.sigmoid, F.tanh | ✅ | ✅ | |
| F.softmax, F.log_softmax | ✅ | ✅ | With gradients |
| F.nll_loss, F.cross_entropy | ✅ | ✅ | With backward |
| F.dropout | ✅ | ✅ | |
| F.leaky_relu, F.elu, F.selu, F.silu, F.mish, F.hardsigmoid, F.hardswish, F.softplus, F.softsign, F.tanhshrink, F.glu | ✅ | ✅ | |
| F.conv2d | ✅ | ❌ | Stub (throws) |
| F.max_pool2d | ✅ | ❌ | Stub (throws) |
| F.linear, F.embedding, F.layer_norm | ✅ | ❌ | Use module classes instead |
| F.threshold | ✅ | ❌ | Stub (throws) |
| F.interpolate, F.pad, F.unfold, F.grid_sample | ✅ | ❌ | |

#### `torch.nn.init`

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| Kaiming uniform/normal | ✅ | ✅ | Used internally by Linear |
| xavier_uniform, xavier_normal | ✅ | ❌ | |
| orthogonal, constant, dirac | ✅ | ❌ | |
| uniform_, normal_, zeros_, ones_ | ✅ | ❌ | |
| trunc_normal_, sparse_ | ✅ | ❌ | |

### Optimizers — `torch.optim`

| Optimizer | PyTorch | torch.js | Notes |
|-----------|---------|----------|-------|
| SGD | ✅ | ✅ | Momentum, Nesterov, weight_decay, dampening |
| Adam | ✅ | ✅ | AMSGrad pending |
| AdamW | ✅ | ✅ | Decoupled weight decay, AMSGrad pending |
| Adagrad | ✅ | ❌ | |
| RMSprop | ✅ | ❌ | |
| LBFGS | ✅ | ❌ | |
| Adadelta | ✅ | ❌ | |
| NAdam | ✅ | ❌ | |
| RAdam | ✅ | ❌ | |
| ASGD | ✅ | ❌ | |
| Adamax | ✅ | ❌ | |
| SparseAdam | ✅ | ❌ | |
| Rprop | ✅ | ❌ | |
| Muon | ✅ | ❌ | |
| Adafactor | ✅ | ❌ | |

#### Learning Rate Schedulers

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, CyclicLR, LambdaLR, MultiplicativeLR, ExponentialLR, SequentialLR, ChainedScheduler, ConstantLR, LinearLR, PolynomialLR | ✅ | ❌ | |
| Stochastic Weight Averaging (SWA) | ✅ | ❌ | |

### Autograd

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| requires_grad flag | ✅ | ✅ | |
| backward() | ✅ | ✅ | Async in torch.js |
| grad_fn (gradient functions) | ✅ | ✅ | Registered per operation |
| is_leaf | ✅ | ✅ | |
| accumulate_grad | ✅ | ✅ | |
| Gradient: add/sub/mul/div | ✅ | ✅ | |
| Gradient: matmul (2D) | ✅ | ✅ | |
| Gradient: reshape/transpose/permute | ✅ | ✅ | |
| Gradient: sum/mean | ✅ | ✅ | |
| Gradient: log_softmax, nll_loss | ✅ | ✅ | |
| Gradient: scalar ops | ✅ | ✅ | |
| Gradient: broadcast | ✅ | ✅ | Backprop with broadcasting |
| torch.no_grad() | ✅ | ❌ | |
| torch.enable_grad() | ✅ | ❌ | |
| torch.inference_mode() | ✅ | ❌ | |
| torch.autograd.grad() | ✅ | ❌ | |
| torch.autograd.functional (vjp, jvp, hessian, jacobian) | ✅ | ❌ | |
| retain_grad() | ✅ | ❌ | |
| detect_anomaly() | ✅ | ❌ | |
| forward-mode AD | ✅ | ❌ | |
| custom autograd.Function | ✅ | ❌ | |

### Data Loading (`torch.utils.data`)

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| DataLoader | ✅ | ❌ | |
| Dataset (Map/Iterable) | ✅ | ❌ | |
| Sampler (Sequential, Random, Subset, Weighted, Distributed) | ✅ | ❌ | |
| random_split, ConcatDataset, Subset | ✅ | ❌ | |
| DataLoader workers (multiprocessing) | ✅ | ❌ | N/A in browser |
| collate_fn | ✅ | ❌ | |

### Device / Backend

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| CPU tensors | ✅ | ❌ | All tensors are WebGPU |
| CUDA tensors | ✅ | ❌ | |
| MPS (Apple Silicon) | ✅ | ❌ | |
| WebGPU | ❌ | ✅ | Primary backend |
| GPU capabilities detection | ✅ | ✅ | `getCapabilities()` |
| Buffer pooling | ✅ (CUDA caching allocator) | ✅ | Custom pool |
| Pipeline caching | ✅ | ✅ | `getOrCreatePipeline` |
| timestamp-query profiling | ✅ | ✅ | When supported |
| customGpu provider (Node.js) | N/A | ✅ | Extension point |

### Utilities

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| Shape utils (numel, infer, validate) | ✅ | ✅ | `torch.utils.shape` |
| Broadcast utils | ✅ | ✅ | `torch.utils.broadcast` |
| Benchmark utils (Timer, Fuzzer, compare) | ✅ | ✅ | `torch.utils.benchmark` |
| Profiler | ✅ `torch.profiler` | ✅ `torch.profiler` | Kineto-style tables |
| Serialization (save/load) | ✅ | ⚠️ Partial | Browser-focused |
| Testing utils (assert_close) | ✅ | ❌ | |
| DLPack interop | ✅ | ❌ | |
| Checkpointing (activation) | ✅ | ❌ | |
| FLOP counter | ✅ | ❌ | |
| Deterministic flags | ✅ | ❌ | |
| TensorBoard integration | ✅ | ❌ | |
| C++ extension build | ✅ | ❌ | N/A in JS |

### Special Math (`torch.special`)

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| ~50 special functions (erf, gamma, digamma, bessel, etc.) | ✅ | ❌ | |

### FFT (`torch.fft`)

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| fft, ifft, fft2, fftn, rfft, irfft, etc. | ✅ | ❌ | |

### Distributions (`torch.distributions`)

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| ~45 distributions (Normal, Beta, Gamma, Poisson, etc.) | ✅ | ❌ | Only rand/randn for tensor creation |
| KL divergences | ✅ | ❌ | |
| Constraints | ✅ | ❌ | |

### Distributed / Parallel (`torch.distributed`, `torch.nn.parallel`)

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| DDP, FSDP | ✅ | ❌ | |
| Process groups (NCCL, Gloo, MPI) | ✅ | ❌ | |
| RPC framework | ✅ | ❌ | |
| Collective ops (all_reduce, all_gather, etc.) | ✅ | ❌ | |
| DeviceMesh | ✅ | ❌ | |
| Distributed checkpointing | ✅ | ❌ | |
| Pipeline parallelism | ✅ | ❌ | |
| DataParallel | ✅ | ❌ | |

### Compilation / JIT (`torch.compile`, `torch.jit`, `torch._dynamo`, `torch._inductor`)

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| torch.compile() | ✅ | ❌ | |
| TorchScript (script/trace) | ✅ | ❌ | |
| FX graph transforms | ✅ | ❌ | |
| PT2 export | ✅ | ❌ | |
| Inductor codegen (Triton, C++) | ✅ | ❌ | |
| AOT Autograd | ✅ | ❌ | |

### Quantization (`torch.quantization`, `torch.ao`)

| Feature | PyTorch | torch.js | Notes |
|---------|---------|----------|-------|
| PTQ, QAT, Dynamic quantization | ✅ | ❌ | |
| Observers, fake quantization | ✅ | ❌ | |
| Quantized modules | ✅ | ❌ | |
| Pruning (L1, Random, Structured) | ✅ | ❌ | |

### Other Major Modules

| Module | PyTorch | torch.js | Notes |
|--------|---------|----------|-------|
| torch.sparse (sparse tensors) | ✅ | ❌ | |
| torch.nested (ragged tensors) | ✅ | ❌ | |
| torch.masked (masked tensors) | ✅ | ❌ | |
| torch.func (vmap, functional transforms) | ✅ | ❌ | |
| torch.onnx (ONNX export) | ✅ | ❌ | |
| torch.multiprocessing | ✅ | ❌ | N/A in browser |
| torch.signal (window functions) | ✅ | ❌ | |
| torch.package (model packaging) | ✅ | ❌ | |
| torch._numpy (NumPy compatibility) | ✅ | ❌ | |
| torch.nn.attention (SDPA, flex_attention) | ✅ | ❌ | |
| torch.nn.utils (clip_grad, prune, spectral_norm, weight_norm) | ✅ | ❌ | |

---

## Summary

### Implemented (~70% of core tensor API)

- ✅ **Tensor creation and manipulation** — full coverage
- ✅ **Element-wise math** — ~60+ operations with aliases
- ✅ **Reductions** — sum, mean, max, min, prod, argmax, argmin, var, std, all, any
- ✅ **Shape operations** — reshape, view, permute, transpose, flatten, squeeze, slice, chunk, etc.
- ✅ **Concatenation/stacking** — cat, stack, vstack, hstack, dstack
- ✅ **Matrix operations** — matmul, mm, bmm, mv, dot, outer, addmm, etc.
- ✅ **Linear algebra** — LU, Cholesky, triangular solve, det, inv (partial), norms
- ✅ **Neural network base** — Module, Parameter, Sequential, ModuleList, ModuleDict
- ✅ **NN layers** — Linear, Embedding, LayerNorm, Flatten
- ✅ **Activations** — 15+ activation functions (class + functional)
- ✅ **Loss functions** — NLLLoss, CrossEntropyLoss
- ✅ **Optimizers** — SGD, Adam, AdamW
- ✅ **Autograd** — basic backward with gradients for core ops
- ✅ **WebGPU backend** — with buffer pooling, pipeline caching, capability detection
- ✅ **Profiler** — Kineto-style profiling with formatted tables
- ✅ **Serialization** — browser save/load

### Not Implemented / Gaps

- ❌ **Convolution** — Conv1d/2d/3d, ConvTranspose (stubs exist)
- ❌ **Pooling** — MaxPool, AvgPool, AdaptivePool
- ❌ **RNN/Transformer** — RNN, LSTM, GRU, Transformer, MultiheadAttention
- ❌ **BatchNorm/InstanceNorm/GroupNorm**
- ❌ **Most loss functions** — only NLL + CrossEntropy
- ❌ **Most optimizers** — Adagrad, RMSprop, LBFGS, etc.
- ❌ **LR schedulers**
- ❌ **Advanced autograd** — no_grad, inference_mode, grad(), vjp/jvp, custom Function
- ❌ **DataLoader/Dataset** — data loading pipeline
- ❌ **torch.fft** — Fourier transforms
- ❌ **torch.special** — special math functions
- ❌ **torch.distributions** — probability distributions
- ❌ **torch.sparse/nested/masked** — specialized tensor types
- ❌ **torch.distributed/parallel** — distributed training
- ❌ **torch.compile/JIT/TorchScript** — compilation
- ❌ **torch.quantization** — quantization
- ❌ **torch.onnx** — ONNX export
- ❌ **torch.func/vmap** — functional transforms
- ❌ **Advanced linalg** — SVD, eig, QR, pinverse, lstsq, solve
- ❌ **CPU fallback** — all tensors must live on GPU

---

## Architecture Notes

### torch.js Design Decisions

1. **Browser-first**: All computation happens on GPU via WebGPU. No CPU tensor fallback.
2. **WGSL shaders**: ~53 compute shaders handle all operations (elementwise, reduce, matmul, etc.)
3. **TypeGPU integration**: Uses the TypeGPU library for type-safe GPU buffer management.
4. **Async-heavy**: GPU read-back is async (`toArray()`, `backward()`, `equal()`).
5. **Buffer pooling**: Reuses GPU buffers to avoid allocation overhead.
6. **Dual export**: ESM exports for browser (`./torch.js`) and separate `core` module.

### PyTorch Design Decisions

1. **C++ core (ATen)**: All operations implemented in C++ with Python bindings.
2. **Multi-backend**: CPU, CUDA, MPS, XPU with unified tensor interface.
3. **TorchDynamo + Inductor**: PT2 compilation stack with `torch.compile()`.
4. **Distributed training**: DDP, FSDP, RPC, pipeline parallelism.
5. **Ecosystem**: Quantization, ONNX export, TensorBoard, distributed checkpointing.
