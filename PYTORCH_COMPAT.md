# PyTorch API Compatibility Report

This report tracks the progress of reimplementing the PyTorch API in `torch.js`.

## Overall Porting Progress

- **Total PyTorch APIs**: 1308 (torch: 874, nn: 166, nn_functional: 139, fft: 23, linalg: 42, special: 57, random: 7, signal: 0)
- **Reimplemented & Verified**: 208
- **Porting Completeness**: 15.90%

```
Progress: [██████----------------------------------] 15.9%
```

## Test Fixture Status

Legend:
- ✅ Tested - Has test coverage against PyTorch fixtures
- ❌ Not Tested - Fixture exists but no test coverage yet

## Summary of Tested Fixtures

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ Tested | 166 | 89.2% |
| ❌ Not Tested | 20 | 10.8% |
| **Total** | 186 | 100% |

## Blas

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.addbmm()` / 	orch.Tensor.addbmm()` | ✅ Tested | 1 | 1e-7 |
| 	orch.addmm()` / 	orch.Tensor.addmm()` | ✅ Tested | 1 | 1e-7 |
| 	orch.addmv()` / 	orch.Tensor.addmv()` | ✅ Tested | 1 | 1e-7 |
| 	orch.addr()` / 	orch.Tensor.addr()` | ✅ Tested | 1 | 1e-7 |
| 	orch.baddbmm()` / 	orch.Tensor.baddbmm()` | ✅ Tested | 1 | 1e-7 |
| 	orch.bmm()` / 	orch.Tensor.bmm()` | ✅ Tested | 1 | 1e-7 |
| 	orch.cumulative_trapezoid()` / 	orch.Tensor.cumulative_trapezoid()` | ✅ Tested | 1 | 1e-7 |
| 	orch.dot()` / 	orch.Tensor.dot()` | ✅ Tested | 1 | 1e-7 |
| 	orch.mm()` / 	orch.Tensor.mm()` | ✅ Tested | 1 | 1e-7 |
| 	orch.mv()` / 	orch.Tensor.mv()` | ✅ Tested | 1 | 1e-7 |
| 	orch.outer()` / 	orch.Tensor.outer()` | ✅ Tested | 1 | 1e-7 |
| 	orch.trapezoid()` / 	orch.Tensor.trapezoid()` | ✅ Tested | 1 | 1e-7 |
| 	orch.vdot()` / 	orch.Tensor.vdot()` | ✅ Tested | 1 | 1e-7 |

## Broadcast Ops

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.add()` / 	orch.Tensor.add()` | ✅ Tested | 3 | 1e-7 |

## Comparison

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.eq()` / 	orch.Tensor.eq()` | ✅ Tested | 1 | 1e-7 |
| 	orch.fmax()` / 	orch.Tensor.fmax()` | ✅ Tested | 1 | 1e-7 |
| 	orch.fmin()` / 	orch.Tensor.fmin()` | ✅ Tested | 1 | 1e-7 |
| 	orch.ge()` / 	orch.Tensor.ge()` | ✅ Tested | 1 | 1e-7 |
| 	orch.gt()` / 	orch.Tensor.gt()` | ✅ Tested | 1 | 1e-7 |
| 	orch.isclose()` / 	orch.Tensor.isclose()` | ✅ Tested | 1 | 1e-7 |
| 	orch.isfinite()` / 	orch.Tensor.isfinite()` | ✅ Tested | 1 | 1e-7 |
| 	orch.isinf()` / 	orch.Tensor.isinf()` | ✅ Tested | 1 | 1e-7 |
| 	orch.isnan()` / 	orch.Tensor.isnan()` | ✅ Tested | 1 | 1e-7 |
| 	orch.isneginf()` / 	orch.Tensor.isneginf()` | ✅ Tested | 1 | 1e-7 |
| 	orch.isposinf()` / 	orch.Tensor.isposinf()` | ✅ Tested | 1 | 1e-7 |
| 	orch.le()` / 	orch.Tensor.le()` | ✅ Tested | 1 | 1e-7 |
| 	orch.lt()` / 	orch.Tensor.lt()` | ✅ Tested | 1 | 1e-7 |
| 	orch.maximum()` / 	orch.Tensor.maximum()` | ✅ Tested | 1 | 1e-7 |
| 	orch.minimum()` / 	orch.Tensor.minimum()` | ✅ Tested | 1 | 1e-7 |
| 	orch.ne()` / 	orch.Tensor.ne()` | ✅ Tested | 1 | 1e-7 |

## Creation

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.arange()` / 	orch.Tensor.arange()` | ✅ Tested | 3 | 1e-7 |
| 	orch.empty_like()` / 	orch.Tensor.empty_like()` | ✅ Tested | 1 | 1e-7 |
| 	orch.eye()` / 	orch.Tensor.eye()` | ✅ Tested | 1 | 1e-7 |
| 	orch.full()` / 	orch.Tensor.full()` | ✅ Tested | 1 | 1e-7 |
| 	orch.full_like()` / 	orch.Tensor.full_like()` | ✅ Tested | 1 | 1e-7 |
| 	orch.heaviside()` / 	orch.Tensor.heaviside()` | ✅ Tested | 1 | 1e-7 |
| 	orch.linspace()` / 	orch.Tensor.linspace()` | ✅ Tested | 1 | 1e-7 |
| 	orch.logspace()` / 	orch.Tensor.logspace()` | ✅ Tested | 1 | 1e-7 |
| 	orch.ones()` / 	orch.Tensor.ones()` | ✅ Tested | 1 | 1e-7 |
| 	orch.ones_like()` / 	orch.Tensor.ones_like()` | ✅ Tested | 1 | 1e-7 |
| 	orch.zeros()` / 	orch.Tensor.zeros()` | ✅ Tested | 1 | 1e-7 |
| 	orch.zeros_like()` / 	orch.Tensor.zeros_like()` | ✅ Tested | 1 | 1e-7 |

## Fuzz Tests

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.add()` / 	orch.Tensor.add()` | ✅ Tested | 4 | 1e-7 |
| 	orch.mul()` / 	orch.Tensor.mul()` | ✅ Tested | 3 | 1e-7 |

## Linalg

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.cholesky()` / 	orch.Tensor.cholesky()` | ✅ Tested | 2 | 1e-7 |
| 	orch.cross()` / 	orch.Tensor.cross()` | ❌ Not Tested | 1 | - |
| 	orch.det()` / 	orch.Tensor.det()` | ✅ Tested | 1 | 1e-7 |
| 	orch.inv()` / 	orch.Tensor.inv()` | ❌ Not Tested | 1 | - |
| 	orch.lu_factor()` / 	orch.Tensor.lu_factor()` | ✅ Tested | 1 | 1e-7 |
| 	orch.matrix_norm()` / 	orch.Tensor.matrix_norm()` | ✅ Tested | 1 | 1e-7 |
| 	orch.matrix_power()` / 	orch.Tensor.matrix_power()` | ✅ Tested | 1 | 1e-7 |
| 	orch.norm()` / 	orch.Tensor.norm()` | ✅ Tested | 2 | 1e-7 |
| 	orch.solve_triangular()` / 	orch.Tensor.solve_triangular()` | ✅ Tested | 1 | 1e-7 |
| 	orch.vander()` / 	orch.Tensor.vander()` | ✅ Tested | 1 | 1e-7 |
| 	orch.vecdot()` / 	orch.Tensor.vecdot()` | ✅ Tested | 1 | 1e-7 |
| 	orch.vector_norm()` / 	orch.Tensor.vector_norm()` | ✅ Tested | 2 | 1e-7 |

## Math

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.abs()` / 	orch.Tensor.abs()` | ✅ Tested | 1 | 1e-7 |
| 	orch.acos()` / 	orch.Tensor.acos()` | ✅ Tested | 1 | 1e-7 |
| 	orch.acosh()` / 	orch.Tensor.acosh()` | ✅ Tested | 1 | 1e-7 |
| 	orch.add()` / 	orch.Tensor.add()` | ✅ Tested | 2 | 1e-7 |
| 	orch.asin()` / 	orch.Tensor.asin()` | ✅ Tested | 1 | 1e-7 |
| 	orch.asinh()` / 	orch.Tensor.asinh()` | ✅ Tested | 1 | 1e-7 |
| 	orch.atan()` / 	orch.Tensor.atan()` | ✅ Tested | 1 | 1e-7 |
| 	orch.atan2()` / 	orch.Tensor.atan2()` | ✅ Tested | 1 | 1e-7 |
| 	orch.atanh()` / 	orch.Tensor.atanh()` | ✅ Tested | 1 | 1e-7 |
| 	orch.bitwise_and()` / 	orch.Tensor.bitwise_and()` | ✅ Tested | 1 | 1e-7 |
| 	orch.bitwise_or()` / 	orch.Tensor.bitwise_or()` | ✅ Tested | 1 | 1e-7 |
| 	orch.bitwise_xor()` / 	orch.Tensor.bitwise_xor()` | ✅ Tested | 1 | 1e-7 |
| 	orch.ceil()` / 	orch.Tensor.ceil()` | ✅ Tested | 1 | 1e-7 |
| 	orch.cos()` / 	orch.Tensor.cos()` | ✅ Tested | 1 | 1e-7 |
| 	orch.cosh()` / 	orch.Tensor.cosh()` | ✅ Tested | 1 | 1e-7 |
| 	orch.div()` / 	orch.Tensor.div()` | ✅ Tested | 1 | 1e-7 |
| 	orch.exp()` / 	orch.Tensor.exp()` | ✅ Tested | 1 | 1e-7 |
| 	orch.exp2()` / 	orch.Tensor.exp2()` | ✅ Tested | 1 | 1e-7 |
| 	orch.floor()` / 	orch.Tensor.floor()` | ✅ Tested | 1 | 1e-7 |
| 	orch.frac()` / 	orch.Tensor.frac()` | ✅ Tested | 1 | 1e-7 |
| 	orch.gelu()` / 	orch.Tensor.gelu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.hypot()` / 	orch.Tensor.hypot()` | ✅ Tested | 1 | 1e-7 |
| 	orch.log()` / 	orch.Tensor.log()` | ✅ Tested | 1 | 1e-7 |
| 	orch.log10()` / 	orch.Tensor.log10()` | ✅ Tested | 1 | 1e-7 |
| 	orch.log1p()` / 	orch.Tensor.log1p()` | ✅ Tested | 1 | 1e-7 |
| 	orch.log2()` / 	orch.Tensor.log2()` | ✅ Tested | 1 | 1e-7 |
| 	orch.logaddexp()` / 	orch.Tensor.logaddexp()` | ✅ Tested | 1 | 1e-7 |
| 	orch.mul()` / 	orch.Tensor.mul()` | ✅ Tested | 1 | 1e-7 |
| 	orch.neg()` / 	orch.Tensor.neg()` | ✅ Tested | 1 | 1e-7 |
| 	orch.pow()` / 	orch.Tensor.pow()` | ✅ Tested | 1 | 1e-7 |
| 	orch.reciprocal()` / 	orch.Tensor.reciprocal()` | ✅ Tested | 1 | 1e-7 |
| 	orch.relu()` / 	orch.Tensor.relu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.round()` / 	orch.Tensor.round()` | ✅ Tested | 1 | 1e-7 |
| 	orch.rsqrt()` / 	orch.Tensor.rsqrt()` | ✅ Tested | 1 | 1e-7 |
| 	orch.sigmoid()` / 	orch.Tensor.sigmoid()` | ✅ Tested | 1 | 1e-7 |
| 	orch.sin()` / 	orch.Tensor.sin()` | ✅ Tested | 1 | 1e-7 |
| 	orch.sinh()` / 	orch.Tensor.sinh()` | ✅ Tested | 1 | 1e-7 |
| 	orch.sqrt()` / 	orch.Tensor.sqrt()` | ✅ Tested | 1 | 1e-7 |
| 	orch.square()` / 	orch.Tensor.square()` | ✅ Tested | 1 | 1e-7 |
| 	orch.sub()` / 	orch.Tensor.sub()` | ✅ Tested | 1 | 1e-7 |
| 	orch.tan()` / 	orch.Tensor.tan()` | ✅ Tested | 1 | 1e-7 |
| 	orch.tanh()` / 	orch.Tensor.tanh()` | ✅ Tested | 1 | 1e-7 |
| 	orch.trunc()` / 	orch.Tensor.trunc()` | ✅ Tested | 1 | 1e-7 |

## Math Ops

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.abs()` / 	orch.Tensor.abs()` | ✅ Tested | 1 | 1e-7 |
| 	orch.add()` / 	orch.Tensor.add()` | ✅ Tested | 1 | 1e-7 |
| 	orch.exp()` / 	orch.Tensor.exp()` | ✅ Tested | 1 | 1e-7 |
| 	orch.mul()` / 	orch.Tensor.mul()` | ✅ Tested | 1 | 1e-7 |
| 	orch.tanh()` / 	orch.Tensor.tanh()` | ✅ Tested | 1 | 1e-7 |

## Matmul

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.matmul()` / 	orch.Tensor.matmul()` | ✅ Tested | 4 | 1e-7 |

## Misc

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.atleast_1d()` / 	orch.Tensor.atleast_1d()` | ✅ Tested | 2 | 1e-7 |
| 	orch.atleast_2d()` / 	orch.Tensor.atleast_2d()` | ✅ Tested | 1 | 1e-7 |
| 	orch.broadcast_to()` / 	orch.Tensor.broadcast_to()` | ✅ Tested | 1 | 1e-7 |
| 	orch.cumprod()` / 	orch.Tensor.cumprod()` | ✅ Tested | 1 | 1e-7 |
| 	orch.cumsum()` / 	orch.Tensor.cumsum()` | ✅ Tested | 1 | 1e-7 |
| 	orch.diag()` / 	orch.Tensor.diag()` | ✅ Tested | 2 | 1e-7 |
| 	orch.flip()` / 	orch.Tensor.flip()` | ✅ Tested | 1 | 1e-7 |
| 	orch.fliplr()` / 	orch.Tensor.fliplr()` | ✅ Tested | 1 | 1e-7 |
| 	orch.flipud()` / 	orch.Tensor.flipud()` | ✅ Tested | 1 | 1e-7 |
| 	orch.triu()` / 	orch.Tensor.triu()` | ✅ Tested | 1 | 1e-7 |

## Nn Functional

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.avg_pool2d()` / 	orch.Tensor.avg_pool2d()` | ❌ Not Tested | 1 | - |
| 	orch.elu()` / 	orch.Tensor.elu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.gelu()` / 	orch.Tensor.gelu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.glu()` / 	orch.Tensor.glu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.hardsigmoid()` / 	orch.Tensor.hardsigmoid()` | ✅ Tested | 1 | 1e-7 |
| 	orch.hardswish()` / 	orch.Tensor.hardswish()` | ✅ Tested | 1 | 1e-7 |
| 	orch.leaky_relu()` / 	orch.Tensor.leaky_relu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.linear()` / 	orch.Tensor.linear()` | ❌ Not Tested | 1 | - |
| 	orch.log_softmax()` / 	orch.Tensor.log_softmax()` | ❌ Not Tested | 1 | - |
| 	orch.max_pool2d()` / 	orch.Tensor.max_pool2d()` | ❌ Not Tested | 1 | - |
| 	orch.mish()` / 	orch.Tensor.mish()` | ✅ Tested | 1 | 1e-7 |
| 	orch.relu()` / 	orch.Tensor.relu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.selu()` / 	orch.Tensor.selu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.sigmoid()` / 	orch.Tensor.sigmoid()` | ✅ Tested | 1 | 1e-7 |
| 	orch.silu()` / 	orch.Tensor.silu()` | ✅ Tested | 1 | 1e-7 |
| 	orch.softmax()` / 	orch.Tensor.softmax()` | ❌ Not Tested | 1 | - |
| 	orch.softplus()` / 	orch.Tensor.softplus()` | ✅ Tested | 1 | 1e-7 |
| 	orch.softsign()` / 	orch.Tensor.softsign()` | ✅ Tested | 1 | 1e-7 |
| 	orch.tanh()` / 	orch.Tensor.tanh()` | ✅ Tested | 1 | 1e-7 |
| 	orch.tanhshrink()` / 	orch.Tensor.tanhshrink()` | ✅ Tested | 1 | 1e-7 |
| 	orch.threshold()` / 	orch.Tensor.threshold()` | ✅ Tested | 1 | 1e-7 |

## Reduction

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.all()` / 	orch.Tensor.all()` | ✅ Tested | 1 | 1e-7 |
| 	orch.amax_dim0()` / 	orch.Tensor.amax_dim0()` | ❌ Not Tested | 1 | - |
| 	orch.amax_dim1()` / 	orch.Tensor.amax_dim1()` | ❌ Not Tested | 1 | - |
| 	orch.amax_dim2()` / 	orch.Tensor.amax_dim2()` | ❌ Not Tested | 1 | - |
| 	orch.amin_dim0()` / 	orch.Tensor.amin_dim0()` | ❌ Not Tested | 1 | - |
| 	orch.amin_dim1()` / 	orch.Tensor.amin_dim1()` | ❌ Not Tested | 1 | - |
| 	orch.amin_dim2()` / 	orch.Tensor.amin_dim2()` | ❌ Not Tested | 1 | - |
| 	orch.any()` / 	orch.Tensor.any()` | ✅ Tested | 1 | 1e-7 |
| 	orch.argmax()` / 	orch.Tensor.argmax()` | ✅ Tested | 2 | 1e-7 |
| 	orch.argmin()` / 	orch.Tensor.argmin()` | ✅ Tested | 1 | 1e-7 |
| 	orch.max()` / 	orch.Tensor.max()` | ✅ Tested | 1 | 1e-7 |
| 	orch.mean()` / 	orch.Tensor.mean()` | ✅ Tested | 1 | 1e-7 |
| 	orch.mean_dim0()` / 	orch.Tensor.mean_dim0()` | ❌ Not Tested | 1 | - |
| 	orch.mean_dim1()` / 	orch.Tensor.mean_dim1()` | ❌ Not Tested | 1 | - |
| 	orch.mean_dim2()` / 	orch.Tensor.mean_dim2()` | ❌ Not Tested | 1 | - |
| 	orch.min()` / 	orch.Tensor.min()` | ✅ Tested | 1 | 1e-7 |
| 	orch.prod()` / 	orch.Tensor.prod()` | ✅ Tested | 1 | 1e-7 |
| 	orch.std()` / 	orch.Tensor.std()` | ✅ Tested | 1 | 1e-7 |
| 	orch.sum()` / 	orch.Tensor.sum()` | ✅ Tested | 1 | 1e-7 |
| 	orch.sum_dim0()` / 	orch.Tensor.sum_dim0()` | ❌ Not Tested | 2 | - |
| 	orch.sum_dim1()` / 	orch.Tensor.sum_dim1()` | ❌ Not Tested | 2 | - |
| 	orch.sum_dim2()` / 	orch.Tensor.sum_dim2()` | ❌ Not Tested | 2 | - |
| 	orch.var()` / 	orch.Tensor.var()` | ✅ Tested | 1 | 1e-7 |

## Reduction Ops

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.max()` / 	orch.Tensor.max()` | ✅ Tested | 5 | 1e-7 |
| 	orch.mean()` / 	orch.Tensor.mean()` | ✅ Tested | 5 | 1e-7 |
| 	orch.sum()` / 	orch.Tensor.sum()` | ✅ Tested | 5 | 1e-7 |

## Shape

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.cat()` / 	orch.Tensor.cat()` | ✅ Tested | 2 | 1e-7 |
| 	orch.chunk()` / 	orch.Tensor.chunk()` | ✅ Tested | 1 | 1e-7 |
| 	orch.dstack()` / 	orch.Tensor.dstack()` | ✅ Tested | 1 | 1e-7 |
| 	orch.flatten()` / 	orch.Tensor.flatten()` | ✅ Tested | 2 | 1e-7 |
| 	orch.hstack()` / 	orch.Tensor.hstack()` | ✅ Tested | 1 | 1e-7 |
| 	orch.permute()` / 	orch.Tensor.permute()` | ✅ Tested | 2 | 1e-7 |
| 	orch.repeat()` / 	orch.Tensor.repeat()` | ✅ Tested | 1 | 1e-7 |
| 	orch.split()` / 	orch.Tensor.split()` | ✅ Tested | 1 | 1e-7 |
| 	orch.squeeze()` / 	orch.Tensor.squeeze()` | ✅ Tested | 2 | 1e-7 |
| 	orch.stack()` / 	orch.Tensor.stack()` | ✅ Tested | 2 | 1e-7 |
| 	orch.tile()` / 	orch.Tensor.tile()` | ✅ Tested | 1 | 1e-7 |
| 	orch.unsqueeze()` / 	orch.Tensor.unsqueeze()` | ✅ Tested | 2 | 1e-7 |
| 	orch.vstack()` / 	orch.Tensor.vstack()` | ✅ Tested | 1 | 1e-7 |

## Shape Ops

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.reshape()` / 	orch.Tensor.reshape()` | ✅ Tested | 3 | 1e-7 |
| 	orch.squeeze()` / 	orch.Tensor.squeeze()` | ✅ Tested | 1 | 1e-7 |
| 	orch.t()` / 	orch.Tensor.t()` | ❌ Not Tested | 2 | - |
| 	orch.unsqueeze()` / 	orch.Tensor.unsqueeze()` | ✅ Tested | 2 | 1e-7 |

## Tensor Creation

| API | Status | Fixture Tests | Avg Precision |
|-----|--------|---------------|---------------|
| 	orch.arange()` / 	orch.Tensor.arange()` | ✅ Tested | 3 | 1e-7 |
| 	orch.eye()` / 	orch.Tensor.eye()` | ✅ Tested | 2 | 1e-7 |
| 	orch.full()` / 	orch.Tensor.full()` | ✅ Tested | 2 | 1e-7 |
| 	orch.linspace()` / 	orch.Tensor.linspace()` | ✅ Tested | 2 | 1e-7 |
| 	orch.ones()` / 	orch.Tensor.ones()` | ✅ Tested | 2 | 1e-7 |
| 	orch.tensor()` / 	orch.Tensor.tensor()` | ✅ Tested | 3 | 1e-7 |
| 	orch.zeros()` / 	orch.Tensor.zeros()` | ✅ Tested | 3 | 1e-7 |

---

## How to Add Test Coverage

1. Check the fixture file in `fixtures/pytorch/` for expected inputs/outputs
2. Create or update a test file in `tests/pytorch-compat/`
3. Run `pnpm test:run` to verify
4. Run `pnpm compat-report` to update this report

*Generated on 2025-12-22*