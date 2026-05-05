import { w as webgpu, _ as __utils_benchmark, a as __utils_data, i as index, l as linalg, n as nn, o as optim, b as __autograd, c as __distributions, d as __sparse, e as no_grad, f as enable_grad, g as inference_mode, h as fft, t as tensor, z as zeros, j as ones, k as full, m as empty, p as zeros_like, q as ones_like, r as full_like, s as empty_like, u as rand, v as randn, x as randint, y as randperm, A as normal, B as eye, C as arange, D as linspace, E as logspace, F as scalar_tensor, G as as_tensor, H as from_numpy, I as manual_seed, J as tril, K as cat, L as stack, M as split, N as vstack, O as hstack, P as dstack, Q as column_stack, R as histc, S as bincount, T as meshgrid, U as cartesian_prod, V as combinations, W as tensor_split, X as trace, Y as unravel_index, Z as Tensor, $ as Slice, a0 as initWebGPU, a1 as SaveFunc, a2 as LoadFunc, a3 as index$1, a4 as syncDevice, a5 as DEBUG, a6 as DEBUG_ASYNC, a7 as warn, a8 as error, a9 as assert, aa as getDevice, ab as getOrCreatePipeline, ac as calculateWorkgroups, ad as readBuffer, ae as createTorch } from './core-CsHVFBX8.js';
export { af as BufferUsage, ag as DType, ah as GradFn, ai as TensorOptions, aj as TypedArray, ak as _internals, al as abs, am as absolute, an as acos, ao as acosh, ap as add, aq as addbmm, ar as addmm, as as addmv, at as addr, au as advancedSlice, av as all, aw as allclose, ax as amax, ay as amin, az as aminmax, aA as any, aB as arccos, aC as arccosh, aD as arcsin, aE as arcsinh, aF as arctan, aG as arctan2, aH as arctanh, aI as argmax, aJ as argmin, aK as argsort, aL as asin, aM as asinh, aN as atan, aO as atan2, aP as atanh, aQ as atleast_1d, aR as atleast_2d, aS as atleast_3d, aT as baddbmm, aU as bitwise_and, aV as bitwise_or, aW as bitwise_xor, aX as bmm, aY as broadcast_to, aZ as ceil, a_ as chain_matmul, a$ as chunk, b0 as clamp, b1 as clamp_max, b2 as clamp_min, b3 as clip, b4 as cos, b5 as cosh, b6 as count_nonzero, b7 as cummax, b8 as cummin, b9 as cumprod, ba as cumsum, bb as cumulative_trapezoid, bc as deg2rad, bd as diag, be as diagonal, bf as digamma, bg as div, bh as divide, bi as dot, bj as einsum, bk as elu, bl as eq, bm as equal, bn as erf, bo as erfc, bp as exp, bq as exp2, br as expm1, bs as fix, bt as flatten, bu as flip, bv as fliplr, bw as flipud, bx as floor, by as fmax, bz as fmin, bA as fmod, bB as frac, bC as gather, bD as ge, bE as gelu, bF as ger, bG as greater, bH as greater_equal, bI as gt, bJ as hardsigmoid, bK as hardswish, bL as heaviside, bM as hypot, bN as i0, bO as index_select, bP as inner, bQ as isclose, bR as isfinite, bS as isinf, bT as isnan, bU as isneginf, bV as isposinf, bW as isreal, bX as kthvalue, bY as le, bZ as leaky_relu, b_ as less, b$ as less_equal, c0 as lgamma, c1 as log, c2 as log10, c3 as log1p, c4 as log2, c5 as log_softmax, c6 as logaddexp, c7 as logcumsumexp, c8 as logical_not, c9 as logsumexp, ca as lt, cb as masked_select, cc as matmul, cd as maximum, ce as minimum, cf as mish, cg as mm, ch as moveaxis, ci as movedim, cj as mul, ck as multinomial, cl as multiply, cm as mv, cn as narrow, co as ne, cp as neg, cq as negative, cr as nonzero, cs as not_equal, ct as outer, cu as permute, cv as pow, cw as prod, cx as rad2deg, cy as reciprocal, cz as relu, cA as remainder, cB as repeat_interleave, cC as roll, cD as rot90, cE as round, cF as row_stack, cG as rsqrt, cH as scatter, cI as scatter_add, cJ as select, cK as selu, cL as sgn, cM as sigmoid, cN as sign, cO as silu, cP as sin, cQ as sinh, cR as softmax, cS as softmin, cT as softplus, cU as softsign, cV as sort, cW as sqrt, cX as square, cY as squeeze, cZ as std, c_ as std_mean, c$ as sub, d0 as subtract, d1 as swapaxes, d2 as swapdims, d3 as take, d4 as tan, d5 as tanh, d6 as tanhshrink, d7 as threshold, d8 as tile, d9 as topk, da as trapezoid, db as trapz, dc as triu, dd as trunc, de as unbind, df as unflatten, dg as unsqueeze, dh as utils, di as var_, dj as vdot, dk as where } from './core-CsHVFBX8.js';
import { l as loadPyTorchZip } from './pytorch_loader-DQJp8N7I.js';

/**
 * GPU capability detection for runtime feature selection.
 * Tests actual GPU behavior since browser/driver bugs vary by platform.
 */
interface GPULimits {
    maxComputeWorkgroupSizeX: number;
    maxComputeWorkgroupSizeY: number;
    maxComputeWorkgroupSizeZ: number;
    maxComputeInvocationsPerWorkgroup: number;
    maxComputeWorkgroupsPerDimension: number;
    maxStorageBufferBindingSize: number;
    maxBufferSize: number;
}
interface GPUCapabilities {
    /** Whether workgroup shared memory works correctly */
    workgroupSharedMemory: boolean;
    /** Whether high-precision timestamp queries are supported */
    timestampQuery: boolean;
    /** Whether subgroups are supported (future) */
    subgroups: boolean;
    /** GPU limits from adapter */
    limits: GPULimits;
    /** Detected platform info for debugging */
    platform: {
        browser: string;
        gpu: string;
    };
}
/**
 * Get cached capabilities. Throws if not yet detected.
 */
declare function getCapabilities(): GPUCapabilities;

/**
 * Main torch object.
 */
declare const torch: {
    webgpu: typeof webgpu;
    utils: {
        benchmark: typeof __utils_benchmark;
        data: typeof __utils_data;
    };
    profiler: typeof index;
    linalg: typeof linalg;
    nn: typeof nn;
    optim: typeof optim;
    autograd: typeof __autograd;
    distributions: typeof __distributions;
    sparse: typeof __sparse;
    no_grad: typeof no_grad;
    enable_grad: typeof enable_grad;
    inference_mode: typeof inference_mode;
    is_grad_enabled: () => boolean;
    fft: typeof fft;
    tensor: typeof tensor;
    zeros: typeof zeros;
    ones: typeof ones;
    full: typeof full;
    empty: typeof empty;
    zeros_like: typeof zeros_like;
    ones_like: typeof ones_like;
    full_like: typeof full_like;
    empty_like: typeof empty_like;
    rand: typeof rand;
    randn: typeof randn;
    randint: typeof randint;
    randperm: typeof randperm;
    normal: typeof normal;
    eye: typeof eye;
    arange: typeof arange;
    linspace: typeof linspace;
    logspace: typeof logspace;
    scalar_tensor: typeof scalar_tensor;
    as_tensor: typeof as_tensor;
    from_numpy: typeof from_numpy;
    manual_seed: typeof manual_seed;
    tril: typeof tril;
    cat: typeof cat;
    stack: typeof stack;
    split: typeof split;
    vstack: typeof vstack;
    row_stack: typeof vstack;
    hstack: typeof hstack;
    dstack: typeof dstack;
    column_stack: typeof column_stack;
    histc: typeof histc;
    bincount: typeof bincount;
    meshgrid: typeof meshgrid;
    cartesian_prod: typeof cartesian_prod;
    combinations: typeof combinations;
    tensor_split: typeof tensor_split;
    trace: typeof trace;
    unravel_index: typeof unravel_index;
    atleast_1d: (input: Tensor) => Tensor;
    atleast_2d: (input: Tensor) => Tensor;
    atleast_3d: (input: Tensor) => Tensor;
    broadcast_to: (input: Tensor, shape: number[]) => Tensor;
    flip: (input: Tensor, dims: number[]) => Tensor;
    fliplr: (input: Tensor) => Tensor;
    flipud: (input: Tensor) => Tensor;
    cumsum: (input: Tensor, dim: number) => Tensor;
    cumprod: (input: Tensor, dim: number) => Tensor;
    triu: (input: Tensor, diagonal?: number) => Tensor;
    diag: (input: Tensor, diagonal?: number) => Tensor;
    heaviside: (input: Tensor, values: Tensor) => Tensor;
    std: (input: Tensor, dim?: number | number[], keepdim?: boolean, unbiased?: boolean) => Tensor;
    var_: (input: Tensor, dim?: number | number[], keepdim?: boolean, unbiased?: boolean) => Tensor;
    std_mean: (input: Tensor, dim?: number | number[], keepdim?: boolean, unbiased?: boolean) => {
        std: Tensor;
        mean: Tensor;
    };
    sort: (input: Tensor, dim?: number, descending?: boolean) => Promise<{
        values: Tensor;
        indices: Tensor;
    }>;
    argsort: (input: Tensor, dim?: number, descending?: boolean) => Promise<Tensor>;
    topk: (input: Tensor, k: number, dim?: number, largest?: boolean, sorted?: boolean) => Promise<{
        values: Tensor;
        indices: Tensor;
    }>;
    kthvalue: (input: Tensor, k: number, dim?: number, keepdim?: boolean) => Promise<{
        values: Tensor;
        indices: Tensor;
    }>;
    cummax: (input: Tensor, dim: number) => Promise<{
        values: Tensor;
        indices: Tensor;
    }>;
    cummin: (input: Tensor, dim: number) => Promise<{
        values: Tensor;
        indices: Tensor;
    }>;
    logsumexp: (input: Tensor, dim?: number | number[], keepdim?: boolean) => Tensor;
    logcumsumexp: (input: Tensor, dim: number) => Promise<Tensor>;
    count_nonzero: (input: Tensor, dim?: number | number[], keepdim?: boolean) => Tensor;
    aminmax: (input: Tensor, dim?: number, keepdim?: boolean) => {
        min: Tensor;
        max: Tensor;
    };
    nonzero: (input: Tensor) => Promise<Tensor>;
    diagonal: (input: Tensor, offset?: number, dim1?: number, dim2?: number) => Tensor;
    masked_select: (input: Tensor, mask: Tensor) => Promise<Tensor>;
    gather: (input: Tensor, dim: number, index: Tensor) => Promise<Tensor>;
    scatter: (input: Tensor, dim: number, index: Tensor, src: Tensor) => Tensor;
    scatter_add: (input: Tensor, dim: number, index: Tensor, src: Tensor) => Tensor;
    repeat_interleave: (input: Tensor, repeats: number, dim?: number) => Promise<Tensor>;
    roll: (input: Tensor, shifts: number | number[], dims?: number | number[]) => Promise<Tensor>;
    rot90: (input: Tensor, k?: number, dims?: number[]) => Promise<Tensor>;
    unflatten: (input: Tensor, dim: number, sizes: number[]) => Tensor;
    clip: (input: Tensor, min?: number, max?: number) => Tensor;
    clamp_min: (input: Tensor, min: number) => Tensor;
    clamp_max: (input: Tensor, max: number) => Tensor;
    fmod: (input: Tensor, other: Tensor | number) => Tensor;
    remainder: (input: Tensor, other: Tensor | number) => Tensor;
    trunc: (input: Tensor) => Tensor;
    fix: (input: Tensor) => Tensor;
    round: (input: Tensor) => Tensor;
    isposinf: (input: Tensor) => Tensor;
    isneginf: (input: Tensor) => Tensor;
    isreal: (input: Tensor) => Tensor;
    isnan: (input: Tensor) => Tensor;
    isinf: (input: Tensor) => Tensor;
    isfinite: (input: Tensor) => Tensor;
    abs: (input: Tensor) => Tensor;
    absolute: (input: Tensor) => Tensor;
    acos: (input: Tensor) => Tensor;
    arccos: (input: Tensor) => Tensor;
    acosh: (input: Tensor) => Tensor;
    arccosh: (input: Tensor) => Tensor;
    asin: (input: Tensor) => Tensor;
    arcsin: (input: Tensor) => Tensor;
    asinh: (input: Tensor) => Tensor;
    arcsinh: (input: Tensor) => Tensor;
    atan: (input: Tensor) => Tensor;
    arctan: (input: Tensor) => Tensor;
    atanh: (input: Tensor) => Tensor;
    arctanh: (input: Tensor) => Tensor;
    atan2: (input: Tensor, other: Tensor) => Tensor;
    arctan2: (input: Tensor, other: Tensor) => Tensor;
    ceil: (input: Tensor) => Tensor;
    floor: (input: Tensor) => Tensor;
    frac: (input: Tensor) => Tensor;
    clamp: (input: Tensor, min?: number, max?: number) => Tensor;
    flatten: (input: Tensor, startDim?: number, endDim?: number) => Tensor;
    squeeze: (input: Tensor, dim?: number) => Tensor;
    unsqueeze: (input: Tensor, dim: number) => Tensor;
    argmax: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
    argmin: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
    amax: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
    amin: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
    all: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
    any: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
    chunk: (input: Tensor, chunks: number, dim?: number) => Tensor[];
    narrow: (input: Tensor, dim: number, start: number, length: number) => Tensor;
    permute: (input: Tensor, dims: number[]) => Tensor;
    movedim: (input: Tensor, source: number | number[], destination: number | number[]) => Tensor;
    moveaxis: (input: Tensor, source: number | number[], destination: number | number[]) => Tensor;
    swapaxes: (input: Tensor, dim0: number, dim1: number) => Tensor;
    swapdims: (input: Tensor, dim0: number, dim1: number) => Tensor;
    tile: (input: Tensor, reps: number[]) => Tensor;
    unbind: (input: Tensor, dim?: number) => Tensor[];
    index_select: (input: Tensor, dim: number, index: Tensor) => Tensor;
    select: (input: Tensor, dim: number, index: number) => Tensor;
    take: (input: Tensor, indices: Tensor) => Tensor;
    where: (condition: Tensor, input: Tensor, other: Tensor) => Tensor;
    eq: (input: Tensor, other: Tensor) => Tensor;
    ne: (input: Tensor, other: Tensor) => Tensor;
    not_equal: (input: Tensor, other: Tensor) => Tensor;
    lt: (input: Tensor, other: Tensor) => Tensor;
    less: (input: Tensor, other: Tensor) => Tensor;
    le: (input: Tensor, other: Tensor) => Tensor;
    less_equal: (input: Tensor, other: Tensor) => Tensor;
    gt: (input: Tensor, other: Tensor) => Tensor;
    greater: (input: Tensor, other: Tensor) => Tensor;
    ge: (input: Tensor, other: Tensor) => Tensor;
    greater_equal: (input: Tensor, other: Tensor) => Tensor;
    maximum: (input: Tensor, other: Tensor) => Tensor;
    minimum: (input: Tensor, other: Tensor) => Tensor;
    fmax: (input: Tensor, other: Tensor) => Tensor;
    fmin: (input: Tensor, other: Tensor) => Tensor;
    equal: (input: Tensor, other: Tensor) => Promise<boolean>;
    isclose: (input: Tensor, other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean) => Tensor;
    allclose: (input: Tensor, other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean) => Promise<boolean>;
    cos: (input: Tensor) => Tensor;
    cosh: (input: Tensor) => Tensor;
    sin: (input: Tensor) => Tensor;
    sinh: (input: Tensor) => Tensor;
    tan: (input: Tensor) => Tensor;
    tanh: (input: Tensor) => Tensor;
    exp: (input: Tensor) => Tensor;
    exp2: (input: Tensor) => Tensor;
    log: (input: Tensor) => Tensor;
    log10: (input: Tensor) => Tensor;
    log2: (input: Tensor) => Tensor;
    log1p: (input: Tensor) => Tensor;
    logaddexp: (input: Tensor, other: Tensor) => Tensor;
    neg: (input: Tensor) => Tensor;
    negative: (input: Tensor) => Tensor;
    prod: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
    pow: (input: Tensor, exponent: number | Tensor) => Tensor;
    reciprocal: (input: Tensor) => Tensor;
    rsqrt: (input: Tensor) => Tensor;
    sqrt: (input: Tensor) => Tensor;
    square: (input: Tensor) => Tensor;
    sigmoid: (input: Tensor) => Tensor;
    relu: (input: Tensor) => Tensor;
    sign: (input: Tensor) => Tensor;
    sgn: (input: Tensor) => Tensor;
    erf: (input: Tensor) => Tensor;
    erfc: (input: Tensor) => Tensor;
    expm1: (input: Tensor) => Tensor;
    deg2rad: (input: Tensor) => Tensor;
    rad2deg: (input: Tensor) => Tensor;
    logical_not: (input: Tensor) => Tensor;
    i0: (input: Tensor) => Tensor;
    lgamma: (input: Tensor) => Tensor;
    digamma: (input: Tensor) => Tensor;
    gelu: (input: Tensor) => Tensor;
    softplus: (input: Tensor) => Tensor;
    silu: (input: Tensor) => Tensor;
    mish: (input: Tensor) => Tensor;
    hardsigmoid: (input: Tensor) => Tensor;
    hardswish: (input: Tensor) => Tensor;
    softsign: (input: Tensor) => Tensor;
    tanhshrink: (input: Tensor) => Tensor;
    leaky_relu: (input: Tensor, negative_slope?: number) => Tensor;
    elu: (input: Tensor, alpha?: number) => Tensor;
    selu: (input: Tensor) => Tensor;
    threshold: (input: Tensor, threshold: number, value: number) => Tensor;
    softmin: (input: Tensor) => Tensor;
    log_softmax: (input: Tensor, dim?: number) => Tensor;
    softmax: (input: Tensor, dim?: number) => Tensor;
    add: (input: Tensor, other: Tensor | number) => Tensor;
    sub: (input: Tensor, other: Tensor | number) => Tensor;
    subtract: (input: Tensor, other: Tensor | number) => Tensor;
    mul: (input: Tensor, other: Tensor | number) => Tensor;
    multiply: (input: Tensor, other: Tensor | number) => Tensor;
    div: (input: Tensor, other: Tensor | number) => Tensor;
    divide: (input: Tensor, other: Tensor | number) => Tensor;
    bitwise_and: (input: Tensor, other: Tensor) => Tensor;
    bitwise_or: (input: Tensor, other: Tensor) => Tensor;
    bitwise_xor: (input: Tensor, other: Tensor) => Tensor;
    hypot: (input: Tensor, other: Tensor) => Tensor;
    matmul: (input: Tensor, other: Tensor) => Tensor;
    mm: (input: Tensor, mat2: Tensor) => Tensor;
    chain_matmul: (...matrices: Tensor[]) => Tensor;
    addmm: (input: Tensor, mat1: Tensor, mat2: Tensor, beta?: number, alpha?: number) => Tensor;
    mv: (input: Tensor, vec: Tensor) => Tensor;
    addmv: (input: Tensor, mat: Tensor, vec: Tensor, beta?: number, alpha?: number) => Tensor;
    outer: (input: Tensor, vec2: Tensor) => Tensor;
    ger: (input: Tensor, vec2: Tensor) => Tensor;
    addr: (input: Tensor, vec1: Tensor, vec2: Tensor, beta?: number, alpha?: number) => Tensor;
    bmm: (input: Tensor, mat2: Tensor) => Tensor;
    baddbmm: (input: Tensor, batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number) => Tensor;
    addbmm: (input: Tensor, batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number) => Tensor;
    dot: (input: Tensor, other: Tensor) => Tensor;
    vdot: (input: Tensor, other: Tensor) => Tensor;
    inner: (input: Tensor, other: Tensor) => Tensor;
    trapezoid: (input: Tensor, dx?: number, dim?: number) => Tensor;
    cumulative_trapezoid: (input: Tensor, dx?: number, dim?: number) => Tensor;
    trapz: (input: Tensor, dx?: number, dim?: number) => Tensor;
    einsum: (equation: string, ...operands: Tensor[]) => Promise<Tensor>;
    advancedSlice: (input: Tensor, indices: (number | Slice)[]) => Promise<Tensor>;
    multinomial: (input: Tensor, num_samples?: number, replacement?: boolean) => Promise<Tensor>;
    init: typeof initWebGPU;
    save: SaveFunc;
    load: LoadFunc;
    loadPyTorch: typeof loadPyTorchZip;
    Tensor: typeof Tensor;
    special: typeof index$1;
    syncDevice: typeof syncDevice;
    DEBUG: typeof DEBUG;
    DEBUG_ASYNC: typeof DEBUG_ASYNC;
    warn: typeof warn;
    error: typeof error;
    assert: typeof assert;
    _internals: {
        getDevice: typeof getDevice;
        getOrCreatePipeline: typeof getOrCreatePipeline;
        calculateWorkgroups: typeof calculateWorkgroups;
        readBuffer: typeof readBuffer;
    };
    createTorch: typeof createTorch;
};

export { DEBUG, DEBUG_ASYNC, LoadFunc, SaveFunc, Tensor, arange, as_tensor, assert, bincount, calculateWorkgroups, cartesian_prod, cat, column_stack, combinations, createTorch, torch as default, dstack, empty, empty_like, error, eye, fft, from_numpy, full, full_like, getCapabilities, getDevice, getOrCreatePipeline, histc, hstack, initWebGPU as init, linalg, linspace, logspace, manual_seed, meshgrid, nn, normal, ones, ones_like, optim, index as profiler, rand, randint, randn, randperm, readBuffer, scalar_tensor, index$1 as special, split, stack, syncDevice, tensor, tensor_split, torch, trace, tril, unravel_index, vstack, warn, webgpu, zeros, zeros_like };
