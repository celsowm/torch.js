import { w as webgpu, _ as __utils_benchmark, i as index, l as linalg, s as syncDevice, n as nn, o as optim, t as tensor, z as zeros, a as ones, f as full, b as zeros_like, c as ones_like, d as full_like, e as empty_like, r as rand, g as randn, h as eye, j as arange, k as linspace, m as logspace, p as manual_seed, q as tril, u as cat, v as stack, x as vstack, y as hstack, A as dstack, B as column_stack, T as Tensor, C as initWebGPU, S as SaveFunc, L as LoadFunc, D as DEBUG, E as DEBUG_ASYNC, F as warn, G as error, H as assert, I as getDevice, J as getOrCreatePipeline, K as calculateWorkgroups, M as readBuffer, N as createTorch } from './core-U79hGaa9.js';
export { O as BufferUsage, P as DType, V as GradFn, Q as TensorOptions, W as TypedArray, U as _internals, a9 as abs, aa as absolute, ab as acos, ad as acosh, bA as add, bW as addbmm, bO as addmm, bQ as addmv, bT as addr, aE as all, bd as allclose, aC as amax, aD as amin, aF as any, ac as arccos, ae as arccosh, ag as arcsin, ai as arcsinh, ak as arctan, ao as arctan2, am as arctanh, aA as argmax, aB as argmin, af as asin, ah as asinh, aj as atan, an as atan2, al as atanh, Y as atleast_1d, Z as atleast_2d, $ as atleast_3d, bV as baddbmm, bH as bitwise_and, bI as bitwise_or, bJ as bitwise_xor, bU as bmm, a0 as broadcast_to, ap as ceil, bN as chain_matmul, aG as chunk, av as clamp, aw as clip, be as cos, bf as cosh, a5 as cumprod, a4 as cumsum, b$ as cumulative_trapezoid, a7 as diag, bF as div, bG as divide, bX as dot, aT as eq, bb as equal, bk as exp, bl as exp2, at as fix, ax as flatten, a1 as flip, a2 as fliplr, a3 as flipud, aq as floor, b9 as fmax, ba as fmin, au as frac, b0 as ge, bS as ger, a$ as greater, b1 as greater_equal, a_ as gt, a8 as heaviside, bK as hypot, aP as index_select, bZ as inner, bc as isclose, b4 as isfinite, b3 as isinf, b2 as isnan, b6 as isneginf, b5 as isposinf, aY as le, aX as less, aZ as less_equal, X as log, bm as log10, bo as log1p, bn as log2, bp as logaddexp, aW as lt, bL as matmul, b7 as maximum, b8 as minimum, bM as mm, aK as moveaxis, aJ as movedim, bD as mul, bE as multiply, bP as mv, aH as narrow, aU as ne, bq as neg, br as negative, aV as not_equal, bR as outer, aI as permute, bt as pow, bs as prod, bu as reciprocal, bz as relu, ar as round, c1 as row_stack, bv as rsqrt, aQ as select, by as sigmoid, bg as sin, bh as sinh, bw as sqrt, bx as square, ay as squeeze, bB as sub, bC as subtract, aL as swapaxes, aM as swapdims, aR as take, bi as tan, bj as tanh, aN as tile, b_ as trapezoid, c0 as trapz, a6 as triu, as as trunc, aO as unbind, az as unsqueeze, R as utils, bY as vdot, aS as where } from './core-U79hGaa9.js';

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
    };
    profiler: typeof index;
    linalg: typeof linalg;
    syncDevice: typeof syncDevice;
    nn: typeof nn;
    optim: typeof optim;
    tensor: typeof tensor;
    zeros: typeof zeros;
    ones: typeof ones;
    full: typeof full;
    zeros_like: typeof zeros_like;
    ones_like: typeof ones_like;
    full_like: typeof full_like;
    empty_like: typeof empty_like;
    rand: typeof rand;
    randn: typeof randn;
    eye: typeof eye;
    arange: typeof arange;
    linspace: typeof linspace;
    logspace: typeof logspace;
    manual_seed: typeof manual_seed;
    tril: typeof tril;
    cat: typeof cat;
    stack: typeof stack;
    vstack: typeof vstack;
    row_stack: typeof vstack;
    hstack: typeof hstack;
    dstack: typeof dstack;
    column_stack: typeof column_stack;
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
    round: (input: Tensor) => Tensor;
    trunc: (input: Tensor) => Tensor;
    fix: (input: Tensor) => Tensor;
    frac: (input: Tensor) => Tensor;
    clamp: (input: Tensor, min?: number, max?: number) => Tensor;
    clip: (input: Tensor, min?: number, max?: number) => Tensor;
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
    isnan: (input: Tensor) => Tensor;
    isinf: (input: Tensor) => Tensor;
    isfinite: (input: Tensor) => Tensor;
    isposinf: (input: Tensor) => Tensor;
    isneginf: (input: Tensor) => Tensor;
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
    init: typeof initWebGPU;
    save: SaveFunc;
    load: LoadFunc;
    Tensor: typeof Tensor;
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

export { DEBUG, DEBUG_ASYNC, LoadFunc, SaveFunc, Tensor, arange, assert, calculateWorkgroups, cat, column_stack, createTorch, torch as default, dstack, empty_like, error, eye, full, full_like, getCapabilities, getDevice, getOrCreatePipeline, hstack, initWebGPU as init, linalg, linspace, logspace, manual_seed, nn, ones, ones_like, optim, index as profiler, rand, randn, readBuffer, stack, syncDevice, tensor, tril, vstack, warn, webgpu, zeros, zeros_like };
