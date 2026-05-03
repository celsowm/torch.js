/**
 * Unified WebGPU types that work for both browser and Node.js (wgpu-native).
 * We use 'any' internally to avoid the complexity of intersecting browser and native types.
 */
type WebGPUDevice = any;
type WebGPUBuffer = any;
type WebGPUComputePipeline = any;
declare const BufferUsage: {
    MAP_READ: number;
    MAP_WRITE: number;
    COPY_SRC: number;
    COPY_DST: number;
    INDEX: number;
    VERTEX: number;
    UNIFORM: number;
    STORAGE: number;
    INDIRECT: number;
    QUERY_RESOLVE: number;
};

/**
 * WebGPU device initialization and management.
 * Optimized for browser by default, but allows custom GPU provider (e.g. for Node.js).
 * Uses TypeGPU for type-safe buffer and shader management.
 * @status implemented
 */

/**
 * Initialize the WebGPU device.
 * Must be called before any tensor operations.
 *
 * @param customGpu Optional WebGPU implementation (e.g. from @torchjsorg/wgpu-native)
 * @status implemented
 * @pytorch N/A (torch.js specific)
 */
declare function initWebGPU(customGpu?: GPU): Promise<void>;
/**
 * Get the initialized WebGPU device.
 * Throws if not initialized.
 */
declare function getDevice(): WebGPUDevice;

/**
 * Supported data types for tensors.
 * @status implemented
 */
type DType = 'float32' | 'float16' | 'int32' | 'uint32' | 'int8' | 'uint8' | 'bool';
type TypedArray = Float32Array | Int32Array | Uint32Array | Int8Array | Uint8Array;

/**
 * GPU buffer management with pooling for performance.
 * @status implemented
 */

/**
 * Read data from a GPU buffer back to CPU.
 */
declare function readBuffer(buffer: WebGPUBuffer, dtype: DType, numElements: number): Promise<TypedArray>;

/**
 * Shader dispatch utilities for WebGPU compute operations.
 * @status implemented
 */

/**
 * Get or create a compute pipeline for the given shader.
 */
declare function getOrCreatePipeline(shaderCode: string, entryPoint?: string): WebGPUComputePipeline;
/**
 * Calculate workgroup count for a given number of elements.
 * Uses a default workgroup size of 256.
 */
declare function calculateWorkgroups(numElements: number, workgroupSize?: number): [number, number, number];
/**
 * Wait for all GPU operations to complete.
 * Useful for timing and synchronization.
 */
declare function syncDevice(): Promise<void>;

/**
 * Result of a benchmark measurement.
 */
declare class Measurement {
    readonly name: string;
    readonly times: number[];
    readonly task_spec: string;
    readonly label: string;
    readonly sub_label: string;
    readonly description: string;
    readonly env: string;
    constructor(name: string, times: number[], task_spec?: string, label?: string, sub_label?: string, description?: string, env?: string);
    get mean(): number;
    get median(): number;
    get iqr(): number;
    toString(): string;
}
/**
 * Timer class for benchmarking code snippets.
 * Generic T represents the return type of the statement being timed.
 */
declare class Timer<T = any> {
    private readonly stmt;
    private readonly setup;
    readonly label: string;
    readonly sub_label: string;
    readonly description: string;
    readonly env: string;
    constructor(stmt: () => T | Promise<T>, setup?: () => void | Promise<void>, label?: string, sub_label?: string, description?: string, env?: string);
    /**
     * Run the statement multiple times and return a Measurement.
     */
    timeit(number?: number): Promise<Measurement>;
    /**
     * Automatically determine number of iterations to get stable results.
     */
    blocked_autorange(min_run_time?: number): Promise<Measurement>;
}

/**
 * Tensor types and interfaces.
 */

/**
 * Gradient function for autograd.
 * backward can be async for operations that need GPU readback.
 */
type GradFn = {
    backward(gradOutput: Tensor): void | Promise<void>;
};
/**
 * Options for tensor creation.
 */
interface TensorOptions {
    dtype?: DType;
    device?: 'webgpu' | 'cpu';
    requires_grad?: boolean;
}
/**
 * Internal tensor data structure.
 */
interface TensorData {
    buffer: GPUBuffer;
    shape: readonly number[];
    dtype: DType;
    device: 'webgpu';
    requires_grad: boolean;
    grad_fn?: GradFn;
}
/**
 * Slice specification for tensor indexing.
 * Supports Python-style slicing: start:stop:step
 */
interface SliceSpec {
    start?: number;
    stop?: number;
    step?: number;
}

/**
 * Core Tensor class - the fundamental data structure of torch.js.
 * @status partial
 * @pytorch torch.Tensor
 */

/**
 * A multi-dimensional array with GPU acceleration.
 */
declare class Tensor {
    private _buffer;
    private _shape;
    private _dtype;
    private _device;
    private _requires_grad;
    private _grad;
    private _grad_fn;
    private _is_leaf;
    /** @internal */
    constructor(data: TensorData);
    static cat(tensors: Tensor[], dim?: number): Tensor;
    /**
     * Manually release the GPU buffer associated with this tensor.
     * Returns it to the pool for reuse.
     */
    destroy(): void;
    tile(reps: readonly number[]): Tensor;
    repeat(sizes: readonly number[]): Tensor;
    get shape(): readonly number[];
    get dtype(): DType;
    get device(): 'webgpu';
    get requires_grad(): boolean;
    get grad(): Tensor | null;
    set grad(value: Tensor | null);
    get grad_fn(): GradFn | null;
    get is_leaf(): boolean;
    /**
     * Transpose property (alias for t()).
     * @status implemented
     */
    get T(): Tensor;
    /**
     * Change if autograd should record operations on this tensor:
     * sets the tensor's requires_grad attribute in-place.
     * @pytorch tensor.requires_grad_()
     */
    requires_grad_(value?: boolean): Tensor;
    detach(): Tensor;
    /**
     * Copies the elements from src into self tensor and returns self.
     * @pytorch tensor.copy_()
     */
    copy_(src: Tensor): Tensor;
    clone(): Tensor;
    /** @internal */
    get buffer(): GPUBuffer;
    add(other: Tensor | number): Tensor;
    sub(other: Tensor | number): Tensor;
    mul(other: Tensor | number): Tensor;
    div(other: Tensor | number): Tensor;
    abs(): Tensor;
    acos(): Tensor;
    asin(): Tensor;
    atan(): Tensor;
    ceil(): Tensor;
    cos(): Tensor;
    cosh(): Tensor;
    exp(): Tensor;
    exp2(): Tensor;
    floor(): Tensor;
    log(base?: number | 'e'): Tensor;
    log10(): Tensor;
    log2(): Tensor;
    log1p(): Tensor;
    neg(): Tensor;
    round(): Tensor;
    sin(): Tensor;
    sinh(): Tensor;
    acosh(): Tensor;
    asinh(): Tensor;
    atanh(): Tensor;
    sqrt(): Tensor;
    tan(): Tensor;
    tanh(): Tensor;
    trunc(): Tensor;
    frac(): Tensor;
    reciprocal(): Tensor;
    rsqrt(): Tensor;
    square(): Tensor;
    sigmoid(): Tensor;
    relu(): Tensor;
    gelu(): Tensor;
    softplus(): Tensor;
    silu(): Tensor;
    mish(): Tensor;
    hardsigmoid(): Tensor;
    hardswish(): Tensor;
    softsign(): Tensor;
    tanhshrink(): Tensor;
    leaky_relu(negative_slope?: number): Tensor;
    elu(alpha?: number): Tensor;
    selu(): Tensor;
    threshold(threshold: number, value: number): Tensor;
    zeros_like(): Tensor;
    ones_like(): Tensor;
    heaviside(values: Tensor): Tensor;
    glu(dim?: number): Tensor;
    pow(exponent: number | Tensor): Tensor;
    atan2(other: Tensor): Tensor;
    hypot(other: Tensor): Tensor;
    logaddexp(other: Tensor): Tensor;
    bitwise_and(other: Tensor): Tensor;
    bitwise_or(other: Tensor): Tensor;
    bitwise_xor(other: Tensor): Tensor;
    clamp(min?: number, max?: number): Tensor;
    masked_fill(mask: Tensor, value: number): Tensor;
    triu(diagonal?: number): Tensor;
    tril(diagonal?: number): Tensor;
    to(dtype: DType): Tensor;
    flip(dims: number[]): Tensor;
    cumsum(dim: number): Tensor;
    cumprod(dim: number): Tensor;
    diag(diagonal?: number): Tensor;
    private _cumOp;
    private _zeros;
    _sliceDimStep(dim: number, start: number, end: number, step?: number): Tensor;
    trapezoid(dx?: number, dim?: number): Tensor;
    cumulative_trapezoid(dx?: number, dim?: number): Tensor;
    index_select(dim: number, index: Tensor): Tensor;
    masked_select(mask: Tensor): Tensor;
    select(dim: number, index: number): Tensor;
    take(indices: Tensor): Tensor;
    where(condition: Tensor, other: Tensor): Tensor;
    eq(other: Tensor | number): Tensor;
    ne(other: Tensor | number): Tensor;
    lt(other: Tensor | number): Tensor;
    gt(other: Tensor | number): Tensor;
    ge(other: Tensor | number): Tensor;
    le(other: Tensor | number): Tensor;
    greater(other: Tensor | number): Tensor;
    greater_equal(other: Tensor | number): Tensor;
    less(other: Tensor | number): Tensor;
    less_equal(other: Tensor | number): Tensor;
    not_equal(other: Tensor | number): Tensor;
    isnan(): Tensor;
    isinf(): Tensor;
    isfinite(): Tensor;
    isposinf(): Tensor;
    isneginf(): Tensor;
    maximum(other: Tensor): Tensor;
    minimum(other: Tensor): Tensor;
    fmax(other: Tensor): Tensor;
    fmin(other: Tensor): Tensor;
    equal(other: Tensor): Promise<boolean>;
    isclose(other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean): Tensor;
    allclose(other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean): Promise<boolean>;
    numel(): number;
    dim(): number;
    size(dim?: number): number | readonly number[];
    stride(dim?: number): number | number[];
    broadcast_to(shape: readonly number[]): Tensor;
    expand(shape: readonly number[]): Tensor;
    reshape(shape: readonly number[]): Tensor;
    view(...args: any[]): Tensor;
    squeeze(dim?: number): Tensor;
    unsqueeze(dim: number): Tensor;
    flatten(startDim?: number, endDim?: number): Tensor;
    /**
     * Returns a contiguous tensor (for now, just returns self as we assume contiguous).
     * @status implemented
     * @pytorch tensor.contiguous()
     */
    contiguous(): Tensor;
    /**
     * Splits tensor into chunks along a dimension.
     * @status implemented
     * @pytorch tensor.split()
     */
    split(split_size: number, dim?: number): Tensor[];
    chunk(chunks: number, dim?: number): Tensor[];
    movedim(source: number | number[], destination: number | number[]): Tensor;
    moveaxis(source: number | number[], destination: number | number[]): Tensor;
    swapaxes(dim0: number, dim1: number): Tensor;
    swapdims(dim0: number, dim1: number): Tensor;
    unbind(dim?: number): Tensor[];
    narrow(dim: number, start: number, length: number): Tensor;
    /**
     * Slice the tensor.
     * @param slices - Array of slice specifications or indices for each dimension.
     * @status implemented
     */
    slice(slices: (SliceSpec | number)[]): Tensor;
    private _sliceMultiDim;
    transpose(dim0: number, dim1: number): Tensor;
    permute(dims: number[]): Tensor;
    t(): Tensor;
    matmul(other: Tensor): Tensor;
    mm(other: Tensor): Tensor;
    private _matmul_impl;
    addmm(mat1: Tensor, mat2: Tensor, beta?: number, alpha?: number): Tensor;
    addmv(mat: Tensor, vec: Tensor, beta?: number, alpha?: number): Tensor;
    outer(vec2: Tensor): Tensor;
    addr(vec1: Tensor, vec2: Tensor, beta?: number, alpha?: number): Tensor;
    baddbmm(batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number): Tensor;
    addbmm(batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number): Tensor;
    mv(vec: Tensor): Tensor;
    bmm(mat2: Tensor): Tensor;
    dot(other: Tensor): Tensor;
    vdot(other: Tensor): Tensor;
    sum(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    mean(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    _expandGradAndAccumulate(scalarGrad: Tensor): void;
    private _reduceDim;
    max(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    min(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    any(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    all(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    prod(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    amax(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    amin(dim?: number | number[] | readonly number[], keepdim?: boolean): Tensor;
    var(dim?: number, correction?: number, keepdim?: boolean): Tensor;
    std(dim?: number, correction?: number, keepdim?: boolean): Tensor;
    /**
     * Returns the indices of maximum values along a dimension.
     * @pytorch tensor.argmax()
     */
    argmax(dim?: number, keepdim?: boolean): Tensor;
    /**
     * Returns the indices of minimum values along a dimension.
     * @pytorch tensor.argmin()
     */
    argmin(dim?: number, keepdim?: boolean): Tensor;
    private _reduceArg;
    toArray(): Promise<any>;
    toNestedArray(): Promise<any>;
    tolist(): Promise<any>;
    item(): Promise<number>;
    backward(gradient?: Tensor): Promise<void>;
    accumulateGrad(grad: Tensor): void;
    private _getState;
    private _nestArray;
    private _unaryOp;
    private _binaryOp;
    private _broadcastBinaryOp;
    private _reduceBroadcastGrad;
    private _sumAlongDim0;
    private _scalarOp;
    private _reduce;
    private _isChromeMac;
    private _reduceWithSharedMemory;
    private _reduceSimple;
    private _compareOp;
}

/**
 * A parameter that can be fuzzed for benchmarks.
 */
declare class FuzzedParameter {
    readonly name: string;
    readonly values: any[];
    readonly distribution?: "uniform" | "loguniform" | undefined;
    constructor(name: string, values: any[], distribution?: "uniform" | "loguniform" | undefined);
}
/**
 * A tensor with randomized shape and values for fuzzing.
 */
declare class FuzzedTensor {
    readonly name: string;
    readonly min_shape: number[];
    readonly max_shape: number[];
    readonly probability_contiguous: number;
    readonly dtype: string;
    constructor(name: string, min_shape: number[], max_shape: number[], probability_contiguous?: number, dtype?: string);
    generate(): Tensor;
}
/**
 * Generator for randomized benchmark inputs.
 */
declare class Fuzzer {
    private readonly parameters;
    private readonly seed?;
    constructor(parameters: (FuzzedParameter | FuzzedTensor)[], seed?: number | undefined);
    /**
     * Returns a generator that yields randomized inputs.
     */
    take(n: number): Generator<Record<string, any>>;
}

/**
 * Utility for comparing multiple benchmark measurements.
 */
declare class Compare {
    private readonly measurements;
    constructor(measurements: Measurement[]);
    /**
     * Print a formatted comparison table.
     */
    print(): void;
}

/**
 * An alias for a fuzzed parameter.
 */
declare class ParameterAlias {
    readonly name: string;
    readonly alias: string;
    constructor(name: string, alias: string);
}
/**
 * Operation-specific fuzzers for standard benchmarks.
 */
declare const op_fuzzers: {
    unary: (name: string) => void;
};

type __utils_benchmark_Compare = Compare;
declare const __utils_benchmark_Compare: typeof Compare;
type __utils_benchmark_FuzzedParameter = FuzzedParameter;
declare const __utils_benchmark_FuzzedParameter: typeof FuzzedParameter;
type __utils_benchmark_FuzzedTensor = FuzzedTensor;
declare const __utils_benchmark_FuzzedTensor: typeof FuzzedTensor;
type __utils_benchmark_Fuzzer = Fuzzer;
declare const __utils_benchmark_Fuzzer: typeof Fuzzer;
type __utils_benchmark_Measurement = Measurement;
declare const __utils_benchmark_Measurement: typeof Measurement;
type __utils_benchmark_ParameterAlias = ParameterAlias;
declare const __utils_benchmark_ParameterAlias: typeof ParameterAlias;
type __utils_benchmark_Timer<T = any> = Timer<T>;
declare const __utils_benchmark_Timer: typeof Timer;
declare const __utils_benchmark_op_fuzzers: typeof op_fuzzers;
declare namespace __utils_benchmark {
  export { __utils_benchmark_Compare as Compare, __utils_benchmark_FuzzedParameter as FuzzedParameter, __utils_benchmark_FuzzedTensor as FuzzedTensor, __utils_benchmark_Fuzzer as Fuzzer, __utils_benchmark_Measurement as Measurement, __utils_benchmark_ParameterAlias as ParameterAlias, __utils_benchmark_Timer as Timer, __utils_benchmark_op_fuzzers as op_fuzzers };
}

/**
 * Tensor creation operations.
 * @status partial
 */

type NestedArray = number | NestedArray[];
/**
 * Create a tensor from array data.
 * @status implemented
 * @pytorch torch.tensor()
 */
declare function tensor(data: NestedArray | number[], options?: TensorOptions): Tensor;
/**
 * Create a tensor filled with zeros.
 * @status implemented
 * @pytorch torch.zeros()
 */
declare function zeros(shape: number[], options?: TensorOptions): Tensor;
/**
 * Create a tensor filled with ones.
 * @status implemented
 * @pytorch torch.ones()
 */
declare function ones(shape: number[], options?: TensorOptions): Tensor;
/**
 * Create a tensor filled with a specific value.
 * @status implemented
 * @pytorch torch.full()
 */
declare function full(shape: number[], fillValue: number, options?: TensorOptions): Tensor;
/**
 * Create a tensor with zeros like another tensor.
 * @status implemented
 * @pytorch torch.zeros_like()
 */
declare function zeros_like(tensor: Tensor, options?: TensorOptions): Tensor;
/**
 * Create a tensor with ones like another tensor.
 * @status implemented
 * @pytorch torch.ones_like()
 */
declare function ones_like(tensor: Tensor, options?: TensorOptions): Tensor;
/**
 * Create a tensor filled with a scalar value, with the same size as input.
 * @status implemented
 * @pytorch torch.full_like()
 */
declare function full_like(input: Tensor, fillValue: number, options?: TensorOptions): Tensor;
/**
 * Returns an uninitialized tensor with the same size as input.
 * @status implemented
 * @pytorch torch.empty_like()
 */
declare function empty_like(input: Tensor, options?: TensorOptions): Tensor;
/**
 * Create a tensor with random values from uniform distribution [0, 1).
 * @status implemented
 * @pytorch torch.rand()
 */
declare function rand(shape: number[], options?: TensorOptions): Tensor;
/**
 * Create a tensor with random values from standard normal distribution.
 * @status implemented
 * @pytorch torch.randn()
 */
declare function randn(shape: number[], options?: TensorOptions): Tensor;
/**
 * Create an identity matrix.
 * @status implemented
 * @pytorch torch.eye()
 */
declare function eye(n: number, m?: number, options?: TensorOptions): Tensor;
/**
 * Create a tensor with values in a range.
 * @status implemented
 * @pytorch torch.arange()
 */
declare function arange(start: number, end?: number, step?: number, options?: TensorOptions): Tensor;
/**
 * Create a tensor with linearly spaced values.
 * @status implemented
 * @pytorch torch.linspace()
 */
declare function linspace(start: number, end: number, steps: number, options?: TensorOptions): Tensor;
/**
 * Create a tensor with logarithmically spaced values.
 * @status implemented
 * @pytorch torch.logspace()
 */
declare function logspace(start: number, end: number, steps: number, base?: number, options?: TensorOptions): Tensor;
/**
 * Lower triangular part of a matrix.
 * @status implemented
 * @pytorch torch.tril()
 */
declare function tril(input: Tensor, diagonal?: number): Tensor;
/**
 * Concatenate tensors along a dimension.
 * @status implemented
 * @pytorch torch.cat()
 */
declare function cat(tensors: Tensor[], dim?: number): Tensor;
/**
 * Stack tensors along a new dimension.
 * @status implemented
 * @pytorch torch.stack()
 */
declare function stack(tensors: Tensor[], dim?: number): Tensor;
/**
 * Stack tensors vertically (row wise).
 * @status implemented
 * @pytorch torch.vstack()
 */
declare function vstack(tensors: Tensor[]): Tensor;
declare const row_stack: typeof vstack;
/**
 * Stack tensors horizontally (column wise).
 * @status implemented
 * @pytorch torch.hstack()
 */
declare function hstack(tensors: Tensor[]): Tensor;
/**
 * Stack tensors depthwise (along third axis).
 * @status implemented
 * @pytorch torch.dstack()
 */
declare function dstack(tensors: Tensor[]): Tensor;
/**
 * Stack 1-D arrays as columns into a 2-D array.
 * @status implemented
 * @pytorch torch.column_stack()
 */
declare function column_stack(tensors: Tensor[]): Tensor;
/**
 * Set the random seed for reproducibility.
 */
declare function manual_seed(seed: number): void;

/**
 * Operations exports.
 * @status partial
 */

declare const atleast_1d: (input: Tensor) => Tensor;
declare const atleast_2d: (input: Tensor) => Tensor;
declare const atleast_3d: (input: Tensor) => Tensor;
declare const broadcast_to: (input: Tensor, shape: number[]) => Tensor;
declare const flip: (input: Tensor, dims: number[]) => Tensor;
declare const fliplr: (input: Tensor) => Tensor;
declare const flipud: (input: Tensor) => Tensor;
declare const cumsum: (input: Tensor, dim: number) => Tensor;
declare const cumprod: (input: Tensor, dim: number) => Tensor;
declare const triu: (input: Tensor, diagonal?: number) => Tensor;
declare const diag: (input: Tensor, diagonal?: number) => Tensor;
declare const heaviside: (input: Tensor, values: Tensor) => Tensor;
declare const abs: (input: Tensor) => Tensor;
declare const absolute: (input: Tensor) => Tensor;
declare const acos: (input: Tensor) => Tensor;
declare const arccos: (input: Tensor) => Tensor;
declare const acosh: (input: Tensor) => Tensor;
declare const arccosh: (input: Tensor) => Tensor;
declare const asin: (input: Tensor) => Tensor;
declare const arcsin: (input: Tensor) => Tensor;
declare const asinh: (input: Tensor) => Tensor;
declare const arcsinh: (input: Tensor) => Tensor;
declare const atan: (input: Tensor) => Tensor;
declare const arctan: (input: Tensor) => Tensor;
declare const atanh: (input: Tensor) => Tensor;
declare const arctanh: (input: Tensor) => Tensor;
declare const atan2: (input: Tensor, other: Tensor) => Tensor;
declare const arctan2: (input: Tensor, other: Tensor) => Tensor;
declare const ceil: (input: Tensor) => Tensor;
declare const floor: (input: Tensor) => Tensor;
declare const round: (input: Tensor) => Tensor;
declare const trunc: (input: Tensor) => Tensor;
declare const fix: (input: Tensor) => Tensor;
declare const frac: (input: Tensor) => Tensor;
declare const clamp: (input: Tensor, min?: number, max?: number) => Tensor;
declare const clip: (input: Tensor, min?: number, max?: number) => Tensor;
declare const flatten: (input: Tensor, startDim?: number, endDim?: number) => Tensor;
declare const squeeze: (input: Tensor, dim?: number) => Tensor;
declare const unsqueeze: (input: Tensor, dim: number) => Tensor;
declare const argmax: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
declare const argmin: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
declare const amax: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
declare const amin: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
declare const all: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
declare const any: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
declare const chunk: (input: Tensor, chunks: number, dim?: number) => Tensor[];
declare const narrow: (input: Tensor, dim: number, start: number, length: number) => Tensor;
declare const permute: (input: Tensor, dims: number[]) => Tensor;
declare const movedim: (input: Tensor, source: number | number[], destination: number | number[]) => Tensor;
declare const moveaxis: (input: Tensor, source: number | number[], destination: number | number[]) => Tensor;
declare const swapaxes: (input: Tensor, dim0: number, dim1: number) => Tensor;
declare const swapdims: (input: Tensor, dim0: number, dim1: number) => Tensor;
declare const tile: (input: Tensor, reps: number[]) => Tensor;
declare const unbind: (input: Tensor, dim?: number) => Tensor[];
declare const index_select: (input: Tensor, dim: number, index: Tensor) => Tensor;
declare const select: (input: Tensor, dim: number, index: number) => Tensor;
declare const take: (input: Tensor, indices: Tensor) => Tensor;
declare const where: (condition: Tensor, input: Tensor, other: Tensor) => Tensor;
declare const eq: (input: Tensor, other: Tensor) => Tensor;
declare const ne: (input: Tensor, other: Tensor) => Tensor;
declare const not_equal: (input: Tensor, other: Tensor) => Tensor;
declare const lt: (input: Tensor, other: Tensor) => Tensor;
declare const less: (input: Tensor, other: Tensor) => Tensor;
declare const le: (input: Tensor, other: Tensor) => Tensor;
declare const less_equal: (input: Tensor, other: Tensor) => Tensor;
declare const gt: (input: Tensor, other: Tensor) => Tensor;
declare const greater: (input: Tensor, other: Tensor) => Tensor;
declare const ge: (input: Tensor, other: Tensor) => Tensor;
declare const greater_equal: (input: Tensor, other: Tensor) => Tensor;
declare const isnan: (input: Tensor) => Tensor;
declare const isinf: (input: Tensor) => Tensor;
declare const isfinite: (input: Tensor) => Tensor;
declare const isposinf: (input: Tensor) => Tensor;
declare const isneginf: (input: Tensor) => Tensor;
declare const maximum: (input: Tensor, other: Tensor) => Tensor;
declare const minimum: (input: Tensor, other: Tensor) => Tensor;
declare const fmax: (input: Tensor, other: Tensor) => Tensor;
declare const fmin: (input: Tensor, other: Tensor) => Tensor;
declare const equal: (input: Tensor, other: Tensor) => Promise<boolean>;
declare const isclose: (input: Tensor, other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean) => Tensor;
declare const allclose: (input: Tensor, other: Tensor, rtol?: number, atol?: number, equal_nan?: boolean) => Promise<boolean>;
declare const cos: (input: Tensor) => Tensor;
declare const cosh: (input: Tensor) => Tensor;
declare const sin: (input: Tensor) => Tensor;
declare const sinh: (input: Tensor) => Tensor;
declare const tan: (input: Tensor) => Tensor;
declare const tanh$1: (input: Tensor) => Tensor;
declare const exp: (input: Tensor) => Tensor;
declare const exp2: (input: Tensor) => Tensor;
declare const log10: (input: Tensor) => Tensor;
declare const log2: (input: Tensor) => Tensor;
declare const log1p: (input: Tensor) => Tensor;
declare const logaddexp: (input: Tensor, other: Tensor) => Tensor;
declare const neg: (input: Tensor) => Tensor;
declare const negative: (input: Tensor) => Tensor;
declare const prod: (input: Tensor, dim?: number, keepdim?: boolean) => Tensor;
declare const pow: (input: Tensor, exponent: number | Tensor) => Tensor;
declare const reciprocal: (input: Tensor) => Tensor;
declare const rsqrt: (input: Tensor) => Tensor;
declare const sqrt: (input: Tensor) => Tensor;
declare const square: (input: Tensor) => Tensor;
declare const sigmoid$1: (input: Tensor) => Tensor;
declare const relu$1: (input: Tensor) => Tensor;
declare const add: (input: Tensor, other: Tensor | number) => Tensor;
declare const sub: (input: Tensor, other: Tensor | number) => Tensor;
declare const subtract: (input: Tensor, other: Tensor | number) => Tensor;
declare const mul: (input: Tensor, other: Tensor | number) => Tensor;
declare const multiply: (input: Tensor, other: Tensor | number) => Tensor;
declare const div: (input: Tensor, other: Tensor | number) => Tensor;
declare const divide: (input: Tensor, other: Tensor | number) => Tensor;
declare const bitwise_and: (input: Tensor, other: Tensor) => Tensor;
declare const bitwise_or: (input: Tensor, other: Tensor) => Tensor;
declare const bitwise_xor: (input: Tensor, other: Tensor) => Tensor;
declare const hypot: (input: Tensor, other: Tensor) => Tensor;
declare const matmul$1: (input: Tensor, other: Tensor) => Tensor;
declare const mm: (input: Tensor, mat2: Tensor) => Tensor;
declare const chain_matmul: (...matrices: Tensor[]) => Tensor;
declare const addmm: (input: Tensor, mat1: Tensor, mat2: Tensor, beta?: number, alpha?: number) => Tensor;
declare const mv: (input: Tensor, vec: Tensor) => Tensor;
declare const addmv: (input: Tensor, mat: Tensor, vec: Tensor, beta?: number, alpha?: number) => Tensor;
declare const outer: (input: Tensor, vec2: Tensor) => Tensor;
declare const ger: (input: Tensor, vec2: Tensor) => Tensor;
declare const addr: (input: Tensor, vec1: Tensor, vec2: Tensor, beta?: number, alpha?: number) => Tensor;
declare const bmm: (input: Tensor, mat2: Tensor) => Tensor;
declare const baddbmm: (input: Tensor, batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number) => Tensor;
declare const addbmm: (input: Tensor, batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number) => Tensor;
declare const dot: (input: Tensor, other: Tensor) => Tensor;
declare const vdot: (input: Tensor, other: Tensor) => Tensor;
declare const inner: (input: Tensor, other: Tensor) => Tensor;
declare const trapezoid: (input: Tensor, dx?: number, dim?: number) => Tensor;
declare const cumulative_trapezoid: (input: Tensor, dx?: number, dim?: number) => Tensor;
declare const trapz: (input: Tensor, dx?: number, dim?: number) => Tensor;

/**
 * Parameter class - a Tensor that is automatically registered as a parameter when assigned.
 * @status implemented
 * @pytorch torch.nn.Parameter
 */

/**
 * A Parameter is a Tensor that is registered as a module parameter.
 */
declare class Parameter extends Tensor {
    /**
     * Create a Parameter from data or another tensor.
     */
    static create(data: Tensor | number[] | number[][] | number[][][] | number[][][][], options?: TensorOptions): Parameter;
}

/**
 * Base Module class for neural network layers.
 * @status implemented
 * @pytorch torch.nn.Module
 */

/**
 * Base class for all neural network modules.
 */
declare abstract class Module {
    private _parameters;
    private _modules;
    private _buffers;
    private _training;
    /**
     * Whether the module is in training mode.
     */
    get training(): boolean;
    /**
     * Register a parameter with the module.
     */
    register_parameter(name: string, param: Parameter | null): void;
    /**
     * Register a submodule with the module.
     */
    add_module(name: string, module: Module | null): void;
    /**
     * Register a buffer (non-parameter tensor) with the module.
     * Buffers are not included in parameters() but are part of the module state.
     * @pytorch module.register_buffer()
     */
    register_buffer(name: string, tensor: Tensor | null): void;
    /**
     * Get a buffer by name.
     */
    get_buffer(name: string): Tensor | undefined;
    /**
     * Returns an iterator over module buffers.
     * @pytorch module.buffers()
     */
    buffers(recurse?: boolean): Generator<Tensor>;
    /**
     * Returns an iterator over named buffers.
     * @pytorch module.named_buffers()
     */
    named_buffers(prefix?: string, recurse?: boolean): Generator<[string, Tensor]>;
    /**
     * Get a parameter by name.
     */
    get_parameter(name: string): Parameter | undefined;
    /**
     * Get a submodule by name.
     */
    get_submodule(name: string): Module | undefined;
    /**
     * Returns an iterator over module parameters.
     * @pytorch module.parameters()
     */
    parameters(recurse?: boolean): Generator<Parameter>;
    /**
     * Returns an iterator over named parameters.
     * @pytorch module.named_parameters()
     */
    named_parameters(prefix?: string, recurse?: boolean): Generator<[string, Parameter]>;
    /**
     * Returns an iterator over child modules.
     * @pytorch module.children()
     */
    children(): Generator<Module>;
    /**
     * Returns an iterator over all modules (including self).
     * @pytorch module.modules()
     */
    modules(): Generator<Module>;
    /**
     * Set the module in training mode.
     * @pytorch module.train()
     */
    train(mode?: boolean): this;
    /**
     * Set the module in evaluation mode.
     * @pytorch module.eval()
     */
    eval(): this;
    /**
     * Zero out all gradients.
     * @pytorch module.zero_grad()
     */
    zero_grad(): void;
    /**
     * Returns a dictionary containing a whole state of the module.
     * @pytorch module.state_dict()
     */
    state_dict(destination?: Record<string, Tensor>, prefix?: string, recurse?: boolean): Record<string, Tensor>;
    /**
     * Copies parameters and buffers from state_dict into this module and its descendants.
     * @pytorch module.load_state_dict()
     */
    load_state_dict(state_dict: Record<string, Tensor>): void;
    /**
     * Forward pass - must be implemented by subclasses.
     */
    abstract forward(...inputs: any[]): any;
    /**
     * Call the module (alias for forward).
     */
    call(...inputs: any[]): any;
}

/**
 * Linear (fully connected) layer.
 * @status implemented
 * @pytorch torch.nn.Linear
 */

/**
 * Applies a linear transformation: y = xW^T + b
 */
declare class Linear extends Module {
    weight: Parameter;
    bias: Parameter | null;
    private in_features;
    private out_features;
    constructor(in_features: number, out_features: number, bias?: boolean);
    forward(input: Tensor): Tensor;
}

/**
 * Sequential container module.
 * @status implemented
 * @pytorch torch.nn.Sequential
 */

/**
 * A sequential container that runs modules in order.
 */
declare class Sequential extends Module {
    private moduleList;
    constructor(...modules: Module[]);
    forward(input: Tensor): Tensor;
    /**
     * Get a module by index.
     */
    get(index: number): Module | undefined;
    /**
     * Get the number of modules.
     */
    get length(): number;
}

/**
 * Flatten module.
 * @status implemented
 * @pytorch torch.nn.Flatten
 */

/**
 * Flattens a contiguous range of dims into a tensor.
 */
declare class Flatten extends Module {
    private start_dim;
    private end_dim;
    constructor(start_dim?: number, end_dim?: number);
    forward(input: Tensor): Tensor;
}

/**
 * Embedding layer for token/position embeddings.
 * @status implemented
 * @pytorch torch.nn.Embedding
 */

/**
 * A lookup table that stores embeddings of a fixed dictionary and size.
 * @pytorch torch.nn.Embedding
 */
declare class Embedding extends Module {
    readonly num_embeddings: number;
    readonly embedding_dim: number;
    weight: Parameter;
    constructor(num_embeddings: number, embedding_dim: number);
    /**
     * Look up embeddings for the given indices.
     * @param input - Tensor of indices with shape [...], dtype int32
     * @returns Tensor of embeddings with shape [..., embedding_dim]
     */
    forward(input: Tensor): Tensor;
}

/**
 * Layer Normalization.
 * @status implemented
 * @pytorch torch.nn.LayerNorm
 */

/**
 * Applies Layer Normalization over a mini-batch of inputs.
 * @pytorch torch.nn.LayerNorm
 */
declare class LayerNorm extends Module {
    readonly normalized_shape: number[];
    readonly eps: number;
    readonly elementwise_affine: boolean;
    weight: Parameter;
    bias: Parameter;
    constructor(normalized_shape: number | number[], eps?: number, elementwise_affine?: boolean);
    forward(input: Tensor): Tensor;
}

/**
 * Container modules for organizing neural network layers.
 * @status implemented
 * @pytorch torch.nn
 */

/**
 * Holds submodules in a list.
 * @pytorch torch.nn.ModuleList
 */
declare class ModuleList extends Module {
    private _list;
    constructor(modules?: Module[]);
    /**
     * Append a module to the list.
     */
    append(module: Module): this;
    /**
     * Get module at index.
     */
    get(index: number): Module;
    /**
     * Number of modules in the list.
     */
    get length(): number;
    /**
     * Iterate over modules.
     */
    [Symbol.iterator](): Generator<Module>;
    /**
     * Forward is not implemented - use modules individually.
     */
    forward(..._inputs: Tensor[]): Tensor;
}
/**
 * Holds submodules in a dictionary.
 * @pytorch torch.nn.ModuleDict
 */
declare class ModuleDict extends Module {
    private _dict;
    constructor(modules?: Record<string, Module>);
    /**
     * Get module by key.
     */
    get(key: string): Module | undefined;
    /**
     * Set module by key.
     */
    set(key: string, module: Module): void;
    /**
     * Check if key exists.
     */
    has(key: string): boolean;
    /**
     * Get all keys.
     */
    keys(): IterableIterator<string>;
    /**
     * Get all values.
     */
    values(): IterableIterator<Module>;
    /**
     * Get all entries.
     */
    entries(): IterableIterator<[string, Module]>;
    /**
     * Number of modules.
     */
    get size(): number;
    /**
     * Forward is not implemented - use modules individually.
     */
    forward(..._inputs: Tensor[]): Tensor;
}

/**
 * Activation function modules.
 * @status implemented
 * @pytorch torch.nn
 */

/**
 * ReLU activation.
 * @pytorch torch.nn.ReLU
 */
declare class ReLU extends Module {
    private inplace;
    constructor(inplace?: boolean);
    forward(input: Tensor): Tensor;
}
/**
 * Sigmoid activation.
 * @pytorch torch.nn.Sigmoid
 */
declare class Sigmoid extends Module {
    forward(input: Tensor): Tensor;
}
/**
 * Tanh activation.
 * @pytorch torch.nn.Tanh
 */
declare class Tanh extends Module {
    forward(input: Tensor): Tensor;
}
/**
 * LogSoftmax activation.
 * @pytorch torch.nn.LogSoftmax
 */
declare class LogSoftmax extends Module {
    private dim;
    constructor(dim?: number);
    forward(input: Tensor): Tensor;
}
/**
 * GELU activation (Gaussian Error Linear Unit).
 * Uses the tanh approximation as in GPT-2/BERT.
 * @pytorch torch.nn.GELU
 */
declare class GELU extends Module {
    forward(input: Tensor): Tensor;
}
/**
 * Softmax activation.
 * @pytorch torch.nn.Softmax
 */
declare class Softmax extends Module {
    private dim;
    constructor(dim?: number);
    forward(input: Tensor): Tensor;
}

/**
 * Dropout module.
 * @status partial
 * @pytorch torch.nn.Dropout
 */

/**
 * During training, randomly zeroes some elements of the input tensor.
 */
declare class Dropout extends Module {
    private p;
    private _inplace;
    constructor(p?: number, inplace?: boolean);
    forward(input: Tensor): Tensor;
}
/**
 * 2D Dropout for channel-wise dropout on 4D tensors.
 * @pytorch torch.nn.Dropout2d
 */
declare class Dropout2d extends Module {
    private p;
    private _inplace;
    constructor(p?: number, inplace?: boolean);
    forward(input: Tensor): Tensor;
}

/**
 * Functional interface for neural network operations.
 * @status partial
 * @pytorch torch.nn.functional
 */

/**
 * Apply ReLU activation function.
 * @status implemented
 * @pytorch F.relu
 */
declare function relu(input: Tensor, inplace?: boolean): Tensor;
/**
 * Apply GELU activation function.
 * Uses the tanh approximation as in GPT-2/BERT.
 * @status implemented
 * @pytorch F.gelu
 */
declare function gelu(input: Tensor): Tensor;
declare function sigmoid(input: Tensor): Tensor;
declare function tanh(input: Tensor): Tensor;
declare function softplus(input: Tensor): Tensor;
declare function silu(input: Tensor): Tensor;
declare function mish(input: Tensor): Tensor;
declare function hardsigmoid(input: Tensor): Tensor;
declare function hardswish(input: Tensor): Tensor;
declare function softsign(input: Tensor): Tensor;
declare function tanhshrink(input: Tensor): Tensor;
declare function leaky_relu(input: Tensor, negative_slope?: number): Tensor;
declare function elu(input: Tensor, alpha?: number): Tensor;
declare function selu(input: Tensor): Tensor;
declare function glu(input: Tensor, dim?: number): Tensor;
declare function threshold(input: Tensor, threshold: number, value: number): Tensor;
/**
 * Apply softmax along the specified dimension.
 * @status implemented
 * @pytorch F.softmax
 */
declare function softmax(input: Tensor, dim?: number): Tensor;
/**
 * Apply log_softmax along the specified dimension.
 * @status implemented
 * @pytorch F.log_softmax
 */
declare function log_softmax(input: Tensor, dim?: number): Tensor;
/**
 * Negative log likelihood loss.
 * @status implemented
 * @pytorch F.nll_loss
 */
declare function nll_loss(input: Tensor, target: Tensor, reduction?: 'mean' | 'sum' | 'none'): Tensor;
/**
 * Cross entropy loss (combines log_softmax and nll_loss).
 * @status implemented
 * @pytorch F.cross_entropy
 */
declare function cross_entropy(input: Tensor, target: Tensor, reduction?: 'mean' | 'sum' | 'none'): Tensor;
/**
 * Apply dropout.
 * @status implemented
 * @pytorch F.dropout
 */
declare function dropout(input: Tensor, p?: number, training?: boolean): Tensor;
/**
 * Apply 2D max pooling.
 * @status partial
 * @pytorch F.max_pool2d
 */
declare function max_pool2d(input: Tensor, kernel_size: number | [number, number], stride?: number | [number, number], padding?: number | [number, number]): Tensor;
/**
 * Apply 2D convolution.
 * @status partial
 * @pytorch F.conv2d
 */
declare function conv2d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | [number, number], padding?: number | [number, number]): Tensor;

declare const functional_conv2d: typeof conv2d;
declare const functional_cross_entropy: typeof cross_entropy;
declare const functional_dropout: typeof dropout;
declare const functional_elu: typeof elu;
declare const functional_gelu: typeof gelu;
declare const functional_glu: typeof glu;
declare const functional_hardsigmoid: typeof hardsigmoid;
declare const functional_hardswish: typeof hardswish;
declare const functional_leaky_relu: typeof leaky_relu;
declare const functional_log_softmax: typeof log_softmax;
declare const functional_max_pool2d: typeof max_pool2d;
declare const functional_mish: typeof mish;
declare const functional_nll_loss: typeof nll_loss;
declare const functional_relu: typeof relu;
declare const functional_selu: typeof selu;
declare const functional_sigmoid: typeof sigmoid;
declare const functional_silu: typeof silu;
declare const functional_softmax: typeof softmax;
declare const functional_softplus: typeof softplus;
declare const functional_softsign: typeof softsign;
declare const functional_tanh: typeof tanh;
declare const functional_tanhshrink: typeof tanhshrink;
declare const functional_threshold: typeof threshold;
declare namespace functional {
  export { functional_conv2d as conv2d, functional_cross_entropy as cross_entropy, functional_dropout as dropout, functional_elu as elu, functional_gelu as gelu, functional_glu as glu, functional_hardsigmoid as hardsigmoid, functional_hardswish as hardswish, functional_leaky_relu as leaky_relu, functional_log_softmax as log_softmax, functional_max_pool2d as max_pool2d, functional_mish as mish, functional_nll_loss as nll_loss, functional_relu as relu, functional_selu as selu, functional_sigmoid as sigmoid, functional_silu as silu, functional_softmax as softmax, functional_softplus as softplus, functional_softsign as softsign, functional_tanh as tanh, functional_tanhshrink as tanhshrink, functional_threshold as threshold };
}

/**
 * Neural network module exports.
 * @status partial
 * @pytorch torch.nn
 */

type nn_Dropout = Dropout;
declare const nn_Dropout: typeof Dropout;
type nn_Dropout2d = Dropout2d;
declare const nn_Dropout2d: typeof Dropout2d;
type nn_Embedding = Embedding;
declare const nn_Embedding: typeof Embedding;
type nn_Flatten = Flatten;
declare const nn_Flatten: typeof Flatten;
type nn_GELU = GELU;
declare const nn_GELU: typeof GELU;
type nn_LayerNorm = LayerNorm;
declare const nn_LayerNorm: typeof LayerNorm;
type nn_Linear = Linear;
declare const nn_Linear: typeof Linear;
type nn_LogSoftmax = LogSoftmax;
declare const nn_LogSoftmax: typeof LogSoftmax;
type nn_Module = Module;
declare const nn_Module: typeof Module;
type nn_ModuleDict = ModuleDict;
declare const nn_ModuleDict: typeof ModuleDict;
type nn_ModuleList = ModuleList;
declare const nn_ModuleList: typeof ModuleList;
type nn_Parameter = Parameter;
declare const nn_Parameter: typeof Parameter;
type nn_ReLU = ReLU;
declare const nn_ReLU: typeof ReLU;
type nn_Sequential = Sequential;
declare const nn_Sequential: typeof Sequential;
type nn_Sigmoid = Sigmoid;
declare const nn_Sigmoid: typeof Sigmoid;
type nn_Softmax = Softmax;
declare const nn_Softmax: typeof Softmax;
type nn_Tanh = Tanh;
declare const nn_Tanh: typeof Tanh;
declare const nn_cross_entropy: typeof cross_entropy;
declare const nn_dropout: typeof dropout;
declare const nn_functional: typeof functional;
declare const nn_gelu: typeof gelu;
declare const nn_log_softmax: typeof log_softmax;
declare const nn_nll_loss: typeof nll_loss;
declare const nn_relu: typeof relu;
declare const nn_softmax: typeof softmax;
declare namespace nn {
  export { nn_Dropout as Dropout, nn_Dropout2d as Dropout2d, nn_Embedding as Embedding, functional as F, nn_Flatten as Flatten, nn_GELU as GELU, nn_LayerNorm as LayerNorm, nn_Linear as Linear, nn_LogSoftmax as LogSoftmax, nn_Module as Module, nn_ModuleDict as ModuleDict, nn_ModuleList as ModuleList, nn_Parameter as Parameter, nn_ReLU as ReLU, nn_Sequential as Sequential, nn_Sigmoid as Sigmoid, nn_Softmax as Softmax, nn_Tanh as Tanh, nn_cross_entropy as cross_entropy, nn_dropout as dropout, nn_functional as functional, nn_gelu as gelu, nn_log_softmax as log_softmax, nn_nll_loss as nll_loss, nn_relu as relu, nn_softmax as softmax };
}

/**
 * Base Optimizer class.
 * @status implemented
 * @pytorch torch.optim.Optimizer
 */

interface ParamGroup {
    params: Tensor[];
    lr: number;
    [key: string]: unknown;
}
/**
 * Base class for all optimizers.
 */
declare abstract class Optimizer {
    param_groups: ParamGroup[];
    protected defaults: Record<string, unknown>;
    constructor(params: Tensor[] | Iterable<Tensor>, defaults: Record<string, unknown>);
    /**
     * Clears the gradients of all optimized parameters.
     * @pytorch optimizer.zero_grad()
     */
    zero_grad(): void;
    /**
     * Performs a single optimization step.
     * @pytorch optimizer.step()
     */
    abstract step(): Promise<void>;
}

/**
 * Stochastic Gradient Descent optimizer.
 * @status implemented
 * @pytorch torch.optim.SGD
 */

interface SGDOptions {
    lr: number;
    momentum?: number;
    weight_decay?: number;
    dampening?: number;
    nesterov?: boolean;
}
/**
 * Implements stochastic gradient descent (optionally with momentum).
 */
declare class SGD extends Optimizer {
    private momentum_buffers;
    constructor(params: Tensor[] | Iterable<Tensor>, options: SGDOptions);
    /**
     * Performs a single optimization step.
     */
    step(): Promise<void>;
    /**
     * Update a parameter's buffer with new values.
     * @internal
     */
    private _updateParamBuffer;
}

/**
 * AdamW optimizer.
 * @status implemented
 * @pytorch torch.optim.AdamW
 */

interface AdamWOptions {
    lr?: number;
    betas?: [number, number];
    eps?: number;
    weight_decay?: number;
    amsgrad?: boolean;
}
/**
 * Implements AdamW algorithm.
 */
declare class AdamW extends Optimizer {
    private state;
    constructor(params: Tensor[] | Iterable<Tensor>, options?: AdamWOptions);
    /**
     * Performs a single optimization step.
     */
    step(): Promise<void>;
    /**
     * Update a parameter's buffer with new values.
     * @internal
     */
    private _updateParamBuffer;
}

/**
 * Adam optimizer.
 * @status implemented
 * @pytorch torch.optim.Adam
 */

interface AdamOptions {
    lr?: number;
    betas?: [number, number];
    eps?: number;
    weight_decay?: number;
    amsgrad?: boolean;
}
/**
 * Implements Adam algorithm.
 */
declare class Adam extends Optimizer {
    private state;
    constructor(params: Tensor[] | Iterable<Tensor>, options?: AdamOptions);
    /**
     * Performs a single optimization step.
     */
    step(): Promise<void>;
    /**
     * Update a parameter's buffer with new values.
     * @internal
     */
    private _updateParamBuffer;
}

/**
 * Optimizer exports.
 * @status partial
 * @pytorch torch.optim
 */

type optim_Adam = Adam;
declare const optim_Adam: typeof Adam;
type optim_AdamOptions = AdamOptions;
type optim_AdamW = AdamW;
declare const optim_AdamW: typeof AdamW;
type optim_AdamWOptions = AdamWOptions;
type optim_Optimizer = Optimizer;
declare const optim_Optimizer: typeof Optimizer;
type optim_ParamGroup = ParamGroup;
type optim_SGD = SGD;
declare const optim_SGD: typeof SGD;
type optim_SGDOptions = SGDOptions;
declare namespace optim {
  export { optim_Adam as Adam, type optim_AdamOptions as AdamOptions, optim_AdamW as AdamW, type optim_AdamWOptions as AdamWOptions, optim_Optimizer as Optimizer, type optim_ParamGroup as ParamGroup, optim_SGD as SGD, type optim_SGDOptions as SGDOptions };
}

/**
 * Computes the LU factorization with partial pivoting of a matrix A.
 * @pytorch torch.linalg.lu_factor
 */
declare const lu_factor: (A: Tensor, pivot?: boolean) => [Tensor, Tensor];
/**
 * Computes the solution
 to a system of linear equations with a triangular coefficient matrix A and multiple right-hand sides B.
* @pytorch torch.linalg.solve_triangular
*/
declare const solve_triangular: (A: Tensor, B: Tensor, upper: boolean, left?: boolean, unitriangular?: boolean) => Tensor;
/**
 * Computes the Cholesky decomposition
 of a complex Hermitian or real symmetric positive-definite matrix.
* @pytorch torch.linalg.cholesky
*/
declare const cholesky: (A: Tensor, upper?: boolean) => Tensor;
/**
 * Computes a vector or matrix norm.
 * @pytorch torch.linalg.norm
 */
declare const norm: (input: Tensor, ord?: number | string, dim?: number | number[], keepdim?: boolean) => Tensor;
/**
 * Computes a vector norm.
 * @pytorch torch.linalg.vector_norm
 */
declare const vector_norm: (input: Tensor, ord?: number | string, dim?: number | number[], keepdim?: boolean) => Tensor;
/**
 * Computes a matrix norm.
 * @pytorch torch.linalg.matrix_norm
 */
declare const matrix_norm: (input: Tensor, ord?: number | string, dim?: [number, number], keepdim?: boolean) => Tensor;
/**
 * Alias for torch.diagonal() with defaults dim1= -2, dim2= -1.
 * @pytorch torch.linalg.diagonal
 */
declare const diagonal: (input: Tensor, offset?: number, dim1?: number, dim2?: number) => Tensor;
/**
 * Alias for torch.matmul()
 * @pytorch torch.linalg.matmul
 */
declare const matmul: (input: Tensor, other: Tensor) => Tensor;
/**
 * Computes the dot product of two batches of vectors along a dimension.
 * @pytorch torch.linalg.vecdot
 */
declare const vecdot: (x: Tensor, y: Tensor, dim?: number) => Tensor;
/**
 * Generates a Vandermonde matrix.
 * @pytorch torch.linalg.vander
 */
declare const vander: (x: Tensor, N?: number, increasing?: boolean) => Tensor;
/**
 * Computes the n-th power of a square matrix for an integer n.
 * @pytorch torch.linalg.matrix_power
 */
declare const matrix_power: (A: Tensor, n: number) => Tensor;
/**
 * Computes the inverse of a square matrix if it exists.
 * @pytorch torch.linalg.inv
 */
declare const inv: (A: Tensor) => Tensor;
/**
 * Computes the determinant of a square matrix.
 * @pytorch torch.linalg.det
 */
declare const det: (A: Tensor) => Tensor;
/**
 * Computes the cross product of two 3-dimensional vectors.
 * @pytorch torch.linalg.cross
 */
declare const cross: (input: Tensor, other: Tensor, dim?: number) => Tensor;

declare const linalg_cholesky: typeof cholesky;
declare const linalg_cross: typeof cross;
declare const linalg_det: typeof det;
declare const linalg_diagonal: typeof diagonal;
declare const linalg_inv: typeof inv;
declare const linalg_lu_factor: typeof lu_factor;
declare const linalg_matmul: typeof matmul;
declare const linalg_matrix_norm: typeof matrix_norm;
declare const linalg_matrix_power: typeof matrix_power;
declare const linalg_norm: typeof norm;
declare const linalg_solve_triangular: typeof solve_triangular;
declare const linalg_vander: typeof vander;
declare const linalg_vecdot: typeof vecdot;
declare const linalg_vector_norm: typeof vector_norm;
declare namespace linalg {
  export { linalg_cholesky as cholesky, linalg_cross as cross, linalg_det as det, linalg_diagonal as diagonal, linalg_inv as inv, linalg_lu_factor as lu_factor, linalg_matmul as matmul, linalg_matrix_norm as matrix_norm, linalg_matrix_power as matrix_power, linalg_norm as norm, linalg_solve_triangular as solve_triangular, linalg_vander as vander, linalg_vecdot as vecdot, linalg_vector_norm as vector_norm };
}

/**
 * PyTorch-compatible WebGPU-style memory management.
 * Since we use WebGPU, this maps to GPU memory stats.
 */
/**
 * Returns a dictionary of WebGPU memory allocator statistics.
 */
declare function memory_stats(): {
    active_bytes: number;
    pooled_bytes: number;
    peak_bytes: number;
    total_allocations: number;
};
/**
 * Returns a human-readable summary of the memory allocator statistics.
 */
declare function memory_summary(): string;
/**
 * Releases all unoccupied cached memory.
 */
declare function empty_cache(): void;
/**
 * Resets the peak memory stats.
 */
declare function reset_peak_memory_stats(): void;

declare const webgpu_empty_cache: typeof empty_cache;
declare const webgpu_memory_stats: typeof memory_stats;
declare const webgpu_memory_summary: typeof memory_summary;
declare const webgpu_reset_peak_memory_stats: typeof reset_peak_memory_stats;
declare namespace webgpu {
  export { webgpu_empty_cache as empty_cache, webgpu_memory_stats as memory_stats, webgpu_memory_summary as memory_summary, webgpu_reset_peak_memory_stats as reset_peak_memory_stats };
}

type SaveFunc = (obj: any, f?: string) => Promise<void>;
type LoadFunc = (f: string | File | Blob) => Promise<any>;

/**
 * Creates the torch object with specific serialization implementations.
 */
declare function createTorch(save: SaveFunc, load: LoadFunc): {
    linalg: typeof linalg;
    syncDevice: typeof syncDevice;
    nn: typeof nn;
    optim: typeof optim;
    webgpu: typeof webgpu;
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
};

declare enum ProfilerActivity {
    CPU = "cpu",
    CUDA = "cuda",// Alias for compatibility
    WEBGPU = "webgpu"
}
interface ProfilerOptions {
    activities?: ProfilerActivity[];
    record_shapes?: boolean;
    profile_memory?: boolean;
    with_stack?: boolean;
}
interface ProfilerEvent {
    name: string;
    activity: ProfilerActivity;
    start: number;
    end?: number;
    duration?: number;
    shapes?: number[][];
    memory_usage?: number;
}
/**
 * Collection of averages grouped by name/shape.
 */
declare class KinetoStep {
    private events;
    constructor(events: ProfilerEvent[]);
    key_averages(group_by_input_shape?: boolean): KinetoStep;
    table(options?: {
        sort_by?: string;
        row_limit?: number;
    }): string;
}
/**
 * Profiler for tracking operation execution.
 */
declare class Profiler {
    private events;
    private active;
    private options;
    constructor(options?: ProfilerOptions);
    start(): void;
    stop(): Promise<void>;
    _record_event(event: ProfilerEvent): void;
    get_events(): ProfilerEvent[];
    key_averages(): KinetoStep;
}
/**
 * High-level profiling context.
 */
declare function profile(options: ProfilerOptions | ((p: Profiler) => Promise<void> | void), fn?: (p: Profiler) => Promise<void> | void): Promise<Profiler>;
/**
 * Synchronous utility to record a function block.
 * For WebGPU, this measures CPU time up to the submission.
 */
declare function record_function<T>(name: string, fn: () => T): T;
/**
 * Asynchronous version of record_function that includes synchronization.
 */
declare function record_function_async<T>(name: string, fn: () => Promise<T>): Promise<T>;

type index_KinetoStep = KinetoStep;
declare const index_KinetoStep: typeof KinetoStep;
type index_Profiler = Profiler;
declare const index_Profiler: typeof Profiler;
type index_ProfilerActivity = ProfilerActivity;
declare const index_ProfilerActivity: typeof ProfilerActivity;
type index_ProfilerEvent = ProfilerEvent;
type index_ProfilerOptions = ProfilerOptions;
declare const index_profile: typeof profile;
declare const index_record_function: typeof record_function;
declare const index_record_function_async: typeof record_function_async;
declare namespace index {
  export { index_KinetoStep as KinetoStep, index_Profiler as Profiler, index_ProfilerActivity as ProfilerActivity, type index_ProfilerEvent as ProfilerEvent, type index_ProfilerOptions as ProfilerOptions, index_profile as profile, index_record_function as record_function, index_record_function_async as record_function_async };
}

/**
 * Debug utilities for torch.js
 *
 * Usage:
 *   DEBUG(() => {
 *     console.log('expensive debug computation', someValue);
 *     // This entire block is stripped in production
 *   });
 *
 * In production builds (NODE_ENV=production), the DEBUG function becomes
 * a no-op and esbuild eliminates the callback entirely via dead code elimination.
 *
 * @status implemented
 */
/**
 * Execute a debug callback. The entire callback is stripped in production builds.
 *
 * @example
 * DEBUG(() => {
 *   const data = await readBuffer(buffer, dtype, size);
 *   console.log('Buffer contents:', data);
 * });
 */
declare function DEBUG(fn: () => void): void;
/**
 * Async version of DEBUG for callbacks that need await.
 */
declare function DEBUG_ASYNC(fn: () => Promise<void>): void;
/**
 * Debug log helper - use inside DEBUG() blocks.
 */
declare function log(message: string, ...args: unknown[]): void;
/**
 * Debug warn helper - use inside DEBUG() blocks.
 */
declare function warn(message: string, ...args: unknown[]): void;
/**
 * Debug error helper - use inside DEBUG() blocks.
 */
declare function error(message: string, ...args: unknown[]): void;
/**
 * Debug assert - use inside DEBUG() blocks.
 */
declare function assert(condition: boolean, message: string): void;

/**
 * torch.js - WebGPU-powered tensor library with PyTorch-compatible API.
 * @status partial
 */

declare const utils: {
    benchmark: typeof __utils_benchmark;
};

declare const _internals: {
    getDevice: typeof getDevice;
    getOrCreatePipeline: typeof getOrCreatePipeline;
    calculateWorkgroups: typeof calculateWorkgroups;
    readBuffer: typeof readBuffer;
};

export { atleast_3d as $, dstack as A, column_stack as B, initWebGPU as C, DEBUG as D, DEBUG_ASYNC as E, warn as F, error as G, assert as H, getDevice as I, getOrCreatePipeline as J, calculateWorkgroups as K, type LoadFunc as L, readBuffer as M, createTorch as N, BufferUsage as O, type DType as P, type TensorOptions as Q, utils as R, type SaveFunc as S, Tensor as T, _internals as U, type GradFn as V, type TypedArray as W, log as X, atleast_1d as Y, atleast_2d as Z, __utils_benchmark as _, ones as a, greater as a$, broadcast_to as a0, flip as a1, fliplr as a2, flipud as a3, cumsum as a4, cumprod as a5, triu as a6, diag as a7, heaviside as a8, abs as a9, argmax as aA, argmin as aB, amax as aC, amin as aD, all as aE, any as aF, chunk as aG, narrow as aH, permute as aI, movedim as aJ, moveaxis as aK, swapaxes as aL, swapdims as aM, tile as aN, unbind as aO, index_select as aP, select as aQ, take as aR, where as aS, eq as aT, ne as aU, not_equal as aV, lt as aW, less as aX, le as aY, less_equal as aZ, gt as a_, absolute as aa, acos as ab, arccos as ac, acosh as ad, arccosh as ae, asin as af, arcsin as ag, asinh as ah, arcsinh as ai, atan as aj, arctan as ak, atanh as al, arctanh as am, atan2 as an, arctan2 as ao, ceil as ap, floor as aq, round as ar, trunc as as, fix as at, frac as au, clamp as av, clip as aw, flatten as ax, squeeze as ay, unsqueeze as az, zeros_like as b, cumulative_trapezoid as b$, ge as b0, greater_equal as b1, isnan as b2, isinf as b3, isfinite as b4, isposinf as b5, isneginf as b6, maximum as b7, minimum as b8, fmax as b9, add as bA, sub as bB, subtract as bC, mul as bD, multiply as bE, div as bF, divide as bG, bitwise_and as bH, bitwise_or as bI, bitwise_xor as bJ, hypot as bK, matmul$1 as bL, mm as bM, chain_matmul as bN, addmm as bO, mv as bP, addmv as bQ, outer as bR, ger as bS, addr as bT, bmm as bU, baddbmm as bV, addbmm as bW, dot as bX, vdot as bY, inner as bZ, trapezoid as b_, fmin as ba, equal as bb, isclose as bc, allclose as bd, cos as be, cosh as bf, sin as bg, sinh as bh, tan as bi, tanh$1 as bj, exp as bk, exp2 as bl, log10 as bm, log2 as bn, log1p as bo, logaddexp as bp, neg as bq, negative as br, prod as bs, pow as bt, reciprocal as bu, rsqrt as bv, sqrt as bw, square as bx, sigmoid$1 as by, relu$1 as bz, ones_like as c, trapz as c0, row_stack as c1, full_like as d, empty_like as e, full as f, randn as g, eye as h, index as i, arange as j, linspace as k, linalg as l, logspace as m, nn as n, optim as o, manual_seed as p, tril as q, rand as r, syncDevice as s, tensor as t, cat as u, stack as v, webgpu as w, vstack as x, hstack as y, zeros as z };
