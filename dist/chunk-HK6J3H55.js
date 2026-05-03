var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// src/dtype.ts
var DTypeInfo = {
  float32: { bytes: 4, arrayType: Float32Array },
  float16: { bytes: 2, arrayType: Float32Array },
  // Use Float32Array, convert at GPU boundary
  int32: { bytes: 4, arrayType: Int32Array },
  uint32: { bytes: 4, arrayType: Uint32Array },
  int8: { bytes: 1, arrayType: Int8Array },
  uint8: { bytes: 1, arrayType: Uint8Array },
  bool: { bytes: 1, arrayType: Uint8Array }
};
function getDTypeBytes(dtype) {
  return DTypeInfo[dtype].bytes;
}
function getTypedArrayConstructor(dtype) {
  return DTypeInfo[dtype].arrayType;
}

// src/backend/webgpu/types.ts
function isNode() {
  return typeof process !== "undefined" && process.versions?.node !== void 0;
}
var _BufferUsage;
var _MapMode;
if (isNode()) {
  _BufferUsage = {
    MAP_READ: 1,
    MAP_WRITE: 2,
    COPY_SRC: 4,
    COPY_DST: 8,
    INDEX: 16,
    VERTEX: 32,
    UNIFORM: 64,
    STORAGE: 128,
    INDIRECT: 256,
    QUERY_RESOLVE: 512
  };
  _MapMode = {
    READ: 1,
    WRITE: 2
  };
} else {
  _BufferUsage = {
    MAP_READ: GPUBufferUsage?.MAP_READ ?? 1,
    MAP_WRITE: GPUBufferUsage?.MAP_WRITE ?? 2,
    COPY_SRC: GPUBufferUsage?.COPY_SRC ?? 4,
    COPY_DST: GPUBufferUsage?.COPY_DST ?? 8,
    INDEX: GPUBufferUsage?.INDEX ?? 16,
    VERTEX: GPUBufferUsage?.VERTEX ?? 32,
    UNIFORM: GPUBufferUsage?.UNIFORM ?? 64,
    STORAGE: GPUBufferUsage?.STORAGE ?? 128,
    INDIRECT: GPUBufferUsage?.INDIRECT ?? 256,
    QUERY_RESOLVE: GPUBufferUsage?.QUERY_RESOLVE ?? 512
  };
  _MapMode = {
    READ: GPUMapMode?.READ ?? 1,
    WRITE: GPUMapMode?.WRITE ?? 2
  };
}
var BufferUsage = _BufferUsage;
var MapMode = _MapMode;

// src/backend/webgpu/tgpu.ts
import tgpu from "typegpu";
import * as d from "typegpu/data";
var root = null;
async function initTypegpu(existingDevice) {
  if (root) return root;
  if (existingDevice) {
    root = tgpu.initFromDevice({ device: existingDevice });
  } else {
    root = await tgpu.init();
  }
  return root;
}
var TensorSchemas = {
  // Float32 array for tensor data
  f32Array: (maxSize) => d.arrayOf(d.f32, maxSize),
  // Int32 array for indices
  i32Array: (maxSize) => d.arrayOf(d.i32, maxSize),
  // Uint32 array for shapes/strides
  u32Array: (maxSize) => d.arrayOf(d.u32, maxSize),
  // Uniform params for elementwise ops
  ElementwiseParams: d.struct({
    size: d.u32,
    _pad1: d.u32,
    _pad2: d.u32,
    _pad3: d.u32
  }),
  // Uniform params for scalar ops
  ScalarParams: d.struct({
    scalar: d.f32,
    size: d.u32,
    _pad1: d.u32,
    _pad2: d.u32
  }),
  // Uniform params for matmul
  MatmulParams: d.struct({
    M: d.u32,
    K: d.u32,
    N: d.u32,
    _pad: d.u32
  }),
  // Uniform params for reduction
  ReduceParams: d.struct({
    inputSize: d.u32,
    _pad1: d.u32,
    _pad2: d.u32,
    _pad3: d.u32
  })
};
var Schemas = {
  // Simple fill operation params
  FillParams: d.struct({
    value: d.f32,
    length: d.u32
  }),
  // RNG params
  RngParams: d.struct({
    seed: d.u32,
    length: d.u32
  }),
  // 2D dimension params (batch_size, num_classes)
  Dims2D: d.struct({
    dim0: d.u32,
    dim1: d.u32
  })
};

// src/backend/webgpu/device.ts
var gpuDevice = null;
var gpuAdapter = null;
var tgpuRoot = null;
var initialized = false;
var initPromise = null;
async function initWebGPU(customGpu) {
  if (initialized) return;
  if (initPromise) return initPromise;
  initPromise = (async () => {
    const gpuProvider = customGpu || (typeof navigator !== "undefined" ? navigator.gpu : null);
    if (!gpuProvider) {
      throw new Error(
        "WebGPU provider not found. In a browser, WebGPU may not be supported. In Node.js, you must provide a WebGPU implementation (e.g. via @torchjsorg/torch.node.js)."
      );
    }
    const adapter = await gpuProvider.requestAdapter({
      powerPreference: "high-performance"
    });
    if (!adapter) {
      throw new Error("Failed to get WebGPU adapter");
    }
    gpuAdapter = adapter;
    const requiredFeatures = [];
    if (adapter.features && adapter.features.has("timestamp-query")) {
      requiredFeatures.push("timestamp-query");
    }
    gpuDevice = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
      }
    });
    if ("lost" in gpuDevice && gpuDevice.lost) {
      gpuDevice.lost.then((info) => {
        console.error("WebGPU device lost:", info.message);
        initialized = false;
        gpuDevice = null;
        initPromise = null;
      });
    }
    initialized = true;
    tgpuRoot = await initTypegpu(gpuDevice);
    await detectCapabilities();
  })();
  return initPromise;
}
function getDevice() {
  if (!gpuDevice) {
    throw new Error("WebGPU not initialized. Call torch.init() first.");
  }
  return gpuDevice;
}
function getAdapter() {
  if (!gpuAdapter) {
    throw new Error("WebGPU not initialized. Call torch.init() first.");
  }
  return gpuAdapter;
}

// src/backend/webgpu/shaders/elementwise.wgsl
var elementwise_default = "@group(0) @binding(0) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<f32>;\n@group(0) @binding(2) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(256)\nfn add(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = a[idx] + b[idx];\n}\n\n@compute @workgroup_size(256)\nfn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = a[idx] - b[idx];\n}\n\n@compute @workgroup_size(256)\nfn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = a[idx] * b[idx];\n}\n\n@compute @workgroup_size(256)\nfn div_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = a[idx] / b[idx];\n}\n\n@compute @workgroup_size(256)\nfn atan2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = atan2(a[idx], b[idx]);\n}\n\n@compute @workgroup_size(256)\nfn hypot_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = sqrt(a[idx] * a[idx] + b[idx] * b[idx]);\n}\n\n@compute @workgroup_size(256)\nfn logaddexp(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = a[idx];\n    let y = b[idx];\n    let max_val = max(x, y);\n    result[idx] = max_val + log(exp(x - max_val) + exp(y - max_val));\n}\n\n@compute @workgroup_size(256)\nfn bitwise_and(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = bitcast<f32>(bitcast<i32>(a[idx]) & bitcast<i32>(b[idx]));\n}\n\n@compute @workgroup_size(256)\nfn bitwise_or(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = bitcast<f32>(bitcast<i32>(a[idx]) | bitcast<i32>(b[idx]));\n}\n\n@compute @workgroup_size(256)\nfn bitwise_xor(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = bitcast<f32>(bitcast<i32>(a[idx]) ^ bitcast<i32>(b[idx]));\n}\n";

// src/backend/webgpu/shaders/scalar.wgsl
var scalar_default = "struct Params {\n    scalar: f32,\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn add_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.length) {\n        return;\n    }\n    result[idx] = input[idx] + params.scalar;\n}\n\n@compute @workgroup_size(256)\nfn sub_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.length) {\n        return;\n    }\n    result[idx] = input[idx] - params.scalar;\n}\n\n@compute @workgroup_size(256)\nfn mul_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.length) {\n        return;\n    }\n    result[idx] = input[idx] * params.scalar;\n}\n\n@compute @workgroup_size(256)\nfn div_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.length) {\n        return;\n    }\n    result[idx] = input[idx] / params.scalar;\n}\n\n@compute @workgroup_size(256)\nfn pow_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.length) {\n        return;\n    }\n    \n    let base = input[idx];\n    let exponent = params.scalar;\n    \n    // Improved precision for integer powers (e.g. 3^2 = 9 exactly)\n    let exp_int = i32(round(exponent));\n    if (abs(exponent - f32(exp_int)) < 1e-6 && exp_int >= 0 && exp_int <= 10) {\n        var res = 1.0;\n        for (var i = 0; i < exp_int; i = i + 1) {\n            res = res * base;\n        }\n        result[idx] = res;\n        } else {\n            result[idx] = pow(base, exponent);\n        }\n    }\n    \n    @compute @workgroup_size(256)\n    fn eq_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n        let idx = global_id.x;\n        if (idx >= params.length) { return; }\n        result[idx] = select(0.0, 1.0, input[idx] == params.scalar);\n    }\n    \n    @compute @workgroup_size(256)\n    fn ne_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n        let idx = global_id.x;\n        if (idx >= params.length) { return; }\n        result[idx] = select(0.0, 1.0, input[idx] != params.scalar);\n    }\n    \n    @compute @workgroup_size(256)\n    fn lt_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n        let idx = global_id.x;\n        if (idx >= params.length) { return; }\n        result[idx] = select(0.0, 1.0, input[idx] < params.scalar);\n    }\n    \n    @compute @workgroup_size(256)\n    fn le_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n        let idx = global_id.x;\n        if (idx >= params.length) { return; }\n        result[idx] = select(0.0, 1.0, input[idx] <= params.scalar);\n    }\n    \n    @compute @workgroup_size(256)\n    fn gt_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n        let idx = global_id.x;\n        if (idx >= params.length) { return; }\n        result[idx] = select(0.0, 1.0, input[idx] > params.scalar);\n    }\n    \n    @compute @workgroup_size(256)\n    fn ge_scalar(@builtin(global_invocation_id) global_id: vec3<u32>) {\n        let idx = global_id.x;\n        if (idx >= params.length) { return; }\n        result[idx] = select(0.0, 1.0, input[idx] >= params.scalar);\n    }\n    ";

// src/backend/webgpu/shaders/unary.wgsl
var unary_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(256)\nfn neg(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = -input[idx];\n}\n\n@compute @workgroup_size(256)\nfn abs_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = abs(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn sqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = sqrt(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn exp_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = exp(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn exp2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = exp2(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn log_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = log(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn log10(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = log(input[idx]) / 2.302585092994046;\n}\n\n@compute @workgroup_size(256)\nfn log2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = log2(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn log1p(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = log(1.0 + input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn tanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = tanh(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = 1.0 / (1.0 + exp(-input[idx]));\n}\n\n@compute @workgroup_size(256)\nfn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = max(0.0, input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn sin_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = sin(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn cos_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = cos(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn tan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = tan(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn acos_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = acos(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn asin_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = asin(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn atan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = atan(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn cosh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = cosh(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn sinh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = sinh(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn acosh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = log(x + sqrt(x * x - 1.0));\n}\n\n@compute @workgroup_size(256)\nfn asinh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = log(x + sqrt(x * x + 1.0));\n}\n\n@compute @workgroup_size(256)\nfn atanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = 0.5 * log((1.0 + x) / (1.0 - x));\n}\n\n@compute @workgroup_size(256)\nfn ceil_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = ceil(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn floor_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = floor(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn round_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = round(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn trunc_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = trunc(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn frac_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let val = input[idx];\n    result[idx] = val - trunc(val);\n}\n\n@compute @workgroup_size(256)\nfn reciprocal_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = 1.0 / input[idx];\n}\n\n@compute @workgroup_size(256)\nfn rsqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    result[idx] = inverseSqrt(input[idx]);\n}\n\n@compute @workgroup_size(256)\nfn square_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let val = input[idx];\n    result[idx] = val * val;\n}\n\nfn erf(x: f32) -> f32 {\n    let s = sign(x);\n    let a = abs(x);\n    let t = 1.0 / (1.0 + 0.3275911 * a);\n    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);\n    return s * y;\n}\n\n@compute @workgroup_size(256)\nfn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))\n    result[idx] = 0.5 * x * (1.0 + erf(x * 0.7071067811865475));\n}\n\n@compute @workgroup_size(256)\nfn softplus_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = select(log(1.0 + exp(x)), x, x > 20.0);\n}\n\n@compute @workgroup_size(256)\nfn silu_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = x * (1.0 / (1.0 + exp(-x)));\n}\n\n@compute @workgroup_size(256)\nfn mish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    let sp = select(log(1.0 + exp(x)), x, x > 20.0);\n    result[idx] = x * tanh(sp);\n}\n\n@compute @workgroup_size(256)\nfn hardsigmoid_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = clamp(x / 6.0 + 0.5, 0.0, 1.0);\n}\n\n@compute @workgroup_size(256)\nfn hardswish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = x * clamp(x / 6.0 + 0.5, 0.0, 1.0);\n}\n\n@compute @workgroup_size(256)\nfn softsign_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = x / (1.0 + abs(x));\n}\n\n@compute @workgroup_size(256)\nfn tanhshrink_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    result[idx] = x - tanh(x);\n}\n\n@compute @workgroup_size(256)\nfn isnan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let bits = bitcast<u32>(input[idx]);\n    let is_nan = ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0u);\n    result[idx] = select(0.0, 1.0, is_nan);\n}\n\n@compute @workgroup_size(256)\nfn isinf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let bits = bitcast<u32>(input[idx]);\n    let is_inf = ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) == 0u);\n    result[idx] = select(0.0, 1.0, is_inf);\n}\n\n@compute @workgroup_size(256)\nfn isfinite_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let bits = bitcast<u32>(input[idx]);\n    let is_nan_or_inf = (bits & 0x7f800000u) == 0x7f800000u;\n    result[idx] = select(1.0, 0.0, is_nan_or_inf);\n}\n\n@compute @workgroup_size(256)\nfn isposinf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let bits = bitcast<u32>(input[idx]);\n    let is_pos_inf = bits == 0x7f800000u;\n    result[idx] = select(0.0, 1.0, is_pos_inf);\n}\n\n@compute @workgroup_size(256)\nfn isneginf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let bits = bitcast<u32>(input[idx]);\n    let is_neg_inf = bits == 0xff800000u;\n    result[idx] = select(0.0, 1.0, is_neg_inf);\n}";

// src/backend/webgpu/shaders/fill.wgsl
var fill_default = "struct Params {\n    value: f32,\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read_write> result: array<f32>;\n@group(0) @binding(1) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn fill(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.length) {\n        return;\n    }\n    result[idx] = params.value;\n}\n";

// src/backend/webgpu/shaders/random.wgsl
var random_default = "struct RNGParams {\n    seed: u32,\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read_write> output: array<f32>;\n@group(0) @binding(1) var<uniform> params: RNGParams;\n\nfn xorshift(state: u32) -> u32 {\n    var x = state;\n    x ^= x << 13u;\n    x ^= x >> 17u;\n    x ^= x << 5u;\n    return x;\n}\n\nfn uint_to_float(x: u32) -> f32 {\n    return f32(x) / 4294967296.0;\n}\n\n@compute @workgroup_size(256)\nfn rand(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.length) {\n        return;\n    }\n    var state = params.seed ^ (idx * 1664525u + 1013904223u);\n    state = xorshift(state);\n    output[idx] = uint_to_float(state);\n}\n\n@compute @workgroup_size(256)\nfn randn(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.length) {\n        return;\n    }\n    var state1 = params.seed ^ (idx * 1664525u + 1013904223u);\n    state1 = xorshift(state1);\n    var state2 = xorshift(state1);\n    let u1 = uint_to_float(state1);\n    let u2 = uint_to_float(state2);\n    let r = sqrt(-2.0 * log(max(u1, 0.0000001)));\n    let theta = 6.283185307179586 * u2;\n    output[idx] = r * cos(theta);\n}\n";

// src/backend/webgpu/shaders/reduce_sum.wgsl
var reduce_sum_default = "struct Params {\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\nvar<workgroup> shared_data: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>,\n       @builtin(local_invocation_id) local_id: vec3<u32>,\n       @builtin(workgroup_id) workgroup_id: vec3<u32>) {\n    let idx = global_id.x;\n    let lid = local_id.x;\n    if (idx < params.length) {\n        shared_data[lid] = input[idx];\n    } else {\n        shared_data[lid] = 0.0;\n    }\n    workgroupBarrier();\n    for (var stride = 128u; stride > 0u; stride >>= 1u) {\n        if (lid < stride) {\n            shared_data[lid] += shared_data[lid + stride];\n        }\n        workgroupBarrier();\n    }\n    if (lid == 0u) {\n        result[workgroup_id.x] = shared_data[0];\n    }\n}\n";

// src/backend/webgpu/shaders/reduce_max.wgsl
var reduce_max_default = "struct Params {\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\nvar<workgroup> shared_data: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>,\n              @builtin(local_invocation_id) local_id: vec3<u32>,\n              @builtin(workgroup_id) workgroup_id: vec3<u32>) {\n    let idx = global_id.x;\n    let lid = local_id.x;\n    if (idx < params.length) {\n        shared_data[lid] = input[idx];\n    } else {\n        shared_data[lid] = -3.4028235e+38;\n    }\n    workgroupBarrier();\n    for (var stride = 128u; stride > 0u; stride >>= 1u) {\n        if (lid < stride) {\n            shared_data[lid] = max(shared_data[lid], shared_data[lid + stride]);\n        }\n        workgroupBarrier();\n    }\n    if (lid == 0u) {\n        result[workgroup_id.x] = shared_data[0];\n    }\n}\n";

// src/backend/webgpu/shaders/reduce_min.wgsl
var reduce_min_default = "struct Params {\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\nvar<workgroup> shared_data: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>,\n              @builtin(local_invocation_id) local_id: vec3<u32>,\n              @builtin(workgroup_id) workgroup_id: vec3<u32>) {\n    let idx = global_id.x;\n    let lid = local_id.x;\n    if (idx < params.length) {\n        shared_data[lid] = input[idx];\n    } else {\n        shared_data[lid] = 3.4028235e+38;\n    }\n    workgroupBarrier();\n    for (var stride = 128u; stride > 0u; stride >>= 1u) {\n        if (lid < stride) {\n            shared_data[lid] = min(shared_data[lid], shared_data[lid + stride]);\n        }\n        workgroupBarrier();\n    }\n    if (lid == 0u) {\n        result[workgroup_id.x] = shared_data[0];\n    }\n}\n";

// src/backend/webgpu/shaders/reduce_any.wgsl
var reduce_any_default = "struct Params {\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\nvar<workgroup> shared_data: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>,\n       @builtin(local_invocation_id) local_id: vec3<u32>,\n       @builtin(workgroup_id) workgroup_id: vec3<u32>) {\n    let idx = global_id.x;\n    let lid = local_id.x;\n    if (idx < params.length) {\n        shared_data[lid] = select(0.0, 1.0, input[idx] != 0.0);\n    } else {\n        shared_data[lid] = 0.0;\n    }\n    workgroupBarrier();\n    for (var stride = 128u; stride > 0u; stride >>= 1u) {\n        if (lid < stride) {\n            shared_data[lid] = max(shared_data[lid], shared_data[lid + stride]);\n        }\n        workgroupBarrier();\n    }\n    if (lid == 0u) {\n        result[workgroup_id.x] = shared_data[0];\n    }\n}\n";

// src/backend/webgpu/shaders/reduce_all.wgsl
var reduce_all_default = "struct Params {\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\nvar<workgroup> shared_data: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>,\n       @builtin(local_invocation_id) local_id: vec3<u32>,\n       @builtin(workgroup_id) workgroup_id: vec3<u32>) {\n    let idx = global_id.x;\n    let lid = local_id.x;\n    if (idx < params.length) {\n        shared_data[lid] = select(0.0, 1.0, input[idx] != 0.0);\n    } else {\n        shared_data[lid] = 1.0;\n    }\n    workgroupBarrier();\n    for (var stride = 128u; stride > 0u; stride >>= 1u) {\n        if (lid < stride) {\n            shared_data[lid] = min(shared_data[lid], shared_data[lid + stride]);\n        }\n        workgroupBarrier();\n    }\n    if (lid == 0u) {\n        result[workgroup_id.x] = shared_data[0];\n    }\n}\n";

// src/backend/webgpu/shaders/reduce_prod.wgsl
var reduce_prod_default = "struct Params {\n    length: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\nvar<workgroup> shared_data: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>,\n       @builtin(local_invocation_id) local_id: vec3<u32>,\n       @builtin(workgroup_id) workgroup_id: vec3<u32>) {\n    let idx = global_id.x;\n    let lid = local_id.x;\n    if (idx < params.length) {\n        shared_data[lid] = input[idx];\n    } else {\n        shared_data[lid] = 1.0;\n    }\n    workgroupBarrier();\n    for (var stride = 128u; stride > 0u; stride >>= 1u) {\n        if (lid < stride) {\n            shared_data[lid] *= shared_data[lid + stride];\n        }\n        workgroupBarrier();\n    }\n    if (lid == 0u) {\n        result[workgroup_id.x] = shared_data[0];\n    }\n}\n";

// src/backend/webgpu/shaders/reduce_simple_sum.wgsl
var reduce_simple_sum_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let tid = global_id.x;\n    let chunk = 256u;\n    let start = tid * chunk;\n\n    var acc: f32 = 0.0;\n    for (var i = 0u; i < chunk; i++) {\n        let idx = start + i;\n        if (idx < arrayLength(&input)) {\n            acc += input[idx];\n        }\n    }\n    result[tid] = acc;\n}\n";

// src/backend/webgpu/shaders/reduce_simple_max.wgsl
var reduce_simple_max_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let tid = global_id.x;\n    let chunk = 256u;\n    let start = tid * chunk;\n\n    var acc: f32 = -3.4028235e+38;\n    for (var i = 0u; i < chunk; i++) {\n        let idx = start + i;\n        if (idx < arrayLength(&input)) {\n            acc = max(acc, input[idx]);\n        }\n    }\n    result[tid] = acc;\n}\n";

// src/backend/webgpu/shaders/reduce_simple_min.wgsl
var reduce_simple_min_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let tid = global_id.x;\n    let chunk = 256u;\n    let start = tid * chunk;\n\n    var acc: f32 = 3.4028235e+38;\n    for (var i = 0u; i < chunk; i++) {\n        let idx = start + i;\n        if (idx < arrayLength(&input)) {\n            acc = min(acc, input[idx]);\n        }\n    }\n    result[tid] = acc;\n}\n";

// src/backend/webgpu/shaders/reduce_simple_any.wgsl
var reduce_simple_any_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let tid = global_id.x;\n    let chunk = 256u;\n    let start = tid * chunk;\n\n    var acc: f32 = 0.0;\n    for (var i = 0u; i < chunk; i++) {\n        let idx = start + i;\n        if (idx < arrayLength(&input)) {\n            if (input[idx] != 0.0) {\n                acc = 1.0;\n                break;\n            }\n        }\n    }\n    result[tid] = acc;\n}\n";

// src/backend/webgpu/shaders/reduce_simple_all.wgsl
var reduce_simple_all_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let tid = global_id.x;\n    let chunk = 256u;\n    let start = tid * chunk;\n\n    var acc: f32 = 1.0;\n    for (var i = 0u; i < chunk; i++) {\n        let idx = start + i;\n        if (idx < arrayLength(&input)) {\n            if (input[idx] == 0.0) {\n                acc = 0.0;\n                break;\n            }\n        }\n    }\n    result[tid] = acc;\n}\n";

// src/backend/webgpu/shaders/reduce_simple_prod.wgsl
var reduce_simple_prod_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(1)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let tid = global_id.x;\n    let chunk = 256u;\n    let start = tid * chunk;\n\n    var acc: f32 = 1.0;\n    for (var i = 0u; i < chunk; i++) {\n        let idx = start + i;\n        if (idx < arrayLength(&input)) {\n            acc *= input[idx];\n        }\n    }\n    result[tid] = acc;\n}\n";

// src/backend/webgpu/shaders/reduce_dim.wgsl
var reduce_dim_default = "struct Params {\n    batch_size: u32,\n    reduce_size: u32,\n    op: u32, // 0: sum, 1: mean, 2: max, 3: min, 4: prod\n    _pad: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let batch_idx = global_id.x;\n    if (batch_idx >= params.batch_size) {\n        return;\n    }\n\n    var acc: f32;\n    // Initialization based on op\n    if (params.op == 0u || params.op == 1u || params.op == 5u) { \n        acc = 0.0; \n    } else if (params.op == 2u) { \n        acc = -3.4028235e+38; \n    } else if (params.op == 3u) { \n        acc = 3.4028235e+38; \n    } else if (params.op == 4u || params.op == 6u) { \n        acc = 1.0; \n    }\n\n    let offset = batch_idx * params.reduce_size;\n    for (var i = 0u; i < params.reduce_size; i++) {\n        let val = input[offset + i];\n        if (params.op == 0u || params.op == 1u) { \n            acc += val; \n        } else if (params.op == 2u) { \n            acc = max(acc, val); \n        } else if (params.op == 3u) { \n            acc = min(acc, val); \n        } else if (params.op == 4u) { \n            acc *= val; \n        } else if (params.op == 5u) {\n            if (val != 0.0) { acc = 1.0; break; }\n        } else if (params.op == 6u) {\n            if (val == 0.0) { acc = 0.0; break; }\n        }\n    }\n\n    if (params.op == 1u) {\n        output[batch_idx] = acc / f32(params.reduce_size);\n    } else {\n        output[batch_idx] = acc;\n    }\n}\n";

// src/backend/webgpu/shaders/matmul.wgsl
var matmul_default = "struct Dims {\n    M: u32,\n    K: u32,\n    N: u32,\n    batch: u32,\n}\n\n@group(0) @binding(0) var<storage, read> A: array<f32>;\n@group(0) @binding(1) var<storage, read> B: array<f32>;\n@group(0) @binding(2) var<storage, read_write> C: array<f32>;\n@group(0) @binding(3) var<uniform> dims: Dims;\n\nconst TILE_SIZE: u32 = 16u;\nvar<workgroup> tileA: array<array<f32, 16>, 16>;\nvar<workgroup> tileB: array<array<f32, 16>, 16>;\n\n@compute @workgroup_size(16, 16)\nfn matmul(@builtin(global_invocation_id) global_id: vec3<u32>,\n          @builtin(local_invocation_id) local_id: vec3<u32>) {\n    let row = global_id.y;\n    let col = global_id.x;\n    let localRow = local_id.y;\n    let localCol = local_id.x;\n    var sum: f32 = 0.0;\n    let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;\n    for (var t = 0u; t < numTiles; t++) {\n        let aRow = row;\n        let aCol = t * TILE_SIZE + localCol;\n        if (aRow < dims.M && aCol < dims.K) {\n            tileA[localRow][localCol] = A[aRow * dims.K + aCol];\n        } else {\n            tileA[localRow][localCol] = 0.0;\n        }\n        let bRow = t * TILE_SIZE + localRow;\n        let bCol = col;\n        if (bRow < dims.K && bCol < dims.N) {\n            tileB[localRow][localCol] = B[bRow * dims.N + bCol];\n        } else {\n            tileB[localRow][localCol] = 0.0;\n        }\n        workgroupBarrier();\n        for (var k = 0u; k < TILE_SIZE; k++) {\n            sum += tileA[localRow][k] * tileB[k][localCol];\n        }\n        workgroupBarrier();\n    }\n    if (row < dims.M && col < dims.N) {\n        C[row * dims.N + col] = sum;\n    }\n}\n\n@compute @workgroup_size(256)\nfn matmul_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    let totalElements = dims.M * dims.N;\n    if (idx >= totalElements) {\n        return;\n    }\n    let row = idx / dims.N;\n    let col = idx % dims.N;\n    var sum: f32 = 0.0;\n    for (var k = 0u; k < dims.K; k++) {\n        sum += A[row * dims.K + k] * B[k * dims.N + col];\n    }\n    C[idx] = sum;\n}\n\n@compute @workgroup_size(256)\nfn matmul_3d(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    let totalElements = dims.batch * dims.M * dims.N;\n    if (idx >= totalElements) {\n        return;\n    }\n    \n    let n = idx % dims.N;\n    let m = (idx / dims.N) % dims.M;\n    let b = idx / (dims.N * dims.M);\n    \n    var sum: f32 = 0.0;\n    for (var k = 0u; k < dims.K; k++) {\n        sum += A[(b * dims.M + m) * dims.K + k] * B[(b * dims.K + k) * dims.N + n];\n    }\n    C[idx] = sum;\n}\n";

// src/backend/webgpu/shaders/transpose.wgsl
var transpose_default = "struct Dims {\n    rows: u32,\n    cols: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> dims: Dims;\n\n@compute @workgroup_size(256)\nfn transpose_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    let total = dims.rows * dims.cols;\n    if (idx >= total) {\n        return;\n    }\n    let row = idx / dims.cols;\n    let col = idx % dims.cols;\n    let out_idx = col * dims.rows + row;\n    output[out_idx] = input[idx];\n}\n";

// src/backend/webgpu/shaders/log_softmax.wgsl
var log_softmax_default = "// Log-softmax shader\n// Computes log(softmax(x)) along the last dimension\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> dims: vec2<u32>;  // (batch_size, num_classes)\n\n@compute @workgroup_size(1)\nfn log_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let batch_idx = gid.x;\n    let num_classes = dims.y;\n\n    if (batch_idx >= dims.x) {\n        return;\n    }\n\n    let offset = batch_idx * num_classes;\n\n    // Find max for numerical stability\n    var max_val = input[offset];\n    for (var i = 1u; i < num_classes; i++) {\n        max_val = max(max_val, input[offset + i]);\n    }\n\n    // Compute sum of exp(x - max)\n    var sum_exp = 0.0;\n    for (var i = 0u; i < num_classes; i++) {\n        sum_exp += exp(input[offset + i] - max_val);\n    }\n\n    // Compute log_softmax = x - max - log(sum_exp)\n    let log_sum_exp = log(sum_exp);\n    for (var i = 0u; i < num_classes; i++) {\n        output[offset + i] = input[offset + i] - max_val - log_sum_exp;\n    }\n}\n";

// src/backend/webgpu/shaders/nll_loss.wgsl
var nll_loss_default = "// NLL Loss shader\n// Computes negative log likelihood loss: -input[targets]\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> targets: array<i32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> dims: vec2<u32>;  // (batch_size, num_classes)\n\n@compute @workgroup_size(256)\nfn nll_loss(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    let batch_size = dims.x;\n    let num_classes = dims.y;\n\n    if (idx >= batch_size) {\n        return;\n    }\n\n    // Get target class for this batch item\n    let target_class = targets[idx];\n\n    // Loss is -log_prob[target_class]\n    let log_prob = input[idx * num_classes + u32(target_class)];\n    output[idx] = -log_prob;\n}\n";

// src/backend/webgpu/shaders/argmax.wgsl
var argmax_default = "// Argmax shader - finds index of max value along the last dimension\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<i32>;\n@group(0) @binding(2) var<uniform> dims: vec2<u32>;  // (batch_size, num_elements)\n\n@compute @workgroup_size(256)\nfn argmax(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let batch_idx = gid.x;\n    let batch_size = dims.x;\n    let num_elements = dims.y;\n\n    if (batch_idx >= batch_size) {\n        return;\n    }\n\n    let offset = batch_idx * num_elements;\n\n    // Find max value and its index\n    var max_val = input[offset];\n    var max_idx = 0u;\n\n    for (var i = 1u; i < num_elements; i++) {\n        let val = input[offset + i];\n        if (val > max_val) {\n            max_val = val;\n            max_idx = i;\n        }\n    }\n\n    output[batch_idx] = i32(max_idx);\n}\n";

// src/backend/webgpu/shaders/argmin.wgsl
var argmin_default = "// Argmin shader - finds index of min value along the last dimension\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<i32>;\n@group(0) @binding(2) var<uniform> dims: vec2<u32>;  // (batch_size, num_elements)\n\n@compute @workgroup_size(256)\nfn argmin(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let batch_idx = gid.x;\n    let batch_size = dims.x;\n    let num_elements = dims.y;\n\n    if (batch_idx >= batch_size) {\n        return;\n    }\n\n    let offset = batch_idx * num_elements;\n\n    // Find min value and its index\n    var min_val = input[offset];\n    var min_idx = 0u;\n\n    for (var i = 1u; i < num_elements; i++) {\n        let val = input[offset + i];\n        if (val < min_val) {\n            min_val = val;\n            min_idx = i;\n        }\n    }\n\n    output[batch_idx] = i32(min_idx);\n}\n";

// src/backend/webgpu/shaders/compare.wgsl
var compare_default = "// Element-wise comparison operations\n\n@group(0) @binding(0) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n\n// Element-wise equality: output[i] = 1.0 if a[i] == b[i], else 0.0\n@compute @workgroup_size(256)\nfn eq(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&a)) {\n        return;\n    }\n    output[idx] = select(0.0, 1.0, a[idx] == b[idx]);\n}\n\n// Element-wise not equal\n@compute @workgroup_size(256)\nfn ne(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&a)) {\n        return;\n    }\n    output[idx] = select(0.0, 1.0, a[idx] != b[idx]);\n}\n\n// Element-wise less than\n@compute @workgroup_size(256)\nfn lt(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&a)) {\n        return;\n    }\n    output[idx] = select(0.0, 1.0, a[idx] < b[idx]);\n}\n\n// Element-wise less than or equal\n@compute @workgroup_size(256)\nfn le(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&a)) {\n        return;\n    }\n    output[idx] = select(0.0, 1.0, a[idx] <= b[idx]);\n}\n\n// Element-wise greater than\n@compute @workgroup_size(256)\nfn gt(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&a)) {\n        return;\n    }\n    output[idx] = select(0.0, 1.0, a[idx] > b[idx]);\n}\n\n// Element-wise greater than or equal\n@compute @workgroup_size(256)\nfn ge(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&a)) {\n        return;\n    }\n    output[idx] = select(0.0, 1.0, a[idx] >= b[idx]);\n}\n\n@compute @workgroup_size(256)\nfn maximum_op(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&output)) { return; }\n    output[idx] = max(a[idx], b[idx]);\n}\n\n@compute @workgroup_size(256)\nfn minimum_op(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&output)) { return; }\n    output[idx] = min(a[idx], b[idx]);\n}\n\n@compute @workgroup_size(256)\nfn fmax_op(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&output)) { return; }\n    // fmax in PyTorch handles NaNs differently (propagates non-NaN if one is NaN)\n    // WGSL max() behavior on NaN is same as JS Math.max (propagates NaN)\n    // To match fmax: if one is NaN, take the other.\n    let va = a[idx];\n    let vb = b[idx];\n    if (va != va) { output[idx] = vb; }\n    else if (vb != vb) { output[idx] = va; }\n    else { output[idx] = max(va, vb); }\n}\n\n@compute @workgroup_size(256)\nfn fmin_op(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&output)) { return; }\n    let va = a[idx];\n    let vb = b[idx];\n    if (va != va) { output[idx] = vb; }\n    else if (vb != vb) { output[idx] = va; }\n    else { output[idx] = min(va, vb); }\n}\n";

// src/backend/webgpu/shaders/index_select.wgsl
var index_select_default = "/**\n * Index select (gather) operation along a dimension.\n * For 2D tensors: output[i, j] = input[indices[i], j] when dim=0\n *                 output[i, j] = input[i, indices[j]] when dim=1\n */\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> indices: array<i32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n\nstruct Params {\n  dim: u32,           // Dimension to index along\n  input_dim0: u32,    // Input shape[0]\n  input_dim1: u32,    // Input shape[1]\n  num_indices: u32,   // Number of indices\n}\n\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn index_select_2d(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let output_size = select(params.num_indices * params.input_dim1,\n                           params.input_dim0 * params.num_indices,\n                           params.dim == 0u);\n  let idx = global_id.x;\n  if (idx >= output_size) {\n    return;\n  }\n\n  if (params.dim == 0u) {\n    // Indexing along dim 0: output[i, j] = input[indices[i], j]\n    let i = idx / params.input_dim1;\n    let j = idx % params.input_dim1;\n    let input_row = u32(indices[i]);\n    output[idx] = input[input_row * params.input_dim1 + j];\n  } else {\n    // Indexing along dim 1: output[i, j] = input[i, indices[j]]\n    let i = idx / params.num_indices;\n    let j = idx % params.num_indices;\n    let input_col = u32(indices[j]);\n    output[idx] = input[i * params.input_dim1 + input_col];\n  }\n}\n\n// For 1D tensors: simple gather\n@compute @workgroup_size(256)\nfn index_select_1d(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x;\n  if (idx >= params.num_indices) {\n    return;\n  }\n\n  let input_idx = u32(indices[idx]);\n  output[idx] = input[input_idx];\n}\n";

// src/backend/webgpu/shaders/expand.wgsl
var expand_default = "struct Params {\n    shape: vec4<u32>,\n    strides: vec4<u32>,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&output)) { return; }\n\n    var input_idx = 0u;\n    var current_idx = idx;\n    \n    // Extract coordinates in order: d3, d2, d1, d0\n    for (var i = 3i; i >= 0; i--) {\n        let dim_size = params.shape[i];\n        // With right-aligned padding, unused leading dims are 1, so this works.\n        let coord = current_idx % dim_size;\n        current_idx /= dim_size;\n        input_idx += coord * params.strides[i];\n    }\n    \n    output[idx] = input[input_idx];\n}\n";

// src/backend/webgpu/shaders/slice.wgsl
var slice_default = "/**\n * Slice operation for tensors.\n * Supports multi-dimensional slicing with start:stop:step semantics.\n */\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n\n// Slice parameters for each dimension (up to 4D)\nstruct SliceParams {\n  // Input shape (padded to 4D)\n  input_shape: vec4<u32>,\n  // Output shape (padded to 4D)\n  output_shape: vec4<u32>,\n  // Start indices for each dimension\n  starts: vec4<i32>,\n  // Step sizes for each dimension\n  steps: vec4<i32>,\n  // Number of dimensions\n  ndim: u32,\n  // Total output elements\n  output_size: u32,\n  _pad0: u32,\n  _pad1: u32,\n}\n\n@group(0) @binding(2) var<uniform> params: SliceParams;\n\n@compute @workgroup_size(256)\nfn slice(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x;\n  if (idx >= params.output_size) {\n    return;\n  }\n\n  // Compute strides (left-aligned)\n  var out_strides: array<u32, 4>;\n  var in_strides: array<u32, 4>;\n  \n  for (var i = 0u; i < 4u; i++) {\n    out_strides[i] = 1u;\n    in_strides[i] = 1u;\n  }\n\n  if (params.ndim > 0u) {\n    out_strides[params.ndim - 1u] = 1u;\n    in_strides[params.ndim - 1u] = 1u;\n    for (var d = i32(params.ndim) - 2; d >= 0; d--) {\n      out_strides[d] = out_strides[d + 1] * params.output_shape[d + 1];\n      in_strides[d] = in_strides[d + 1] * params.input_shape[d + 1];\n    }\n  }\n\n  // Convert flat output index to multi-dimensional output coordinates\n  var out_coords: array<u32, 4>;\n  var remaining = idx;\n\n  for (var d = 0u; d < params.ndim; d++) {\n    out_coords[d] = remaining / out_strides[d];\n    remaining = remaining % out_strides[d];\n  }\n\n  // Map output coordinates to input coordinates using slice params\n  var in_idx = 0u;\n  for (var d = 0u; d < params.ndim; d++) {\n    let in_coord = u32(params.starts[d]) + out_coords[d] * u32(params.steps[d]);\n    in_idx += in_coord * in_strides[d];\n  }\n\n  output[idx] = input[in_idx];\n}\n";

// src/backend/webgpu/shaders/broadcast.wgsl
var broadcast_default = "/**\n * Element-wise binary operations with broadcasting support.\n * Supports up to 4D tensors with NumPy-style broadcasting.\n */\n\n@group(0) @binding(0) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n\nstruct BroadcastParams {\n  // Output shape (padded to 4D, right-aligned)\n  output_shape: vec4<u32>,\n  // Input A strides (0 means broadcast this dim)\n  a_strides: vec4<u32>,\n  // Input B strides (0 means broadcast this dim)\n  b_strides: vec4<u32>,\n  // Number of dimensions\n  ndim: u32,\n  // Total output elements\n  output_size: u32,\n  // Operation type: 0=add, 1=sub, 2=mul, 3=div\n  op: u32,\n  _pad: u32,\n}\n\n@group(0) @binding(3) var<uniform> params: BroadcastParams;\n\n@compute @workgroup_size(256)\nfn broadcast_add(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x;\n  if (idx >= params.output_size) {\n    return;\n  }\n\n  // Convert flat index to multi-dimensional coordinates\n  let offset = 4u - params.ndim;\n  var remaining = idx;\n  var a_idx = 0u;\n  var b_idx = 0u;\n\n  // Compute output strides\n  var out_strides: vec4<u32>;\n  out_strides[3] = 1u;\n  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }\n  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }\n  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }\n\n  // For each dimension, compute coordinate and add to input indices\n  for (var d = 0u; d < params.ndim; d++) {\n    let dim_idx = offset + d;\n    let coord = remaining / out_strides[dim_idx];\n    remaining = remaining % out_strides[dim_idx];\n\n    a_idx += coord * params.a_strides[dim_idx];\n    b_idx += coord * params.b_strides[dim_idx];\n  }\n\n  output[idx] = a[a_idx] + b[b_idx];\n}\n\n@compute @workgroup_size(256)\nfn broadcast_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x;\n  if (idx >= params.output_size) {\n    return;\n  }\n\n  let offset = 4u - params.ndim;\n  var remaining = idx;\n  var a_idx = 0u;\n  var b_idx = 0u;\n\n  var out_strides: vec4<u32>;\n  out_strides[3] = 1u;\n  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }\n  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }\n  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }\n\n  for (var d = 0u; d < params.ndim; d++) {\n    let dim_idx = offset + d;\n    let coord = remaining / out_strides[dim_idx];\n    remaining = remaining % out_strides[dim_idx];\n    a_idx += coord * params.a_strides[dim_idx];\n    b_idx += coord * params.b_strides[dim_idx];\n  }\n\n  output[idx] = a[a_idx] - b[b_idx];\n}\n\n@compute @workgroup_size(256)\nfn broadcast_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x;\n  if (idx >= params.output_size) {\n    return;\n  }\n\n  let offset = 4u - params.ndim;\n  var remaining = idx;\n  var a_idx = 0u;\n  var b_idx = 0u;\n\n  var out_strides: vec4<u32>;\n  out_strides[3] = 1u;\n  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }\n  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }\n  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }\n\n  for (var d = 0u; d < params.ndim; d++) {\n    let dim_idx = offset + d;\n    let coord = remaining / out_strides[dim_idx];\n    remaining = remaining % out_strides[dim_idx];\n    a_idx += coord * params.a_strides[dim_idx];\n    b_idx += coord * params.b_strides[dim_idx];\n  }\n\n  output[idx] = a[a_idx] * b[b_idx];\n}\n\n@compute @workgroup_size(256)\nfn broadcast_div(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x;\n  if (idx >= params.output_size) {\n    return;\n  }\n\n  let offset = 4u - params.ndim;\n  var remaining = idx;\n  var a_idx = 0u;\n  var b_idx = 0u;\n\n  var out_strides: vec4<u32>;\n  out_strides[3] = 1u;\n  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }\n  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }\n  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }\n\n  for (var d = 0u; d < params.ndim; d++) {\n    let dim_idx = offset + d;\n    let coord = remaining / out_strides[dim_idx];\n    remaining = remaining % out_strides[dim_idx];\n    a_idx += coord * params.a_strides[dim_idx];\n    b_idx += coord * params.b_strides[dim_idx];\n  }\n\n  output[idx] = a[a_idx] / b[b_idx];\n}\n\n@compute @workgroup_size(256)\nfn broadcast_pow(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x;\n  if (idx >= params.output_size) {\n    return;\n  }\n\n  let offset = 4u - params.ndim;\n  var remaining = idx;\n  var a_idx = 0u;\n  var b_idx = 0u;\n\n  var out_strides: vec4<u32>;\n  out_strides[3] = 1u;\n  if (params.ndim > 1u) { out_strides[2] = params.output_shape[3]; }\n  if (params.ndim > 2u) { out_strides[1] = params.output_shape[3] * params.output_shape[2]; }\n  if (params.ndim > 3u) { out_strides[0] = params.output_shape[3] * params.output_shape[2] * params.output_shape[1]; }\n\n  for (var d = 0u; d < params.ndim; d++) {\n    let dim_idx = offset + d;\n    let coord = remaining / out_strides[dim_idx];\n    remaining = remaining % out_strides[dim_idx];\n    a_idx += coord * params.a_strides[dim_idx];\n    b_idx += coord * params.b_strides[dim_idx];\n  }\n\n  output[idx] = pow(a[a_idx], b[b_idx]);\n}\n";

// src/backend/webgpu/shaders/nll_loss_backward.wgsl
var nll_loss_backward_default = "// NLL Loss backward shader\n// Computes gradient: -1/batch_size at target index, 0 elsewhere\n\n@group(0) @binding(0) var<storage, read> targets: array<i32>;\n@group(0) @binding(1) var<storage, read_write> grad_input: array<f32>;\n\nstruct Params {\n  batch_size: u32,\n  num_classes: u32,\n  scale: f32,  // 1.0 for sum reduction, 1/batch_size for mean\n  _pad: u32,\n}\n\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn nll_loss_backward(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let idx = gid.x;\n  let total_size = params.batch_size * params.num_classes;\n\n  if (idx >= total_size) {\n    return;\n  }\n\n  // Compute batch index and class index\n  let batch_idx = idx / params.num_classes;\n  let class_idx = idx % params.num_classes;\n\n  // Get target for this batch\n  let target_class = u32(targets[batch_idx]);\n\n  // Gradient is -scale at target index, 0 elsewhere\n  if (class_idx == target_class) {\n    grad_input[idx] = -params.scale;\n  } else {\n    grad_input[idx] = 0.0;\n  }\n}\n";

// src/backend/webgpu/shaders/log_softmax_backward.wgsl
var log_softmax_backward_default = "/**\n * Log softmax backward gradient computation.\n * grad[i,j] = gradOutput[i,j] - softmax[i,j] * sum_k(gradOutput[i,k])\n * @status implemented\n */\n\n@group(0) @binding(0) var<storage, read> grad_output: array<f32>;\n@group(0) @binding(1) var<storage, read> softmax: array<f32>;\n@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;\n\nstruct Dims {\n  batch_size: u32,\n  num_classes: u32,\n}\n@group(0) @binding(3) var<uniform> dims: Dims;\n\n@compute @workgroup_size(256)\nfn log_softmax_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x;\n  let total = dims.batch_size * dims.num_classes;\n  if (idx >= total) { return; }\n\n  let i = idx / dims.num_classes;  // batch index\n  let j = idx % dims.num_classes;  // class index\n\n  // Sum gradOutput along row i\n  var grad_sum = 0.0;\n  for (var k = 0u; k < dims.num_classes; k++) {\n    grad_sum += grad_output[i * dims.num_classes + k];\n  }\n\n  // grad[i,j] = gradOutput[i,j] - softmax[i,j] * grad_sum\n  grad_input[idx] = grad_output[idx] - softmax[idx] * grad_sum;\n}\n";

// src/backend/webgpu/shaders/masked_fill.wgsl
var masked_fill_default = "struct Params {\n  value: f32,\n  size: u32,\n  _pad1: u32,\n  _pad2: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> mask: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let idx = gid.x;\n  if (idx >= params.size) { return; }\n\n  // mask == 0 means we use original value, mask != 0 means we use fill value\n  if (mask[idx] != 0.0) {\n    output[idx] = params.value;\n  } else {\n    output[idx] = input[idx];\n  }\n}\n";

// src/backend/webgpu/shaders/where.wgsl
var where_default = "@group(0) @binding(0) var<storage, read> condition: array<f32>;\n@group(0) @binding(1) var<storage, read> x: array<f32>;\n@group(0) @binding(2) var<storage, read> y: array<f32>;\n@group(0) @binding(3) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&result)) { return; }\n    if (condition[idx] > 0.0) {\n        result[idx] = x[idx];\n    } else {\n        result[idx] = y[idx];\n    }\n}\n";

// src/backend/webgpu/shaders/transpose_nd.wgsl
var transpose_nd_default = "struct Params {\n  src_strides: vec4<u32>,\n  dst_shape: vec4<u32>,\n  dst_strides: vec4<u32>,\n  ndim: u32,\n  total: u32,\n  _pad1: u32,\n  _pad2: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let dst_idx = gid.x;\n  if (dst_idx >= params.total) { return; }\n\n  // Convert dst_idx to coordinates\n  var coords: array<u32, 4>;\n  var rem = dst_idx;\n  for (var d = i32(params.ndim) - 1; d >= 0; d--) {\n    coords[d] = rem % params.dst_shape[d];\n    rem = rem / params.dst_shape[d];\n  }\n\n  // Compute src index using transposed strides\n  var src_idx = 0u;\n  for (var d = 0u; d < params.ndim; d++) {\n    src_idx += coords[d] * params.src_strides[d];\n  }\n\n  output[dst_idx] = input[src_idx];\n}\n";

// src/backend/webgpu/shaders/reduce_broadcast_grad.wgsl
var reduce_broadcast_grad_default = "struct Params { \n  batch_size: u32, \n  features: u32 \n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let j = global_id.x;\n  if (j >= params.features) { return; }\n\n  var sum = 0.0;\n  for (var i = 0u; i < params.batch_size; i++) {\n    sum += input[i * params.features + j];\n  }\n  output[j] = sum;\n}\n";

// src/backend/webgpu/shaders/embedding.wgsl
var embedding_default = "@group(0) @binding(0) var<storage, read> weight: array<f32>;\n@group(0) @binding(1) var<storage, read> indices: array<i32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n\nstruct Params {\n  num_embeddings: u32,\n  embedding_dim: u32,\n  num_indices: u32,\n  _pad: u32,\n}\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let idx = gid.x;\n  let total = params.num_indices * params.embedding_dim;\n  if (idx >= total) { return; }\n\n  let token_idx = idx / params.embedding_dim;\n  let embed_idx = idx % params.embedding_dim;\n\n  let token_id = indices[token_idx];\n  // Clamp to valid range\n  let safe_token = clamp(u32(token_id), 0u, params.num_embeddings - 1u);\n\n  output[idx] = weight[safe_token * params.embedding_dim + embed_idx];\n}\n";

// src/backend/webgpu/shaders/layernorm.wgsl
var layernorm_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> gamma: array<f32>;\n@group(0) @binding(2) var<storage, read> beta: array<f32>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n\nstruct Params {\n  batch_size: u32,\n  normalized_size: u32,\n  eps: f32,\n  _pad: u32,\n}\n@group(0) @binding(4) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let batch_idx = gid.x;\n  if (batch_idx >= params.batch_size) { return; }\n\n  let offset = batch_idx * params.normalized_size;\n\n  // Compute mean\n  var sum = 0.0;\n  for (var i = 0u; i < params.normalized_size; i++) {\n    sum += input[offset + i];\n  }\n  let mean = sum / f32(params.normalized_size);\n\n  // Compute variance\n  var var_sum = 0.0;\n  for (var i = 0u; i < params.normalized_size; i++) {\n    let diff = input[offset + i] - mean;\n    var_sum += diff * diff;\n  }\n  let variance = var_sum / f32(params.normalized_size);\n  let inv_std = 1.0 / sqrt(variance + params.eps);\n\n  // Normalize and apply affine transform\n  for (var i = 0u; i < params.normalized_size; i++) {\n    let normalized = (input[offset + i] - mean) * inv_std;\n    output[offset + i] = gamma[i] * normalized + beta[i];\n  }\n}\n";

// src/backend/webgpu/shaders/tril.wgsl
var tril_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n\nstruct Params {\n  rows: u32,\n  cols: u32,\n  diagonal: i32,\n  _pad: u32,\n}\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let idx = gid.x;\n  let total = params.rows * params.cols;\n  if (idx >= total) { return; }\n\n  let row = idx / params.cols;\n  let col = idx % params.cols;\n\n  // Keep element if col <= row + diagonal\n  if (i32(col) <= i32(row) + params.diagonal) {\n    output[idx] = input[idx];\n  } else {\n    output[idx] = 0.0;\n  }\n}\n";

// src/backend/webgpu/shaders/triu.wgsl
var triu_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n\nstruct Params {\n  rows: u32,\n  cols: u32,\n  diagonal: i32,\n  _pad: u32,\n}\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let idx = gid.x;\n  let total = params.rows * params.cols;\n  if (idx >= total) { return; }\n\n  let row = idx / params.cols;\n  let col = idx % params.cols;\n\n  if (i32(col) >= i32(row) + params.diagonal) {\n    output[idx] = input[idx];\n  } else {\n    output[idx] = 0.0;\n  }\n}\n";

// src/backend/webgpu/shaders/flip.wgsl
var flip_default = "struct Params {\n    batch_size: u32,\n    dim_size: u32,\n    stride: u32,\n    _pad: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    let total = arrayLength(&output);\n    if (idx >= total) { return; }\n\n    let stride_after = params.stride;\n    let dim_size = params.dim_size;\n    \n    let after_idx = idx % stride_after;\n    let mid_idx = (idx / stride_after) % dim_size;\n    let before_idx = idx / (stride_after * dim_size);\n    \n    let flipped_mid = dim_size - 1u - mid_idx;\n    let input_idx = (before_idx * dim_size + flipped_mid) * stride_after + after_idx;\n    \n    output[idx] = input[input_idx];\n}\n";

// src/backend/webgpu/shaders/heaviside.wgsl
var heaviside_default = "@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> values: array<f32>;\n@group(0) @binding(2) var<storage, read_write> result: array<f32>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let idx = gid.x;\n    if (idx >= arrayLength(&result)) { return; }\n    let x = input[idx];\n    if (x < 0.0) {\n        result[idx] = 0.0;\n    } else if (x > 0.0) {\n        result[idx] = 1.0;\n    } else {\n        result[idx] = values[idx];\n    }\n}\n";

// src/backend/webgpu/shaders/cumsum.wgsl
var cumsum_default = "struct Params {\n    batch_size: u32,\n    reduce_size: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let batch_idx = gid.x;\n    if (batch_idx >= params.batch_size) { return; }\n\n    let offset = batch_idx * params.reduce_size;\n    var acc: f32 = 0.0;\n    for (var i = 0u; i < params.reduce_size; i++) {\n        acc += input[offset + i];\n        output[offset + i] = acc;\n    }\n}\n";

// src/backend/webgpu/shaders/cumprod.wgsl
var cumprod_default = "struct Params {\n    batch_size: u32,\n    reduce_size: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let batch_idx = gid.x;\n    if (batch_idx >= params.batch_size) { return; }\n\n    let offset = batch_idx * params.reduce_size;\n    var acc: f32 = 1.0;\n    for (var i = 0u; i < params.reduce_size; i++) {\n        acc *= input[offset + i];\n        output[offset + i] = acc;\n    }\n}\n";

// src/backend/webgpu/shaders/diag_vec_to_mtx.wgsl
var diag_vec_to_mtx_default = "struct Params {\n    dim: u32,\n    offset: i32,\n    _pad1: u32,\n    _pad2: u32,\n}\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x;\n    if (i >= params.dim) { return; }\n    \n    let offset = params.offset;\n    let n = params.dim + u32(abs(f32(offset)));\n    \n    var row: u32;\n    var col: u32;\n    if (offset >= 0) {\n        row = i;\n        col = i + u32(offset);\n    } else {\n        row = i + u32(-offset);\n        col = i;\n    }\n    \n    output[row * n + col] = input[i];\n}\n";

// src/backend/webgpu/shaders/diag_mtx_to_vec.wgsl
var diag_mtx_to_vec_default = "struct Params {\n    rows: u32,\n    cols: u32,\n    offset: i32,\n    diag_len: u32,\n}\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x;\n    if (i >= params.diag_len) { return; }\n    \n    let offset = params.offset;\n    var row: u32;\n    var col: u32;\n    if (offset >= 0) {\n        row = i;\n        col = i + u32(offset);\n    } else {\n        row = i + u32(-offset);\n        col = i;\n    }\n    \n    output[i] = input[row * params.cols + col];\n}\n";

// src/backend/webgpu/shaders/clamp.wgsl
var clamp_default = "struct Params {\n    min_val: f32,\n    max_val: f32,\n    total: u32,\n    _pad: u32,\n}\n\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let idx = global_id.x;\n    if (idx >= params.total) { return; }\n    \n    result[idx] = clamp(input[idx], params.min_val, params.max_val);\n}\n";

// src/backend/webgpu/shaders/cholesky.wgsl
var cholesky_default = "struct Dims {\n    N: u32,\n    batch: u32,\n    k: u32, // current step\n}\n\n@group(0) @binding(0) var<storage, read_write> A: array<f32>;\n@group(0) @binding(1) var<uniform> dims: Dims;\n\n// Pass 1: Update column k\n@compute @workgroup_size(256)\nfn cholesky_step1(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let b = global_id.x;\n    if (b >= dims.batch) {\n        return;\n    }\n    \n    let n = dims.N;\n    let k = dims.k;\n    let offset = b * n * n;\n    \n    let akk = A[offset + k * n + k];\n    if (akk <= 0.0) {\n        // Matrix is not positive definite\n        // In PyTorch, this would throw an error or return NaN\n        // For now we just set to NaN or something\n        A[offset + k * n + k] = bitcast<f32>(0x7fc00000u);\n        return;\n    }\n    \n    let sqrt_akk = sqrt(akk);\n    A[offset + k * n + k] = sqrt_akk;\n    \n    for (var i = k + 1u; i < n; i++) {\n        A[offset + i * n + k] /= sqrt_akk;\n    }\n}\n\n// Pass 2: Rank-1 update of the remaining submatrix\n@compute @workgroup_size(16, 16)\nfn cholesky_step2(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let i = global_id.y + dims.k + 1u;\n    let j = global_id.x + dims.k + 1u;\n    let b = global_id.z;\n    \n    let n = dims.N;\n    let k = dims.k;\n    \n    if (b >= dims.batch || i >= n || j >= n || j > i) {\n        return;\n    }\n    \n    let offset = b * n * n;\n    let val = A[offset + i * n + k] * A[offset + j * n + k];\n    A[offset + i * n + j] -= val;\n}\n\n// Full Cholesky for small matrices in a single pass (if N is small)\n@compute @workgroup_size(256)\nfn cholesky_small(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let b = global_id.x;\n    if (b >= dims.batch) {\n        return;\n    }\n    \n    let n = dims.N;\n    let offset = b * n * n;\n    \n    for (var k = 0u; k < n; k++) {\n        var akk = A[offset + k * n + k];\n        for (var s = 0u; s < k; s++) {\n            let lks = A[offset + k * n + s];\n            akk -= lks * lks;\n        }\n        \n        if (akk <= 0.0) {\n            A[offset + k * n + k] = bitcast<f32>(0x7fc00000u);\n            return;\n        }\n        \n        let sqrt_akk = sqrt(akk);\n        A[offset + k * n + k] = sqrt_akk;\n        \n        for (var i = k + 1u; i < n; i++) {\n            var aik = A[offset + i * n + k];\n            for (var s = 0u; s < k; s++) {\n                aik -= A[offset + i * n + s] * A[offset + k * n + s];\n            }\n            A[offset + i * n + k] = aik / sqrt_akk;\n        }\n    }\n    \n    // Zero out upper triangle\n    for (var i = 0u; i < n; i++) {\n        for (var j = i + 1u; j < n; j++) {\n            A[offset + i * n + j] = 0.0;\n        }\n    }\n}\n";

// src/backend/webgpu/shaders/triangular_solve.wgsl
var triangular_solve_default = "struct Dims {\n    N: u32,\n    M: u32, // num RHS columns (for B)\n    batch: u32,\n    k: u32, // current step\n}\n\n@group(0) @binding(0) var<storage, read> A: array<f32>;\n@group(0) @binding(1) var<storage, read_write> B: array<f32>;\n@group(0) @binding(2) var<uniform> dims: Dims;\n\n// Forward substitution step (for lower triangular)\n// AX = B => X_kj = (B_kj - sum_{s=0}^{k-1} A_ks * X_sj) / A_kk\n@compute @workgroup_size(256)\nfn forward_sub_step(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let j = global_id.x; // column of B\n    let b = global_id.y; // batch\n    if (j >= dims.M || b >= dims.batch) {\n        return;\n    }\n    \n    let n = dims.N;\n    let k = dims.k;\n    let offsetA = b * n * n;\n    let offsetB = b * n * dims.M;\n    \n    var val = B[offsetB + k * dims.M + j];\n    for (var s = 0u; s < k; s++) {\n        val -= A[offsetA + k * n + s] * B[offsetB + s * dims.M + j];\n    }\n    \n    B[offsetB + k * dims.M + j] = val / A[offsetA + k * n + k];\n}\n\n// Backward substitution step (for upper triangular)\n// AX = B => X_kj = (B_kj - sum_{s=k+1}^{n-1} A_ks * X_sj) / A_kk\n@compute @workgroup_size(256)\nfn backward_sub_step(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let j = global_id.x; // column of B\n    let b = global_id.y; // batch\n    if (j >= dims.M || b >= dims.batch) {\n        return;\n    }\n    \n    let n = dims.N;\n    let k = dims.k;\n    let offsetA = b * n * n;\n    let offsetB = b * n * dims.M;\n    \n    var val = B[offsetB + k * dims.M + j];\n    for (var s = k + 1u; s < n; s++) {\n        val -= A[offsetA + k * n + s] * B[offsetB + s * dims.M + j];\n    }\n    \n    B[offsetB + k * dims.M + j] = val / A[offsetA + k * n + k];\n}\n";

// src/backend/webgpu/shaders/lu.wgsl
var lu_default = "struct Dims {\n    N: u32,\n    batch: u32,\n    k: u32,\n}\n\n@group(0) @binding(0) var<storage, read_write> A: array<f32>;\n@group(0) @binding(1) var<uniform> dims: Dims;\n@group(0) @binding(2) var<storage, read_write> P: array<u32>;\n\n@compute @workgroup_size(256)\nfn lu_pivot(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let b = global_id.x;\n    if (b >= dims.batch) {\n        return;\n    }\n    \n    // Explicit use of P to ensure it's in the layout\n    if (P[0] == 0xffffffffu) { return; }\n\n    let n = dims.N;\n    let k = dims.k;\n    let offset = b * n * n;\n    let pOffset = b * n;\n    \n    var maxVal: f32 = 0.0;\n    var pivotRow: u32 = k;\n    for (var i = k; i < n; i = i + 1u) {\n        let val = abs(A[offset + i * n + k]);\n        if (val > maxVal) {\n            maxVal = val;\n            pivotRow = i;\n        }\n    }\n    \n    if (pivotRow != k) {\n        for (var j = 0u; j < n; j = j + 1u) {\n            let temp = A[offset + k * n + j];\n            A[offset + k * n + j] = A[offset + pivotRow * n + j];\n            A[offset + pivotRow * n + j] = temp;\n        }\n        let tempP = P[pOffset + k];\n        P[pOffset + k] = P[pOffset + pivotRow];\n        P[pOffset + pivotRow] = tempP;\n    }\n}\n\n@compute @workgroup_size(16, 16)\nfn lu_update(@builtin(global_invocation_id) global_id: vec3<u32>) {\n    let i = global_id.y + dims.k + 1u;\n    let j = global_id.x + dims.k + 1u;\n    let b = global_id.z;\n    \n    let n = dims.N;\n    let k = dims.k;\n    \n    if (b >= dims.batch || i >= n) {\n        return;\n    }\n    \n    // Explicit use of P to ensure it's in the layout\n    if (P[0] == 0xffffffffu) { return; }\n\n    let offset = b * n * n;\n    \n    if (j == k + 1u) {\n        let pivotVal = A[offset + k * n + k];\n        if (abs(pivotVal) > 1e-9) {\n            A[offset + i * n + k] /= pivotVal;\n        }\n    }\n    \n    workgroupBarrier();\n    \n    if (j < n) {\n        A[offset + i * n + j] -= A[offset + i * n + k] * A[offset + k * n + j];\n    }\n}\n";

// src/backend/webgpu/capabilities.ts
var capabilities = null;
async function detectCapabilities() {
  if (capabilities) return capabilities;
  const device = getDevice();
  const adapter = getAdapter();
  const adapterLimits = adapter.limits;
  const limits = {
    maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: adapterLimits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: adapterLimits.maxComputeWorkgroupSizeZ,
    maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupsPerDimension: adapterLimits.maxComputeWorkgroupsPerDimension,
    maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
    maxBufferSize: adapterLimits.maxBufferSize
  };
  const platform = {
    browser: detectBrowser(),
    gpu: "unknown"
  };
  const workgroupSharedMemory = await testWorkgroupSharedMemory(device);
  capabilities = {
    workgroupSharedMemory,
    timestampQuery: !!(adapter.features && adapter.features.has("timestamp-query")),
    subgroups: false,
    // Future: test subgroup operations
    limits,
    platform
  };
  console.log("[torch.js] GPU capabilities:", capabilities);
  return capabilities;
}
function getCapabilities() {
  if (!capabilities) {
    throw new Error("GPU capabilities not detected. Call torch.init() first.");
  }
  return capabilities;
}
async function testWorkgroupSharedMemory(device) {
  try {
    const inputBuffer = device.createBuffer({
      size: 20,
      // 5 floats
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(inputBuffer.getMappedRange()).set([1, 2, 3, 4, 5]);
    inputBuffer.unmap();
    const outputBuffer = device.createBuffer({
      size: 4,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC
    });
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM,
      mappedAtCreation: true
    });
    new Uint32Array(paramsBuffer.getMappedRange())[0] = 5;
    paramsBuffer.unmap();
    const shaderCode = reduce_sum_default;
    const shaderModule = device.createShaderModule({ code: shaderCode });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "main" }
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer, offset: 0, size: inputBuffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const stagingBuffer = device.createBuffer({
      size: 4,
      usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST
    });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, 4);
    device.queue.submit([copyEncoder.finish()]);
    await stagingBuffer.mapAsync(MapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange())[0];
    stagingBuffer.unmap();
    inputBuffer.destroy();
    outputBuffer.destroy();
    paramsBuffer.destroy();
    stagingBuffer.destroy();
    const works = Math.abs(result - 15) < 1e-3;
    console.log("[torch.js] Shared memory test result:", result, "expected: 15, works:", works);
    return works;
  } catch (e) {
    console.warn("[torch.js] Workgroup shared memory test failed:", e);
    return false;
  }
}
function detectBrowser() {
  if (typeof navigator === "undefined") return "node";
  const ua = navigator.userAgent;
  if (ua.includes("Firefox")) return "firefox";
  if (ua.includes("Chrome")) return "chrome";
  if (ua.includes("Safari")) return "safari";
  return "unknown";
}

// src/backend/webgpu/buffer.ts
var GPUBufferPool = class {
  pools = /* @__PURE__ */ new Map();
  maxPoolSize = 32;
  // Tracking
  activeBytes = 0;
  pooledBytes = 0;
  peakBytes = 0;
  totalAllocations = 0;
  /**
   * Acquire a buffer of the given size in bytes.
   * Returns a pooled buffer if available, otherwise creates a new one.
   */
  acquire(sizeBytes, usage) {
    const device = getDevice();
    const alignedSize = Math.max(4, Math.ceil(sizeBytes / 4) * 4);
    const pool = this.pools.get(alignedSize);
    if (pool && pool.length > 0) {
      const buffer2 = pool.pop();
      this.pooledBytes -= alignedSize;
      this.activeBytes += alignedSize;
      this.updatePeak();
      return buffer2;
    }
    const buffer = device.createBuffer({
      size: alignedSize,
      usage: usage | BufferUsage.COPY_SRC | BufferUsage.COPY_DST
    });
    this.activeBytes += alignedSize;
    this.totalAllocations++;
    this.updatePeak();
    return buffer;
  }
  /**
   * Release a buffer back to the pool for reuse.
   */
  release(buffer) {
    const size = buffer.size;
    this.activeBytes -= size;
    let pool = this.pools.get(size);
    if (!pool) {
      pool = [];
      this.pools.set(size, pool);
    }
    if (pool.length < this.maxPoolSize) {
      pool.push(buffer);
      this.pooledBytes += size;
    } else {
      buffer.destroy();
    }
  }
  /**
   * Clear all pooled buffers.
   */
  clear() {
    for (const pool of this.pools.values()) {
      for (const buffer of pool) {
        buffer.destroy();
      }
    }
    this.pools.clear();
    this.pooledBytes = 0;
  }
  getStats() {
    return {
      activeBytes: this.activeBytes,
      pooledBytes: this.pooledBytes,
      peakBytes: this.peakBytes,
      allocationCount: this.totalAllocations
    };
  }
  resetPeak() {
    this.peakBytes = this.activeBytes;
  }
  updatePeak() {
    this.peakBytes = Math.max(this.peakBytes, this.activeBytes);
  }
};
var bufferPool = new GPUBufferPool();
function createStorageBuffer(sizeBytes) {
  return bufferPool.acquire(sizeBytes, BufferUsage.STORAGE);
}
function createBufferWithData(data, _dtype) {
  const sizeBytes = data.byteLength;
  const buffer = bufferPool.acquire(sizeBytes, BufferUsage.STORAGE);
  const device = getDevice();
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}
async function readBuffer(buffer, dtype, numElements) {
  const device = getDevice();
  const bytesPerElement = getDTypeBytes(dtype);
  const sizeBytes = numElements * bytesPerElement;
  const alignedSize = Math.max(4, Math.ceil(sizeBytes / 4) * 4);
  const stagingBuffer = device.createBuffer({
    size: alignedSize,
    usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, alignedSize);
  device.queue.submit([commandEncoder.finish()]);
  await stagingBuffer.mapAsync(MapMode.READ);
  const mappedRange = stagingBuffer.getMappedRange();
  let result;
  switch (dtype) {
    case "float32":
    case "float16":
      result = new Float32Array(numElements);
      result.set(new Float32Array(mappedRange, 0, numElements));
      break;
    case "int32":
      result = new Int32Array(numElements);
      result.set(new Int32Array(mappedRange, 0, numElements));
      break;
    case "uint32":
      result = new Uint32Array(numElements);
      result.set(new Uint32Array(mappedRange, 0, numElements));
      break;
    case "int8":
      result = new Int8Array(numElements);
      result.set(new Int8Array(mappedRange, 0, numElements));
      break;
    case "uint8":
    case "bool":
      result = new Uint8Array(numElements);
      result.set(new Uint8Array(mappedRange, 0, numElements));
      break;
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
  stagingBuffer.unmap();
  stagingBuffer.destroy();
  return result;
}

// src/backend/webgpu/dispatch.ts
var pipelineCache = /* @__PURE__ */ new Map();
function getOrCreatePipeline(shaderCode, entryPoint = "main") {
  const cacheKey = `${shaderCode}:${entryPoint}`;
  let pipeline = pipelineCache.get(cacheKey);
  if (pipeline) {
    return pipeline;
  }
  const device = getDevice();
  const shaderModule = device.createShaderModule({
    code: shaderCode
  });
  if (typeof shaderModule.getCompilationInfo === "function") {
    shaderModule.getCompilationInfo().then((info) => {
      for (const msg of info.messages) {
        const level = msg.type === "error" ? "error" : msg.type === "warning" ? "warn" : "log";
        console[level](`[WGSL ${entryPoint}] ${msg.type}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      }
    });
  }
  pipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint
    }
  });
  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}
function calculateWorkgroups(numElements, workgroupSize = 256) {
  const numWorkgroups = Math.ceil(numElements / workgroupSize);
  if (numWorkgroups <= 65535) {
    return [numWorkgroups, 1, 1];
  } else if (numWorkgroups <= 65535 * 65535) {
    const x = Math.ceil(Math.sqrt(numWorkgroups));
    const y = Math.ceil(numWorkgroups / x);
    return [x, y, 1];
  } else {
    throw new Error(`Tensor too large: ${numElements} elements exceeds WebGPU limits`);
  }
}
async function syncDevice() {
  const device = getDevice();
  await device.queue.onSubmittedWorkDone();
}

// src/profiler/index.ts
var profiler_exports = {};
__export(profiler_exports, {
  KinetoStep: () => KinetoStep,
  Profiler: () => Profiler,
  ProfilerActivity: () => ProfilerActivity,
  profile: () => profile,
  record_function: () => record_function,
  record_function_async: () => record_function_async
});
var ProfilerActivity = /* @__PURE__ */ ((ProfilerActivity2) => {
  ProfilerActivity2["CPU"] = "cpu";
  ProfilerActivity2["CUDA"] = "cuda";
  ProfilerActivity2["WEBGPU"] = "webgpu";
  return ProfilerActivity2;
})(ProfilerActivity || {});
var KinetoStep = class {
  constructor(events) {
    this.events = events;
  }
  key_averages(group_by_input_shape = false) {
    return this;
  }
  table(options = {}) {
    const { sort_by = "duration", row_limit = 50 } = options;
    const stats = /* @__PURE__ */ new Map();
    for (const e of this.events) {
      const key = e.name;
      const entry = stats.get(key) || { count: 0, total_duration: 0, total_memory: 0 };
      entry.count++;
      entry.total_duration += e.duration || 0;
      entry.total_memory += e.memory_usage || 0;
      stats.set(key, entry);
    }
    const sorted = Array.from(stats.entries()).sort((a, b) => b[1].total_duration - a[1].total_duration).slice(0, row_limit);
    const lines = ["\n-------------------------------------------------------"];
    lines.push(`${"Name".padEnd(30)} | ${"Count".padStart(8)} | ${"Total Time".padStart(12)} | ${"CPU Mem".padStart(10)}`);
    lines.push("-------------------------------------------------------");
    for (const [name, data] of sorted) {
      lines.push(
        `${name.padEnd(30)} | ${data.count.toString().padStart(8)} | ${data.total_duration.toFixed(3).padStart(9)} ms | ${(data.total_memory / 1024).toFixed(0).padStart(7)} KB`
      );
    }
    lines.push("------------------------------------------------------- \n");
    return lines.join("\n");
  }
};
var activeProfiler = null;
var Profiler = class {
  events = [];
  active = false;
  options;
  constructor(options = {}) {
    this.options = {
      activities: ["cpu" /* CPU */, "webgpu" /* WEBGPU */],
      record_shapes: false,
      profile_memory: false,
      ...options
    };
  }
  start() {
    this.events = [];
    this.active = true;
    activeProfiler = this;
  }
  async stop() {
    await syncDevice();
    this.active = false;
    activeProfiler = null;
  }
  _record_event(event) {
    if (this.active) {
      this.events.push(event);
    }
  }
  get_events() {
    return this.events;
  }
  key_averages() {
    return new KinetoStep(this.events);
  }
};
async function profile(options, fn) {
  const actualOptions = typeof options === "function" ? {} : options;
  const actualFn = typeof options === "function" ? options : fn;
  if (!actualFn) throw new Error("Profile function is required");
  const p = new Profiler(actualOptions);
  p.start();
  const res = actualFn(p);
  if (res instanceof Promise) await res;
  await p.stop();
  return p;
}
function record_function(name, fn) {
  if (!activeProfiler) return fn();
  const start = performance.now();
  const startMem = bufferPool.getStats().activeBytes;
  const result = fn();
  const end = performance.now();
  const endMem = bufferPool.getStats().activeBytes;
  activeProfiler._record_event({
    name,
    activity: "cpu" /* CPU */,
    start,
    end,
    duration: end - start,
    memory_usage: endMem - startMem
  });
  return result;
}
async function record_function_async(name, fn) {
  if (!activeProfiler) return fn();
  const start = performance.now();
  const startMem = bufferPool.getStats().activeBytes;
  const result = await fn();
  await syncDevice();
  const end = performance.now();
  const endMem = bufferPool.getStats().activeBytes;
  activeProfiler._record_event({
    name,
    activity: "webgpu" /* WEBGPU */,
    start,
    end,
    duration: end - start,
    memory_usage: endMem - startMem
  });
  return result;
}

// src/utils/shape.ts
function numel(shape) {
  if (shape.length === 0) return 1;
  return shape.reduce((a, b) => a * b, 1);
}
function validateShape(shape) {
  for (let i = 0; i < shape.length; i++) {
    if (!Number.isInteger(shape[i]) || shape[i] < 0) {
      throw new Error(`Invalid shape dimension at index ${i}: ${shape[i]}`);
    }
  }
}
function inferShape(shape, totalElements) {
  const negOneIndex = shape.indexOf(-1);
  if (negOneIndex === -1) {
    return [...shape];
  }
  if (shape.lastIndexOf(-1) !== negOneIndex) {
    throw new Error("Only one dimension can be -1 in reshape");
  }
  let knownProduct = 1;
  for (let i = 0; i < shape.length; i++) {
    if (i !== negOneIndex) {
      knownProduct *= shape[i];
    }
  }
  if (totalElements % knownProduct !== 0) {
    throw new Error(`Cannot reshape tensor of ${totalElements} elements to shape [${shape.join(", ")}]`);
  }
  const result = [...shape];
  result[negOneIndex] = totalElements / knownProduct;
  return result;
}

// src/utils/broadcast.ts
function broadcastShapes(shapeA, shapeB) {
  const maxDim = Math.max(shapeA.length, shapeB.length);
  const result = new Array(maxDim);
  for (let i = 0; i < maxDim; i++) {
    const dimA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1;
    const dimB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;
    if (dimA === dimB) {
      result[maxDim - 1 - i] = dimA;
    } else if (dimA === 1) {
      result[maxDim - 1 - i] = dimB;
    } else if (dimB === 1) {
      result[maxDim - 1 - i] = dimA;
    } else {
      throw new Error(
        `Shapes [${shapeA.join(", ")}] and [${shapeB.join(", ")}] are not broadcastable`
      );
    }
  }
  return result;
}
function needsBroadcast(shapeA, shapeB) {
  if (shapeA.length !== shapeB.length) return true;
  for (let i = 0; i < shapeA.length; i++) {
    if (shapeA[i] !== shapeB[i]) return true;
  }
  return false;
}

// src/ops/creation.ts
function inferShapeFromData(data) {
  if (typeof data === "number") {
    return [];
  }
  if (!Array.isArray(data) || data.length === 0) {
    return [0];
  }
  const first = data[0];
  const restShape = inferShapeFromData(first);
  return [data.length, ...restShape];
}
function flattenData(data) {
  if (typeof data === "number") {
    return [data];
  }
  const result = [];
  for (const item of data) {
    result.push(...flattenData(item));
  }
  return result;
}
function tensor(data, options = {}) {
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  const shape = inferShapeFromData(data);
  const flat = flattenData(data);
  const TypedArrayCtor = getTypedArrayConstructor(dtype);
  const typedData = new TypedArrayCtor(flat);
  const buffer = createBufferWithData(typedData, dtype);
  return new Tensor({
    buffer,
    shape,
    dtype,
    device: "webgpu",
    requires_grad
  });
}
function zeros(shape, options = {}) {
  validateShape(shape);
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  return _fill(shape, 0, dtype, requires_grad);
}
function ones(shape, options = {}) {
  validateShape(shape);
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  return _fill(shape, 1, dtype, requires_grad);
}
function full(shape, fillValue, options = {}) {
  validateShape(shape);
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  return _fill(shape, fillValue, dtype, requires_grad);
}
function zeros_like(tensor2, options = {}) {
  return zeros([...tensor2.shape], {
    dtype: options.dtype ?? tensor2.dtype,
    requires_grad: options.requires_grad ?? tensor2.requires_grad
  });
}
function ones_like(tensor2, options = {}) {
  return ones([...tensor2.shape], {
    dtype: options.dtype ?? tensor2.dtype,
    requires_grad: options.requires_grad ?? tensor2.requires_grad
  });
}
function full_like(input, fillValue, options = {}) {
  return full([...input.shape], fillValue, {
    dtype: options.dtype ?? input.dtype,
    requires_grad: options.requires_grad ?? input.requires_grad
  });
}
function empty_like(input, options = {}) {
  return zeros_like(input, options);
}
function rand(shape, options = {}) {
  validateShape(shape);
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  return _random(shape, "rand", dtype, requires_grad);
}
function randn(shape, options = {}) {
  validateShape(shape);
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  return _random(shape, "randn", dtype, requires_grad);
}
function eye(n, m, options = {}) {
  m = m ?? n;
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  const data = new Float32Array(n * m);
  const minDim = Math.min(n, m);
  for (let i = 0; i < minDim; i++) {
    data[i * m + i] = 1;
  }
  const buffer = createBufferWithData(data, dtype);
  return new Tensor({
    buffer,
    shape: [n, m],
    dtype,
    device: "webgpu",
    requires_grad
  });
}
function arange(start, end, step = 1, options = {}) {
  if (end === void 0) {
    end = start;
    start = 0;
  }
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  const length = Math.ceil((end - start) / step);
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    data[i] = start + i * step;
  }
  const buffer = createBufferWithData(data, dtype);
  return new Tensor({
    buffer,
    shape: [length],
    dtype,
    device: "webgpu",
    requires_grad
  });
}
function linspace(start, end, steps, options = {}) {
  const dtype = options.dtype ?? "float32";
  const requires_grad = options.requires_grad ?? false;
  const data = new Float32Array(steps);
  if (steps === 1) {
    data[0] = start;
  } else {
    const stepSize = (end - start) / (steps - 1);
    for (let i = 0; i < steps; i++) {
      data[i] = start + i * stepSize;
    }
  }
  const buffer = createBufferWithData(data, dtype);
  return new Tensor({
    buffer,
    shape: [steps],
    dtype,
    device: "webgpu",
    requires_grad
  });
}
function logspace(start, end, steps, base = 10, options = {}) {
  const result = linspace(start, end, steps, options);
  if (base === Math.E) {
    return result.exp();
  }
  return result.mul(Math.log(base)).exp();
}
function tril(input, diagonal = 0) {
  if (input.shape.length !== 2) {
    throw new Error("tril currently only supports 2D tensors");
  }
  const [rows, cols] = input.shape;
  const device = getDevice();
  const outputBuffer = createStorageBuffer(input.numel() * getDTypeBytes(input.dtype));
  const paramsData = new ArrayBuffer(16);
  new Uint32Array(paramsData, 0, 2).set([rows, cols]);
  new Int32Array(paramsData, 8, 1)[0] = diagonal;
  const paramsBuffer = device.createBuffer({
    size: 16,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);
  const pipeline = getOrCreatePipeline(tril_default, "main");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: paramsBuffer } }
    ]
  });
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(input.numel()));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  return new Tensor({
    buffer: outputBuffer,
    shape: [rows, cols],
    dtype: input.dtype,
    device: "webgpu",
    requires_grad: false
  });
}
function cat(tensors, dim = 0) {
  return Tensor.cat(tensors, dim);
}
function stack(tensors, dim = 0) {
  if (tensors.length === 0) {
    throw new Error("stack requires at least one tensor");
  }
  const expanded = tensors.map((t) => t.unsqueeze(dim));
  return cat(expanded, dim);
}
function vstack(tensors) {
  if (tensors.length === 0) {
    throw new Error("vstack requires at least one tensor");
  }
  const first = tensors[0];
  if (first.dim() === 1) {
    return stack(tensors, 0);
  }
  return cat(tensors, 0);
}
var row_stack = vstack;
function hstack(tensors) {
  if (tensors.length === 0) {
    throw new Error("hstack requires at least one tensor");
  }
  const first = tensors[0];
  if (first.dim() === 1) {
    return cat(tensors, 0);
  }
  return cat(tensors, 1);
}
function dstack(tensors) {
  if (tensors.length === 0) {
    throw new Error("dstack requires at least one tensor");
  }
  const processed = tensors.map((t) => {
    if (t.dim() === 1) return t.unsqueeze(0).unsqueeze(1);
    if (t.dim() === 2) return t.unsqueeze(2);
    return t;
  });
  return cat(processed, 2);
}
function column_stack(tensors) {
  if (tensors.length === 0) {
    throw new Error("column_stack requires at least one tensor");
  }
  const processed = tensors.map((t) => {
    if (t.dim() === 1) return t.unsqueeze(1);
    return t;
  });
  return hstack(processed);
}
function _fill(shape, value, dtype, requires_grad) {
  const device = getDevice();
  const n = numel(shape);
  const outputBuffer = createStorageBuffer(n * getDTypeBytes(dtype));
  const paramsData = new ArrayBuffer(8);
  new Float32Array(paramsData, 0, 1)[0] = value;
  new Uint32Array(paramsData, 4, 1)[0] = n;
  const paramsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);
  const pipeline = getOrCreatePipeline(fill_default, "fill");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
    ]
  });
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(n));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  return new Tensor({
    buffer: outputBuffer,
    shape,
    dtype,
    device: "webgpu",
    requires_grad
  });
}
var globalSeed = Math.floor(Math.random() * 2147483647);
function manual_seed(seed) {
  globalSeed = seed;
}
function _random(shape, op, dtype, requires_grad) {
  const device = getDevice();
  const n = numel(shape);
  const outputBuffer = createStorageBuffer(n * getDTypeBytes(dtype));
  globalSeed = globalSeed * 1103515245 + 12345 & 2147483647;
  const paramsData = new Uint32Array([globalSeed, n]);
  const paramsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);
  const pipeline = getOrCreatePipeline(random_default, op);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
    ]
  });
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(n));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  return new Tensor({
    buffer: outputBuffer,
    shape,
    dtype,
    device: "webgpu",
    requires_grad
  });
}

// src/tensor/Tensor.ts
var Tensor = class _Tensor {
  _buffer;
  _shape;
  _dtype;
  _device;
  _requires_grad;
  _grad = null;
  _grad_fn = null;
  _is_leaf = true;
  /** @internal */
  constructor(data) {
    this._buffer = data.buffer;
    this._shape = data.shape;
    this._dtype = data.dtype;
    this._device = data.device;
    this._requires_grad = data.requires_grad;
    this._grad_fn = data.grad_fn ?? null;
    this._is_leaf = !data.grad_fn;
  }
  static cat(tensors, dim = 0) {
    if (tensors.length === 0) {
      throw new Error("cat requires at least one tensor");
    }
    if (tensors.length === 1) {
      return tensors[0];
    }
    const first = tensors[0];
    const ndim = first.shape.length;
    if (dim < 0) dim += ndim;
    for (let i = 1; i < tensors.length; i++) {
      if (tensors[i].shape.length !== ndim) {
        throw new Error("All tensors must have the same number of dimensions");
      }
      for (let d2 = 0; d2 < ndim; d2++) {
        if (d2 !== dim && tensors[i].shape[d2] !== first.shape[d2]) {
          throw new Error(`Tensor shapes don't match at dimension ${d2}`);
        }
      }
    }
    const outputShape = [...first.shape];
    outputShape[dim] = tensors.reduce((sum, t) => sum + t.shape[dim], 0);
    const totalElements = outputShape.reduce((a, b) => a * b, 1);
    const device = getDevice();
    const bytesPerElement = getDTypeBytes(first.dtype);
    const outputBuffer = createStorageBuffer(totalElements * bytesPerElement);
    const commandEncoder = device.createCommandEncoder();
    const outerSize = numel(first.shape.slice(0, dim));
    const innerSize = numel(first.shape.slice(dim + 1));
    const outDimSize = outputShape[dim];
    let dimOffset = 0;
    for (const t of tensors) {
      const tDimSize = t.shape[dim];
      const copySize = tDimSize * innerSize * bytesPerElement;
      for (let i = 0; i < outerSize; i++) {
        const srcOffset = i * tDimSize * innerSize * bytesPerElement;
        const dstOffset = (i * outDimSize * innerSize + dimOffset * innerSize) * bytesPerElement;
        commandEncoder.copyBufferToBuffer(t.buffer, srcOffset, outputBuffer, dstOffset, copySize);
      }
      dimOffset += tDimSize;
    }
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({
      buffer: outputBuffer,
      shape: outputShape,
      dtype: first.dtype,
      device: "webgpu",
      requires_grad: tensors.some((t) => t.requires_grad)
    });
  }
  /**
   * Manually release the GPU buffer associated with this tensor.
   * Returns it to the pool for reuse.
   */
  destroy() {
    if (this._buffer) {
      bufferPool.release(this._buffer);
      this._buffer = null;
    }
    if (this._grad) {
      this._grad.destroy();
      this._grad = null;
    }
    this._grad_fn = null;
  }
  tile(reps) {
    return this.repeat(reps);
  }
  repeat(sizes) {
    if (sizes.length < this._shape.length) {
      throw new Error(`Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor`);
    }
    let tensor2 = this;
    if (sizes.length > this._shape.length) {
      const diff = sizes.length - this._shape.length;
      for (let i = 0; i < diff; i++) {
        tensor2 = tensor2.unsqueeze(0);
      }
    }
    for (let i = 0; i < sizes.length; i++) {
      const repeatTimes = sizes[i];
      if (repeatTimes > 1) {
        const copies = [];
        for (let k = 0; k < repeatTimes; k++) {
          copies.push(tensor2);
        }
        tensor2 = _Tensor.cat(copies, i);
      }
    }
    return tensor2;
  }
  // ============ Properties ============
  get shape() {
    return this._shape;
  }
  get dtype() {
    return this._dtype;
  }
  get device() {
    return this._device;
  }
  get requires_grad() {
    return this._requires_grad;
  }
  get grad() {
    return this._grad;
  }
  set grad(value) {
    this._grad = value;
  }
  get grad_fn() {
    return this._grad_fn;
  }
  get is_leaf() {
    return this._is_leaf;
  }
  /**
   * Transpose property (alias for t()).
   * @status implemented
   */
  get T() {
    return this.t();
  }
  /**
   * Change if autograd should record operations on this tensor: 
   * sets the tensor's requires_grad attribute in-place.
   * @pytorch tensor.requires_grad_()
   */
  requires_grad_(value = true) {
    this._requires_grad = value;
    return this;
  }
  detach() {
    return new _Tensor({
      ...this._getState(),
      requires_grad: false,
      grad_fn: void 0
    });
  }
  /**
   * Copies the elements from src into self tensor and returns self.
   * @pytorch tensor.copy_()
   */
  copy_(src) {
    if (this.numel() !== src.numel()) {
      throw new Error(`copy_: size mismatch, self has ${this.numel()} elements, src has ${src.numel()}`);
    }
    const device = getDevice();
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(src.buffer, 0, this._buffer, 0, this._buffer.size);
    device.queue.submit([commandEncoder.finish()]);
    return this;
  }
  clone() {
    const device = getDevice();
    const sizeBytes = this.numel() * getDTypeBytes(this._dtype);
    const alignedSize = Math.ceil(sizeBytes / 4) * 4;
    const newBuffer = device.createBuffer({
      size: alignedSize,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST
    });
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this._buffer, 0, newBuffer, 0, alignedSize);
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({
      buffer: newBuffer,
      shape: [...this._shape],
      dtype: this._dtype,
      device: "webgpu",
      requires_grad: this._requires_grad
    });
  }
  /** @internal */
  get buffer() {
    return this._buffer;
  }
  // ============ Basic Math ============
  add(other) {
    return record_function("torch.add", () => {
      if (typeof other === "number") return this._scalarOp("add_scalar", other);
      return this._binaryOp("add", other);
    });
  }
  sub(other) {
    return record_function("torch.sub", () => {
      if (typeof other === "number") return this._scalarOp("sub_scalar", other);
      return this._binaryOp("sub", other);
    });
  }
  mul(other) {
    return record_function("torch.mul", () => {
      if (typeof other === "number") return this._scalarOp("mul_scalar", other);
      return this._binaryOp("mul", other);
    });
  }
  div(other) {
    return record_function("torch.div", () => {
      if (typeof other === "number") return this._scalarOp("div_scalar", other);
      return this._binaryOp("div_op", other);
    });
  }
  // ============ Unary Math ============
  abs() {
    return record_function("torch.abs", () => this._unaryOp("abs_op"));
  }
  acos() {
    return record_function("torch.acos", () => this._unaryOp("acos_op"));
  }
  asin() {
    return record_function("torch.asin", () => this._unaryOp("asin_op"));
  }
  atan() {
    return record_function("torch.atan", () => this._unaryOp("atan_op"));
  }
  ceil() {
    return record_function("torch.ceil", () => this._unaryOp("ceil_op"));
  }
  cos() {
    return record_function("torch.cos", () => this._unaryOp("cos_op"));
  }
  cosh() {
    return record_function("torch.cosh", () => this._unaryOp("cosh_op"));
  }
  exp() {
    return record_function("torch.exp", () => this._unaryOp("exp_op"));
  }
  exp2() {
    return record_function("torch.exp2", () => this._unaryOp("exp2_op"));
  }
  floor() {
    return record_function("torch.floor", () => this._unaryOp("floor_op"));
  }
  log(base = Math.E) {
    return record_function("torch.log", () => {
      if (base === Math.E || base === "e") return this._unaryOp("log_op");
      return this._unaryOp("log_op").div(Math.log(base));
    });
  }
  log10() {
    return record_function("torch.log10", () => this._unaryOp("log10"));
  }
  log2() {
    return record_function("torch.log2", () => this._unaryOp("log2_op"));
  }
  log1p() {
    return record_function("torch.log1p", () => this._unaryOp("log1p"));
  }
  neg() {
    return record_function("torch.neg", () => this._unaryOp("neg"));
  }
  round() {
    return record_function("torch.round", () => this._unaryOp("round_op"));
  }
  sin() {
    return record_function("torch.sin", () => this._unaryOp("sin_op"));
  }
  sinh() {
    return record_function("torch.sinh", () => this._unaryOp("sinh_op"));
  }
  acosh() {
    return record_function("torch.acosh", () => this._unaryOp("acosh_op"));
  }
  asinh() {
    return record_function("torch.asinh", () => this._unaryOp("asinh_op"));
  }
  atanh() {
    return record_function("torch.atanh", () => this._unaryOp("atanh_op"));
  }
  sqrt() {
    return record_function("torch.sqrt", () => this._unaryOp("sqrt_op"));
  }
  tan() {
    return record_function("torch.tan", () => this._unaryOp("tan_op"));
  }
  tanh() {
    return record_function("torch.tanh", () => this._unaryOp("tanh_op"));
  }
  trunc() {
    return record_function("torch.trunc", () => this._unaryOp("trunc_op"));
  }
  frac() {
    return record_function("torch.frac", () => this._unaryOp("frac_op"));
  }
  reciprocal() {
    return record_function("torch.reciprocal", () => this._unaryOp("reciprocal_op"));
  }
  rsqrt() {
    return record_function("torch.rsqrt", () => this._unaryOp("rsqrt_op"));
  }
  square() {
    return record_function("torch.square", () => this._unaryOp("square_op"));
  }
  sigmoid() {
    return record_function("torch.sigmoid", () => this._unaryOp("sigmoid"));
  }
  relu() {
    return record_function("torch.relu", () => this._unaryOp("relu"));
  }
  gelu() {
    return record_function("torch.gelu", () => this._unaryOp("gelu"));
  }
  softplus() {
    return record_function("torch.softplus", () => this._unaryOp("softplus_op"));
  }
  silu() {
    return record_function("torch.silu", () => this._unaryOp("silu_op"));
  }
  mish() {
    return record_function("torch.mish", () => this._unaryOp("mish_op"));
  }
  hardsigmoid() {
    return record_function("torch.hardsigmoid", () => this._unaryOp("hardsigmoid_op"));
  }
  hardswish() {
    return record_function("torch.hardswish", () => this._unaryOp("hardswish_op"));
  }
  softsign() {
    return record_function("torch.softsign", () => this._unaryOp("softsign_op"));
  }
  tanhshrink() {
    return record_function("torch.tanhshrink", () => this._unaryOp("tanhshrink_op"));
  }
  leaky_relu(negative_slope = 0.01) {
    return record_function("torch.leaky_relu", () => {
      const pos = this.relu();
      const neg = this.neg().relu().neg();
      return pos.add(neg.mul(negative_slope));
    });
  }
  elu(alpha = 1) {
    return record_function("torch.elu", () => {
      const neg_part = this.neg().relu().neg();
      return this.relu().add(neg_part.exp().sub(1).mul(alpha));
    });
  }
  selu() {
    return record_function("torch.selu", () => {
      return this.elu(1.6732632423543772).mul(1.0507009873554805);
    });
  }
  threshold(threshold, value) {
    return record_function("torch.threshold", () => {
      throw new Error("threshold not implemented");
    });
  }
  zeros_like() {
    return this.mul(0);
  }
  ones_like() {
    return this.mul(0).add(1);
  }
  heaviside(values) {
    return record_function("torch.heaviside", () => {
      const v_expanded = values.expand([...this._shape]);
      const device = getDevice();
      const outputBuffer = createStorageBuffer(this.numel() * 4);
      const pipeline = getOrCreatePipeline(heaviside_default, "main");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: v_expanded.buffer, offset: 0, size: v_expanded.buffer.size } },
          { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      const res = new _Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: "webgpu", requires_grad: false });
      v_expanded.destroy();
      return res;
    });
  }
  glu(dim = -1) {
    return record_function("torch.glu", () => {
      const [a, b] = this.chunk(2, dim);
      return a.mul(b.sigmoid());
    });
  }
  pow(exponent) {
    return record_function("torch.pow", () => {
      if (typeof exponent === "number") return this._scalarOp("pow_scalar", exponent);
      return this._binaryOp("pow_tensor", exponent);
    });
  }
  // ============ Advanced Pointwise ============
  atan2(other) {
    return record_function("torch.atan2", () => this._binaryOp("atan2_op", other));
  }
  hypot(other) {
    return record_function("torch.hypot", () => this._binaryOp("hypot_op", other));
  }
  logaddexp(other) {
    return record_function("torch.logaddexp", () => this._binaryOp("logaddexp", other));
  }
  bitwise_and(other) {
    return record_function("torch.bitwise_and", () => this._binaryOp("bitwise_and", other));
  }
  bitwise_or(other) {
    return record_function("torch.bitwise_or", () => this._binaryOp("bitwise_or", other));
  }
  bitwise_xor(other) {
    return record_function("torch.bitwise_xor", () => this._binaryOp("bitwise_xor", other));
  }
  clamp(min = -Infinity, max = Infinity) {
    return record_function("torch.clamp", () => {
      const device = getDevice();
      const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
      const paramsData = new ArrayBuffer(16);
      new Float32Array(paramsData, 0, 2).set([min, max]);
      new Uint32Array(paramsData, 8, 1)[0] = this.numel();
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(clamp_default, "main");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      return new _Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad });
    });
  }
  masked_fill(mask, value) {
    return record_function("torch.masked_fill", () => {
      const device = getDevice();
      const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
      const paramsData = new ArrayBuffer(16);
      new Float32Array(paramsData, 0, 1)[0] = value;
      new Uint32Array(paramsData, 4, 1)[0] = this.numel();
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(masked_fill_default, "main");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: mask.buffer, offset: 0, size: mask.buffer.size } },
          { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      return new _Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad });
    });
  }
  triu(diagonal = 0) {
    if (this._shape.length !== 2) throw new Error("triu currently only supports 2D tensors");
    const [rows, cols] = this._shape;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * 4);
    const paramsData = new Uint32Array([rows, cols, diagonal, 0]);
    const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(triu_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({ buffer: outputBuffer, shape: [rows, cols], dtype: this._dtype, device: "webgpu", requires_grad: false });
  }
  tril(diagonal = 0) {
    if (this._shape.length !== 2) throw new Error("tril currently only supports 2D tensors");
    const [rows, cols] = this._shape;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const paramsData = new ArrayBuffer(16);
    new Uint32Array(paramsData, 0, 2).set([rows, cols]);
    new Int32Array(paramsData, 8, 1)[0] = diagonal;
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(tril_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({
      buffer: outputBuffer,
      shape: [rows, cols],
      dtype: this._dtype,
      device: "webgpu",
      requires_grad: false
    });
  }
  to(dtype) {
    if (this._dtype === dtype) return this;
    if (this._dtype === "float32" && dtype === "float32") return this;
    if (getDTypeBytes(this._dtype) === getDTypeBytes(dtype)) {
      const res = this.clone();
      res._dtype = dtype;
      return res;
    }
    throw new Error(`Tensor.to: conversion from ${this._dtype} to ${dtype} not yet implemented`);
  }
  flip(dims) {
    let current = this;
    for (let dim of dims) {
      if (dim < 0) dim += current._shape.length;
      const dimSize = current._shape[dim];
      const stride = current.stride()[dim];
      const device = getDevice();
      const outputBuffer = createStorageBuffer(current.numel() * 4);
      const paramsData = new Uint32Array([0, dimSize, stride, 0]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(flip_default, "main");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: current._buffer, offset: 0, size: current._buffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(current.numel()));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      const next = new _Tensor({ buffer: outputBuffer, shape: [...current._shape], dtype: current._dtype, device: "webgpu", requires_grad: false });
      if (current !== this) current.destroy();
      current = next;
    }
    return current;
  }
  cumsum(dim) {
    return this._cumOp(dim, cumsum_default);
  }
  cumprod(dim) {
    return this._cumOp(dim, cumprod_default);
  }
  diag(diagonal = 0) {
    const device = getDevice();
    if (this._shape.length === 1) {
      const n = this._shape[0];
      const m = n + Math.abs(diagonal);
      const res = this._zeros([m, m]);
      const outputBuffer = res.buffer;
      const paramsData = new Int32Array([n, diagonal, 0, 0]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(diag_vec_to_mtx_default, "main");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(n));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      return res;
    } else if (this._shape.length === 2) {
      const [rows, cols] = this._shape;
      let diagLen;
      if (diagonal >= 0) {
        diagLen = Math.max(0, Math.min(rows, cols - diagonal));
      } else {
        diagLen = Math.max(0, Math.min(rows + diagonal, cols));
      }
      const outputBuffer = createStorageBuffer(diagLen * 4);
      const paramsData = new Int32Array([rows, cols, diagonal, diagLen]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(diag_mtx_to_vec_default, "main");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(diagLen));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      return new _Tensor({ buffer: outputBuffer, shape: [diagLen], dtype: this._dtype, device: "webgpu", requires_grad: false });
    }
    throw new Error("diag expects 1D or 2D tensor");
  }
  _cumOp(dim, shader) {
    if (dim < 0) dim += this._shape.length;
    const permutation = Array.from({ length: this._shape.length }, (_, i) => i);
    permutation.splice(dim, 1);
    permutation.push(dim);
    const transposed = this.permute(permutation);
    const reduceSize = this._shape[dim];
    const batchSize = transposed.numel() / reduceSize;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(transposed.numel() * 4);
    const paramsData = new Uint32Array([batchSize, reduceSize, 0, 0]);
    const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(shader, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: transposed.buffer, offset: 0, size: transposed.buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(batchSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const resTransposed = new _Tensor({ buffer: outputBuffer, shape: transposed.shape, dtype: this._dtype, device: "webgpu", requires_grad: false });
    const invPerm = new Array(this._shape.length);
    for (let i = 0; i < permutation.length; i++) invPerm[permutation[i]] = i;
    const final = resTransposed.permute(invPerm);
    transposed.destroy();
    resTransposed.destroy();
    return final;
  }
  _zeros(shape) {
    const n = numel(shape);
    const device = getDevice();
    const outputBuffer = createStorageBuffer(n * 4);
    const paramsData = new ArrayBuffer(8);
    new Float32Array(paramsData, 0, 1)[0] = 0;
    new Uint32Array(paramsData, 4, 1)[0] = n;
    const paramsBuffer = device.createBuffer({ size: 8, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(fill_default, "fill");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(n));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({ buffer: outputBuffer, shape, dtype: "float32", device: "webgpu", requires_grad: false });
  }
  _sliceDimStep(dim, start, end, step = 1) {
    if (dim < 0) dim += this._shape.length;
    const dimSize = this._shape[dim];
    if (start < 0) start += dimSize;
    if (end < 0) end += dimSize;
    if (start < 0) start = 0;
    if (end > dimSize) end = dimSize;
    if (step === 1) {
      return this.narrow(dim, start, Math.max(0, end - start));
    }
    const count = Math.ceil(Math.max(0, end - start) / step);
    const indices = new Int32Array(count);
    for (let i = 0; i < count; i++) {
      indices[i] = start + i * step;
    }
    const device = getDevice();
    const indexBuf = createStorageBuffer(count * 4);
    device.queue.writeBuffer(indexBuf, 0, indices);
    const idxTensor = new _Tensor({
      buffer: indexBuf,
      shape: [count],
      dtype: "int32",
      device: "webgpu",
      requires_grad: false
    });
    const res = this.index_select(dim, idxTensor);
    idxTensor.destroy();
    return res;
  }
  trapezoid(dx = 1, dim = -1) {
    if (dim < 0) dim += this.shape.length;
    const n = this.shape[dim];
    const left = this._sliceDimStep(dim, 0, n - 1);
    const right = this._sliceDimStep(dim, 1, n);
    return left.add(right).mul(dx * 0.5).sum(dim);
  }
  cumulative_trapezoid(dx = 1, dim = -1) {
    if (dim < 0) dim += this.shape.length;
    const n = this.shape[dim];
    const left = this._sliceDimStep(dim, 0, n - 1);
    const right = this._sliceDimStep(dim, 1, n);
    return left.add(right).mul(dx * 0.5).cumsum(dim);
  }
  index_select(dim, index) {
    if (dim < 0) dim += this._shape.length;
    if (index.dim() !== 1) throw new Error("index_select: index must be 1D");
    if (this._shape.length > 2) throw new Error("index_select currently supports up to 2D tensors");
    const device = getDevice();
    const inputDim0 = this._shape.length === 1 ? this._shape[0] : this._shape[0];
    const inputDim1 = this._shape.length === 1 ? 1 : this._shape[1];
    const outputShape = [...this._shape];
    outputShape[dim] = index.shape[0];
    const outputSize = numel(outputShape);
    const outputBuffer = createStorageBuffer(outputSize * getDTypeBytes(this._dtype));
    const paramsData = new Uint32Array([dim, inputDim0, inputDim1, index.shape[0]]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const shaderName = this._shape.length === 1 ? "index_select_1d" : "index_select_2d";
    const pipeline = getOrCreatePipeline(index_select_default, shaderName);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: index.buffer, offset: 0, size: index.buffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({ buffer: outputBuffer, shape: outputShape, dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad });
  }
  masked_select(mask) {
    throw new Error("masked_select not yet fully implemented");
  }
  select(dim, index) {
    return this.narrow(dim, index, 1).squeeze(dim);
  }
  take(indices) {
    return this.flatten().index_select(0, indices);
  }
  where(condition, other) {
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const pipeline = getOrCreatePipeline(where_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: condition.buffer, offset: 0, size: condition.buffer.size } },
        { binding: 1, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 2, resource: { buffer: other.buffer, offset: 0, size: other.buffer.size } },
        { binding: 3, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad || other.requires_grad });
  }
  // ============ Comparison ============
  eq(other) {
    if (typeof other === "number") return this._scalarOp("eq_scalar", other);
    return this._compareOp("eq", other);
  }
  ne(other) {
    if (typeof other === "number") return this._scalarOp("ne_scalar", other);
    return this._compareOp("ne", other);
  }
  lt(other) {
    if (typeof other === "number") return this._scalarOp("lt_scalar", other);
    return this._compareOp("lt", other);
  }
  gt(other) {
    if (typeof other === "number") return this._scalarOp("gt_scalar", other);
    return this._compareOp("gt", other);
  }
  ge(other) {
    if (typeof other === "number") return this._scalarOp("ge_scalar", other);
    return this._compareOp("ge", other);
  }
  le(other) {
    if (typeof other === "number") return this._scalarOp("le_scalar", other);
    return this._compareOp("le", other);
  }
  greater(other) {
    return this.gt(other);
  }
  greater_equal(other) {
    return this.ge(other);
  }
  less(other) {
    return this.lt(other);
  }
  less_equal(other) {
    return this.le(other);
  }
  not_equal(other) {
    return this.ne(other);
  }
  isnan() {
    return this._unaryOp("isnan_op");
  }
  isinf() {
    return this._unaryOp("isinf_op");
  }
  isfinite() {
    return this._unaryOp("isfinite_op");
  }
  isposinf() {
    return this._unaryOp("isposinf_op");
  }
  isneginf() {
    return this._unaryOp("isneginf_op");
  }
  maximum(other) {
    return this._compareOp("maximum_op", other);
  }
  minimum(other) {
    return this._compareOp("minimum_op", other);
  }
  fmax(other) {
    return this._compareOp("fmax_op", other);
  }
  fmin(other) {
    return this._compareOp("fmin_op", other);
  }
  async equal(other) {
    if (this._shape.length !== other._shape.length) return false;
    for (let i = 0; i < this._shape.length; i++) {
      if (this._shape[i] !== other._shape[i]) return false;
    }
    const mask = this.isclose(other, 0, 0, true);
    const diff = mask.all();
    const result = await diff.item();
    mask.destroy();
    diff.destroy();
    return result === 1;
  }
  isclose(other, rtol = 1e-5, atol = 1e-8, equal_nan = false) {
    const self_finite = this.isfinite();
    const other_finite = other.isfinite();
    const both_finite = self_finite.mul(other_finite);
    const diff = this.sub(other).abs();
    const tol = other.abs().mul(rtol).add(atol);
    const close_finite = diff.le(tol).mul(both_finite);
    const both_inf = this.isinf().mul(other.isinf());
    const close_inf = both_inf.mul(this.eq(other));
    let res = close_finite.add(close_inf).clamp(0, 1);
    if (equal_nan) {
      const both_nan = this.isnan().mul(other.isnan());
      res = res.add(both_nan).clamp(0, 1);
    }
    return res;
  }
  async allclose(other, rtol = 1e-5, atol = 1e-8, equal_nan = false) {
    const res = this.isclose(other, rtol, atol, equal_nan).all();
    const result = await res.item();
    res.destroy();
    return result === 1;
  }
  // ============ Shape Operations ============
  numel() {
    return numel(this._shape);
  }
  dim() {
    return this._shape.length;
  }
  size(dim) {
    if (dim === void 0) return this._shape;
    if (dim < 0) dim += this._shape.length;
    return this._shape[dim];
  }
  stride(dim) {
    const ndim = this._shape.length;
    const strides = new Array(ndim);
    let s = 1;
    for (let i = ndim - 1; i >= 0; i--) {
      strides[i] = s;
      s *= this._shape[i];
    }
    if (dim === void 0) return strides;
    if (dim < 0) dim += ndim;
    return strides[dim];
  }
  broadcast_to(shape) {
    return this.expand(shape);
  }
  expand(shape) {
    const ndim = shape.length;
    const inDim = this._shape.length;
    if (ndim < inDim) throw new Error("expand: rank cannot be reduced");
    const paddedIn = new Array(ndim).fill(1);
    for (let i = 0; i < inDim; i++) paddedIn[ndim - inDim + i] = this._shape[i];
    const currentStrides = this.stride();
    const inStrides = new Array(ndim).fill(0);
    for (let i = 0; i < inDim; i++) inStrides[ndim - inDim + i] = currentStrides[i];
    const finalShape = shape.map((s, i) => s === -1 ? paddedIn[i] : s);
    for (let i = 0; i < ndim; i++) {
      if (finalShape[i] !== paddedIn[i] && paddedIn[i] !== 1) {
        throw new Error(`expand: incompatible shapes: ${this._shape} to ${finalShape}`);
      }
    }
    const outputSize = numel(finalShape);
    const device = getDevice();
    const outputBuffer = createStorageBuffer(outputSize * 4);
    const pShape = [1, 1, 1, 1];
    const pStrides = [0, 0, 0, 0];
    for (let i = 0; i < Math.min(ndim, 4); i++) {
      const finalDimIdx = ndim - 1 - i;
      const pIdx = 3 - i;
      pShape[pIdx] = finalShape[finalDimIdx];
      if (i < inDim) {
        const inDimIdx = inDim - 1 - i;
        pStrides[pIdx] = this._shape[inDimIdx] === 1 ? 0 : currentStrides[inDimIdx];
      } else {
        pStrides[pIdx] = 0;
      }
    }
    const paramsData = new Uint32Array([...pShape, ...pStrides]);
    const paramsBuffer = device.createBuffer({ size: 32, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(expand_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({ buffer: outputBuffer, shape: finalShape, dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad });
  }
  reshape(shape) {
    const newShape = inferShape(shape, this.numel());
    validateShape(newShape);
    const self = this;
    let grad_fn;
    if (this._requires_grad) {
      const oldShape = [...this._shape];
      grad_fn = {
        backward(gradOutput) {
          self.accumulateGrad(gradOutput.reshape(oldShape));
        }
      };
    }
    return new _Tensor({
      buffer: this._buffer,
      shape: newShape,
      dtype: this._dtype,
      device: this._device,
      requires_grad: this._requires_grad,
      grad_fn
    });
  }
  view(...args) {
    const shape = args.length === 1 && Array.isArray(args[0]) ? args[0] : args;
    return this.reshape(shape);
  }
  squeeze(dim) {
    let newShape;
    if (dim === void 0) {
      newShape = this._shape.filter((d2) => d2 !== 1);
    } else {
      if (dim < 0) dim += this._shape.length;
      if (this._shape[dim] !== 1) return this;
      newShape = [...this._shape];
      newShape.splice(dim, 1);
    }
    return this.reshape(newShape);
  }
  unsqueeze(dim) {
    if (dim < 0) dim += this._shape.length + 1;
    const newShape = [...this._shape];
    newShape.splice(dim, 0, 1);
    return this.reshape(newShape);
  }
  flatten(startDim = 0, endDim = -1) {
    if (startDim < 0) startDim += this._shape.length;
    if (endDim < 0) endDim += this._shape.length;
    const before = this._shape.slice(0, startDim);
    const middle = this._shape.slice(startDim, endDim + 1);
    const after = this._shape.slice(endDim + 1);
    const flattenedDim = middle.reduce((a, b) => a * b, 1);
    const newShape = [...before, flattenedDim, ...after];
    return this.reshape(newShape);
  }
  /**
   * Returns a contiguous tensor (for now, just returns self as we assume contiguous).
   * @status implemented
   * @pytorch tensor.contiguous()
   */
  contiguous() {
    return this;
  }
  /**
   * Splits tensor into chunks along a dimension.
   * @status implemented
   * @pytorch tensor.split()
   */
  split(split_size, dim = 0) {
    if (dim < 0) dim += this._shape.length;
    const dimSize = this._shape[dim];
    const numChunks = Math.ceil(dimSize / split_size);
    const results = [];
    for (let i = 0; i < numChunks; i++) {
      const start = i * split_size;
      const end = Math.min(start + split_size, dimSize);
      const slices = [];
      for (let d2 = 0; d2 < this._shape.length; d2++) {
        if (d2 === dim) {
          slices.push({ start, stop: end });
        } else {
          slices.push({ start: 0, stop: this._shape[d2] });
        }
      }
      results.push(this._sliceMultiDim(slices));
    }
    return results;
  }
  chunk(chunks, dim = 0) {
    if (dim < 0) dim += this._shape.length;
    const split_size = Math.ceil(this._shape[dim] / chunks);
    return this.split(split_size, dim);
  }
  movedim(source, destination) {
    const src = Array.isArray(source) ? source : [source];
    const dst = Array.isArray(destination) ? destination : [destination];
    if (src.length !== dst.length) {
      throw new Error(`movedim: source and destination length mismatch: ${src.length} vs ${dst.length}`);
    }
    const ndim = this._shape.length;
    const order = Array.from({ length: ndim }, (_, i) => i);
    const srcNormalized = src.map((d2) => d2 < 0 ? d2 + ndim : d2);
    const dstNormalized = dst.map((d2) => d2 < 0 ? d2 + ndim : d2);
    const remaining = order.filter((d2) => !srcNormalized.includes(d2));
    const updates = srcNormalized.map((s, i) => ({ s, d: dstNormalized[i] }));
    updates.sort((a, b) => a.d - b.d);
    for (const update of updates) {
      remaining.splice(update.d, 0, update.s);
    }
    return this.permute(remaining);
  }
  moveaxis(source, destination) {
    return this.movedim(source, destination);
  }
  swapaxes(dim0, dim1) {
    return this.transpose(dim0, dim1);
  }
  swapdims(dim0, dim1) {
    return this.transpose(dim0, dim1);
  }
  unbind(dim = 0) {
    if (dim < 0) dim += this._shape.length;
    const size = this._shape[dim];
    const results = [];
    for (let i = 0; i < size; i++) {
      results.push(this.narrow(dim, i, 1).squeeze(dim));
    }
    return results;
  }
  narrow(dim, start, length) {
    if (dim < 0) dim += this._shape.length;
    const slices = new Array(this._shape.length).fill({ start: 0, step: 1 });
    for (let i = 0; i < this._shape.length; i++) {
      slices[i] = { start: 0, stop: this._shape[i], step: 1 };
    }
    slices[dim] = { start, stop: start + length, step: 1 };
    return this.slice(slices);
  }
  /**
   * Slice the tensor.
   * @param slices - Array of slice specifications or indices for each dimension.
   * @status implemented
   */
  slice(slices) {
    if (slices.length > this._shape.length) {
      throw new Error(`Too many slices provided: ${slices.length} > ${this._shape.length}`);
    }
    const normSlices = [];
    for (let i = 0; i < slices.length; i++) {
      const s = slices[i];
      if (typeof s === "number") {
        let idx = s;
        if (idx < 0) idx += this._shape[i];
        normSlices.push({ start: idx, stop: idx + 1, step: 1 });
      } else {
        const spec = { ...s };
        const dimSize = this._shape[i];
        spec.step = spec.step ?? 1;
        if (spec.step > 0) {
          spec.start = spec.start ?? 0;
          spec.stop = spec.stop ?? dimSize;
        } else {
          spec.start = spec.start ?? dimSize - 1;
          spec.stop = spec.stop ?? -dimSize - 1;
        }
        if (spec.start < 0) spec.start += dimSize;
        if (spec.stop < 0 && spec.stop > -dimSize - 1) spec.stop += dimSize;
        normSlices.push(spec);
      }
    }
    for (let i = slices.length; i < this._shape.length; i++) {
      normSlices.push({ start: 0, stop: this._shape[i], step: 1 });
    }
    const newShape = [];
    for (let i = 0; i < this._shape.length; i++) {
      const s = normSlices[i];
      let len = 0;
      if (s.step > 0) {
        len = Math.max(0, Math.ceil((s.stop - s.start) / s.step));
      } else {
        len = Math.max(0, Math.ceil((s.start - s.stop) / -s.step));
      }
      newShape.push(len);
    }
    const outputSize = numel(newShape);
    const device = getDevice();
    const outputBuffer = createStorageBuffer(outputSize * getDTypeBytes(this._dtype));
    const pad4 = (arr) => {
      const result = new Int32Array(4);
      for (let i = 0; i < Math.min(arr.length, 4); i++) {
        result[i] = arr[i];
      }
      return result;
    };
    const pad4u = (arr) => {
      const result = new Uint32Array(4);
      for (let i = 0; i < Math.min(arr.length, 4); i++) {
        result[i] = arr[i];
      }
      return result;
    };
    const paramsData = new ArrayBuffer(96);
    const view = new DataView(paramsData);
    const inputShape4 = pad4u([...this._shape]);
    for (let i = 0; i < 4; i++) view.setUint32(i * 4, inputShape4[i], true);
    const outputShape4 = pad4u(newShape);
    for (let i = 0; i < 4; i++) view.setUint32(16 + i * 4, outputShape4[i], true);
    const starts4 = pad4(normSlices.map((s) => s.start));
    for (let i = 0; i < 4; i++) view.setInt32(32 + i * 4, starts4[i], true);
    const steps4 = pad4(normSlices.map((s) => s.step));
    for (let i = 0; i < 4; i++) view.setInt32(48 + i * 4, steps4[i], true);
    view.setUint32(64, this._shape.length, true);
    view.setUint32(68, outputSize, true);
    const paramsBuffer = device.createBuffer({
      size: 96,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(slice_default, "slice");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({
      buffer: outputBuffer,
      shape: newShape,
      dtype: this._dtype,
      device: "webgpu",
      requires_grad: this._requires_grad
    });
  }
  _sliceMultiDim(slices) {
    const newSlices = slices.map((s) => ({ start: s.start, stop: s.stop, step: 1 }));
    return this.slice(newSlices);
  }
  transpose(dim0, dim1) {
    if (dim0 < 0) dim0 += this._shape.length;
    if (dim1 < 0) dim1 += this._shape.length;
    if (dim0 === dim1) return this;
    const dims = Array.from({ length: this._shape.length }, (_, i) => i);
    [dims[dim0], dims[dim1]] = [dims[dim1], dims[dim0]];
    return this.permute(dims);
  }
  permute(dims) {
    if (dims.length !== this._shape.length) {
      throw new Error(`permute: expected ${this._shape.length} dimensions, got ${dims.length}`);
    }
    const newDims = dims.map((d2) => d2 < 0 ? d2 + this._shape.length : d2);
    const uniqueDims = new Set(newDims);
    if (uniqueDims.size !== this._shape.length) {
      throw new Error("permute: duplicate dimensions");
    }
    const newShape = newDims.map((d2) => this._shape[d2]);
    const computeStrides = (shape) => {
      const strides = new Array(shape.length).fill(1);
      for (let i = shape.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * shape[i + 1];
      return strides;
    };
    const srcStrides = computeStrides(this._shape);
    const permutedSrcStrides = newDims.map((d2) => srcStrides[d2]);
    const dstStrides = computeStrides(newShape);
    const device = getDevice();
    const totalElements = this.numel();
    const outputBuffer = createStorageBuffer(totalElements * getDTypeBytes(this._dtype));
    const pad4 = (arr) => {
      const res = new Uint32Array(4);
      for (let i = 0; i < Math.min(arr.length, 4); i++) res[i] = arr[i];
      return res;
    };
    const paramsData = new ArrayBuffer(64);
    const view = new DataView(paramsData);
    const s4 = pad4(permutedSrcStrides);
    for (let i = 0; i < 4; i++) view.setUint32(i * 4, s4[i], true);
    const sh4 = pad4(newShape);
    for (let i = 0; i < 4; i++) view.setUint32(16 + i * 4, sh4[i], true);
    const d4 = pad4(dstStrides);
    for (let i = 0; i < 4; i++) view.setUint32(32 + i * 4, d4[i], true);
    view.setUint32(48, this._shape.length, true);
    view.setUint32(52, totalElements, true);
    const paramsBuffer = device.createBuffer({ size: 64, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(transpose_nd_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(totalElements));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const self = this;
    let grad_fn;
    if (this._requires_grad) {
      grad_fn = {
        backward(gradOutput) {
          const invDims = new Array(dims.length);
          for (let i = 0; i < dims.length; i++) invDims[newDims[i]] = i;
          self.accumulateGrad(gradOutput.permute(invDims));
        }
      };
    }
    return new _Tensor({ buffer: outputBuffer, shape: newShape, dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad, grad_fn });
  }
  t() {
    if (this._shape.length !== 2) throw new Error("t() only supports 2D tensors");
    const [M, N] = this._shape;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const dimsData = new Uint32Array([M, N]);
    const dimsBuffer = device.createBuffer({ size: 8, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(dimsBuffer, 0, dimsData);
    const pipeline = getOrCreatePipeline(transpose_default, "transpose_2d");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(M * N));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const self = this;
    let grad_fn;
    if (this._requires_grad) {
      grad_fn = {
        backward(gradOutput) {
          self.accumulateGrad(gradOutput.t());
        }
      };
    }
    return new _Tensor({ buffer: outputBuffer, shape: [N, M], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad, grad_fn });
  }
  // ============ Matrix Multiplication ============
  matmul(other) {
    return record_function("torch.matmul", () => this._matmul_impl(other));
  }
  mm(other) {
    return this.matmul(other);
  }
  _matmul_impl(other) {
    const dimSelf = this._shape.length;
    const dimOther = other._shape.length;
    if (dimSelf === 1 && dimOther === 1) {
      if (this._shape[0] !== other._shape[0]) throw new Error("Shape mismatch for dot product");
      return this.reshape([1, this._shape[0]]).matmul(other.reshape([other._shape[0], 1])).reshape([]);
    }
    if (dimSelf === 2 && dimOther === 2) {
      const [M, K1] = this._shape;
      const [K2, N] = other._shape;
      if (K1 !== K2) throw new Error("matmul: incompatible shapes");
      const device = getDevice();
      const outputBuffer = createStorageBuffer(M * N * getDTypeBytes(this._dtype));
      const dimsData = new Uint32Array([M, K1, N, 1]);
      const dimsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(dimsBuffer, 0, dimsData);
      const pipeline = getOrCreatePipeline(matmul_default, "matmul_2d");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
          { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 3, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(M * N));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      const self = this;
      const requires_grad = this._requires_grad || other._requires_grad;
      let grad_fn;
      if (requires_grad) {
        grad_fn = {
          backward(gradOutput) {
            if (self._requires_grad) {
              self.accumulateGrad(gradOutput.matmul(other.t()));
            }
            if (other._requires_grad) {
              other.accumulateGrad(self.t().matmul(gradOutput));
            }
          }
        };
      }
      return new _Tensor({ buffer: outputBuffer, shape: [M, N], dtype: this._dtype, device: "webgpu", requires_grad, grad_fn });
    }
    if (dimSelf === 2 && dimOther === 1) {
      const [M, K1] = this._shape;
      const [K2] = other._shape;
      if (K1 !== K2) throw new Error("matmul: incompatible shapes");
      return this.matmul(other.unsqueeze(1)).squeeze(1);
    }
    if (dimSelf === 3 && dimOther === 3) {
      const [B1, M, K1] = this._shape;
      const [B2, K2, N] = other._shape;
      if (B1 !== B2 || K1 !== K2) throw new Error("matmul: incompatible shapes");
      const device = getDevice();
      const outputBuffer = createStorageBuffer(B1 * M * N * getDTypeBytes(this._dtype));
      const dimsData = new Uint32Array([M, K1, N, B1]);
      const dimsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(dimsBuffer, 0, dimsData);
      const pipeline = getOrCreatePipeline(matmul_default, "matmul_3d");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
          { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
          { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 3, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(B1 * M * N));
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      return new _Tensor({ buffer: outputBuffer, shape: [B1, M, N], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad || other._requires_grad });
    }
    throw new Error("Unsupported matmul dimensions");
  }
  addmm(mat1, mat2, beta = 1, alpha = 1) {
    const product = mat1.mm(mat2);
    return this.mul(beta).add(product.mul(alpha));
  }
  addmv(mat, vec, beta = 1, alpha = 1) {
    const product = mat.matmul(vec);
    return this.mul(beta).add(product.mul(alpha));
  }
  outer(vec2) {
    return this.unsqueeze(1).matmul(vec2.unsqueeze(0));
  }
  addr(vec1, vec2, beta = 1, alpha = 1) {
    const product = vec1.outer(vec2);
    return this.mul(beta).add(product.mul(alpha));
  }
  baddbmm(batch1, batch2, beta = 1, alpha = 1) {
    const product = batch1.matmul(batch2);
    return this.mul(beta).add(product.mul(alpha));
  }
  addbmm(batch1, batch2, beta = 1, alpha = 1) {
    const product = batch1.matmul(batch2);
    const summed = product.sum(0);
    return this.mul(beta).add(summed.mul(alpha));
  }
  mv(vec) {
    return this.matmul(vec);
  }
  bmm(mat2) {
    return this.matmul(mat2);
  }
  dot(other) {
    return this.mul(other).sum();
  }
  vdot(other) {
    return this.dot(other);
  }
  // ============ Reductions ============
  sum(dim, keepdim = false) {
    if (dim === void 0) {
      const reduceResult2 = this._reduce("sum");
      const self2 = this;
      if (this._requires_grad) {
        const grad_fn = {
          backward(gradOutput) {
            self2._expandGradAndAccumulate(gradOutput);
          }
        };
        return new _Tensor({ ...reduceResult2._getState(), requires_grad: true, grad_fn });
      }
      return reduceResult2;
    }
    if (Array.isArray(dim) || typeof dim === "object") {
      const dims = Array.from(dim).sort((a, b) => b - a);
      let res = this;
      for (const d2 of dims) {
        const nextRes = res.sum(d2, keepdim);
        if (res !== this) res.destroy();
        res = nextRes;
      }
      return res;
    }
    const reduceResult = this._reduceDim("sum", dim, keepdim);
    const self = this;
    if (this._requires_grad) {
      const grad_fn = {
        backward(gradOutput) {
          let g = gradOutput;
          if (!keepdim) g = g.unsqueeze(dim);
          const expandedGrad = ones_like(self).mul(g);
          self.accumulateGrad(expandedGrad);
        }
      };
      return new _Tensor({ ...reduceResult._getState(), requires_grad: true, grad_fn });
    }
    return reduceResult;
  }
  mean(dim, keepdim = false) {
    if (dim === void 0) {
      const s = this.sum();
      const n2 = this.numel();
      const res = s.div(n2);
      const self2 = this;
      if (this._requires_grad) {
        const grad_fn = {
          backward(gradOutput) {
            self2._expandGradAndAccumulate(gradOutput.div(n2));
          }
        };
        return new _Tensor({ ...res._getState(), requires_grad: true, grad_fn });
      }
      return res;
    }
    if (Array.isArray(dim) || typeof dim === "object") {
      const dims = Array.from(dim).sort((a, b) => b - a);
      let res = this;
      for (const d2 of dims) {
        const nextRes = res.mean(d2, keepdim);
        if (res !== this) res.destroy();
        res = nextRes;
      }
      return res;
    }
    const reduceResult = this._reduceDim("mean", dim, keepdim);
    const n = this._shape[dim];
    const self = this;
    if (this._requires_grad) {
      const grad_fn = {
        backward(gradOutput) {
          let g = gradOutput.div(n);
          if (!keepdim) g = g.unsqueeze(dim);
          const expandedGrad = ones_like(self).mul(g);
          self.accumulateGrad(expandedGrad);
        }
      };
      return new _Tensor({ ...reduceResult._getState(), requires_grad: true, grad_fn });
    }
    return reduceResult;
  }
  _expandGradAndAccumulate(scalarGrad) {
    const expandedGrad = ones_like(this).mul(scalarGrad);
    this.accumulateGrad(expandedGrad);
  }
  _reduceDim(opName, dim, keepdim) {
    if (dim < 0) dim += this._shape.length;
    const permutation = Array.from({ length: this._shape.length }, (_, i) => i);
    permutation.splice(dim, 1);
    permutation.push(dim);
    const transposed = this.permute(permutation);
    const reduceSize = this._shape[dim];
    const batchSize = transposed.numel() / reduceSize;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(batchSize * getDTypeBytes(this._dtype));
    const opMap = {
      "sum": 0,
      "mean": 1,
      "max_reduce": 2,
      "min_reduce": 3,
      "prod": 4,
      "any": 5,
      "all": 6
    };
    const op = opMap[opName] ?? 0;
    const paramsData = new Uint32Array([batchSize, reduceSize, op, 0]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(reduce_dim_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: transposed.buffer, offset: 0, size: transposed.buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(batchSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    let finalShape = this._shape.filter((_, i) => i !== dim);
    if (keepdim) {
      finalShape.splice(dim, 0, 1);
    }
    const result = new _Tensor({
      buffer: outputBuffer,
      shape: finalShape,
      dtype: this._dtype,
      device: "webgpu",
      requires_grad: this._requires_grad
    });
    transposed.destroy();
    return result;
  }
  max(dim, keepdim = false) {
    if (dim === void 0) return this._reduce("max_reduce");
    if (Array.isArray(dim) || typeof dim === "object") {
      const dims = Array.from(dim).sort((a, b) => b - a);
      let res = this;
      for (const d2 of dims) {
        const nextRes = res.max(d2, keepdim);
        if (res !== this) res.destroy();
        res = nextRes;
      }
      return res;
    }
    return this._reduceDim("max_reduce", dim, keepdim);
  }
  min(dim, keepdim = false) {
    if (dim === void 0) return this._reduce("min_reduce");
    if (Array.isArray(dim) || typeof dim === "object") {
      const dims = Array.from(dim).sort((a, b) => b - a);
      let res = this;
      for (const d2 of dims) {
        const nextRes = res.min(d2, keepdim);
        if (res !== this) res.destroy();
        res = nextRes;
      }
      return res;
    }
    return this._reduceDim("min_reduce", dim, keepdim);
  }
  any(dim, keepdim = false) {
    if (dim === void 0) return this._reduce("any");
    if (Array.isArray(dim) || typeof dim === "object") {
      const dims = Array.from(dim).sort((a, b) => b - a);
      let res = this;
      for (const d2 of dims) {
        const nextRes = res.any(d2, keepdim);
        if (res !== this) res.destroy();
        res = nextRes;
      }
      return res;
    }
    return this._reduceDim("any", dim, keepdim);
  }
  all(dim, keepdim = false) {
    if (dim === void 0) return this._reduce("all");
    if (Array.isArray(dim) || typeof dim === "object") {
      const dims = Array.from(dim).sort((a, b) => b - a);
      let res = this;
      for (const d2 of dims) {
        const nextRes = res.all(d2, keepdim);
        if (res !== this) res.destroy();
        res = nextRes;
      }
      return res;
    }
    return this._reduceDim("all", dim, keepdim);
  }
  prod(dim, keepdim = false) {
    if (dim === void 0) return this._reduce("prod");
    if (Array.isArray(dim) || typeof dim === "object") {
      const dims = Array.from(dim).sort((a, b) => b - a);
      let res = this;
      for (const d2 of dims) {
        const nextRes = res.prod(d2, keepdim);
        if (res !== this) res.destroy();
        res = nextRes;
      }
      return res;
    }
    return this._reduceDim("prod", dim, keepdim);
  }
  amax(dim, keepdim = false) {
    return this.max(dim, keepdim);
  }
  amin(dim, keepdim = false) {
    return this.min(dim, keepdim);
  }
  var(dim, correction = 1, keepdim = false) {
    let mean = this.mean(dim, true);
    let diff = this.sub(mean);
    let sq_diff = diff.square();
    let sum_sq = sq_diff.sum(dim, keepdim);
    let n;
    if (dim === void 0) {
      n = this.numel();
    } else {
      const d2 = dim < 0 ? dim + this._shape.length : dim;
      n = this._shape[d2];
    }
    return sum_sq.div(n - correction);
  }
  std(dim, correction = 1, keepdim = false) {
    return this.var(dim, correction, keepdim).sqrt();
  }
  /**
   * Returns the indices of maximum values along a dimension.
   * @pytorch tensor.argmax()
   */
  argmax(dim, keepdim = false) {
    if (dim === void 0) {
      return this.flatten().argmax(0, keepdim);
    }
    return this._reduceArg("argmax", dim, keepdim);
  }
  /**
   * Returns the indices of minimum values along a dimension.
   * @pytorch tensor.argmin()
   */
  argmin(dim, keepdim = false) {
    if (dim === void 0) {
      return this.flatten().argmin(0, keepdim);
    }
    return this._reduceArg("argmin", dim, keepdim);
  }
  _reduceArg(op, dim, keepdim) {
    if (dim < 0) dim += this._shape.length;
    const permutation = Array.from({ length: this._shape.length }, (_, i) => i);
    permutation.splice(dim, 1);
    permutation.push(dim);
    const transposed = this.permute(permutation);
    const reduceSize = this._shape[dim];
    const batchSize = transposed.numel() / reduceSize;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(batchSize * 4);
    const dimsData = new Uint32Array([batchSize, reduceSize]);
    const dimsBuffer = device.createBuffer({
      size: 8,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(dimsBuffer, 0, dimsData);
    const shader = op === "argmax" ? argmax_default : argmin_default;
    const pipeline = getOrCreatePipeline(shader, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: transposed.buffer, offset: 0, size: transposed.buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(batchSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    let finalShape = this._shape.filter((_, i) => i !== dim);
    if (keepdim) {
      finalShape.splice(dim, 0, 1);
    }
    const result = new _Tensor({
      buffer: outputBuffer,
      shape: finalShape,
      dtype: "int32",
      device: "webgpu",
      requires_grad: false
    });
    transposed.destroy();
    return result;
  }
  // ============ Data Access ============
  async toArray() {
    const data = await readBuffer(this._buffer, this._dtype, this.numel());
    return Array.from(data);
  }
  async toNestedArray() {
    const flat = await this.toArray();
    return this._nestArray(flat, [...this._shape]);
  }
  async tolist() {
    return this.toNestedArray();
  }
  async item() {
    if (this.numel() !== 1) throw new Error("item() only works for scalar tensors");
    const data = await readBuffer(this._buffer, this._dtype, 1);
    return data[0];
  }
  // ============ Autograd ============
  async backward(gradient) {
    if (!this._requires_grad && !this._grad_fn) return;
    const grad = gradient || ones([], { dtype: this._dtype });
    this.accumulateGrad(grad);
  }
  accumulateGrad(grad) {
    if (this._grad_fn) {
      this._grad_fn.backward(grad);
    }
    if (this._is_leaf && this._requires_grad) {
      if (!this._grad) {
        this._grad = grad.clone();
      } else {
        const oldGrad = this._grad;
        this._grad = oldGrad.add(grad);
        oldGrad.destroy();
      }
    }
  }
  // ============ Internal Helpers ============
  _getState() {
    return { buffer: this._buffer, shape: [...this._shape], dtype: this._dtype, device: this._device, requires_grad: this._requires_grad, grad_fn: this._grad_fn ?? void 0 };
  }
  _nestArray(flat, shape, offset = 0) {
    if (shape.length === 0) return flat[offset];
    if (shape.length === 1) return flat.slice(offset, offset + shape[0]);
    const result = [];
    const stride = shape.slice(1).reduce((a, b) => a * b, 1);
    for (let i = 0; i < shape[0]; i++) result.push(this._nestArray(flat, shape.slice(1), offset + i * stride));
    return result;
  }
  _unaryOp(op) {
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const pipeline = getOrCreatePipeline(unary_default, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const self = this;
    let grad_fn;
    if (this._requires_grad) {
      grad_fn = {
        backward(gradOutput) {
          let grad_self;
          if (op === "neg") {
            grad_self = gradOutput.neg();
          } else if (op === "abs_op") {
            grad_self = gradOutput.mul(self.div(self.abs().add(1e-7)));
          } else {
            grad_self = gradOutput;
          }
          self.accumulateGrad(grad_self);
        }
      };
    }
    return new _Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad, grad_fn });
  }
  _binaryOp(op, other) {
    if (needsBroadcast(this._shape, other._shape)) return this._broadcastBinaryOp(op, other);
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const pipeline = getOrCreatePipeline(elementwise_default, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const self = this;
    const requires_grad = this._requires_grad || other._requires_grad;
    let grad_fn;
    if (requires_grad) {
      grad_fn = {
        backward(gradOutput) {
          if (self._requires_grad) {
            let grad_self;
            if (op === "add" || op === "sub") {
              grad_self = gradOutput;
            } else if (op === "mul") {
              grad_self = gradOutput.mul(other);
            } else {
              grad_self = gradOutput;
            }
            self.accumulateGrad(grad_self);
          }
          if (other._requires_grad) {
            let grad_other;
            if (op === "add") {
              grad_other = gradOutput;
            } else if (op === "sub") {
              grad_other = gradOutput.neg();
            } else if (op === "mul") {
              grad_other = gradOutput.mul(self);
            } else {
              grad_other = gradOutput;
            }
            other.accumulateGrad(grad_other);
          }
        }
      };
    }
    return new _Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: "webgpu", requires_grad, grad_fn });
  }
  _broadcastBinaryOp(op, other) {
    const device = getDevice();
    const outputShape = broadcastShapes(this._shape, other._shape);
    const outputSize = numel(outputShape);
    const computeStrides = (shape, outShape) => {
      const strides = [0, 0, 0, 0];
      const ndim = outShape.length;
      let stride = 1;
      for (let i = shape.length - 1; i >= 0; i--) {
        const dimIdx = 4 - ndim + (i + (ndim - shape.length));
        strides[dimIdx] = shape[i] === 1 ? 0 : stride;
        stride *= shape[i];
      }
      return strides;
    };
    const aStrides = computeStrides(this._shape, outputShape);
    const bStrides = computeStrides(other._shape, outputShape);
    const paddedOut = [1, 1, 1, 1];
    for (let i = 0; i < outputShape.length; i++) paddedOut[4 - outputShape.length + i] = outputShape[i];
    const outputBuffer = createStorageBuffer(outputSize * getDTypeBytes(this._dtype));
    const paramsData = new Uint32Array([...paddedOut, ...aStrides, ...bStrides, outputShape.length, outputSize, 0, 0]);
    const paramsBuffer = device.createBuffer({ size: 64, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const entryPoints = {
      "add": "broadcast_add",
      "sub": "broadcast_sub",
      "mul": "broadcast_mul",
      "div_op": "broadcast_div",
      "pow_tensor": "broadcast_pow"
    };
    const pipeline = getOrCreatePipeline(broadcast_default, entryPoints[op] || "broadcast_add");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 3, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const self = this;
    const requires_grad = this._requires_grad || other._requires_grad;
    let grad_fn;
    if (requires_grad) {
      grad_fn = {
        backward(gradOutput) {
          if (self._requires_grad) {
            let grad_self;
            if (op === "add" || op === "sub") {
              grad_self = self._reduceBroadcastGrad(gradOutput);
            } else if (op === "mul") {
              grad_self = self._reduceBroadcastGrad(gradOutput.mul(other));
            } else if (op === "pow_tensor") {
              grad_self = self._reduceBroadcastGrad(gradOutput.mul(other).mul(self.pow(other.sub(1))));
            } else {
              grad_self = self._reduceBroadcastGrad(gradOutput);
            }
            self.accumulateGrad(grad_self);
          }
          if (other._requires_grad) {
            let grad_other;
            if (op === "add") {
              grad_other = other._reduceBroadcastGrad(gradOutput);
            } else if (op === "sub") {
              grad_other = other._reduceBroadcastGrad(gradOutput.neg());
            } else if (op === "mul") {
              grad_other = other._reduceBroadcastGrad(gradOutput.mul(self));
            } else if (op === "pow_tensor") {
              grad_other = other._reduceBroadcastGrad(gradOutput.mul(self.pow(other)).mul(self.abs().log()));
            } else {
              grad_other = other._reduceBroadcastGrad(gradOutput);
            }
            other.accumulateGrad(grad_other);
          }
        }
      };
    }
    return new _Tensor({ buffer: outputBuffer, shape: outputShape, dtype: this._dtype, device: "webgpu", requires_grad, grad_fn });
  }
  _reduceBroadcastGrad(grad) {
    if (!needsBroadcast(this._shape, grad._shape)) return grad;
    if (this._shape.length === 1 && grad._shape.length === 2) return this._sumAlongDim0(grad);
    return grad;
  }
  _sumAlongDim0(input) {
    if (input.shape.length !== 2) throw new Error("_sumAlongDim0 only supports 2D tensors");
    const [batchSize, features] = input.shape;
    const device = getDevice();
    const outputBuffer = createStorageBuffer(features * getDTypeBytes(input.dtype));
    const paramsData = new Uint32Array([batchSize, features]);
    const paramsBuffer = device.createBuffer({ size: 8, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(reduce_broadcast_grad_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer, offset: 0, size: input.buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(features));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({ buffer: outputBuffer, shape: [features], dtype: input.dtype, device: "webgpu", requires_grad: false });
  }
  _scalarOp(op, scalar) {
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const paramsData = new ArrayBuffer(8);
    new Float32Array(paramsData, 0, 1)[0] = scalar;
    new Uint32Array(paramsData, 4, 1)[0] = this.numel();
    const paramsBuffer = device.createBuffer({ size: 8, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(scalar_default, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const self = this;
    let grad_fn;
    if (this._requires_grad) {
      grad_fn = {
        backward(gradOutput) {
          let grad_self;
          if (op === "add_scalar" || op === "sub_scalar") {
            grad_self = gradOutput;
          } else if (op === "mul_scalar") {
            grad_self = gradOutput.mul(scalar);
          } else if (op === "div_scalar") {
            grad_self = gradOutput.div(scalar);
          } else {
            grad_self = gradOutput;
          }
          self.accumulateGrad(grad_self);
        }
      };
    }
    return new _Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad, grad_fn });
  }
  _reduce(op) {
    const capabilities2 = getCapabilities();
    if (capabilities2.workgroupSharedMemory && !this._isChromeMac()) return this._reduceWithSharedMemory(op);
    return this._reduceSimple(op);
  }
  _isChromeMac() {
    if (typeof navigator === "undefined") return false;
    return navigator.userAgent.includes("Macintosh") && navigator.userAgent.includes("Chrome");
  }
  _reduceWithSharedMemory(op) {
    const device = getDevice();
    const n = this.numel();
    let currentBuffer = this._buffer;
    let currentLength = n;
    const shaderMap = {
      "sum": reduce_sum_default,
      "max_reduce": reduce_max_default,
      "min_reduce": reduce_min_default,
      "any": reduce_any_default,
      "all": reduce_all_default,
      "prod": reduce_prod_default
    };
    const shader = shaderMap[op] || reduce_sum_default;
    while (currentLength > 1) {
      const numWorkgroups = Math.ceil(currentLength / 256);
      const outputBuffer = createStorageBuffer(numWorkgroups * getDTypeBytes(this._dtype));
      const paramsData = new Uint32Array([currentLength, 0, 0, 0]);
      const paramsBuffer = device.createBuffer({ size: 16, usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const pipeline = getOrCreatePipeline(shader, "main");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: currentBuffer, offset: 0, size: currentBuffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
          { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(numWorkgroups, 1, 1);
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      currentBuffer = outputBuffer;
      currentLength = numWorkgroups;
    }
    return new _Tensor({ buffer: currentBuffer, shape: [], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad });
  }
  _reduceSimple(op) {
    const device = getDevice();
    const n = this.numel();
    const shaderMap = {
      "sum": reduce_simple_sum_default,
      "max_reduce": reduce_simple_max_default,
      "min_reduce": reduce_simple_min_default,
      "any": reduce_simple_any_default,
      "all": reduce_simple_all_default,
      "prod": reduce_simple_prod_default
    };
    const shader = shaderMap[op] || reduce_simple_sum_default;
    const inputCopy = device.createBuffer({ size: this._buffer.size, usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(this._buffer, 0, inputCopy, 0, this._buffer.size);
    device.queue.submit([copyEncoder.finish()]);
    let currentBuffer = inputCopy;
    let currentLength = n;
    while (currentLength > 1) {
      const numWorkgroups = Math.ceil(currentLength / 256);
      const outputBuffer = device.createBuffer({ size: Math.max(4, numWorkgroups * getDTypeBytes(this._dtype)), usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST });
      const pipeline = getOrCreatePipeline(shader, "main");
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: currentBuffer, offset: 0, size: currentBuffer.size } },
          { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(numWorkgroups, 1, 1);
      passEncoder.end();
      device.queue.submit([commandEncoder.finish()]);
      currentBuffer = outputBuffer;
      currentLength = numWorkgroups;
    }
    return new _Tensor({ buffer: currentBuffer, shape: [], dtype: this._dtype, device: "webgpu", requires_grad: this._requires_grad });
  }
  _compareOp(op, other) {
    if (needsBroadcast(this._shape, other._shape)) throw new Error("Broadcasting not yet implemented for comparison");
    const device = getDevice();
    const outputBuffer = createStorageBuffer(this.numel() * getDTypeBytes(this._dtype));
    const pipeline = getOrCreatePipeline(compare_default, op);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._buffer, offset: 0, size: this._buffer.size } },
        { binding: 1, resource: { buffer: other._buffer, offset: 0, size: other._buffer.size } },
        { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(this.numel()));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new _Tensor({ buffer: outputBuffer, shape: [...this._shape], dtype: this._dtype, device: "webgpu", requires_grad: false });
  }
};

export {
  __export,
  getDTypeBytes,
  getTypedArrayConstructor,
  BufferUsage,
  log_softmax_default,
  nll_loss_default,
  nll_loss_backward_default,
  log_softmax_backward_default,
  embedding_default,
  layernorm_default,
  cholesky_default,
  triangular_solve_default,
  lu_default,
  getCapabilities,
  initWebGPU,
  getDevice,
  bufferPool,
  createStorageBuffer,
  readBuffer,
  getOrCreatePipeline,
  calculateWorkgroups,
  syncDevice,
  profiler_exports,
  Tensor,
  tensor,
  zeros,
  ones,
  full,
  zeros_like,
  ones_like,
  full_like,
  empty_like,
  rand,
  randn,
  eye,
  arange,
  linspace,
  logspace,
  tril,
  cat,
  stack,
  vstack,
  row_stack,
  hstack,
  dstack,
  column_stack,
  manual_seed
};
//# sourceMappingURL=chunk-HK6J3H55.js.map