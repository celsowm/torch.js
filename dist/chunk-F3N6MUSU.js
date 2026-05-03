import {
  BufferUsage,
  Tensor,
  __export,
  arange,
  bufferPool,
  calculateWorkgroups,
  cat,
  cholesky_default,
  column_stack,
  createStorageBuffer,
  dstack,
  embedding_default,
  empty_like,
  eye,
  full,
  full_like,
  getDTypeBytes,
  getDevice,
  getOrCreatePipeline,
  hstack,
  initWebGPU,
  layernorm_default,
  linspace,
  log_softmax_backward_default,
  log_softmax_default,
  logspace,
  lu_default,
  manual_seed,
  nll_loss_backward_default,
  nll_loss_default,
  ones,
  ones_like,
  profiler_exports,
  rand,
  randn,
  readBuffer,
  row_stack,
  stack,
  syncDevice,
  tensor,
  triangular_solve_default,
  tril,
  vstack,
  zeros,
  zeros_like
} from "./chunk-HK6J3H55.js";

// src/core.ts
var core_exports = {};
__export(core_exports, {
  DEBUG: () => DEBUG,
  DEBUG_ASYNC: () => DEBUG_ASYNC,
  Tensor: () => Tensor,
  _internals: () => _internals,
  abs: () => abs,
  absolute: () => absolute,
  acos: () => acos,
  acosh: () => acosh,
  add: () => add,
  addbmm: () => addbmm,
  addmm: () => addmm,
  addmv: () => addmv,
  addr: () => addr,
  all: () => all,
  allclose: () => allclose,
  amax: () => amax,
  amin: () => amin,
  any: () => any,
  arange: () => arange,
  arccos: () => arccos,
  arccosh: () => arccosh,
  arcsin: () => arcsin,
  arcsinh: () => arcsinh,
  arctan: () => arctan,
  arctan2: () => arctan2,
  arctanh: () => arctanh,
  argmax: () => argmax,
  argmin: () => argmin,
  asin: () => asin,
  asinh: () => asinh,
  assert: () => assert,
  atan: () => atan,
  atan2: () => atan2,
  atanh: () => atanh,
  atleast_1d: () => atleast_1d,
  atleast_2d: () => atleast_2d,
  atleast_3d: () => atleast_3d,
  baddbmm: () => baddbmm,
  bitwise_and: () => bitwise_and,
  bitwise_or: () => bitwise_or,
  bitwise_xor: () => bitwise_xor,
  bmm: () => bmm,
  broadcast_to: () => broadcast_to,
  cat: () => cat,
  ceil: () => ceil,
  chain_matmul: () => chain_matmul,
  chunk: () => chunk,
  clamp: () => clamp,
  clip: () => clip,
  column_stack: () => column_stack,
  cos: () => cos,
  cosh: () => cosh,
  createTorch: () => createTorch,
  cumprod: () => cumprod,
  cumsum: () => cumsum,
  cumulative_trapezoid: () => cumulative_trapezoid,
  diag: () => diag,
  div: () => div,
  divide: () => divide,
  dot: () => dot,
  dstack: () => dstack,
  empty_like: () => empty_like,
  eq: () => eq,
  equal: () => equal,
  error: () => error,
  exp: () => exp,
  exp2: () => exp2,
  eye: () => eye,
  fix: () => fix,
  flatten: () => flatten,
  flip: () => flip,
  fliplr: () => fliplr,
  flipud: () => flipud,
  floor: () => floor,
  fmax: () => fmax,
  fmin: () => fmin,
  frac: () => frac,
  full: () => full,
  full_like: () => full_like,
  ge: () => ge,
  ger: () => ger,
  greater: () => greater,
  greater_equal: () => greater_equal,
  gt: () => gt,
  heaviside: () => heaviside,
  hstack: () => hstack,
  hypot: () => hypot,
  index_select: () => index_select,
  init: () => initWebGPU,
  inner: () => inner,
  isclose: () => isclose,
  isfinite: () => isfinite,
  isinf: () => isinf,
  isnan: () => isnan,
  isneginf: () => isneginf,
  isposinf: () => isposinf,
  le: () => le,
  less: () => less,
  less_equal: () => less_equal,
  linalg: () => linalg_exports,
  linspace: () => linspace,
  log: () => log3,
  log10: () => log10,
  log1p: () => log1p,
  log2: () => log2,
  logaddexp: () => logaddexp,
  logspace: () => logspace,
  lt: () => lt,
  manual_seed: () => manual_seed,
  matmul: () => matmul,
  maximum: () => maximum,
  minimum: () => minimum,
  mm: () => mm,
  moveaxis: () => moveaxis,
  movedim: () => movedim,
  mul: () => mul,
  multiply: () => multiply,
  mv: () => mv,
  narrow: () => narrow,
  ne: () => ne,
  neg: () => neg,
  negative: () => negative,
  nn: () => nn_exports,
  not_equal: () => not_equal,
  ones: () => ones,
  ones_like: () => ones_like,
  optim: () => optim_exports,
  outer: () => outer,
  permute: () => permute,
  pow: () => pow,
  prod: () => prod,
  profiler: () => profiler_exports,
  rand: () => rand,
  randn: () => randn,
  reciprocal: () => reciprocal,
  relu: () => relu2,
  round: () => round,
  row_stack: () => row_stack,
  rsqrt: () => rsqrt,
  select: () => select,
  sigmoid: () => sigmoid2,
  sin: () => sin,
  sinh: () => sinh,
  sqrt: () => sqrt,
  square: () => square,
  squeeze: () => squeeze,
  stack: () => stack,
  sub: () => sub,
  subtract: () => subtract,
  swapaxes: () => swapaxes,
  swapdims: () => swapdims,
  syncDevice: () => syncDevice,
  take: () => take,
  tan: () => tan,
  tanh: () => tanh2,
  tensor: () => tensor,
  tile: () => tile,
  trapezoid: () => trapezoid,
  trapz: () => trapz,
  tril: () => tril,
  triu: () => triu,
  trunc: () => trunc,
  unbind: () => unbind,
  unsqueeze: () => unsqueeze,
  utils: () => utils,
  vdot: () => vdot,
  vstack: () => vstack,
  warn: () => warn,
  webgpu: () => webgpu_exports,
  where: () => where,
  zeros: () => zeros,
  zeros_like: () => zeros_like
});

// src/nn/index.ts
var nn_exports = {};
__export(nn_exports, {
  Dropout: () => Dropout,
  Dropout2d: () => Dropout2d,
  Embedding: () => Embedding,
  F: () => functional_exports,
  Flatten: () => Flatten,
  GELU: () => GELU,
  LayerNorm: () => LayerNorm,
  Linear: () => Linear,
  LogSoftmax: () => LogSoftmax,
  Module: () => Module,
  ModuleDict: () => ModuleDict,
  ModuleList: () => ModuleList,
  Parameter: () => Parameter,
  ReLU: () => ReLU,
  Sequential: () => Sequential,
  Sigmoid: () => Sigmoid,
  Softmax: () => Softmax,
  Tanh: () => Tanh,
  cross_entropy: () => cross_entropy,
  dropout: () => dropout,
  functional: () => functional_exports,
  gelu: () => gelu,
  log_softmax: () => log_softmax,
  nll_loss: () => nll_loss,
  relu: () => relu,
  softmax: () => softmax
});

// src/nn/module.ts
var Module = class {
  _parameters = /* @__PURE__ */ new Map();
  _modules = /* @__PURE__ */ new Map();
  _buffers = /* @__PURE__ */ new Map();
  _training = true;
  /**
   * Whether the module is in training mode.
   */
  get training() {
    return this._training;
  }
  /**
   * Register a parameter with the module.
   */
  register_parameter(name, param) {
    if (param === null) {
      this._parameters.delete(name);
    } else {
      this._parameters.set(name, param);
    }
  }
  /**
   * Register a submodule with the module.
   */
  add_module(name, module) {
    if (module === null) {
      this._modules.delete(name);
    } else {
      this._modules.set(name, module);
    }
  }
  /**
   * Register a buffer (non-parameter tensor) with the module.
   * Buffers are not included in parameters() but are part of the module state.
   * @pytorch module.register_buffer()
   */
  register_buffer(name, tensor2) {
    if (tensor2 === null) {
      this._buffers.delete(name);
    } else {
      this._buffers.set(name, tensor2);
      this[name] = tensor2;
    }
  }
  /**
   * Get a buffer by name.
   */
  get_buffer(name) {
    return this._buffers.get(name);
  }
  /**
   * Returns an iterator over module buffers.
   * @pytorch module.buffers()
   */
  *buffers(recurse = true) {
    for (const buffer of this._buffers.values()) {
      yield buffer;
    }
    if (recurse) {
      for (const module of this._modules.values()) {
        yield* module.buffers(recurse);
      }
    }
  }
  /**
   * Returns an iterator over named buffers.
   * @pytorch module.named_buffers()
   */
  *named_buffers(prefix = "", recurse = true) {
    for (const [name, buffer] of this._buffers.entries()) {
      yield [prefix ? `${prefix}.${name}` : name, buffer];
    }
    if (recurse) {
      for (const [name, module] of this._modules.entries()) {
        yield* module.named_buffers(prefix ? `${prefix}.${name}` : name, recurse);
      }
    }
  }
  /**
   * Get a parameter by name.
   */
  get_parameter(name) {
    return this._parameters.get(name);
  }
  /**
   * Get a submodule by name.
   */
  get_submodule(name) {
    return this._modules.get(name);
  }
  /**
   * Returns an iterator over module parameters.
   * @pytorch module.parameters()
   */
  *parameters(recurse = true) {
    for (const param of this._parameters.values()) {
      yield param;
    }
    if (recurse) {
      for (const module of this._modules.values()) {
        yield* module.parameters(recurse);
      }
    }
  }
  /**
   * Returns an iterator over named parameters.
   * @pytorch module.named_parameters()
   */
  *named_parameters(prefix = "", recurse = true) {
    for (const [name, param] of this._parameters.entries()) {
      yield [prefix ? `${prefix}.${name}` : name, param];
    }
    if (recurse) {
      for (const [name, module] of this._modules.entries()) {
        yield* module.named_parameters(prefix ? `${prefix}.${name}` : name, recurse);
      }
    }
  }
  /**
   * Returns an iterator over child modules.
   * @pytorch module.children()
   */
  *children() {
    yield* this._modules.values();
  }
  /**
   * Returns an iterator over all modules (including self).
   * @pytorch module.modules()
   */
  *modules() {
    yield this;
    for (const module of this._modules.values()) {
      yield* module.modules();
    }
  }
  /**
   * Set the module in training mode.
   * @pytorch module.train()
   */
  train(mode = true) {
    this._training = mode;
    for (const module of this._modules.values()) {
      module.train(mode);
    }
    return this;
  }
  /**
   * Set the module in evaluation mode.
   * @pytorch module.eval()
   */
  eval() {
    return this.train(false);
  }
  /**
   * Zero out all gradients.
   * @pytorch module.zero_grad()
   */
  zero_grad() {
    for (const param of this.parameters()) {
      param.grad = null;
    }
  }
  /**
   * Returns a dictionary containing a whole state of the module.
   * @pytorch module.state_dict()
   */
  state_dict(destination = {}, prefix = "", recurse = true) {
    for (const [name, param] of this.named_parameters(prefix, recurse)) {
      destination[name] = param;
    }
    for (const [name, buffer] of this.named_buffers(prefix, recurse)) {
      destination[name] = buffer;
    }
    return destination;
  }
  /**
   * Copies parameters and buffers from state_dict into this module and its descendants.
   * @pytorch module.load_state_dict()
   */
  load_state_dict(state_dict) {
    const own_state = this.state_dict();
    for (const name in state_dict) {
      if (Object.prototype.hasOwnProperty.call(own_state, name)) {
        const param = own_state[name];
        const saved = state_dict[name];
        if (param.numel() !== saved.numel()) {
          throw new Error(`Shape mismatch for key ${name}: expected ${param.shape}, got ${saved.shape}`);
        }
        param.copy_(saved);
      }
    }
  }
  /**
   * Call the module (alias for forward).
   */
  call(...inputs) {
    return this.forward(...inputs);
  }
};

// src/nn/parameter.ts
var Parameter = class extends Tensor {
  /**
   * Create a Parameter from data or another tensor.
   */
  static create(data, options = {}) {
    const opts = { requires_grad: true, ...options };
    if (data instanceof Tensor) {
      data.requires_grad_(opts.requires_grad ?? true);
      return data;
    }
    const t = tensor(data, opts);
    return t;
  }
};

// src/nn/linear.ts
var Linear = class extends Module {
  weight;
  bias;
  in_features;
  out_features;
  constructor(in_features, out_features, bias = true) {
    super();
    this.in_features = in_features;
    this.out_features = out_features;
    const k = Math.sqrt(1 / in_features);
    const weightData = randn([out_features, in_features]).mul(k);
    this.weight = Parameter.create(weightData);
    this.register_parameter("weight", this.weight);
    if (bias) {
      const biasData = zeros([out_features]);
      this.bias = Parameter.create(biasData);
      this.register_parameter("bias", this.bias);
    } else {
      this.bias = null;
    }
  }
  forward(input) {
    if (input.shape.length === 1) {
      input = input.unsqueeze(0);
    }
    let output = input.matmul(this.weight.t());
    if (this.bias) {
      output = output.add(this.bias);
    }
    return output;
  }
};

// src/nn/sequential.ts
var Sequential = class extends Module {
  moduleList;
  constructor(...modules) {
    super();
    this.moduleList = modules;
    modules.forEach((mod, idx) => {
      this.add_module(String(idx), mod);
    });
  }
  forward(input) {
    let output = input;
    for (const module of this.moduleList) {
      output = module.forward(output);
    }
    return output;
  }
  /**
   * Get a module by index.
   */
  get(index) {
    return this.moduleList[index];
  }
  /**
   * Get the number of modules.
   */
  get length() {
    return this.moduleList.length;
  }
};

// src/nn/flatten.ts
var Flatten = class extends Module {
  start_dim;
  end_dim;
  constructor(start_dim = 1, end_dim = -1) {
    super();
    this.start_dim = start_dim;
    this.end_dim = end_dim;
  }
  forward(input) {
    return input.flatten(this.start_dim, this.end_dim);
  }
};

// src/nn/embedding.ts
var Embedding = class extends Module {
  num_embeddings;
  embedding_dim;
  weight;
  constructor(num_embeddings, embedding_dim) {
    super();
    this.num_embeddings = num_embeddings;
    this.embedding_dim = embedding_dim;
    const weight = randn([num_embeddings, embedding_dim], { requires_grad: true });
    this.weight = Parameter.create(weight);
    this.register_parameter("weight", this.weight);
  }
  /**
   * Look up embeddings for the given indices.
   * @param input - Tensor of indices with shape [...], dtype int32
   * @returns Tensor of embeddings with shape [..., embedding_dim]
   */
  forward(input) {
    const device = getDevice();
    const inputShape = input.shape;
    const numIndices = input.numel();
    const outputShape = [...inputShape, this.embedding_dim];
    const outputSize = numIndices * this.embedding_dim;
    const outputBuffer = createStorageBuffer(outputSize * getDTypeBytes("float32"));
    const paramsData = new Uint32Array([
      this.num_embeddings,
      this.embedding_dim,
      numIndices,
      0
      // padding
    ]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(embedding_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.weight.buffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(outputSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    const result = new Tensor({
      buffer: outputBuffer,
      shape: outputShape,
      dtype: "float32",
      device: "webgpu",
      requires_grad: this.weight.requires_grad
    });
    return result;
  }
};

// src/nn/layernorm.ts
var LayerNorm = class extends Module {
  normalized_shape;
  eps;
  elementwise_affine;
  weight;
  bias;
  constructor(normalized_shape, eps = 1e-5, elementwise_affine = true) {
    super();
    this.normalized_shape = Array.isArray(normalized_shape) ? normalized_shape : [normalized_shape];
    this.eps = eps;
    this.elementwise_affine = elementwise_affine;
    if (this.normalized_shape.length !== 1) {
      throw new Error("LayerNorm currently only supports 1D normalized_shape");
    }
    const size = this.normalized_shape[0];
    const weightTensor = ones([size], { requires_grad: true });
    const biasTensor = zeros([size], { requires_grad: true });
    this.weight = Parameter.create(weightTensor);
    this.bias = Parameter.create(biasTensor);
    this.register_parameter("weight", this.weight);
    this.register_parameter("bias", this.bias);
  }
  forward(input) {
    const device = getDevice();
    const shape = input.shape;
    const normalizedSize = this.normalized_shape[0];
    if (shape[shape.length - 1] !== normalizedSize) {
      throw new Error(
        `Expected last dim ${normalizedSize}, got ${shape[shape.length - 1]}`
      );
    }
    const batchSize = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const totalSize = input.numel();
    const outputBuffer = createStorageBuffer(totalSize * getDTypeBytes("float32"));
    const paramsData = new ArrayBuffer(16);
    new Uint32Array(paramsData, 0, 2).set([batchSize, normalizedSize]);
    new Float32Array(paramsData, 8, 1)[0] = this.eps;
    new Uint32Array(paramsData, 12, 1)[0] = 0;
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const pipeline = getOrCreatePipeline(layernorm_default, "main");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input.buffer } },
        { binding: 1, resource: { buffer: this.weight.buffer } },
        { binding: 2, resource: { buffer: this.bias.buffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
        { binding: 4, resource: { buffer: paramsBuffer } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(batchSize));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    return new Tensor({
      buffer: outputBuffer,
      shape: [...shape],
      dtype: "float32",
      device: "webgpu",
      requires_grad: input.requires_grad
    });
  }
};

// src/nn/containers.ts
var ModuleList = class extends Module {
  _list = [];
  constructor(modules) {
    super();
    if (modules) {
      for (let i = 0; i < modules.length; i++) {
        this.add_module(String(i), modules[i]);
        this._list.push(modules[i]);
      }
    }
  }
  /**
   * Append a module to the list.
   */
  append(module) {
    const idx = this._list.length;
    this.add_module(String(idx), module);
    this._list.push(module);
    return this;
  }
  /**
   * Get module at index.
   */
  get(index) {
    return this._list[index];
  }
  /**
   * Number of modules in the list.
   */
  get length() {
    return this._list.length;
  }
  /**
   * Iterate over modules.
   */
  *[Symbol.iterator]() {
    yield* this._list;
  }
  /**
   * Forward is not implemented - use modules individually.
   */
  forward(..._inputs) {
    throw new Error("ModuleList has no forward method. Iterate over modules instead.");
  }
};
var ModuleDict = class extends Module {
  _dict = /* @__PURE__ */ new Map();
  constructor(modules) {
    super();
    if (modules) {
      for (const [key, module] of Object.entries(modules)) {
        this.add_module(key, module);
        this._dict.set(key, module);
      }
    }
  }
  /**
   * Get module by key.
   */
  get(key) {
    return this._dict.get(key);
  }
  /**
   * Set module by key.
   */
  set(key, module) {
    this.add_module(key, module);
    this._dict.set(key, module);
  }
  /**
   * Check if key exists.
   */
  has(key) {
    return this._dict.has(key);
  }
  /**
   * Get all keys.
   */
  keys() {
    return this._dict.keys();
  }
  /**
   * Get all values.
   */
  values() {
    return this._dict.values();
  }
  /**
   * Get all entries.
   */
  entries() {
    return this._dict.entries();
  }
  /**
   * Number of modules.
   */
  get size() {
    return this._dict.size;
  }
  /**
   * Forward is not implemented - use modules individually.
   */
  forward(..._inputs) {
    throw new Error("ModuleDict has no forward method. Access modules by key instead.");
  }
};

// src/nn/functional.ts
var functional_exports = {};
__export(functional_exports, {
  conv2d: () => conv2d,
  cross_entropy: () => cross_entropy,
  dropout: () => dropout,
  elu: () => elu,
  gelu: () => gelu,
  glu: () => glu,
  hardsigmoid: () => hardsigmoid,
  hardswish: () => hardswish,
  leaky_relu: () => leaky_relu,
  log_softmax: () => log_softmax,
  max_pool2d: () => max_pool2d,
  mish: () => mish,
  nll_loss: () => nll_loss,
  relu: () => relu,
  selu: () => selu,
  sigmoid: () => sigmoid,
  silu: () => silu,
  softmax: () => softmax,
  softplus: () => softplus,
  softsign: () => softsign,
  tanh: () => tanh,
  tanhshrink: () => tanhshrink,
  threshold: () => threshold
});
function relu(input, inplace = false) {
  if (inplace) {
    throw new Error("inplace ReLU not yet supported");
  }
  return input.relu();
}
function gelu(input) {
  return input.gelu();
}
function sigmoid(input) {
  return input.sigmoid();
}
function tanh(input) {
  return input.tanh();
}
function softplus(input) {
  return input.softplus();
}
function silu(input) {
  return input.silu();
}
function mish(input) {
  return input.mish();
}
function hardsigmoid(input) {
  return input.hardsigmoid();
}
function hardswish(input) {
  return input.hardswish();
}
function softsign(input) {
  return input.softsign();
}
function tanhshrink(input) {
  return input.tanhshrink();
}
function leaky_relu(input, negative_slope = 0.01) {
  return input.leaky_relu(negative_slope);
}
function elu(input, alpha = 1) {
  return input.elu(alpha);
}
function selu(input) {
  return input.selu();
}
function glu(input, dim = -1) {
  return input.glu(dim);
}
function threshold(input, threshold2, value) {
  return input.threshold(threshold2, value);
}
function softmax(input, dim = -1) {
  return log_softmax(input, dim).exp();
}
function _logSoftmaxBackward(gradOutput, softmax2) {
  const [batchSize, numClasses] = gradOutput.shape;
  const device = getDevice();
  const outputBuffer = createStorageBuffer(gradOutput.numel() * getDTypeBytes(gradOutput.dtype));
  const dimsData = new Uint32Array([batchSize, numClasses]);
  const dimsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
  });
  device.queue.writeBuffer(dimsBuffer, 0, dimsData);
  const pipeline = getOrCreatePipeline(log_softmax_backward_default, "log_softmax_backward");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gradOutput.buffer, offset: 0, size: gradOutput.buffer.size } },
      { binding: 1, resource: { buffer: softmax2.buffer, offset: 0, size: softmax2.buffer.size } },
      { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 3, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } }
    ]
  });
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...calculateWorkgroups(gradOutput.numel()));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  return new Tensor({
    buffer: outputBuffer,
    shape: [...gradOutput.shape],
    dtype: gradOutput.dtype,
    device: "webgpu",
    requires_grad: false
  });
}
function log_softmax(input, dim = -1) {
  let ndim = input.shape.length;
  if (dim < 0) {
    dim += ndim;
  }
  if (dim === ndim - 1) {
    if (ndim > 2) {
      const C = input.shape[ndim - 1];
      const flatInput = input.reshape([-1, C]);
      const flatResult = log_softmax(flatInput, 1);
      return flatResult.reshape(input.shape);
    }
  } else {
    throw new Error(`log_softmax currently only supports dim=-1 (last dimension) for ND tensors. Got dim=${dim}, ndim=${ndim}`);
  }
  if (input.shape.length !== 2) {
    throw new Error(`log_softmax implementation error: expected 2D tensor, got ${input.shape.length}D`);
  }
  if (dim !== 1) {
    throw new Error("log_softmax currently only supports dim=1 (last dimension) for 2D tensors");
  }
  const [batchSize, numClasses] = input.shape;
  const device = getDevice();
  const outputBuffer = createStorageBuffer(input.numel() * getDTypeBytes(input.dtype));
  const dimsData = new Uint32Array([batchSize, numClasses]);
  const dimsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
  });
  device.queue.writeBuffer(dimsBuffer, 0, dimsData);
  const pipeline = getOrCreatePipeline(log_softmax_default, "log_softmax");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer, offset: 0, size: input.buffer.size } },
      { binding: 1, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 2, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } }
    ]
  });
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(batchSize, 1, 1);
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  const resultHolder = {};
  let grad_fn;
  if (input.requires_grad) {
    const inputRef = input;
    grad_fn = {
      backward(gradOutput) {
        const logSoftmaxOutput = resultHolder.tensor;
        const softmax2 = logSoftmaxOutput.exp();
        const grad = _logSoftmaxBackward(gradOutput, softmax2);
        inputRef.accumulateGrad(grad);
      }
    };
  }
  const result = new Tensor({
    buffer: outputBuffer,
    shape: [...input.shape],
    dtype: input.dtype,
    device: "webgpu",
    requires_grad: input.requires_grad,
    grad_fn
  });
  resultHolder.tensor = result;
  return result;
}
function nll_loss(input, target, reduction = "mean") {
  if (input.shape.length !== 2) {
    throw new Error("nll_loss currently only supports 2D input tensors");
  }
  if (target.shape.length !== 1) {
    throw new Error("nll_loss currently only supports 1D target tensors");
  }
  if (input.shape[0] !== target.shape[0]) {
    throw new Error(`Batch size mismatch: input ${input.shape[0]} vs target ${target.shape[0]}`);
  }
  const [batchSize, numClasses] = input.shape;
  const device = getDevice();
  const outputBuffer = createStorageBuffer(batchSize * getDTypeBytes(input.dtype));
  const dimsData = new Uint32Array([batchSize, numClasses]);
  const dimsBuffer = device.createBuffer({
    size: 8,
    usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
  });
  device.queue.writeBuffer(dimsBuffer, 0, dimsData);
  const pipeline = getOrCreatePipeline(nll_loss_default, "nll_loss");
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer, offset: 0, size: input.buffer.size } },
      { binding: 1, resource: { buffer: target.buffer, offset: 0, size: target.buffer.size } },
      { binding: 2, resource: { buffer: outputBuffer, offset: 0, size: outputBuffer.size } },
      { binding: 3, resource: { buffer: dimsBuffer, offset: 0, size: dimsBuffer.size } }
    ]
  });
  const workgroups = calculateWorkgroups(batchSize);
  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroups);
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
  const lossPerItem = new Tensor({
    buffer: outputBuffer,
    shape: [batchSize],
    dtype: input.dtype,
    device: "webgpu",
    requires_grad: input.requires_grad
  });
  let result;
  if (reduction === "mean") {
    result = lossPerItem.mean();
  } else if (reduction === "sum") {
    result = lossPerItem.sum();
  } else {
    result = lossPerItem;
  }
  if (input.requires_grad) {
    const inputRef = input;
    const targetRef = target;
    const scale = reduction === "mean" ? 1 / batchSize : 1;
    const grad_fn = {
      backward(_gradOutput) {
        const gradInputBuffer = createStorageBuffer(
          inputRef.numel() * getDTypeBytes(inputRef.dtype)
        );
        const paramsData = new ArrayBuffer(16);
        new Uint32Array(paramsData, 0, 1)[0] = batchSize;
        new Uint32Array(paramsData, 4, 1)[0] = numClasses;
        new Float32Array(paramsData, 8, 1)[0] = scale;
        new Uint32Array(paramsData, 12, 1)[0] = 0;
        const paramsBuffer = device.createBuffer({
          size: 16,
          usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
        });
        device.queue.writeBuffer(paramsBuffer, 0, paramsData);
        const pipeline2 = getOrCreatePipeline(nll_loss_backward_default, "nll_loss_backward");
        const bindGroup2 = device.createBindGroup({
          layout: pipeline2.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: targetRef.buffer, offset: 0, size: targetRef.buffer.size } },
            { binding: 1, resource: { buffer: gradInputBuffer, offset: 0, size: gradInputBuffer.size } },
            { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
          ]
        });
        const commandEncoder2 = device.createCommandEncoder();
        const passEncoder2 = commandEncoder2.beginComputePass();
        passEncoder2.setPipeline(pipeline2);
        passEncoder2.setBindGroup(0, bindGroup2);
        passEncoder2.dispatchWorkgroups(...calculateWorkgroups(inputRef.numel()));
        passEncoder2.end();
        device.queue.submit([commandEncoder2.finish()]);
        const gradInput = new Tensor({
          buffer: gradInputBuffer,
          shape: [...inputRef.shape],
          dtype: inputRef.dtype,
          device: "webgpu",
          requires_grad: false
        });
        if (inputRef.grad_fn) {
          inputRef.grad_fn.backward(gradInput);
        }
      }
    };
    result = new Tensor({
      buffer: result.buffer,
      shape: [...result.shape],
      dtype: result.dtype,
      device: result.device,
      requires_grad: true,
      grad_fn
    });
  }
  return result;
}
function cross_entropy(input, target, reduction = "mean") {
  return nll_loss(log_softmax(input, -1), target, reduction);
}
function dropout(input, p = 0.5, training = true) {
  if (!training || p === 0) {
    return input;
  }
  if (p === 1) {
    return input.mul(0);
  }
  return input.mul(1 - p);
}
function max_pool2d(input, kernel_size, stride, padding = 0) {
  void input;
  void kernel_size;
  void stride;
  void padding;
  throw new Error("max_pool2d not yet implemented");
}
function conv2d(input, weight, bias, stride = 1, padding = 0) {
  void input;
  void weight;
  void bias;
  void stride;
  void padding;
  throw new Error("conv2d not yet implemented");
}

// src/nn/activation.ts
var ReLU = class extends Module {
  inplace;
  constructor(inplace = false) {
    super();
    this.inplace = inplace;
  }
  forward(input) {
    return relu(input, this.inplace);
  }
};
var Sigmoid = class extends Module {
  forward(input) {
    return input.sigmoid();
  }
};
var Tanh = class extends Module {
  forward(input) {
    return input.tanh();
  }
};
var LogSoftmax = class extends Module {
  dim;
  constructor(dim = -1) {
    super();
    this.dim = dim;
  }
  forward(input) {
    return log_softmax(input, this.dim);
  }
};
var GELU = class extends Module {
  forward(input) {
    return gelu(input);
  }
};
var Softmax = class extends Module {
  dim;
  constructor(dim = -1) {
    super();
    this.dim = dim;
  }
  forward(input) {
    return softmax(input, this.dim);
  }
};

// src/nn/dropout.ts
var Dropout = class extends Module {
  p;
  _inplace;
  constructor(p = 0.5, inplace = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }
  forward(input) {
    return dropout(input, this.p, this.training);
  }
};
var Dropout2d = class extends Module {
  p;
  _inplace;
  constructor(p = 0.5, inplace = false) {
    super();
    this.p = p;
    this._inplace = inplace;
  }
  forward(input) {
    return dropout(input, this.p, this.training);
  }
};

// src/optim/index.ts
var optim_exports = {};
__export(optim_exports, {
  Adam: () => Adam,
  AdamW: () => AdamW,
  Optimizer: () => Optimizer,
  SGD: () => SGD
});

// src/optim/optimizer.ts
var Optimizer = class {
  param_groups;
  defaults;
  constructor(params, defaults) {
    this.defaults = defaults;
    this.param_groups = [];
    const paramArray = Array.isArray(params) ? params : Array.from(params);
    this.param_groups.push({
      params: paramArray,
      lr: defaults.lr ?? 0.01,
      ...defaults
    });
  }
  /**
   * Clears the gradients of all optimized parameters.
   * @pytorch optimizer.zero_grad()
   */
  zero_grad() {
    for (const group of this.param_groups) {
      for (const param of group.params) {
        param.grad = null;
      }
    }
  }
};

// src/optim/sgd.ts
var SGD = class extends Optimizer {
  momentum_buffers;
  constructor(params, options) {
    super(params, {
      lr: options.lr,
      momentum: options.momentum ?? 0,
      weight_decay: options.weight_decay ?? 0,
      dampening: options.dampening ?? 0,
      nesterov: options.nesterov ?? false
    });
    this.momentum_buffers = /* @__PURE__ */ new Map();
    if (options.nesterov && (!options.momentum || options.dampening !== 0)) {
      throw new Error("Nesterov momentum requires a momentum and zero dampening");
    }
  }
  /**
   * Performs a single optimization step.
   */
  async step() {
    for (const group of this.param_groups) {
      const lr = group.lr;
      const momentum = group.momentum ?? 0;
      const weight_decay = group.weight_decay ?? 0;
      const dampening = group.dampening ?? 0;
      const nesterov = group.nesterov ?? false;
      for (const param of group.params) {
        if (!param.grad) {
          continue;
        }
        let grad = param.grad;
        if (weight_decay !== 0) {
          grad = grad.add(param.mul(weight_decay));
        }
        if (momentum !== 0) {
          let buf = this.momentum_buffers.get(param);
          if (!buf) {
            buf = grad.clone();
            this.momentum_buffers.set(param, buf);
          } else {
            buf = buf.mul(momentum).add(grad.mul(1 - dampening));
            this.momentum_buffers.set(param, buf);
          }
          if (nesterov) {
            grad = grad.add(buf.mul(momentum));
          } else {
            grad = buf;
          }
        }
        const paramData = await param.toArray();
        const gradData = await grad.toArray();
        const newParamData = paramData.map((p, i) => p - lr * gradData[i]);
        await this._updateParamBuffer(param, newParamData);
      }
    }
  }
  /**
   * Update a parameter's buffer with new values.
   * @internal
   */
  async _updateParamBuffer(param, newData) {
    const device = getDevice();
    const data = new Float32Array(newData);
    device.queue.writeBuffer(param.buffer, 0, data);
  }
};

// src/optim/adamw.ts
var AdamW = class extends Optimizer {
  state;
  constructor(params, options = {}) {
    super(params, {
      lr: options.lr ?? 1e-3,
      betas: options.betas ?? [0.9, 0.999],
      eps: options.eps ?? 1e-8,
      weight_decay: options.weight_decay ?? 0.01,
      amsgrad: options.amsgrad ?? false
    });
    this.state = /* @__PURE__ */ new Map();
  }
  /**
   * Performs a single optimization step.
   */
  async step() {
    for (const group of this.param_groups) {
      const lr = group.lr;
      const [beta1, beta2] = group.betas;
      const eps = group.eps;
      const weight_decay = group.weight_decay;
      const amsgrad = group.amsgrad;
      for (const param of group.params) {
        if (!param.grad) {
          continue;
        }
        const grad = param.grad;
        let state = this.state.get(param);
        if (!state) {
          state = {
            step: 0,
            exp_avg: zeros_like(param),
            exp_avg_sq: zeros_like(param),
            max_exp_avg_sq: amsgrad ? zeros_like(param) : void 0
          };
          this.state.set(param, state);
        }
        state.step += 1;
        state.exp_avg = state.exp_avg.mul(beta1).add(grad.mul(1 - beta1));
        state.exp_avg_sq = state.exp_avg_sq.mul(beta2).add(grad.pow(2).mul(1 - beta2));
        const bias_correction1 = 1 - Math.pow(beta1, state.step);
        const bias_correction2 = 1 - Math.pow(beta2, state.step);
        let denom;
        if (amsgrad && state.max_exp_avg_sq) {
          state.max_exp_avg_sq = state.max_exp_avg_sq.max().eq(state.max_exp_avg_sq).mul(state.max_exp_avg_sq).add(
            state.max_exp_avg_sq.lt(state.exp_avg_sq).mul(state.exp_avg_sq)
          );
          throw new Error("AMSGrad not yet fully implemented");
        }
        denom = state.exp_avg_sq.div(bias_correction2).sqrt().add(eps);
        const step_size = lr / bias_correction1;
        const pData = await param.toArray();
        const mData = await state.exp_avg.toArray();
        const vData = await state.exp_avg_sq.toArray();
        const denomData = await denom.toArray();
        const newParamData = new Float32Array(pData.length);
        for (let i = 0; i < pData.length; i++) {
          let p = pData[i];
          const m = mData[i];
          p = p * (1 - lr * weight_decay);
          p = p - step_size * (m / denomData[i]);
          newParamData[i] = p;
        }
        await this._updateParamBuffer(param, newParamData);
      }
    }
  }
  /**
   * Update a parameter's buffer with new values.
   * @internal
   */
  async _updateParamBuffer(param, newData) {
    const device = getDevice();
    device.queue.writeBuffer(param.buffer, 0, newData);
  }
};

// src/optim/adam.ts
var Adam = class extends Optimizer {
  state;
  constructor(params, options = {}) {
    super(params, {
      lr: options.lr ?? 1e-3,
      betas: options.betas ?? [0.9, 0.999],
      eps: options.eps ?? 1e-8,
      weight_decay: options.weight_decay ?? 0,
      amsgrad: options.amsgrad ?? false
    });
    this.state = /* @__PURE__ */ new Map();
  }
  /**
   * Performs a single optimization step.
   */
  async step() {
    for (const group of this.param_groups) {
      const lr = group.lr;
      const [beta1, beta2] = group.betas;
      const eps = group.eps;
      const weight_decay = group.weight_decay;
      const amsgrad = group.amsgrad;
      for (const param of group.params) {
        if (!param.grad) {
          continue;
        }
        let grad = param.grad;
        if (weight_decay !== 0) {
          grad = grad.add(param.mul(weight_decay));
        }
        let state = this.state.get(param);
        if (!state) {
          state = {
            step: 0,
            exp_avg: zeros_like(param),
            exp_avg_sq: zeros_like(param),
            max_exp_avg_sq: amsgrad ? zeros_like(param) : void 0
          };
          this.state.set(param, state);
        }
        state.step += 1;
        state.exp_avg = state.exp_avg.mul(beta1).add(grad.mul(1 - beta1));
        state.exp_avg_sq = state.exp_avg_sq.mul(beta2).add(grad.pow(2).mul(1 - beta2));
        const bias_correction1 = 1 - Math.pow(beta1, state.step);
        const bias_correction2 = 1 - Math.pow(beta2, state.step);
        let denom;
        if (amsgrad && state.max_exp_avg_sq) {
          throw new Error("AMSGrad not yet fully implemented");
        }
        denom = state.exp_avg_sq.div(bias_correction2).sqrt().add(eps);
        const step_size = lr / bias_correction1;
        const pData = await param.toArray();
        const mData = await state.exp_avg.toArray();
        const denomData = await denom.toArray();
        const newParamData = new Float32Array(pData.length);
        for (let i = 0; i < pData.length; i++) {
          let p = pData[i];
          const m = mData[i];
          p = p - step_size * (m / denomData[i]);
          newParamData[i] = p;
        }
        await this._updateParamBuffer(param, newParamData);
      }
    }
  }
  /**
   * Update a parameter's buffer with new values.
   * @internal
   */
  async _updateParamBuffer(param, newData) {
    const device = getDevice();
    device.queue.writeBuffer(param.buffer, 0, newData);
  }
};

// src/linalg/index.ts
var linalg_exports = {};
__export(linalg_exports, {
  cholesky: () => cholesky,
  cross: () => cross,
  det: () => det,
  diagonal: () => diagonal,
  inv: () => inv,
  lu_factor: () => lu_factor,
  matmul: () => matmul2,
  matrix_norm: () => matrix_norm,
  matrix_power: () => matrix_power,
  norm: () => norm,
  solve_triangular: () => solve_triangular,
  vander: () => vander,
  vecdot: () => vecdot,
  vector_norm: () => vector_norm
});

// src/ops/index.ts
var ops_exports = {};
__export(ops_exports, {
  abs: () => abs,
  absolute: () => absolute,
  acos: () => acos,
  acosh: () => acosh,
  add: () => add,
  addbmm: () => addbmm,
  addmm: () => addmm,
  addmv: () => addmv,
  addr: () => addr,
  all: () => all,
  allclose: () => allclose,
  amax: () => amax,
  amin: () => amin,
  any: () => any,
  arange: () => arange,
  arccos: () => arccos,
  arccosh: () => arccosh,
  arcsin: () => arcsin,
  arcsinh: () => arcsinh,
  arctan: () => arctan,
  arctan2: () => arctan2,
  arctanh: () => arctanh,
  argmax: () => argmax,
  argmin: () => argmin,
  asin: () => asin,
  asinh: () => asinh,
  atan: () => atan,
  atan2: () => atan2,
  atanh: () => atanh,
  atleast_1d: () => atleast_1d,
  atleast_2d: () => atleast_2d,
  atleast_3d: () => atleast_3d,
  baddbmm: () => baddbmm,
  bitwise_and: () => bitwise_and,
  bitwise_or: () => bitwise_or,
  bitwise_xor: () => bitwise_xor,
  bmm: () => bmm,
  broadcast_to: () => broadcast_to,
  cat: () => cat,
  ceil: () => ceil,
  chain_matmul: () => chain_matmul,
  chunk: () => chunk,
  clamp: () => clamp,
  clip: () => clip,
  column_stack: () => column_stack,
  cos: () => cos,
  cosh: () => cosh,
  cumprod: () => cumprod,
  cumsum: () => cumsum,
  cumulative_trapezoid: () => cumulative_trapezoid,
  diag: () => diag,
  div: () => div,
  divide: () => divide,
  dot: () => dot,
  dstack: () => dstack,
  empty_like: () => empty_like,
  eq: () => eq,
  equal: () => equal,
  exp: () => exp,
  exp2: () => exp2,
  eye: () => eye,
  fix: () => fix,
  flatten: () => flatten,
  flip: () => flip,
  fliplr: () => fliplr,
  flipud: () => flipud,
  floor: () => floor,
  fmax: () => fmax,
  fmin: () => fmin,
  frac: () => frac,
  full: () => full,
  full_like: () => full_like,
  ge: () => ge,
  ger: () => ger,
  greater: () => greater,
  greater_equal: () => greater_equal,
  gt: () => gt,
  heaviside: () => heaviside,
  hstack: () => hstack,
  hypot: () => hypot,
  index_select: () => index_select,
  inner: () => inner,
  isclose: () => isclose,
  isfinite: () => isfinite,
  isinf: () => isinf,
  isnan: () => isnan,
  isneginf: () => isneginf,
  isposinf: () => isposinf,
  le: () => le,
  less: () => less,
  less_equal: () => less_equal,
  linspace: () => linspace,
  log: () => log,
  log10: () => log10,
  log1p: () => log1p,
  log2: () => log2,
  logaddexp: () => logaddexp,
  logspace: () => logspace,
  lt: () => lt,
  manual_seed: () => manual_seed,
  matmul: () => matmul,
  maximum: () => maximum,
  minimum: () => minimum,
  mm: () => mm,
  moveaxis: () => moveaxis,
  movedim: () => movedim,
  mul: () => mul,
  multiply: () => multiply,
  mv: () => mv,
  narrow: () => narrow,
  ne: () => ne,
  neg: () => neg,
  negative: () => negative,
  not_equal: () => not_equal,
  ones: () => ones,
  ones_like: () => ones_like,
  outer: () => outer,
  permute: () => permute,
  pow: () => pow,
  prod: () => prod,
  rand: () => rand,
  randn: () => randn,
  reciprocal: () => reciprocal,
  relu: () => relu2,
  round: () => round,
  row_stack: () => row_stack,
  rsqrt: () => rsqrt,
  select: () => select,
  sigmoid: () => sigmoid2,
  sin: () => sin,
  sinh: () => sinh,
  sqrt: () => sqrt,
  square: () => square,
  squeeze: () => squeeze,
  stack: () => stack,
  sub: () => sub,
  subtract: () => subtract,
  swapaxes: () => swapaxes,
  swapdims: () => swapdims,
  take: () => take,
  tan: () => tan,
  tanh: () => tanh2,
  tensor: () => tensor,
  tile: () => tile,
  trapezoid: () => trapezoid,
  trapz: () => trapz,
  tril: () => tril,
  triu: () => triu,
  trunc: () => trunc,
  unbind: () => unbind,
  unsqueeze: () => unsqueeze,
  vdot: () => vdot,
  vstack: () => vstack,
  where: () => where,
  zeros: () => zeros,
  zeros_like: () => zeros_like
});
var atleast_1d = (input) => {
  if (input.dim() >= 1) return input;
  return input.reshape([1]);
};
var atleast_2d = (input) => {
  if (input.dim() >= 2) return input;
  if (input.dim() === 0) return input.reshape([1, 1]);
  return input.reshape([1, input.shape[0]]);
};
var atleast_3d = (input) => {
  if (input.dim() >= 3) return input;
  if (input.dim() === 0) return input.reshape([1, 1, 1]);
  if (input.dim() === 1) return input.reshape([1, input.shape[0], 1]);
  return input.reshape([input.shape[0], input.shape[1], 1]);
};
var broadcast_to = (input, shape) => input.broadcast_to(shape);
var flip = (input, dims) => input.flip(dims);
var fliplr = (input) => input.flip([1]);
var flipud = (input) => input.flip([0]);
var cumsum = (input, dim) => input.cumsum(dim);
var cumprod = (input, dim) => input.cumprod(dim);
var triu = (input, diagonal2 = 0) => input.triu(diagonal2);
var diag = (input, diagonal2 = 0) => input.diag(diagonal2);
var heaviside = (input, values) => input.heaviside(values);
var abs = (input) => input.abs();
var absolute = abs;
var acos = (input) => input.acos();
var arccos = acos;
var acosh = (input) => input.acosh();
var arccosh = acosh;
var asin = (input) => input.asin();
var arcsin = asin;
var asinh = (input) => input.asinh();
var arcsinh = asinh;
var atan = (input) => input.atan();
var arctan = atan;
var atanh = (input) => input.atanh();
var arctanh = atanh;
var atan2 = (input, other) => input.atan2(other);
var arctan2 = atan2;
var ceil = (input) => input.ceil();
var floor = (input) => input.floor();
var round = (input) => input.round();
var trunc = (input) => input.trunc();
var fix = trunc;
var frac = (input) => input.frac();
var clamp = (input, min, max) => input.clamp(min, max);
var clip = clamp;
var flatten = (input, startDim, endDim) => input.flatten(startDim, endDim);
var squeeze = (input, dim) => input.squeeze(dim);
var unsqueeze = (input, dim) => input.unsqueeze(dim);
var argmax = (input, dim, keepdim) => input.argmax(dim, keepdim);
var argmin = (input, dim, keepdim) => input.argmin(dim, keepdim);
var amax = (input, dim, keepdim) => input.amax(dim, keepdim);
var amin = (input, dim, keepdim) => input.amin(dim, keepdim);
var all = (input, dim, keepdim) => input.all(dim, keepdim);
var any = (input, dim, keepdim) => input.any(dim, keepdim);
var chunk = (input, chunks, dim) => input.chunk(chunks, dim);
var narrow = (input, dim, start, length) => input.narrow(dim, start, length);
var permute = (input, dims) => input.permute(dims);
var movedim = (input, source, destination) => input.movedim(source, destination);
var moveaxis = movedim;
var swapaxes = (input, dim0, dim1) => input.swapaxes(dim0, dim1);
var swapdims = swapaxes;
var tile = (input, reps) => input.tile(reps);
var unbind = (input, dim) => input.unbind(dim);
var index_select = (input, dim, index) => input.index_select(dim, index);
var select = (input, dim, index) => input.select(dim, index);
var take = (input, indices) => input.take(indices);
var where = (condition, input, other) => input.where(condition, other);
var eq = (input, other) => input.eq(other);
var ne = (input, other) => input.ne(other);
var not_equal = ne;
var lt = (input, other) => input.lt(other);
var less = lt;
var le = (input, other) => input.le(other);
var less_equal = le;
var gt = (input, other) => input.gt(other);
var greater = gt;
var ge = (input, other) => input.ge(other);
var greater_equal = ge;
var isnan = (input) => input.isnan();
var isinf = (input) => input.isinf();
var isfinite = (input) => input.isfinite();
var isposinf = (input) => input.isposinf();
var isneginf = (input) => input.isneginf();
var maximum = (input, other) => input.maximum(other);
var minimum = (input, other) => input.minimum(other);
var fmax = (input, other) => input.fmax(other);
var fmin = (input, other) => input.fmin(other);
var equal = (input, other) => input.equal(other);
var isclose = (input, other, rtol, atol, equal_nan) => input.isclose(other, rtol, atol, equal_nan);
var allclose = (input, other, rtol, atol, equal_nan) => input.allclose(other, rtol, atol, equal_nan);
var cos = (input) => input.cos();
var cosh = (input) => input.cosh();
var sin = (input) => input.sin();
var sinh = (input) => input.sinh();
var tan = (input) => input.tan();
var tanh2 = (input) => input.tanh();
var exp = (input) => input.exp();
var exp2 = (input) => input.exp2();
var log = (input) => input.log();
var log10 = (input) => input.log10();
var log2 = (input) => input.log2();
var log1p = (input) => input.log1p();
var logaddexp = (input, other) => input.logaddexp(other);
var neg = (input) => input.neg();
var negative = neg;
var prod = (input, dim, keepdim) => input.prod(dim, keepdim);
var pow = (input, exponent) => input.pow(exponent);
var reciprocal = (input) => input.reciprocal();
var rsqrt = (input) => input.rsqrt();
var sqrt = (input) => input.sqrt();
var square = (input) => input.square();
var sigmoid2 = (input) => input.sigmoid();
var relu2 = (input) => input.relu();
var add = (input, other) => input.add(other);
var sub = (input, other) => input.sub(other);
var subtract = sub;
var mul = (input, other) => input.mul(other);
var multiply = mul;
var div = (input, other) => input.div(other);
var divide = div;
var bitwise_and = (input, other) => input.bitwise_and(other);
var bitwise_or = (input, other) => input.bitwise_or(other);
var bitwise_xor = (input, other) => input.bitwise_xor(other);
var hypot = (input, other) => input.hypot(other);
var matmul = (input, other) => input.matmul(other);
var mm = (input, mat2) => input.mm(mat2);
var chain_matmul = (...matrices) => {
  if (matrices.length === 0) throw new Error("chain_matmul: expected at least one tensor");
  let result = matrices[0];
  for (let i = 1; i < matrices.length; i++) {
    result = result.mm(matrices[i]);
  }
  return result;
};
var addmm = (input, mat1, mat2, beta, alpha) => input.addmm(mat1, mat2, beta, alpha);
var mv = (input, vec) => input.mv(vec);
var addmv = (input, mat, vec, beta, alpha) => input.addmv(mat, vec, beta, alpha);
var outer = (input, vec2) => input.outer(vec2);
var ger = outer;
var addr = (input, vec1, vec2, beta, alpha) => input.addr(vec1, vec2, beta, alpha);
var bmm = (input, mat2) => input.bmm(mat2);
var baddbmm = (input, batch1, batch2, beta, alpha) => input.baddbmm(batch1, batch2, beta, alpha);
var addbmm = (input, batch1, batch2, beta, alpha) => input.addbmm(batch1, batch2, beta, alpha);
var dot = (input, other) => input.dot(other);
var vdot = (input, other) => input.vdot(other);
var inner = dot;
var trapezoid = (input, dx, dim) => input.trapezoid(dx, dim);
var cumulative_trapezoid = (input, dx, dim) => input.cumulative_trapezoid(dx, dim);
var trapz = trapezoid;

// src/linalg/index.ts
var lu_factor = (A, pivot = true) => {
  if (!pivot) throw new Error("linalg.lu_factor: pivot=false not yet implemented");
  if (A.dim() < 2) throw new Error("linalg.lu_factor: expected tensor with at least 2 dimensions");
  const n = A.shape[A.dim() - 1];
  const m = A.shape[A.dim() - 2];
  if (n !== m) throw new Error("linalg.lu_factor: expected square matrix");
  const batch = A.dim() > 2 ? A.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
  const device = getDevice();
  const LU = A.clone();
  const P = arange(n).unsqueeze(0).expand([batch, n]).to("uint32").clone();
  console.log("lu_factor P details:", {
    numel: P.numel(),
    dtype: P.dtype,
    shape: P.shape,
    buffer_size: P.buffer.size
  });
  const pipelinePivot = getOrCreatePipeline(lu_default, "lu_pivot");
  const pipelineUpdate = getOrCreatePipeline(lu_default, "lu_update");
  for (let k = 0; k < n; k++) {
    const paramsData = new Uint32Array([n, batch, k, 0]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const bindGroup = device.createBindGroup({
      layout: pipelinePivot.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: LU.buffer, offset: 0, size: LU.buffer.size } },
        { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
        { binding: 2, resource: { buffer: P.buffer, offset: 0, size: P.buffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipelinePivot);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(batch / 256), 1, 1);
    passEncoder.end();
    if (k < n - 1) {
      const remaining = n - 1 - k;
      const bindGroupUpdate = device.createBindGroup({
        layout: pipelineUpdate.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: LU.buffer, offset: 0, size: LU.buffer.size } },
          { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } },
          { binding: 2, resource: { buffer: P.buffer, offset: 0, size: P.buffer.size } }
        ]
      });
      const passEncoderUpdate = commandEncoder.beginComputePass();
      passEncoderUpdate.setPipeline(pipelineUpdate);
      passEncoderUpdate.setBindGroup(0, bindGroupUpdate);
      passEncoderUpdate.dispatchWorkgroups(Math.ceil(remaining / 16), Math.ceil(remaining / 16), batch);
      passEncoderUpdate.end();
    }
    device.queue.submit([commandEncoder.finish()]);
  }
  return [LU, P];
};
var solve_triangular = (A, B, upper, left = true, unitriangular = false) => {
  if (!left) throw new Error("linalg.solve_triangular: right-side solve (left=false) not yet implemented");
  if (unitriangular) throw new Error("linalg.solve_triangular: unitriangular=true not yet implemented");
  if (A.dim() < 2 || B.dim() < 2) throw new Error("linalg.solve_triangular: expected tensors with at least 2 dimensions");
  const n = A.shape[A.dim() - 1];
  const m = B.shape[B.dim() - 1];
  const batch = A.dim() > 2 ? A.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
  const device = getDevice();
  const X = B.clone();
  const pipeline = getOrCreatePipeline(triangular_solve_default, upper ? "backward_sub_step" : "forward_sub_step");
  for (let step = 0; step < n; step++) {
    const k = upper ? n - 1 - step : step;
    const paramsData = new Uint32Array([n, m, batch, k]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: A.buffer, offset: 0, size: A.buffer.size } },
        { binding: 1, resource: { buffer: X.buffer, offset: 0, size: X.buffer.size } },
        { binding: 2, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(m / 256), batch, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
  }
  return X;
};
var cholesky = (A, upper = false) => {
  if (A.dim() < 2) throw new Error("linalg.cholesky: expected tensor with at least 2 dimensions");
  const n = A.shape[A.dim() - 1];
  const m = A.shape[A.dim() - 2];
  if (n !== m) throw new Error("linalg.cholesky: expected square matrix");
  const batch = A.dim() > 2 ? A.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
  const device = getDevice();
  const res = A.clone();
  const buffer = res.buffer;
  if (n <= 16) {
    const pipeline = getOrCreatePipeline(cholesky_default, "cholesky_small");
    const paramsData = new Uint32Array([n, batch, 0, 0]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer, offset: 0, size: buffer.size } },
        { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
      ]
    });
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...calculateWorkgroups(batch));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
  } else {
    const pipeline1 = getOrCreatePipeline(cholesky_default, "cholesky_step1");
    const pipeline2 = getOrCreatePipeline(cholesky_default, "cholesky_step2");
    for (let k = 0; k < n; k++) {
      const paramsData = new Uint32Array([n, batch, k, 0]);
      const paramsBuffer = device.createBuffer({
        size: 16,
        usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
      });
      device.queue.writeBuffer(paramsBuffer, 0, paramsData);
      const bindGroup = device.createBindGroup({
        layout: pipeline1.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer, offset: 0, size: buffer.size } },
          { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
        ]
      });
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline1);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(...calculateWorkgroups(batch));
      passEncoder.end();
      if (k < n - 1) {
        const remaining = n - 1 - k;
        const pipeline22 = getOrCreatePipeline(cholesky_default, "cholesky_step2");
        const bindGroup2 = device.createBindGroup({
          layout: pipeline22.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer, offset: 0, size: buffer.size } },
            { binding: 1, resource: { buffer: paramsBuffer, offset: 0, size: paramsBuffer.size } }
          ]
        });
        const passEncoder2 = commandEncoder.beginComputePass();
        passEncoder2.setPipeline(pipeline22);
        passEncoder2.setBindGroup(0, bindGroup2);
        passEncoder2.dispatchWorkgroups(
          Math.ceil(remaining / 16),
          Math.ceil(remaining / 16),
          batch
        );
        passEncoder2.end();
      }
      device.queue.submit([commandEncoder.finish()]);
    }
    return res.tril();
  }
  if (upper) {
    return res.transpose(-2, -1);
  }
  return res;
};
var norm = (input, ord, dim, keepdim = false) => {
  if (typeof dim === "number") dim = [dim];
  if (dim === void 0 && input.dim() === 2 || dim !== void 0 && dim.length === 2) {
    return matrix_norm(input, ord, dim, keepdim);
  }
  return vector_norm(input, ord, dim, keepdim);
};
var vector_norm = (input, ord = 2, dim, keepdim = false) => {
  if (typeof dim === "number") dim = [dim];
  if (ord === 2 || ord === void 0) {
    return input.pow(2).sum(dim, keepdim).sqrt();
  }
  if (ord === 1) {
    return input.abs().sum(dim, keepdim);
  }
  if (ord === Infinity || ord === "inf") {
    return input.abs().amax(dim, keepdim);
  }
  if (ord === -Infinity || ord === "-inf") {
    return input.abs().amin(dim, keepdim);
  }
  if (ord === 0) {
    return input.ne(0).sum(dim, keepdim);
  }
  if (typeof ord === "number") {
    return input.abs().pow(ord).sum(dim, keepdim).pow(1 / ord);
  }
  throw new Error(`linalg.vector_norm: ord=${ord} not supported`);
};
var matrix_norm = (input, ord = "fro", dim = [-2, -1], keepdim = false) => {
  if (ord === "fro" || ord === 2) {
    return input.pow(2).sum(dim, keepdim).sqrt();
  }
  if (ord === "nuc") {
    throw new Error('linalg.matrix_norm: ord="nuc" (nuclear norm) not yet implemented');
  }
  if (ord === 1) {
    const absSum = input.abs().sum([dim[0]], keepdim);
    const reduceDim = dim[1] < 0 ? dim[1] : keepdim ? dim[1] : dim[1] - 1;
    return absSum.amax([reduceDim], keepdim);
  }
  if (ord === Infinity || ord === "inf") {
    const absSum = input.abs().sum([dim[1]], keepdim);
    const reduceDim = dim[0] < 0 ? dim[0] : keepdim ? dim[0] : dim[0] - 1;
    return absSum.amax([reduceDim], keepdim);
  }
  throw new Error(`linalg.matrix_norm: ord=${ord} not supported`);
};
var diagonal = (input, offset = 0, dim1 = -2, dim2 = -1) => {
  if (input.dim() === 2) {
    return diag(input, offset);
  }
  throw new Error("linalg.diagonal: only 2D tensors supported currently");
};
var matmul2 = matmul;
var vecdot = (x, y, dim = -1) => {
  return x.mul(y).sum(dim);
};
var vander = (x, N, increasing = false) => {
  const n = x.shape[0];
  const m = N ?? n;
  const xCol = x.unsqueeze(1);
  if (!increasing) {
    const exponents = arange(m - 1, -1, -1).to(x.dtype).unsqueeze(0);
    return xCol.pow(exponents);
  } else {
    const exponents = arange(m).to(x.dtype).unsqueeze(0);
    return xCol.pow(exponents);
  }
};
var matrix_power = (A, n) => {
  if (A.dim() < 2) throw new Error("linalg.matrix_power: expected tensor with at least 2 dimensions");
  const h = A.shape[A.dim() - 2];
  const w = A.shape[A.dim() - 1];
  if (h !== w) throw new Error("linalg.matrix_power: expected square matrix");
  if (n === 0) {
    return eye(h, A.device).expand(A.shape);
  }
  if (n < 0) {
    return matrix_power(inv(A), -n);
  }
  let res = A;
  for (let i = 1; i < n; i++) {
    res = matmul(res, A);
  }
  return res;
};
var inv = (A) => {
  const [LU, P] = lu_factor(A);
  const n = A.shape[A.dim() - 1];
  const batch = A.dim() > 2 ? A.shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
  const I = eye(n).expand(A.shape).clone();
  throw new Error("linalg.inv: permutation handling not yet implemented");
};
var det = (A) => {
  const [LU, P] = lu_factor(A);
  const n = A.shape[A.dim() - 1];
  const diagU = diag(LU);
  let d = diagU.prod(-1);
  return d;
};
var cross = (input, other, dim = -1) => {
  if (input.shape[dim] !== 3 || other.shape[dim] !== 3) {
    throw new Error("linalg.cross: expected dimension of size 3");
  }
  throw new Error("linalg.cross: not yet implemented");
};

// src/webgpu.ts
var webgpu_exports = {};
__export(webgpu_exports, {
  empty_cache: () => empty_cache,
  memory_stats: () => memory_stats,
  memory_summary: () => memory_summary,
  reset_peak_memory_stats: () => reset_peak_memory_stats
});
function memory_stats() {
  const stats = bufferPool.getStats();
  return {
    active_bytes: stats.activeBytes,
    pooled_bytes: stats.pooledBytes,
    peak_bytes: stats.peakBytes,
    total_allocations: stats.allocationCount
  };
}
function memory_summary() {
  const stats = memory_stats();
  const toMB = (bytes) => (bytes / 1024 / 1024).toFixed(2);
  return `
GPU Memory Summary
-------------------------------------------------------
Active Memory:   ${toMB(stats.active_bytes)} MB
Pooled Memory:   ${toMB(stats.pooled_bytes)} MB
Peak Memory:     ${toMB(stats.peak_bytes)} MB
Total Allocs:    ${stats.total_allocations}
-------------------------------------------------------
`;
}
function empty_cache() {
  bufferPool.clear();
}
function reset_peak_memory_stats() {
  bufferPool.resetPeak();
}

// src/utils/benchmark/index.ts
var benchmark_exports = {};
__export(benchmark_exports, {
  Compare: () => Compare,
  FuzzedParameter: () => FuzzedParameter,
  FuzzedTensor: () => FuzzedTensor,
  Fuzzer: () => Fuzzer,
  Measurement: () => Measurement,
  ParameterAlias: () => ParameterAlias,
  Timer: () => Timer,
  op_fuzzers: () => op_fuzzers
});

// src/utils/benchmark/timer.ts
var Measurement = class {
  constructor(name, times, task_spec = "", label = "", sub_label = "", description = "", env = "") {
    this.name = name;
    this.times = times;
    this.task_spec = task_spec;
    this.label = label;
    this.sub_label = sub_label;
    this.description = description;
    this.env = env;
  }
  get mean() {
    return this.times.reduce((a, b) => a + b, 0) / this.times.length;
  }
  get median() {
    const sorted = [...this.times].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }
  get iqr() {
    if (this.times.length < 4) return 0;
    const sorted = [...this.times].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    return q3 - q1;
  }
  toString() {
    const median = (this.median * 1e3 * 1e3).toFixed(2);
    return `${this.label}${this.sub_label ? ": " + this.sub_label : ""}  ${median} us (${this.times.length} iterations)`;
  }
};
var Timer = class {
  constructor(stmt, setup = () => {
  }, label = "", sub_label = "", description = "", env = "") {
    this.stmt = stmt;
    this.setup = setup;
    this.label = label;
    this.sub_label = sub_label;
    this.description = description;
    this.env = env;
  }
  /**
   * Run the statement multiple times and return a Measurement.
   */
  async timeit(number = 100) {
    await this.setup();
    await syncDevice();
    const times = [];
    for (let i = 0; i < number; i++) {
      const start = performance.now();
      const res = this.stmt();
      if (res instanceof Promise) await res;
      await syncDevice();
      const end = performance.now();
      times.push((end - start) / 1e3);
    }
    return new Measurement(
      "",
      times,
      "",
      this.label,
      this.sub_label,
      this.description,
      this.env
    );
  }
  /**
   * Automatically determine number of iterations to get stable results.
   */
  async blocked_autorange(min_run_time = 0.2) {
    await this.setup();
    await syncDevice();
    let number = 1;
    let totalTime = 0;
    while (totalTime < 0.01) {
      const start = performance.now();
      for (let i = 0; i < number; i++) {
        const res = this.stmt();
        if (res instanceof Promise) await res;
      }
      await syncDevice();
      totalTime = (performance.now() - start) / 1e3;
      if (totalTime < 0.01) number *= 10;
    }
    const actual_number = Math.max(number, Math.ceil(min_run_time / (totalTime / number)));
    return this.timeit(actual_number);
  }
};

// src/utils/benchmark/fuzzer.ts
var FuzzedParameter = class {
  constructor(name, values, distribution) {
    this.name = name;
    this.values = values;
    this.distribution = distribution;
  }
};
var FuzzedTensor = class {
  constructor(name, min_shape, max_shape, probability_contiguous = 1, dtype = "float32") {
    this.name = name;
    this.min_shape = min_shape;
    this.max_shape = max_shape;
    this.probability_contiguous = probability_contiguous;
    this.dtype = dtype;
  }
  generate() {
    const shape = this.min_shape.map((min, i) => {
      const max = this.max_shape[i];
      return Math.floor(Math.random() * (max - min + 1)) + min;
    });
    return randn(shape);
  }
};
var Fuzzer = class {
  constructor(parameters, seed) {
    this.parameters = parameters;
    this.seed = seed;
    if (seed !== void 0) {
    }
  }
  /**
   * Returns a generator that yields randomized inputs.
   */
  *take(n) {
    for (let i = 0; i < n; i++) {
      const result = {};
      for (const p of this.parameters) {
        if (p instanceof FuzzedParameter) {
          result[p.name] = p.values[Math.floor(Math.random() * p.values.length)];
        } else if (p instanceof FuzzedTensor) {
          result[p.name] = p.generate();
        }
      }
      yield result;
    }
  }
};

// src/utils/benchmark/compare.ts
var Compare = class {
  constructor(measurements) {
    this.measurements = measurements;
  }
  /**
   * Print a formatted comparison table.
   */
  print() {
    console.log("\n[Benchmark Comparison]");
    console.log("--------------------------------------------------------------------------------");
    console.log(`${"Label".padEnd(20)} | ${"Sub-label".padEnd(20)} | ${"Median (us)".padStart(12)} | ${"IQR".padStart(10)}`);
    console.log("--------------------------------------------------------------------------------");
    for (const m of this.measurements) {
      const median = (m.median * 1e3).toFixed(2);
      const iqr = (m.iqr * 1e3).toFixed(2);
      console.log(
        `${m.label.padEnd(20)} | ${m.sub_label.padEnd(20)} | ${median.padStart(12)} | ${iqr.padStart(10)}`
      );
    }
    console.log("--------------------------------------------------------------------------------\n");
  }
};

// src/utils/benchmark/index.ts
var ParameterAlias = class {
  constructor(name, alias) {
    this.name = name;
    this.alias = alias;
  }
};
var op_fuzzers = {
  // Add common op fuzzers here as needed
  unary: (name) => {
  }
};

// src/debug.ts
function DEBUG(fn) {
  if (true) {
    fn();
  }
}
function DEBUG_ASYNC(fn) {
  if (true) {
    fn();
  }
}
function log3(message, ...args) {
  console.log(`[torch.js] ${message}`, ...args);
}
function warn(message, ...args) {
  console.warn(`[torch.js] ${message}`, ...args);
}
function error(message, ...args) {
  console.error(`[torch.js] ${message}`, ...args);
}
function assert(condition, message) {
  if (!condition) {
    throw new Error(`[torch.js] Assertion failed: ${message}`);
  }
}

// src/create_torch.ts
function createTorch(save, load) {
  return {
    // Initialization
    init: initWebGPU,
    // Serialization
    save,
    load,
    // Operations (spread all ops)
    ...ops_exports,
    // Linear Algebra module
    linalg: linalg_exports,
    // Synchronization
    syncDevice,
    // Neural network module
    nn: nn_exports,
    // Optimizer module
    optim: optim_exports,
    // WebGPU/Memory management
    webgpu: webgpu_exports
  };
}

// src/core.ts
var utils = { benchmark: benchmark_exports };
var _internals = {
  getDevice,
  getOrCreatePipeline,
  calculateWorkgroups,
  readBuffer
};

export {
  atleast_1d,
  atleast_2d,
  atleast_3d,
  broadcast_to,
  flip,
  fliplr,
  flipud,
  cumsum,
  cumprod,
  triu,
  diag,
  heaviside,
  abs,
  absolute,
  acos,
  arccos,
  acosh,
  arccosh,
  asin,
  arcsin,
  asinh,
  arcsinh,
  atan,
  arctan,
  atanh,
  arctanh,
  atan2,
  arctan2,
  ceil,
  floor,
  round,
  trunc,
  fix,
  frac,
  clamp,
  clip,
  flatten,
  squeeze,
  unsqueeze,
  argmax,
  argmin,
  amax,
  amin,
  all,
  any,
  chunk,
  narrow,
  permute,
  movedim,
  moveaxis,
  swapaxes,
  swapdims,
  tile,
  unbind,
  index_select,
  select,
  take,
  where,
  eq,
  ne,
  not_equal,
  lt,
  less,
  le,
  less_equal,
  gt,
  greater,
  ge,
  greater_equal,
  isnan,
  isinf,
  isfinite,
  isposinf,
  isneginf,
  maximum,
  minimum,
  fmax,
  fmin,
  equal,
  isclose,
  allclose,
  cos,
  cosh,
  sin,
  sinh,
  tan,
  tanh2 as tanh,
  exp,
  exp2,
  log10,
  log2,
  log1p,
  logaddexp,
  neg,
  negative,
  prod,
  pow,
  reciprocal,
  rsqrt,
  sqrt,
  square,
  sigmoid2 as sigmoid,
  relu2 as relu,
  add,
  sub,
  subtract,
  mul,
  multiply,
  div,
  divide,
  bitwise_and,
  bitwise_or,
  bitwise_xor,
  hypot,
  matmul,
  mm,
  chain_matmul,
  addmm,
  mv,
  addmv,
  outer,
  ger,
  addr,
  bmm,
  baddbmm,
  addbmm,
  dot,
  vdot,
  inner,
  trapezoid,
  cumulative_trapezoid,
  trapz,
  nn_exports,
  optim_exports,
  linalg_exports,
  webgpu_exports,
  createTorch,
  DEBUG,
  DEBUG_ASYNC,
  log3 as log,
  warn,
  error,
  assert,
  utils,
  _internals,
  core_exports
};
//# sourceMappingURL=chunk-F3N6MUSU.js.map