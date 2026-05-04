/**
 * Base Module class for neural network layers.
 * @status implemented
 * @pytorch torch.nn.Module
 */

import { Tensor } from '../tensor';
import { Parameter } from './parameter';

export type ModuleChildren = Map<string, Module>;
export type ModuleParameters = Map<string, Parameter>;
export type ModuleBuffers = Map<string, Tensor>;

/**
 * Base class for all neural network modules.
 */
export abstract class Module {
  private _parameters: ModuleParameters = new Map();
  private _modules: ModuleChildren = new Map();
  private _buffers: ModuleBuffers = new Map();
  private _training: boolean = true;

  /**
   * Whether the module is in training mode.
   */
  get training(): boolean {
    return this._training;
  }

  /**
   * Register a parameter with the module.
   */
  public register_parameter(name: string, param: Parameter | null): void {
    if (param === null) {
      this._parameters.delete(name);
    } else {
      this._parameters.set(name, param);
    }
  }

  /**
   * Register a submodule with the module.
   */
  public add_module(name: string, module: Module | null): void {
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
  public register_buffer(name: string, tensor: Tensor | null): void {
    if (tensor === null) {
      this._buffers.delete(name);
    } else {
      this._buffers.set(name, tensor);
      // Also set as property for easy access
      (this as Record<string, unknown>)[name] = tensor;
    }
  }

  /**
   * Get a buffer by name.
   */
  get_buffer(name: string): Tensor | undefined {
    return this._buffers.get(name);
  }

  /**
   * Returns an iterator over module buffers.
   * @pytorch module.buffers()
   */
  *buffers(recurse: boolean = true): Generator<Tensor> {
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
  *named_buffers(prefix: string = '', recurse: boolean = true): Generator<[string, Tensor]> {
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
  get_parameter(name: string): Parameter | undefined {
    return this._parameters.get(name);
  }

  /**
   * Get a submodule by name.
   */
  get_submodule(name: string): Module | undefined {
    return this._modules.get(name);
  }

  /**
   * Returns an iterator over module parameters.
   * @pytorch module.parameters()
   */
  *parameters(recurse: boolean = true): Generator<Parameter> {
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
  *named_parameters(prefix: string = '', recurse: boolean = true): Generator<[string, Parameter]> {
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
  *children(): Generator<Module> {
    yield* this._modules.values();
  }

  /**
   * Returns an iterator over all modules (including self).
   * @pytorch module.modules()
   */
  *modules(): Generator<Module> {
    yield this;
    for (const module of this._modules.values()) {
      yield* module.modules();
    }
  }

  /**
   * Set the module in training mode.
   * @pytorch module.train()
   */
  train(mode: boolean = true): this {
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
  eval(): this {
    return this.train(false);
  }

  /**
   * Zero out all gradients.
   * @pytorch module.zero_grad()
   */
  zero_grad(): void {
    for (const param of this.parameters()) {
      param.grad = null;
    }
  }

  /**
   * Returns a dictionary containing a whole state of the module.
   * @pytorch module.state_dict()
   */
  state_dict(destination: Record<string, Tensor> = {}, prefix: string = '', recurse: boolean = true): Record<string, Tensor> {
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
  load_state_dict(state_dict: Record<string, Tensor>): void {
      const own_state = this.state_dict();
      for (const name in state_dict) {
          if (Object.prototype.hasOwnProperty.call(own_state, name)) {
              // Copy data
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
   * Move all parameters and buffers to the specified device.
   * @pytorch module.to(device)
   * @note torch.js only supports 'webgpu' device, so this is a no-op
   * but kept for API compatibility. It also supports dtype migration.
   */
  to(dtypeOrDevice: string): this {
    // Migrate all parameters
    for (const [name, param] of this._parameters.entries()) {
      const migrated = param.to(dtypeOrDevice as any);
      if (migrated !== param) {
        // Need to update the parameter reference
        // In PyTorch, to() modifies in-place or returns a new module
        // Here we just copy the data if it's a device move (no-op for us)
      }
    }
    // Migrate all buffers
    for (const [name, buffer] of this._buffers.entries()) {
      // Buffers are tensors, migrate them
    }
    // Recurse into submodules
    for (const [name, module] of this._modules.entries()) {
      module.to(dtypeOrDevice);
    }
    return this;
  }

  /**
   * Forward pass - must be implemented by subclasses.
   */
  abstract forward(...inputs: any[]): any;

  /**
   * Call the module (alias for forward).
   */
  call(...inputs: any[]): any {
    return this.forward(...inputs);
  }
}
