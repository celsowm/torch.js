/**
 * Container modules for organizing neural network layers.
 * @status implemented
 * @pytorch torch.nn
 */

import { Tensor } from '../tensor';
import { Module } from './module';

/**
 * Holds submodules in a list.
 * @pytorch torch.nn.ModuleList
 */
export class ModuleList extends Module {
  private _list: Module[] = [];

  constructor(modules?: Module[]) {
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
  append(module: Module): this {
    const idx = this._list.length;
    this.add_module(String(idx), module);
    this._list.push(module);
    return this;
  }

  /**
   * Get module at index.
   */
  get(index: number): Module {
    return this._list[index];
  }

  /**
   * Number of modules in the list.
   */
  get length(): number {
    return this._list.length;
  }

  /**
   * Iterate over modules.
   */
  *[Symbol.iterator](): Generator<Module> {
    yield* this._list;
  }

  /**
   * Forward is not implemented - use modules individually.
   */
  forward(..._inputs: Tensor[]): Tensor {
    throw new Error('ModuleList has no forward method. Iterate over modules instead.');
  }
}

/**
 * Holds submodules in a dictionary.
 * @pytorch torch.nn.ModuleDict
 */
export class ModuleDict extends Module {
  private _dict: Map<string, Module> = new Map();

  constructor(modules?: Record<string, Module>) {
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
  get(key: string): Module | undefined {
    return this._dict.get(key);
  }

  /**
   * Set module by key.
   */
  set(key: string, module: Module): void {
    this.add_module(key, module);
    this._dict.set(key, module);
  }

  /**
   * Check if key exists.
   */
  has(key: string): boolean {
    return this._dict.has(key);
  }

  /**
   * Get all keys.
   */
  keys(): IterableIterator<string> {
    return this._dict.keys();
  }

  /**
   * Get all values.
   */
  values(): IterableIterator<Module> {
    return this._dict.values();
  }

  /**
   * Get all entries.
   */
  entries(): IterableIterator<[string, Module]> {
    return this._dict.entries();
  }

  /**
   * Number of modules.
   */
  get size(): number {
    return this._dict.size;
  }

  /**
   * Forward is not implemented - use modules individually.
   */
  forward(..._inputs: Tensor[]): Tensor {
    throw new Error('ModuleDict has no forward method. Access modules by key instead.');
  }
}
