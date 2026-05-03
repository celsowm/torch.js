/**
 * Sequential container module.
 * @status implemented
 * @pytorch torch.nn.Sequential
 */

import { Tensor } from '../tensor';
import { Module } from './module';

/**
 * A sequential container that runs modules in order.
 */
export class Sequential extends Module {
  private moduleList: Module[];

  constructor(...modules: Module[]) {
    super();
    this.moduleList = modules;
    modules.forEach((mod, idx) => {
      this.add_module(String(idx), mod);
    });
  }

  forward(input: Tensor): Tensor {
    let output = input;
    for (const module of this.moduleList) {
      output = module.forward(output);
    }
    return output;
  }

  /**
   * Get a module by index.
   */
  get(index: number): Module | undefined {
    return this.moduleList[index];
  }

  /**
   * Get the number of modules.
   */
  get length(): number {
    return this.moduleList.length;
  }
}
