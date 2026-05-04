/**
 * torch.autograd - Automatic differentiation engine.
 * @pytorch torch.autograd
 */

import { Tensor } from './tensor/index';
import { is_grad_enabled } from './grad_mode';

/**
 * Computes and returns the sum of gradients of outputs with respect to the inputs.
 * @pytorch torch.autograd.grad(outputs, inputs, grad_outputs, retain_graph, create_graph)
 *
 * @param outputs - Tensor(s) for which gradients are computed
 * @param inputs - Input tensor(s) w.r.t. which gradients are computed
 * @param grad_outputs - Gradient tensor(s) for outputs (default: ones)
 * @param retain_graph - If false, the computation graph is freed after
 * @param create_graph - If true, gradients of gradients are also constructed
 * @param allow_unused - If true, return zeros for disconnected inputs
 * @returns Array of gradient tensors, one per input
 */
export function grad(
  outputs: Tensor | Tensor[],
  inputs: Tensor | Tensor[],
  grad_outputs?: Tensor | Tensor[],
  retain_graph: boolean = false,
  create_graph: boolean = false,
  allow_unused: boolean = false,
): Tensor[] {
  if (!is_grad_enabled()) {
    throw new Error('grad() requires gradients to be enabled. Use torch.enable_grad() or remove no_grad().');
  }

  const outputList = Array.isArray(outputs) ? outputs : [outputs];
  const inputList = Array.isArray(inputs) ? inputs : [inputs];

  // Sum outputs into a single scalar
  let total = outputList[0];
  for (let i = 1; i < outputList.length; i++) {
    total = total.add(outputList[i]);
  }

  // Use provided grad_outputs or default to ones
  let gradOut: Tensor;
  if (grad_outputs) {
    const gradList = Array.isArray(grad_outputs) ? grad_outputs : [grad_outputs];
    gradOut = gradList.length === 1 ? gradList[0] : gradList.reduce((a, b) => a.add(b));
  } else {
    gradOut = total.ones_like();
  }

  // Backward pass populates .grad on leaf tensors
  total.backward(gradOut, retain_graph, create_graph);

  // Collect gradients
  const result: (Tensor | null)[] = [];
  for (const inp of inputList) {
    if (inp.grad !== undefined && inp.grad !== null) {
      result.push(inp.grad.clone());
    } else if (allow_unused) {
      result.push(null as any);
    } else {
      throw new Error(
        `One of the differentiated tensors does not have a gradient. ` +
        `This likely means it is not connected to the output graph. ` +
        `Use allow_unused=true to return null for disconnected tensors.`
      );
    }
  }

  return result as Tensor[];
}

/**
 * Compute the vector-Jacobian product.
 * @pytorch torch.autograd.functional.vjp
 */
export function vjp(
  func: (x: Tensor) => Tensor,
  inputs: Tensor,
  v: Tensor,
  create_graph: boolean = false,
): [Tensor, Tensor] {
  const output = func(inputs);
  const grads = grad(output, inputs, v, false, create_graph);
  return [output, grads[0]];
}

/**
 * Custom autograd function base class.
 * @pytorch torch.autograd.Function
 *
 * @example
 * ```ts
 * class MyFn extends Function {
 *   static forward(ctx, x) { ctx.save(x); return x.mul(2); }
 *   static backward(ctx, g) { const [x] = ctx.saved; return g.mul(2); }
 * }
 * const y = MyFn.apply(x);
 * ```
 */
export class Function {
  static forward(ctx: FunctionContext, ...args: any[]): Tensor {
    throw new Error('forward() must be implemented by subclass');
  }

  static backward(ctx: FunctionContext, ...args: any[]): Tensor | Tensor[] {
    throw new Error('backward() must be implemented by subclass');
  }

  static apply(...args: any[]): Tensor {
    const ctx = new FunctionContext();
    const result = this.forward(ctx, ...args);
    return result;
  }
}

/**
 * Context for saving data during forward pass of custom Functions.
 */
export class FunctionContext {
  public saved_tensors: Tensor[] = [];
  public saved_data: Record<string, any> = {};

  save_for_backward(...tensors: Tensor[]): void {
    this.saved_tensors = tensors;
  }

  mark_dirty(...tensors: Tensor[]): void {}
  mark_non_differentiable(...tensors: Tensor[]): void {}
}

export default {
  grad,
  vjp,
  Function,
};
