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
 * Compute the Jacobian-vector product.
 * JVP computes J @ v where J is the Jacobian of func w.r.t. inputs.
 * @pytorch torch.autograd.functional.jvp
 */
export async function jvp(
  func: (x: Tensor) => Tensor,
  inputs: Tensor,
  v: Tensor,
  create_graph: boolean = false,
): Promise<[Tensor, Tensor]> {
  const output = func(inputs);
  const outShape = output.shape;
  const outNumel = outShape.reduce((a, b) => a * b, 1);

  if (outNumel === 1) {
    // Scalar output: JVP = grad(output) * v[0]
    const [gradResult] = grad(output, inputs, undefined, false, create_graph);
    const vData = await v.toArray();
    const vScalar = vData[0] || 1;
    const jvpResult = gradResult.mul(vScalar);
    return [output, jvpResult];
  }

  // Vector output: JVP = sum over i of (v_i * grad(output_i))
  const outputFlat = output.reshape([outNumel]);
  const vData = await v.toArray();
  
  let result: Tensor | null = null;
  for (let i = 0; i < outNumel; i++) {
    const out_i = outputFlat.select(0, i);
    const v_i = vData[i] || 0;
    if (v_i === 0) continue;
    const [grad_i] = grad(out_i, inputs, undefined, true, create_graph);
    const weighted = grad_i.mul(v_i);
    if (result === null) {
      result = weighted;
    } else {
      result = result.add(weighted);
    }
  }

  return [output, result || inputs.zeros_like()];
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

    // Find input tensors from args
    const inputTensors: Tensor[] = args.filter((a): a is Tensor => 
      a && typeof a === 'object' && 'shape' in a
    );

    // Attach grad_fn if any inputs require grad
    const requiresGrad = inputTensors.some(t => t.requires_grad);
    if (requiresGrad) {
      (result as any)._grad_fn = {
        backward: (gradOutput: any) => {
          const gradInputs = this.backward(ctx, gradOutput);
          const gradArray = Array.isArray(gradInputs) ? gradInputs : [gradInputs];
          for (let i = 0; i < Math.min(inputTensors.length, gradArray.length); i++) {
            if (inputTensors[i].requires_grad && gradArray[i]) {
              (inputTensors[i] as any).accumulateGrad(gradArray[i]);
            }
          }
        },
        _next_tensors: inputTensors,
      };
    }

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
  jvp,
  Function,
};
