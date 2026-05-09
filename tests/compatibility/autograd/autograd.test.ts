import { describe, it, expect, beforeAll } from 'vitest';
import * as torchModule from '../../../src/index';
// Get the actual torch object - in ESM, import * gives namespace, need to access default or named export
const torch = (torchModule as any).torch || (torchModule as any).default || torchModule;

describe('torch.autograd', () => {
  beforeAll(async () => {
    await torch.init();
  });

  describe('torch.autograd.grad', () => {
    it('computes gradient of scalar output w.r.t. input', async () => {
      const x = torch.tensor([2.0, 3.0], { requires_grad: true });
      const y = x.pow(2).sum();
      const [grad] = torch.autograd.grad(y, x);
      const data = Array.from(await grad.toArray());
      // dy/dx = 2x => [4, 6]
      expect(data[0]).toBeCloseTo(4.0, 4);
      expect(data[1]).toBeCloseTo(6.0, 4);
    });

    it('computes gradient with custom grad_outputs', async () => {
      const x = torch.tensor([2.0, 3.0], { requires_grad: true });
      const y = x.pow(2);
      const gradOut = torch.tensor([1.0, 2.0]);
      const [grad] = torch.autograd.grad(y, x, gradOut);
      const data = Array.from(await grad.toArray());
      // dy/dx * gradOut = 2x * [1, 2] => [4, 12]
      expect(data[0]).toBeCloseTo(4.0, 4);
      expect(data[1]).toBeCloseTo(12.0, 4);
    });

    it('computes gradients w.r.t. multiple inputs', async () => {
      const x = torch.tensor([1.0, 2.0], { requires_grad: true });
      const w = torch.tensor([3.0, 4.0], { requires_grad: true });
      const y = x.mul(w).sum();
      const [gradX, gradW] = torch.autograd.grad(y, [x, w]);
      const dataX = Array.from(await gradX.toArray());
      const dataW = Array.from(await gradW.toArray());
      // dy/dx = w => [3, 4], dy/dw = x => [1, 2]
      expect(dataX[0]).toBeCloseTo(3.0, 4);
      expect(dataX[1]).toBeCloseTo(4.0, 4);
      expect(dataW[0]).toBeCloseTo(1.0, 4);
      expect(dataW[1]).toBeCloseTo(2.0, 4);
    });

    it('computes second-order gradients with create_graph', async () => {
      const x = torch.tensor([2.0], { requires_grad: true });
      const y = x.pow(3);
      const [grad1] = torch.autograd.grad(y, x, torch.ones_like(y), true, true);
      const [grad2] = torch.autograd.grad(grad1, x, torch.ones_like(grad1));
      const data = Array.from(await grad2.toArray());
      // dy/dx = 3x^2, d2y/dx2 = 6x => 12
      expect(data[0]).toBeCloseTo(12.0, 4);
    });
  });

  describe('torch.autograd.vjp', () => {
    it('computes vector-Jacobian product for simple function', async () => {
      const x = torch.tensor([1.0, 2.0], { requires_grad: true });
      const v = torch.tensor([1.0, 1.0]);
      const [output, vjpResult] = torch.autograd.vjp(
        (t) => t.pow(2),
        x,
        v,
      );
      const outputData = Array.from(await output.toArray());
      const vjpData = Array.from(await vjpResult.toArray());
      // f(x) = x^2 => output = [1, 4]
      // J = diag(2x) = [2, 4], v^T J = [2, 4]
      expect(outputData[0]).toBeCloseTo(1.0, 4);
      expect(outputData[1]).toBeCloseTo(4.0, 4);
      expect(vjpData[0]).toBeCloseTo(2.0, 4);
      expect(vjpData[1]).toBeCloseTo(4.0, 4);
    });

    it('computes VJP for linear function', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0], { requires_grad: true });
      const v = torch.tensor([1.0, 1.0]);
      // f(x) = [x0 + x1, x1 + x2]
      const [output, vjpResult] = torch.autograd.vjp(
        (t) => torch.cat([t.slice([{start: 0, stop: 2}]).sum(-1, true), t.slice([{start: 1, stop: 3}]).sum(-1, true)]),
        x,
        v,
      );
      const vjpData = Array.from(await vjpResult.toArray());
      // J = [[1,1,0],[0,1,1]], v^T J = [1, 2, 1]
      expect(vjpData[0]).toBeCloseTo(1.0, 4);
      expect(vjpData[1]).toBeCloseTo(2.0, 4);
      expect(vjpData[2]).toBeCloseTo(1.0, 4);
    });

    it('computes VJP with create_graph=true', async () => {
      const x = torch.tensor([2.0], { requires_grad: true });
      const v = torch.tensor([1.0]);
      const [output, vjpResult] = torch.autograd.vjp(
        (t) => t.pow(3),
        x,
        v,
        true,
      );
      const vjpData = Array.from(await vjpResult.toArray());
      // f(x)=x^3, J=3x^2=12, v^T*J = 12
      expect(vjpData[0]).toBeCloseTo(12.0, 4);
    });
  });

  describe('torch.autograd.jvp', () => {
    it('computes Jacobian-vector product for simple function', async () => {
      const x = torch.tensor([1.0, 2.0], { requires_grad: true });
      const v = torch.tensor([3.0, 4.0]);
      // JVP for f(x) = x^2: J = diag(2x) = [2, 4], J*v = [6, 16]
      const [output, jvpResult] = await torch.autograd.jvp(
        (t) => t.pow(2),
        x,
        v,
      );
      const outputData = Array.from(await output.toArray());
      const jvpData = Array.from(await jvpResult.toArray());
      expect(outputData[0]).toBeCloseTo(1.0, 4);
      expect(outputData[1]).toBeCloseTo(4.0, 4);
      expect(jvpData[0]).toBeCloseTo(6.0, 4);
      expect(jvpData[1]).toBeCloseTo(16.0, 4);
    });

    it('computes JVP for sum function', async () => {
      const x = torch.tensor([1.0, 2.0, 3.0], { requires_grad: true });
      const v = torch.tensor([1.0, 1.0, 1.0]);
      const [output, jvpResult] = await torch.autograd.jvp(
        (t) => t.sum(-1, true),
        x,
        v,
      );
      const jvpData = Array.from(await jvpResult.toArray());
      // f(x) = sum(x), J = [1,1,1], J*v = 3
      expect(jvpData[0]).toBeCloseTo(3.0, 4);
    });

    it('computes JVP for scalar output', async () => {
      const x = torch.tensor([2.0, 3.0], { requires_grad: true });
      const v = torch.tensor([1.0, 0.0]);
      const [output, jvpResult] = await torch.autograd.jvp(
        (t) => t.pow(2).sum(-1, true),
        x,
        v,
      );
      const jvpData = Array.from(await jvpResult.toArray());
      // f(x)=x0^2+x1^2, df/dx=[2x0,2x1]=[4,6], J*v = 4*1+6*0 = 4
      expect(jvpData[0]).toBeCloseTo(4.0, 4);
    });
  });

  describe('torch.autograd.Function', () => {
    it('custom Function with simple forward and backward', async () => {
      class DoubleFn extends torch.autograd.Function {
        static forward(ctx: any, x: any) {
          ctx.save_for_backward(x);
          return x.mul(2);
        }
        static backward(ctx: any, grad_output: any) {
          return grad_output.mul(2);
        }
      }

      const x = torch.tensor([1.0, 2.0, 3.0], { requires_grad: true });
      const y = DoubleFn.apply(x);
      const loss = y.sum();
      loss.backward();
      const gradData = Array.from(await x.grad!.toArray());
      // dy/dx = 2 for each element, d(loss)/dy = 1, so grad = 2
      expect(gradData[0]).toBeCloseTo(2.0, 4);
      expect(gradData[1]).toBeCloseTo(2.0, 4);
      expect(gradData[2]).toBeCloseTo(2.0, 4);
    });

    it('custom Function for ReLU-like operation', async () => {
      class ReLUFn extends torch.autograd.Function {
        static forward(ctx: any, x: any) {
          ctx.save_for_backward(x);
          return x.clamp_min(0);
        }
        static backward(ctx: any, grad_output: any) {
          const [x] = ctx.saved_tensors;
          const mask = x.gt(0);
          return grad_output.mul(mask);
        }
      }

      const x = torch.tensor([-1.0, 0.0, 2.0, 5.0], { requires_grad: true });
      const y = ReLUFn.apply(x);
      const loss = y.sum();
      loss.backward();
      const gradData = Array.from(await x.grad!.toArray());
      // gradient is 1 where x > 0, else 0
      expect(gradData[0]).toBeCloseTo(0.0, 4);
      expect(gradData[1]).toBeCloseTo(0.0, 4);
      expect(gradData[2]).toBeCloseTo(1.0, 4);
      expect(gradData[3]).toBeCloseTo(1.0, 4);
    });

    it('custom Function with multiple inputs', async () => {
      class MulAddFn extends torch.autograd.Function {
        static forward(ctx: any, x: any, y: any) {
          ctx.save_for_backward(x, y);
          return x.mul(y).add(x);
        }
        static backward(ctx: any, grad_output: any) {
          const [x, y] = ctx.saved_tensors;
          // d/dx = y + 1, d/dy = x
          const gradX = grad_output.mul(y.add(1));
          const gradY = grad_output.mul(x);
          return [gradX, gradY];
        }
      }

      const x = torch.tensor([2.0, 3.0], { requires_grad: true });
      const y = torch.tensor([4.0, 5.0], { requires_grad: true });
      const z = MulAddFn.apply(x, y);
      const loss = z.sum();
      loss.backward();
      const gradX = Array.from(await x.grad!.toArray());
      const gradY = Array.from(await y.grad!.toArray());
      // d/dx = y+1 = [5,6], d/dy = x = [2,3]
      expect(gradX[0]).toBeCloseTo(5.0, 4);
      expect(gradX[1]).toBeCloseTo(6.0, 4);
      expect(gradY[0]).toBeCloseTo(2.0, 4);
      expect(gradY[1]).toBeCloseTo(3.0, 4);
    });
  });

  describe('torch.autograd.gradcheck', () => {
    it('verifies analytical gradients match numerical for linear function', async () => {
      const x = torch.tensor([2.0, 3.0], { requires_grad: true });
      const eps = 1e-3;
      // Numerical gradient via finite differences
      const x0 = Array.from(await x.toArray());
      const f0 = Array.from(await x.pow(2).sum(-1, true).toArray());
      const numericalGrad: number[] = [];
      for (let i = 0; i < 2; i++) {
        const xPlus = torch.tensor([...x0], { requires_grad: false });
        const plusData = Array.from(await xPlus.toArray()) as number[];
        plusData[i] += eps;
        const xPerturbed = torch.tensor(plusData, { requires_grad: false });
        const fPlus = Array.from(await xPerturbed.pow(2).sum(-1, true).toArray());
        numericalGrad.push((fPlus[0] - f0[0]) / eps);
      }
      // Analytical gradient
      const y = x.pow(2).sum(-1, true);
      y.backward(torch.ones_like(y));
      const analyticalGrad = Array.from(await x.grad!.toArray());
      // Compare
      for (let i = 0; i < 2; i++) {
        expect(analyticalGrad[i]).toBeCloseTo(numericalGrad[i], 2);
      }
    });

    it('verifies gradients for matrix multiplication', async () => {
      const A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], { requires_grad: true });
      const B = torch.tensor([[5.0], [6.0]]);
      const eps = 1e-3;
      const y = torch.matmul(A, B);
      const loss = y.sum();
      loss.backward();
      const analyticalGrad = Array.from(await A.grad!.toArray());
      // Numerical check on A[0,0]
      const AData = [[1.0, 2.0], [3.0, 4.0]];
      AData[0][0] += eps;
      const Ap = torch.tensor(AData.flat(), { requires_grad: false }).reshape([2, 2]);
      const yp = torch.matmul(Ap, B);
      const lossP = Array.from(await yp.sum().toArray());
      const loss0 = Array.from(await y.sum().toArray());
      const numerical = (lossP[0] - loss0[0]) / eps;
      expect(analyticalGrad[0]).toBeCloseTo(numerical, 2);
    });

    it('verifies gradients through composition of operations', async () => {
      const x = torch.tensor([1.0, 2.0], { requires_grad: true });
      const eps = 1e-3;
      // f(x) = sum(exp(x))
      const y = x.exp().sum(-1, true);
      y.backward(torch.ones_like(y));
      const analyticalGrad = Array.from(await x.grad!.toArray());
      // Numerical
      const xData = Array.from(await x.toArray()) as number[];
      const f0 = Array.from(await torch.tensor(xData).exp().sum(-1, true).toArray());
      const numericalGrad: number[] = [];
      for (let i = 0; i < 2; i++) {
        const xp = [...xData];
        xp[i] += eps;
        const fPlus = Array.from(await torch.tensor(xp).exp().sum(-1, true).toArray());
        numericalGrad.push((fPlus[0] - f0[0]) / eps);
      }
      for (let i = 0; i < 2; i++) {
        expect(analyticalGrad[i]).toBeCloseTo(numericalGrad[i], 2);
      }
    });
  });
});
