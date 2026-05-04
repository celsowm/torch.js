import { describe, it, expect, beforeAll } from 'vitest';
import { createTorch } from '../../../src/create_torch';
import { Module } from '../../../src/nn/module';
import { Parameter } from '../../../src/nn/parameter';
import { Linear } from '../../../src/nn/linear';
import { Sequential } from '../../../src/nn/sequential';
import { Embedding } from '../../../src/nn/embedding';
import { LayerNorm } from '../../../src/nn/layernorm';
import { BatchNorm1d, BatchNorm2d } from '../../../src/nn/batchnorm';
import { Conv1d, Conv2d } from '../../../src/nn/conv';
import { MaxPool2d, AvgPool2d } from '../../../src/nn/pooling';
import { Flatten } from '../../../src/nn/flatten';
import { ModuleList, ModuleDict } from '../../../src/nn/containers';
import { ReLU } from '../../../src/nn/activation';

const torch = createTorch(async () => {}, async () => ({}));

describe('nn.Module', () => {
  beforeAll(async () => { await torch.init(); });

  it('can be subclassed with a forward method', async () => {
    class MyModule extends Module {
      forward(x: any) { return x; }
    }
    const mod = new MyModule();
    const x = torch.tensor([1.0, 2.0, 3.0]);
    expect(mod.forward(x).shape).toEqual([3]);
  });

  it('registers and iterates over parameters', () => {
    class MyModule extends Module {
      weight: Parameter; bias: Parameter;
      constructor() {
        super();
        this.weight = Parameter.create(torch.randn([3, 2]));
        this.register_parameter('weight', this.weight);
        this.bias = Parameter.create(torch.zeros([2]));
        this.register_parameter('bias', this.bias);
      }
      forward(x: any) { return x; }
    }
    expect(Array.from(new MyModule().parameters()).length).toBe(2);
  });

  it('named_parameters includes prefixes for submodules', () => {
    class Inner extends Module {
      w: Parameter;
      constructor() { super(); this.w = Parameter.create(torch.randn([2, 2])); this.register_parameter('w', this.w); }
      forward(x: any) { return x; }
    }
    class Outer extends Module {
      inner: Inner;
      constructor() { super(); this.inner = new Inner(); this.add_module('inner', this.inner); }
      forward(x: any) { return x; }
    }
    const named = Array.from(new Outer().named_parameters());
    expect(named.length).toBe(1);
    expect(named[0][0]).toBe('inner.w');
  });

  it('supports train/eval mode toggling', () => {
    class M extends Module {
      lin: Linear;
      constructor() { super(); this.lin = new Linear(4, 2); this.add_module('lin', this.lin); }
      forward(x: any) { return this.lin.forward(x); }
    }
    const m = new M();
    expect(m.training).toBe(true);
    m.eval(); expect(m.training).toBe(false);
    m.train(); expect(m.training).toBe(true);
  });

  it('propagates training mode to submodules', () => {
    class Child extends Module { forward(x: any) { return x; } }
    class Parent extends Module {
      child: Child;
      constructor() { super(); this.child = new Child(); this.add_module('child', this.child); }
      forward(x: any) { return x; }
    }
    const p = new Parent();
    expect(p.child.training).toBe(true);
    p.eval();
    expect(p.child.training).toBe(false);
  });

  it('supports zero_grad', () => {
    class M extends Module {
      w: Parameter;
      constructor() { super(); this.w = Parameter.create(torch.randn([2, 2])); this.register_parameter('w', this.w); }
      forward(x: any) { return x; }
    }
    const m = new M(); m.zero_grad();
    expect(m.w.grad).toBe(null);
  });

  it('to(device) returns self', () => {
    class M extends Module {
      w: Parameter;
      constructor() { super(); this.w = Parameter.create(torch.randn([2])); this.register_parameter('w', this.w); }
      forward(x: any) { return x; }
    }
    expect(new M().to('webgpu')).toBeDefined();
  });
});

describe('nn.Linear', () => {
  beforeAll(async () => { await torch.init(); });

  it('weight shape [out, in], bias shape [out]', () => {
    const l = new Linear(10, 5);
    expect(l.weight.shape).toEqual([5, 10]);
    expect(l.bias!.shape).toEqual([5]);
  });

  it('no bias when bias=false', () => {
    expect(new Linear(8, 4, false).bias).toBe(null);
  });

  it('forward [batch, in] -> [batch, out]', async () => {
    const l = new Linear(16, 8);
    expect(l.forward(torch.randn([32, 16])).shape).toEqual([32, 8]);
  });

  it('1D input unsqueezed', async () => {
    const l = new Linear(10, 5);
    const out = l.forward(torch.randn([10]));
    expect(out.shape[1]).toBe(5);
  });

  it('parameters include weight and bias', () => {
    expect(Array.from(new Linear(4, 2).parameters()).length).toBe(2);
  });
});

describe('nn.Sequential', () => {
  beforeAll(async () => { await torch.init(); });

  it('forward through chain', async () => {
    const s = new Sequential(new Linear(10, 8), new ReLU(), new Linear(8, 4));
    expect(s.forward(torch.randn([2, 10])).shape).toEqual([2, 4]);
  });

  it('get module by index', () => {
    const s = new Sequential(new Linear(10, 5), new Linear(5, 3));
    expect(s.get(0)).toBeInstanceOf(Linear);
    expect(s.get(1)).toBeInstanceOf(Linear);
    expect(s.length).toBe(2);
  });

  it('collects all parameters', () => {
    expect(Array.from(new Sequential(new Linear(10, 5), new Linear(5, 3)).parameters()).length).toBe(4);
  });

  it('nested sequential', async () => {
    const s = new Sequential(new Sequential(new Linear(8, 4), new ReLU()), new Linear(4, 2));
    expect(s.forward(torch.randn([1, 8])).shape).toEqual([1, 2]);
  });
});

describe('nn.Embedding', () => {
  beforeAll(async () => { await torch.init(); });

  it('weight shape [num, dim]', () => {
    const e = new Embedding(100, 64);
    expect(e.weight.shape).toEqual([100, 64]);
    expect(e.num_embeddings).toBe(100);
    expect(e.embedding_dim).toBe(64);
  });

  it('1D indices -> [N, dim]', async () => {
    const e = new Embedding(50, 32);
    expect(e.forward(torch.tensor([0, 1, 2, 3], { dtype: 'int32' })).shape).toEqual([4, 32]);
  });

  it('2D indices -> [B, N, dim]', async () => {
    const e = new Embedding(20, 16);
    expect(e.forward(torch.tensor([[0, 1], [2, 3]], { dtype: 'int32' })).shape).toEqual([2, 2, 16]);
  });

  it('weight is a parameter', () => {
    const e = new Embedding(10, 8);
    const params = Array.from(e.parameters());
    expect(params.length).toBe(1);
    expect(params[0].shape).toEqual([10, 8]);
  });
});

describe('nn.LayerNorm', () => {
  beforeAll(async () => { await torch.init(); });

  it('weight and bias shape [size]', () => {
    const ln = new LayerNorm(64);
    expect(ln.weight.shape).toEqual([64]);
    expect(ln.bias.shape).toEqual([64]);
    expect(ln.eps).toBe(1e-5);
  });

  it('custom eps', () => { expect(new LayerNorm(32, 1e-3).eps).toBe(1e-3); });

  it('forward preserves shape', async () => {
    expect(new LayerNorm(8).forward(torch.randn([4, 8])).shape).toEqual([4, 8]);
  });

  it('array normalized_shape', () => { expect(new LayerNorm([16]).normalized_shape).toEqual([16]); });

  it('2 parameters', () => { expect(Array.from(new LayerNorm(10).parameters()).length).toBe(2); });
});

describe('nn.BatchNorm1d', () => {
  beforeAll(async () => { await torch.init(); });

  it('creates weight, bias, running_mean, running_var', () => {
    const bn = new BatchNorm1d(16);
    expect(bn.weight.shape).toEqual([16]);
    expect(bn.bias.shape).toEqual([16]);
    expect(bn.running_mean.shape).toEqual([16]);
    expect(bn.running_var.shape).toEqual([16]);
  });

  it('2D input [N, C]', async () => { expect(new BatchNorm1d(8).forward(torch.randn([4, 8])).shape).toEqual([4, 8]); });
  it('3D input [N, C, L]', async () => { expect(new BatchNorm1d(8).forward(torch.randn([2, 8, 16])).shape).toEqual([2, 8, 16]); });

  it('eval mode uses running stats', async () => {
    const bn = new BatchNorm1d(4); bn.eval();
    expect(bn.forward(torch.randn([2, 4])).shape).toEqual([2, 4]);
  });

  it('affine=false', () => {
    const bn = new BatchNorm1d(8, 1e-5, 0.1, false);
    expect(bn.weight).toBe(null); expect(bn.bias).toBe(null);
  });
});

describe('nn.BatchNorm2d', () => {
  beforeAll(async () => { await torch.init(); });
  it('running stats shape', () => {
    const bn = new BatchNorm2d(32);
    expect(bn.running_mean.shape).toEqual([32]);
  });
  it('4D input [N, C, H, W]', async () => {
    expect(new BatchNorm2d(16).forward(torch.randn([2, 16, 8, 8])).shape).toEqual([2, 16, 8, 8]);
  });
});

describe('nn.Conv1d', () => {
  beforeAll(async () => { await torch.init(); });

  it('weight [out, in/g, k], bias [out]', () => {
    const c = new Conv1d(3, 16, 5);
    expect(c.weight.shape).toEqual([16, 3, 5]);
    expect(c.bias!.shape).toEqual([16]);
  });

  it('forward output shape', async () => {
    // out = (32 - 5)/1 + 1 = 28
    expect(new Conv1d(3, 8, 5, 1, 0).forward(torch.randn([2, 3, 32])).shape).toEqual([2, 8, 28]);
  });

  it('padding', async () => {
    // out = (16 + 2 - 3)/1 + 1 = 16
    expect(new Conv1d(3, 8, 3, 1, 1).forward(torch.randn([1, 3, 16])).shape).toEqual([1, 8, 16]);
  });

  it('no bias', () => { expect(new Conv1d(4, 8, 3, 1, 0, 1, 1, false).bias).toBe(null); });
});

describe('nn.Conv2d', () => {
  beforeAll(async () => { await torch.init(); });

  it('weight [out, in/g, kH, kW]', () => {
    expect(new Conv2d(3, 16, [3, 3]).weight.shape).toEqual([16, 3, 3, 3]);
  });

  it('scalar kernel', () => { expect(new Conv2d(3, 8, 5).weight.shape).toEqual([8, 3, 5, 5]); });

  it('forward output shape', async () => {
    expect(new Conv2d(3, 8, 3, 1, 0).forward(torch.randn([2, 3, 32, 32])).shape).toEqual([2, 8, 30, 30]);
  });

  it('stride', async () => {
    expect(new Conv2d(3, 8, 3, 2, 1).forward(torch.randn([1, 3, 32, 32])).shape).toEqual([1, 8, 16, 16]);
  });

  it('dilation', async () => {
    expect(new Conv2d(3, 4, 3, 1, 0, 2).forward(torch.randn([1, 3, 32, 32])).shape).toEqual([1, 4, 28, 28]);
  });

  it('groups', async () => {
    expect(new Conv2d(4, 8, 3, 1, 0, 1, 2).forward(torch.randn([1, 4, 16, 16])).shape).toEqual([1, 8, 14, 14]);
  });
});

describe('nn.MaxPool2d', () => {
  beforeAll(async () => { await torch.init(); });
  it('kernel 2 -> half', async () => { expect(new MaxPool2d(2).forward(torch.randn([2, 3, 32, 32])).shape).toEqual([2, 3, 16, 16]); });
  it('non-square kernel', async () => { expect(new MaxPool2d([3, 2]).forward(torch.randn([1, 2, 24, 24])).shape).toEqual([1, 2, 8, 12]); });
  it('stride', async () => { expect(new MaxPool2d(3, 2).forward(torch.randn([1, 1, 10, 10])).shape).toEqual([1, 1, 4, 4]); });
  it('padding', async () => { expect(new MaxPool2d(2, 2, 1).forward(torch.randn([1, 1, 8, 8])).shape).toEqual([1, 1, 4, 4]); });
});

describe('nn.AvgPool2d', () => {
  beforeAll(async () => { await torch.init(); });
  it('kernel 2 -> half', async () => { expect(new AvgPool2d(2).forward(torch.randn([2, 3, 32, 32])).shape).toEqual([2, 3, 16, 16]); });
  it('stride', async () => { expect(new AvgPool2d(3, 2).forward(torch.randn([1, 1, 10, 10])).shape).toEqual([1, 1, 4, 4]); });
});

describe('nn.Flatten', () => {
  beforeAll(async () => { await torch.init(); });
  it('default start_dim=1', async () => { expect(new Flatten().forward(torch.randn([2, 3, 4, 5])).shape).toEqual([2, 60]); });
  it('start_dim=0', async () => { expect(new Flatten(0).forward(torch.randn([2, 3, 4])).shape).toEqual([24]); });
  it('end_dim', async () => { expect(new Flatten(1, 2).forward(torch.randn([2, 3, 4, 5])).shape).toEqual([2, 12, 5]); });
});

describe('nn.ModuleList', () => {
  beforeAll(async () => { await torch.init(); });
  it('length and iteration', () => {
    const ml = new ModuleList([new Linear(10, 5), new Linear(5, 3), new Linear(3, 1)]);
    expect(ml.length).toBe(3);
    expect(Array.from(ml).length).toBe(3);
  });
  it('get by index', () => {
    const ml = new ModuleList([new ReLU(), new ReLU()]);
    expect(ml.get(0)).toBeInstanceOf(ReLU);
  });
  it('append', () => {
    const ml = new ModuleList();
    ml.append(new Linear(4, 2)).append(new Linear(2, 1));
    expect(ml.length).toBe(2);
  });
  it('parameters', () => {
    expect(Array.from(new ModuleList([new Linear(4, 2), new Linear(2, 1)]).parameters()).length).toBe(4);
  });
});

describe('nn.ModuleDict', () => {
  beforeAll(async () => { await torch.init(); });
  it('get and size', () => {
    const md = new ModuleDict({ l1: new Linear(10, 5), l2: new Linear(5, 3) });
    expect(md.get('l1')).toBeInstanceOf(Linear);
    expect(md.size).toBe(2);
  });
  it('set and has', () => {
    const md = new ModuleDict();
    md.set('r', new ReLU());
    expect(md.has('r')).toBe(true);
  });
  it('keys/values/entries', () => {
    const md = new ModuleDict({ a: new Linear(4, 2), b: new Linear(2, 1) });
    expect(Array.from(md.keys())).toEqual(['a', 'b']);
  });
  it('parameters', () => {
    expect(Array.from(new ModuleDict({ l1: new Linear(4, 2), l2: new Linear(2, 1) }).parameters()).length).toBe(4);
  });
});
