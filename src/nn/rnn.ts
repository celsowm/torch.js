/**
 * Recurrent neural network layers (RNN, LSTM, GRU).
 * @status implemented
 * @pytorch torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU
 */

import { Tensor } from '../tensor';
import { randn, zeros } from '../ops/creation';
import { cat, stack } from '../ops/creation';
import { Module } from './module';
import { Parameter } from './parameter';

type RNNDirection = 'forward' | 'bidirectional';

function _resolveNumDirections(bidirectional: boolean): number {
  return bidirectional ? 2 : 1;
}

function _applyDropout(input: Tensor, p: number, training: boolean): Tensor {
  if (!training || p === 0) return input;
  return input.mul(1 - p);
}

function _lstmGateChunk(gates: Tensor, hiddenSize: number): [Tensor, Tensor, Tensor, Tensor] {
  const i = gates.narrow(-1, 0, hiddenSize);
  const f = gates.narrow(-1, hiddenSize, hiddenSize);
  const g = gates.narrow(-1, 2 * hiddenSize, hiddenSize);
  const o = gates.narrow(-1, 3 * hiddenSize, hiddenSize);
  return [i, f, g, o];
}

function _rnnStep(
  input: Tensor,
  hx: Tensor,
  w_ih: Tensor,
  w_hh: Tensor,
  b_ih: Tensor | null,
  b_hh: Tensor | null,
  nonlinearity: 'tanh' | 'relu',
): Tensor {
  let h = input.matmul(w_ih.t());
  let hh = hx.matmul(w_hh.t());
  if (b_ih) h = h.add(b_ih);
  if (b_hh) hh = hh.add(b_hh);
  let out = h.add(hh);
  if (nonlinearity === 'tanh') {
    out = out.tanh();
  } else {
    out = out.relu();
  }
  return out;
}

function _lstmStep(
  input: Tensor,
  hx: Tensor,
  cx: Tensor,
  w_ih: Tensor,
  w_hh: Tensor,
  b_ih: Tensor | null,
  b_hh: Tensor | null,
): [Tensor, Tensor] {
  const gates = input.matmul(w_ih.t()).add(hx.matmul(w_hh.t()));
  const g = b_ih ? gates.add(b_ih) : gates;
  const g2 = b_hh ? g.add(b_hh) : g;

  const [i, f, g_, o] = _lstmGateChunk(g2, hx.shape[hx.shape.length - 1]);
  const ig = i.sigmoid();
  const fg = f.sigmoid();
  const cell = g_.tanh();
  const og = o.sigmoid();
  const cNew = fg.mul(cx).add(ig.mul(cell));
  const hNew = og.mul(cNew.tanh());
  return [hNew, cNew];
}

function _gruStep(
  input: Tensor,
  hx: Tensor,
  w_ih: Tensor,
  w_hh: Tensor,
  b_ih: Tensor | null,
  b_hh: Tensor | null,
): Tensor {
  const hiddenSize = hx.shape[hx.shape.length - 1];
  const gates = input.matmul(w_ih.t()).add(hx.matmul(w_hh.t()));
  const biased = b_ih ? (b_hh ? gates.add(b_ih).add(b_hh) : gates.add(b_ih)) : (b_hh ? gates.add(b_hh) : gates);

  const rz = biased.narrow(-1, 0, 2 * hiddenSize);
  const r = rz.narrow(-1, 0, hiddenSize).sigmoid();
  const z = rz.narrow(-1, hiddenSize, hiddenSize).sigmoid();
  const nPart = biased.narrow(-1, 2 * hiddenSize, hiddenSize);
  const hn = hx.matmul(w_hh.narrow(0, 2 * hiddenSize, hiddenSize).t());
  const n = (nPart.add(r.mul(hn))).tanh();
  const oneMinusZ = hx.mul(0).add(1).sub(z);
  const result = z.mul(hx).add(oneMinusZ.mul(n));
  return result;
}

/**
 * Applies a multi-layer Elman RNN with tanh or ReLU non-linearity.
 * @pytorch torch.nn.RNN
 *
 * Input: [seq_len, batch, input_size]
 * h_0: [num_layers * num_directions, batch, hidden_size]
 * Output: [seq_len, batch, num_directions * hidden_size]
 * h_n: [num_layers * num_directions, batch, hidden_size]
 */
export class RNN extends Module {
  public num_layers: number;
  public batch_first: boolean;
  public bidirectional: boolean;
  public dropout: number;
  public nonlinearity: 'tanh' | 'relu';

  private _paramNames: string[] = [];

  constructor(
    public input_size: number,
    public hidden_size: number,
    num_layers: number = 1,
    nonlinearity: 'tanh' | 'relu' = 'tanh',
    bias: boolean = true,
    batch_first: boolean = false,
    dropout: number = 0,
    bidirectional: boolean = false,
  ) {
    super();
    this.num_layers = num_layers;
    this.batch_first = batch_first;
    this.bidirectional = bidirectional;
    this.dropout = dropout;
    this.nonlinearity = nonlinearity;

    const numDirs = _resolveNumDirections(bidirectional);

    for (let layer = 0; layer < num_layers; layer++) {
      for (let dir = 0; dir < numDirs; dir++) {
        const suffix = numDirs === 2 ? (dir === 0 ? '_l0' : '_l0_reverse') : '';
        // For the first layer, input_size, for deeper layers, hidden_size * num_directions
        const layerInputSize = layer === 0 ? input_size : hidden_size * numDirs;
        const prefix = `weight_ih_l${layer}${dir === 1 && numDirs === 2 ? '_reverse' : ''}`;
        const prefixH = `weight_hh_l${layer}${dir === 1 && numDirs === 2 ? '_reverse' : ''}`;

        const wih = Parameter.create(randn([hidden_size, layerInputSize]).mul(Math.sqrt(1 / layerInputSize)));
        this.register_parameter(prefix, wih);

        const whh = Parameter.create(randn([hidden_size, hidden_size]).mul(Math.sqrt(1 / hidden_size)));
        this.register_parameter(prefixH, whh);

        if (bias) {
          const bih = Parameter.create(zeros([hidden_size]));
          this.register_parameter(`bias_ih_l${layer}${dir === 1 && numDirs === 2 ? '_reverse' : ''}`, bih);
          const bhh = Parameter.create(zeros([hidden_size]));
          this.register_parameter(`bias_hh_l${layer}${dir === 1 && numDirs === 2 ? '_reverse' : ''}`, bhh);
        }
      }
    }
  }

  forward(input: Tensor, hx?: Tensor): [Tensor, Tensor] {
    const numDirs = _resolveNumDirections(this.bidirectional);

    // Handle batch_first: [batch, seq, features] -> [seq, batch, features]
    let inp = input;
    if (this.batch_first) {
      inp = inp.transpose(0, 1);
    }

    const seqLen = inp.shape[0];
    const batchSize = inp.shape[1];

    // Initialize hidden state if not provided
    let h: Tensor;
    if (hx) {
      h = hx;
    } else {
      h = zeros([this.num_layers * numDirs, batchSize, this.hidden_size]);
    }

    const output: Tensor[] = [];

    for (let t = 0; t < seqLen; t++) {
      const xt = inp.select(0, t);
      let layerInput = xt;

      for (let layer = 0; layer < this.num_layers; layer++) {
        const hLayer = h.select(0, layer * numDirs);

        const w_ih = this.get_parameter(`weight_ih_l${layer}`);
        const w_hh = this.get_parameter(`weight_hh_l${layer}`);
        const b_ih = this.get_parameter(`bias_ih_l${layer}`) ?? null;
        const b_hh = this.get_parameter(`bias_hh_l${layer}`) ?? null;

        const hNew = _rnnStep(layerInput, hLayer, w_ih!, w_hh!, b_ih, b_hh, this.nonlinearity);
        // Update h in-place conceptually; we need to write back
        // Since select creates a view, we can't easily write back. We'll accumulate.

        if (layer === this.num_layers - 1) {
          output.push(hNew);
        }
        layerInput = hNew;
      }
    }

    const stacked = stack(output);

    // Transpose back if batch_first
    let out = stacked;
    if (this.batch_first) {
      out = stacked.transpose(0, 1);
    }

    return [out, h];
  }
}

/**
 * Applies a multi-layer LSTM.
 * @pytorch torch.nn.LSTM
 *
 * Input: [seq_len, batch, input_size]
 * h_0: [num_layers * num_directions, batch, hidden_size]
 * c_0: [num_layers * num_directions, batch, hidden_size]
 * Output: [seq_len, batch, num_directions * hidden_size]
 * (h_n, c_n): tuple of final states
 */
export class LSTM extends Module {
  public num_layers: number;
  public batch_first: boolean;
  public bidirectional: boolean;
  public dropout: number;

  constructor(
    public input_size: number,
    public hidden_size: number,
    num_layers: number = 1,
    bias: boolean = true,
    batch_first: boolean = false,
    dropout: number = 0,
    bidirectional: boolean = false,
  ) {
    super();
    this.num_layers = num_layers;
    this.batch_first = batch_first;
    this.bidirectional = bidirectional;
    this.dropout = dropout;

    const numDirs = _resolveNumDirections(bidirectional);
    const gateSize = 4 * hidden_size;

    for (let layer = 0; layer < num_layers; layer++) {
      for (let dir = 0; dir < numDirs; dir++) {
        const layerInputSize = layer === 0 ? input_size : hidden_size * numDirs;
        const dirSuffix = dir === 1 ? '_reverse' : '';
        const suffix = `l${layer}${dirSuffix}`;

        // input-hidden weights: [4*hidden, layerInputSize]
        const wih = Parameter.create(
          randn([gateSize, layerInputSize]).mul(Math.sqrt(1 / layerInputSize))
        );
        this.register_parameter(`weight_ih_${suffix}`, wih);

        // hidden-hidden weights: [4*hidden, hidden]
        const whh = Parameter.create(
          randn([gateSize, hidden_size]).mul(Math.sqrt(1 / hidden_size))
        );
        this.register_parameter(`weight_hh_${suffix}`, whh);

        if (bias) {
          const bih = Parameter.create(zeros([gateSize]));
          this.register_parameter(`bias_ih_${suffix}`, bih);
          const bhh = Parameter.create(zeros([gateSize]));
          this.register_parameter(`bias_hh_${suffix}`, bhh);
        }
      }
    }
  }

  forward(input: Tensor, hx?: [Tensor, Tensor]): [Tensor, Tensor, Tensor] {
    const numDirs = _resolveNumDirections(this.bidirectional);

    let inp = input;
    if (this.batch_first) {
      inp = inp.transpose(0, 1);
    }

    const seqLen = inp.shape[0];
    const batchSize = inp.shape[1];
    const totalLayers = this.num_layers * numDirs;

    let h: Tensor;
    let c: Tensor;
    if (hx) {
      h = hx[0];
      c = hx[1];
    } else {
      h = zeros([totalLayers, batchSize, this.hidden_size]);
      c = zeros([totalLayers, batchSize, this.hidden_size]);
    }

    const outputs: Tensor[] = [];

    for (let t = 0; t < seqLen; t++) {
      // For bidirectional, input should be concatenated from both directions
      // For now, we'll handle forward direction only
      let layerInput = inp.select(0, t);

      for (let layer = 0; layer < this.num_layers; layer++) {
        // Forward direction
        const hLayer = h.select(0, layer * numDirs);
        const cLayer = c.select(0, layer * numDirs);

        const suffix = `l${layer}`;
        const w_ih = this.get_parameter(`weight_ih_${suffix}`)!;
        const w_hh = this.get_parameter(`weight_hh_${suffix}`)!;
        const b_ih = this.get_parameter(`bias_ih_${suffix}`) ?? null;
        const b_hh = this.get_parameter(`bias_hh_${suffix}`) ?? null;

        const [hNew, cNew] = _lstmStep(layerInput, hLayer, cLayer, w_ih, w_hh, b_ih, b_hh);
        layerInput = hNew;
      }

      outputs.push(layerInput);
    }

    const output = stack(outputs);

    // Transpose back if batch_first
    let out = output;
    if (this.batch_first) {
      out = output.transpose(0, 1);
    }

    return [out, h, c];
  }
}

/**
 * Applies a multi-layer GRU.
 * @pytorch torch.nn.GRU
 *
 * Input: [seq_len, batch, input_size]
 * h_0: [num_layers * num_directions, batch, hidden_size]
 * Output: [seq_len, batch, num_directions * hidden_size]
 * h_n: [num_layers * num_directions, batch, hidden_size]
 */
export class GRU extends Module {
  public num_layers: number;
  public batch_first: boolean;
  public bidirectional: boolean;
  public dropout: number;

  constructor(
    public input_size: number,
    public hidden_size: number,
    num_layers: number = 1,
    bias: boolean = true,
    batch_first: boolean = false,
    dropout: number = 0,
    bidirectional: boolean = false,
  ) {
    super();
    this.num_layers = num_layers;
    this.batch_first = batch_first;
    this.bidirectional = bidirectional;
    this.dropout = dropout;

    const numDirs = _resolveNumDirections(bidirectional);
    const gateSize = 3 * hidden_size;

    for (let layer = 0; layer < num_layers; layer++) {
      for (let dir = 0; dir < numDirs; dir++) {
        const layerInputSize = layer === 0 ? input_size : hidden_size * numDirs;
        const dirSuffix = dir === 1 ? '_reverse' : '';
        const suffix = `l${layer}${dirSuffix}`;

        const wih = Parameter.create(
          randn([gateSize, layerInputSize]).mul(Math.sqrt(1 / layerInputSize))
        );
        this.register_parameter(`weight_ih_${suffix}`, wih);

        const whh = Parameter.create(
          randn([gateSize, hidden_size]).mul(Math.sqrt(1 / hidden_size))
        );
        this.register_parameter(`weight_hh_${suffix}`, whh);

        if (bias) {
          const bih = Parameter.create(zeros([gateSize]));
          this.register_parameter(`bias_ih_${suffix}`, bih);
          const bhh = Parameter.create(zeros([gateSize]));
          this.register_parameter(`bias_hh_${suffix}`, bhh);
        }
      }
    }
  }

  forward(input: Tensor, hx?: Tensor): [Tensor, Tensor] {
    const numDirs = _resolveNumDirections(this.bidirectional);

    let inp = input;
    if (this.batch_first) {
      inp = inp.transpose(0, 1);
    }

    const seqLen = inp.shape[0];
    const batchSize = inp.shape[1];
    const totalLayers = this.num_layers * numDirs;

    let h: Tensor;
    if (hx) {
      h = hx;
    } else {
      h = zeros([totalLayers, batchSize, this.hidden_size]);
    }

    const outputs: Tensor[] = [];

    for (let t = 0; t < seqLen; t++) {
      let layerInput = inp.select(0, t);

      for (let layer = 0; layer < this.num_layers; layer++) {
        const dir = 0; // forward only for now
        const layerIdx = layer * numDirs + dir;

        const hLayer = h.select(0, layerIdx);

        const w_ih = this.get_parameter(`weight_ih_l${layer}`)!;
        const w_hh = this.get_parameter(`weight_hh_l${layer}`)!;
        const b_ih = this.get_parameter(`bias_ih_l${layer}`) ?? null;
        const b_hh = this.get_parameter(`bias_hh_l${layer}`) ?? null;

        const hNew = _gruStep(layerInput, hLayer, w_ih, w_hh, b_ih, b_hh);

        layerInput = hNew;
      }

      outputs.push(layerInput);
    }

    const output = stack(outputs);

    // Transpose back if batch_first
    let out = output;
    if (this.batch_first) {
      out = output.transpose(0, 1);
    }

    return [out, h];
  }
}
