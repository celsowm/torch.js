/**
 * Transformer modules for neural networks.
 * @status partial
 * @pytorch torch.nn.transformer
 */

import { Tensor } from '../tensor';
import { Module } from '../nn/module';
import { Parameter } from '../nn/parameter';
import { Linear } from '../nn/linear';
import { LayerNorm } from '../nn/layernorm';
import { Dropout } from '../nn/dropout';
import { MultiheadAttention } from '../nn/attention';
import * as F from '../nn/functional';

/**
 * Transformer Encoder following PyTorch's nn.TransformerEncoder API.
 * 
 * @example
 * ```typescript
 * const encoderLayer = new TransformerEncoderLayer({ d_model: 512, nhead: 8 });
 * const transformerEncoder = new TransformerEncoder({ encoderLayer, numLayers: 6 });
 * const output = transformerEncoder.forward(src);
 * ```
 */
export class TransformerEncoder extends Module {
  private layers: Module[];
  private norm: LayerNorm | null;

  constructor({
    encoderLayer,
    numLayers,
    norm = null,
  }: {
    encoderLayer: TransformerEncoderLayer | TransformerEncoderLayerParams;
    numLayers: number;
    norm?: LayerNorm | null;
  }) {
    super();

    // If given params, create layers
    if (!(encoderLayer instanceof TransformerEncoderLayer)) {
      this.layers = Array.from({ length: numLayers }, (_, i) =>
        new TransformerEncoderLayer(encoderLayer)
      );
    } else {
      this.layers = Array.from({ length: numLayers }, () => encoderLayer);
    }
    this.norm = norm;
  }

  forward(
    src: Tensor,
    mask?: Tensor,
    srcKeyPaddingMask?: Tensor,
    isCausal: boolean = false,
  ): Tensor {
    let output = src;
    
    for (const layer of this.layers) {
      if (layer instanceof TransformerEncoderLayer) {
        output = layer.forward(output, mask, srcKeyPaddingMask, isCausal);
      } else {
        output = layer.forward(output);
      }
    }

    if (this.norm) {
      output = this.norm.forward(output);
    }

    return output;
  }
}

/**
 * Parameters for creating a TransformerEncoderLayer.
 */
export interface TransformerEncoderLayerParams {
  d_model: number;
  nhead: number;
  dimFeedforward?: number;
  dropout?: number;
  activation?: 'relu' | 'gelu';
  layerNormEps?: number;
  batchFirst?: boolean;
}

/**
 * Transformer Encoder Layer following PyTorch's nn.TransformerEncoderLayer API.
 * 
 * Architecture:
 * 1. Multi-head self-attention
 * 2. Add & Norm (residual + layer norm)
 * 3. Feed-forward network
 * 4. Add & Norm (residual + layer norm)
 */
export class TransformerEncoderLayer extends Module {
  private selfAttn: MultiheadAttention;
  private linear1: Linear;
  private linear2: Linear;
  private norm1: LayerNorm;
  private norm2: LayerNorm;
  private dropout1: Dropout;
  private dropout2: Dropout;
  private dModel: number;
  private dimFeedforward: number;
  private activation: 'relu' | 'gelu';

  constructor({
    d_model,
    nhead,
    dimFeedforward = 2048,
    dropout = 0.1,
    activation = 'relu',
    layerNormEps = 1e-5,
    batchFirst = true,
  }: TransformerEncoderLayerParams) {
    super();

    this.dModel = d_model;
    this.dimFeedforward = dimFeedforward;
    this.activation = activation;

    // Multi-head self-attention
    this.selfAttn = new MultiheadAttention(
      d_model,
      nhead,
      dropout,
      true, // bias
      batchFirst,
    );

    // Feed-forward network
    this.linear1 = new Linear(d_model, dimFeedforward);
    this.linear2 = new Linear(dimFeedforward, d_model);

    // Layer norms
    this.norm1 = new LayerNorm(d_model, layerNormEps);
    this.norm2 = new LayerNorm(d_model, layerNormEps);

    // Dropouts
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
  }

  forward(
    src: Tensor,
    mask?: Tensor,
    srcKeyPaddingMask?: Tensor,
    isCausal: boolean = false,
  ): Tensor {
    // Self-attention with residual
    const attnOutput = this.selfAttn.forward(src, src, src, mask, false, srcKeyPaddingMask);
    // MultiheadAttention returns [output, attn_weights]
    const attnOut = attnOutput[0];
    let x = src.add(attnOut);
    x = this.norm1.forward(x);

    // Feed-forward with residual
    let ff = this.linear1.forward(x);
    ff = this.activation === 'relu' ? ff.relu() : ff.gelu();
    ff = this.dropout1.forward(ff);
    ff = this.linear2.forward(ff);
    ff = this.dropout2.forward(ff);
    
    x = x.add(ff);
    x = this.norm2.forward(x);

    return x;
  }
}

/**
 * Transformer Decoder following PyTorch's nn.TransformerDecoder API.
 */
export class TransformerDecoder extends Module {
  private layers: Module[];
  private norm: LayerNorm | null;

  constructor({
    decoderLayer,
    numLayers,
    norm = null,
  }: {
    decoderLayer: TransformerDecoderLayer | TransformerDecoderLayerParams;
    numLayers: number;
    norm?: LayerNorm | null;
  }) {
    super();

    if (!(decoderLayer instanceof TransformerDecoderLayer)) {
      this.layers = Array.from({ length: numLayers }, () =>
        new TransformerDecoderLayer(decoderLayer)
      );
    } else {
      this.layers = Array.from({ length: numLayers }, () => decoderLayer);
    }
    this.norm = norm;
  }

  forward(
    tgt: Tensor,
    memory: Tensor,
    tgtMask?: Tensor,
    memoryMask?: Tensor,
    tgtKeyPaddingMask?: Tensor,
    memoryKeyPaddingMask?: Tensor,
  ): Tensor {
    let output = tgt;

    for (const layer of this.layers) {
      if (layer instanceof TransformerDecoderLayer) {
        output = layer.forward(
          output,
          memory,
          tgtMask,
          memoryMask,
          tgtKeyPaddingMask,
          memoryKeyPaddingMask,
        );
      } else {
        output = layer.forward(output);
      }
    }

    if (this.norm) {
      output = this.norm.forward(output);
    }

    return output;
  }
}

/**
 * Parameters for creating a TransformerDecoderLayer.
 */
export interface TransformerDecoderLayerParams {
  d_model: number;
  nhead: number;
  dimFeedforward?: number;
  dropout?: number;
  activation?: 'relu' | 'gelu';
  layerNormEps?: number;
  batchFirst?: boolean;
}

/**
 * Transformer Decoder Layer following PyTorch's nn.TransformerDecoderLayer API.
 * 
 * Architecture:
 * 1. Self-attention on target
 * 2. Cross-attention (target queries, memory keys/values)
 * 3. Feed-forward network
 * Each with residual connections and layer norm
 */
export class TransformerDecoderLayer extends Module {
  private selfAttn: MultiheadAttention;
  private multiheadAttn: MultiheadAttention;
  private linear1: Linear;
  private linear2: Linear;
  private norm1: LayerNorm;
  private norm2: LayerNorm;
  private norm3: LayerNorm;
  private dropout1: Dropout;
  private dropout2: Dropout;
  private dropout3: Dropout;
  private dModel: number;
  private dimFeedforward: number;
  private activation: 'relu' | 'gelu';

  constructor({
    d_model,
    nhead,
    dimFeedforward = 2048,
    dropout = 0.1,
    activation = 'relu',
    layerNormEps = 1e-5,
    batchFirst = true,
  }: TransformerDecoderLayerParams) {
    super();

    this.dModel = d_model;
    this.dimFeedforward = dimFeedforward;
    this.activation = activation;

    // Self-attention
    this.selfAttn = new MultiheadAttention(
      d_model,
      nhead,
      dropout,
      true, // bias
      batchFirst,
    );

    // Cross-attention
    this.multiheadAttn = new MultiheadAttention(
      d_model,
      nhead,
      dropout,
      true, // bias
      batchFirst,
    );

    // Feed-forward
    this.linear1 = new Linear(d_model, dimFeedforward);
    this.linear2 = new Linear(dimFeedforward, d_model);

    // Layer norms
    this.norm1 = new LayerNorm(d_model, layerNormEps);
    this.norm2 = new LayerNorm(d_model, layerNormEps);
    this.norm3 = new LayerNorm(d_model, layerNormEps);

    // Dropouts
    this.dropout1 = new Dropout(dropout);
    this.dropout2 = new Dropout(dropout);
    this.dropout3 = new Dropout(dropout);
  }

  forward(
    tgt: Tensor,
    memory: Tensor,
    tgtMask?: Tensor,
    memoryMask?: Tensor,
    tgtKeyPaddingMask?: Tensor,
    memoryKeyPaddingMask?: Tensor,
  ): Tensor {
    // Self-attention on target
    const selfAttnOutput = this.selfAttn.forward(tgt, tgt, tgt, tgtMask, false, tgtKeyPaddingMask);
    const selfAttnOut = selfAttnOutput[0];
    let x = tgt.add(selfAttnOut);
    x = this.norm1.forward(x);

    // Cross-attention (target attends to memory)
    const crossAttnOutput = this.multiheadAttn.forward(x, memory, memory, memoryMask, false, memoryKeyPaddingMask);
    const crossAttnOut = crossAttnOutput[0];
    x = x.add(crossAttnOut);
    x = this.norm2.forward(x);

    // Feed-forward
    let ff = this.linear1.forward(x);
    ff = this.activation === 'relu' ? ff.relu() : ff.gelu();
    ff = this.dropout1.forward(ff);
    ff = this.linear2.forward(ff);
    ff = this.dropout2.forward(ff);

    x = x.add(ff);
    x = this.norm3.forward(x);

    return x;
  }
}
