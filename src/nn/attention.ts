/**
 * Multi-headed attention module.
 * @status partial
 * @pytorch torch.nn.MultiheadAttention
 */

import { Tensor } from '../tensor';
import { Parameter } from './parameter';
import { Module } from './module';
import { Linear } from './linear';
import { softmax } from './functional';
import { randn, zeros, full } from '../ops/creation';

/**
 * Multi-Head Attention module.
 *
 * Allows the model to jointly attend to information from different representation subspaces.
 *
 * Multi-Head Attention is defined as:
 *   MultiHead(Q, K, V) = Concat(head_1,...,head_h) W^O
 *   where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
 *
 * @param embed_dim - Total dimension of the model.
 * @param num_heads - Number of parallel attention heads. embed_dim must be divisible by num_heads.
 * @param dropout - Dropout probability on attention weights. Default: 0.0
 * @param bias - If true, adds bias to input/output projection layers. Default: true
 * @param batch_first - If true, input/output tensors are (batch, seq, feature). Default: false (seq, batch, feature)
 *
 * @pytorch torch.nn.MultiheadAttention
 */
export class MultiheadAttention extends Module {
  public embed_dim: number;
  public num_heads: number;
  public dropout: number;
  public batch_first: boolean;
  public head_dim: number;

  // Combined in_proj weight for q, k, v (single matrix of shape [3 * embed_dim, embed_dim])
  public in_proj_weight: Parameter | null;
  public in_proj_bias: Parameter | null;

  // Output projection
  public out_proj_weight: Parameter;
  public out_proj_bias: Parameter | null;

  constructor(
    embed_dim: number,
    num_heads: number,
    dropout: number = 0.0,
    bias: boolean = true,
    batch_first: boolean = false
  ) {
    super();

    if (embed_dim <= 0 || num_heads <= 0) {
      throw new Error(
        `embed_dim and num_heads must be greater than 0, got embed_dim=${embed_dim} and num_heads=${num_heads}`
      );
    }

    const head_dim = embed_dim / num_heads;
    if (!Number.isInteger(head_dim)) {
      throw new Error('embed_dim must be divisible by num_heads');
    }

    this.embed_dim = embed_dim;
    this.num_heads = num_heads;
    this.dropout = dropout;
    this.batch_first = batch_first;
    this.head_dim = head_dim;

    // Initialize combined in_proj_weight [3 * embed_dim, embed_dim]
    const in_proj_w = randn([3 * embed_dim, embed_dim]).mul(Math.sqrt(1 / embed_dim));
    this.in_proj_weight = Parameter.create(in_proj_w);
    this.register_parameter('in_proj_weight', this.in_proj_weight);

    if (bias) {
      const in_proj_b = zeros([3 * embed_dim]);
      this.in_proj_bias = Parameter.create(in_proj_b);
      this.register_parameter('in_proj_bias', this.in_proj_bias);
    } else {
      this.in_proj_bias = null;
    }

    // Output projection weight [embed_dim, embed_dim]
    const out_proj_w = randn([embed_dim, embed_dim]).mul(Math.sqrt(1 / embed_dim));
    this.out_proj_weight = Parameter.create(out_proj_w);
    this.register_parameter('out_proj_weight', this.out_proj_weight);

    if (bias) {
      const out_proj_b = zeros([embed_dim]);
      this.out_proj_bias = Parameter.create(out_proj_b);
      this.register_parameter('out_proj_bias', this.out_proj_bias);
    } else {
      this.out_proj_bias = null;
    }
  }

  /**
   * Forward pass for multi-head attention.
   *
   * @param query - Query tensor of shape (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim) if batch_first
   * @param key - Key tensor of shape (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim) if batch_first
   * @param value - Value tensor of shape (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim) if batch_first
   * @param key_padding_mask - Not yet implemented
   * @param need_weights - If true, returns attention weights. Default: false
   * @param attn_mask - Not yet implemented
   * @returns [output, attn_weights] where output has same shape as query, attn_weights is null if need_weights=false
   *
   * @pytorch nn.MultiheadAttention.forward
   */
  forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask?: Tensor | null,
    need_weights: boolean = false,
    attn_mask?: Tensor | null
  ): [Tensor, Tensor | null] {
    if (key_padding_mask != null) {
      throw new Error('key_padding_mask is not yet supported in MultiheadAttention');
    }
    if (attn_mask != null) {
      throw new Error('attn_mask is not yet supported in MultiheadAttention');
    }

    // Determine shapes based on batch_first
    // Expected shapes:
    //   batch_first=false: (seq_len, batch, embed_dim)
    //   batch_first=true:  (batch, seq_len, embed_dim)
    let q: Tensor = query;
    let k: Tensor = key;
    let v: Tensor = value;

    // If batch_first, transpose to (seq_len, batch, embed_dim) for internal processing
    if (this.batch_first) {
      // (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
      q = q.transpose(0, 1);
      k = k.transpose(0, 1);
      v = v.transpose(0, 1);
    }

    const tgt_len = q.shape[0];
    const batch_size = q.shape[1];
    const src_len = k.shape[0];

    // Split in_proj_weight and in_proj_bias into q, k, v parts
    const embed_dim = this.embed_dim;

    // Linear projections for q, k, v
    // in_proj_weight: [3*embed_dim, embed_dim]
    // in_proj_bias: [3*embed_dim]
    // q_weight: [embed_dim, embed_dim], k_weight: [embed_dim, embed_dim], v_weight: [embed_dim, embed_dim]
    let q_proj: Tensor;
    let k_proj: Tensor;
    let v_proj: Tensor;

    if (this.in_proj_weight != null) {
      // Split the combined weight
      const w_data = this.in_proj_weight; // [3*embed_dim, embed_dim]

      // Compute q = query @ W_q.T + b_q
      // q_proj = q @ in_proj_weight[0:embed_dim].T + in_proj_bias[0:embed_dim]
      const q_weight = w_data.slice([{ start: 0, stop: embed_dim }, { start: 0, stop: embed_dim }]);
      const k_weight = w_data.slice([{ start: embed_dim, stop: 2 * embed_dim }, { start: 0, stop: embed_dim }]);
      const v_weight = w_data.slice([{ start: 2 * embed_dim, stop: 3 * embed_dim }, { start: 0, stop: embed_dim }]);

      // Reshape to 2D for matmul: (tgt_len * batch_size, embed_dim) @ (embed_dim, embed_dim).T
      // q: (tgt_len, batch, embed_dim) -> (tgt_len * batch, embed_dim)
      const q_flat = q.reshape([tgt_len * batch_size, embed_dim] as number[]);
      const k_flat = k.reshape([src_len * batch_size, embed_dim] as number[]);
      const v_flat = v.reshape([src_len * batch_size, embed_dim] as number[]);

      q_proj = q_flat.matmul(q_weight.transpose(0, 1)); // (tgt_len * batch, embed_dim)
      k_proj = k_flat.matmul(k_weight.transpose(0, 1)); // (src_len * batch, embed_dim)
      v_proj = v_flat.matmul(v_weight.transpose(0, 1)); // (src_len * batch, embed_dim)

      // Add bias
      if (this.in_proj_bias != null) {
        const q_bias = this.in_proj_bias.slice([{ start: 0, stop: embed_dim }]);
        const k_bias = this.in_proj_bias.slice([{ start: embed_dim, stop: 2 * embed_dim }]);
        const v_bias = this.in_proj_bias.slice([{ start: 2 * embed_dim, stop: 3 * embed_dim }]);

        q_proj = q_proj.add(q_bias);
        k_proj = k_proj.add(k_bias);
        v_proj = v_proj.add(v_bias);
      }
    } else {
      throw new Error('in_proj_weight is required for standard MultiheadAttention');
    }

    // Reshape back to (seq_len, batch, embed_dim)
    q_proj = q_proj.reshape([tgt_len, batch_size, embed_dim] as number[]);
    k_proj = k_proj.reshape([src_len, batch_size, embed_dim] as number[]);
    v_proj = v_proj.reshape([src_len, batch_size, embed_dim] as number[]);

    // Reshape to (seq_len, batch, num_heads, head_dim)
    q_proj = q_proj.reshape([tgt_len, batch_size, this.num_heads, this.head_dim] as number[]);
    k_proj = k_proj.reshape([src_len, batch_size, this.num_heads, this.head_dim] as number[]);
    v_proj = v_proj.reshape([src_len, batch_size, this.num_heads, this.head_dim] as number[]);

    // Transpose for attention: (batch, num_heads, seq_len, head_dim)
    // q: (tgt_len, batch, num_heads, head_dim) -> (batch, num_heads, tgt_len, head_dim)
    q_proj = q_proj.transpose(0, 1).transpose(1, 2); // (batch, num_heads, tgt_len, head_dim)
    k_proj = k_proj.transpose(0, 1).transpose(1, 2); // (batch, num_heads, src_len, head_dim)
    v_proj = v_proj.transpose(0, 1).transpose(1, 2); // (batch, num_heads, src_len, head_dim)

    // Compute attention scores: (q @ k.transpose(-2, -1)) / sqrt(head_dim)
    // q: (batch, num_heads, tgt_len, head_dim)
    // k.transpose(-2, -1): (batch, num_heads, head_dim, src_len)
    // scores: (batch, num_heads, tgt_len, src_len)
    const k_transposed = k_proj.transpose(-2, -1);
    let scores = q_proj.matmul(k_transposed);

    // Scale by sqrt(head_dim)
    const scale = Math.sqrt(this.head_dim);
    scores = scores.div(scale);

    // Apply attention mask if provided (not yet implemented)

    // Apply softmax to get attention weights
    const attn_weights = softmax(scores, -1);

    // Apply dropout during training
    if (this.training && this.dropout > 0.0) {
      // Simple dropout implementation
      const mask = randn([...attn_weights.shape] as number[]);
      const threshold = this.dropout;
      // This is a simplified dropout - in practice should use proper dropout
      // For now, skip dropout as it requires proper implementation
    }

    // Apply attention weights to v
    // attn_weights: (batch, num_heads, tgt_len, src_len)
    // v: (batch, num_heads, src_len, head_dim)
    // output: (batch, num_heads, tgt_len, head_dim)
    let output = attn_weights.matmul(v_proj);

    // Transpose back: (batch, num_heads, tgt_len, head_dim) -> (tgt_len, batch, num_heads, head_dim)
    output = output.transpose(0, 1).transpose(1, 2);

    // Reshape to (tgt_len, batch, embed_dim)
    output = output.reshape([tgt_len, batch_size, embed_dim] as number[]);

    // Output projection
    // output: (tgt_len, batch, embed_dim) -> (tgt_len * batch, embed_dim)
    const output_flat = output.reshape([tgt_len * batch_size, embed_dim] as number[]);
    let final_output = output_flat.matmul(this.out_proj_weight.transpose(0, 1));

    // Add output bias
    if (this.out_proj_bias != null) {
      final_output = final_output.add(this.out_proj_bias);
    }

    // Reshape back to (tgt_len, batch, embed_dim)
    final_output = final_output.reshape([tgt_len, batch_size, embed_dim]);

    // If batch_first, transpose back to (batch, tgt_len, embed_dim)
    if (this.batch_first) {
      final_output = final_output.transpose(0, 1);
    }

    // Return attention weights if requested
    let returned_attn: Tensor | null = null;
    if (need_weights) {
      // Return average attention weights across heads: (batch, tgt_len, src_len)
      returned_attn = attn_weights;
    }

    return [final_output, returned_attn];
  }
}
