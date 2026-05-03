/**
 * Linear (fully connected) layer.
 * @status implemented
 * @pytorch torch.nn.Linear
 */

import { Tensor } from '../tensor';
import { randn, zeros } from '../ops/creation';
import { Module } from './module';
import { Parameter } from './parameter';

/**
 * Applies a linear transformation: y = xW^T + b
 */
export class Linear extends Module {
  public weight: Parameter;
  public bias: Parameter | null;

  private in_features: number;
  private out_features: number;

  constructor(in_features: number, out_features: number, bias: boolean = true) {
    super();
    this.in_features = in_features;
    this.out_features = out_features;

    // Initialize weight with Kaiming uniform (simplified)
    // PyTorch uses: weight ~ U(-sqrt(k), sqrt(k)) where k = 1/in_features
    const k = Math.sqrt(1 / in_features);
    const weightData = randn([out_features, in_features]).mul(k);
    this.weight = Parameter.create(weightData);
    this.register_parameter('weight', this.weight);

    if (bias) {
      const biasData = zeros([out_features]);
      this.bias = Parameter.create(biasData);
      this.register_parameter('bias', this.bias);
    } else {
      this.bias = null;
    }
  }

  forward(input: Tensor): Tensor {
    // input: [batch_size, in_features]
    // weight: [out_features, in_features]
    // output = input @ weight.T + bias

    // Handle different input shapes
    if (input.shape.length === 1) {
      // Single sample: [in_features] -> [1, in_features]
      input = input.unsqueeze(0);
    }

    // Compute: input @ weight.T
    let output = input.matmul(this.weight.t());

    // Add bias if present
    if (this.bias) {
      // Broadcast bias across batch dimension
      // For now, we need to add bias manually per row
      // TODO: Implement proper broadcasting
      output = output.add(this.bias);
    }

    return output;
  }
}
