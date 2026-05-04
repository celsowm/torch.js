/**
 * Loss functions for neural network training.
 * All loss classes extend Module and provide forward methods matching PyTorch API.
 * @status implemented
 * @pytorch torch.nn.losses
 */

import { Tensor } from '../tensor';
import { Module } from './module';
import { arange, zeros, full } from '../ops/creation';

// ─── Helpers ────────────────────────────────────────────────────────────────

/**
 * Apply reduction to a tensor.
 */
function apply_reduction(
  t: Tensor,
  reduction: 'mean' | 'sum' | 'none' | 'batchmean'
): Tensor {
  switch (reduction) {
    case 'mean':
      return t.mean();
    case 'sum':
      return t.sum();
    case 'batchmean':
      return t.sum().div(t.shape[0]);
    case 'none':
      return t;
  }
}

// ─── NLLLoss ────────────────────────────────────────────────────────────────

/**
 * Negative log likelihood loss.
 * Expects input to be log-probabilities [N, C], target is class indices [N].
 * @pytorch torch.nn.NLLLoss
 */
export class NLLLoss extends Module {
  public weight: Tensor | null;
  public reduction: 'mean' | 'sum' | 'none';

  constructor(
    weight?: Tensor,
    reduction: 'mean' | 'sum' | 'none' = 'mean'
  ) {
    super();
    this.weight = weight ?? null;
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return nll_loss(input, target, this.weight, this.reduction);
  }
}

/**
 * Functional NLL loss.
 * @pytorch F.nll_loss
 */
export function nll_loss(
  input: Tensor,
  target: Tensor,
  weight: Tensor | null = null,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  if (input.shape.length !== 2) {
    throw new Error(`nll_loss: expected 2D input [N, C], got ${input.shape.length}D`);
  }
  if (target.shape.length !== 1) {
    throw new Error(`nll_loss: expected 1D target [N], got ${target.shape.length}D`);
  }

  const [batchSize, numClasses] = input.shape;

  // Gather log-probs at target indices using linear indexing
  // linear_idx[i] = i * numClasses + target[i]
  const rowOffsets = arange(0, batchSize, 1, { dtype: input.dtype }).mul(numClasses);
  const linearIdx = rowOffsets.add(target);
  const flatInput = input.reshape([-1]);
  const gathered = flatInput.index_select(0, linearIdx);

  // NLL loss = -gathered log-probs
  let loss = gathered.neg();

  if (weight !== null) {
    const weightAtTarget = weight.index_select(0, target);
    loss = loss.mul(weightAtTarget);
  }

  return apply_reduction(loss, reduction);
}

// ─── CrossEntropyLoss ───────────────────────────────────────────────────────

/**
 * Cross entropy loss combining log_softmax + NLLLoss.
 * Supports label smoothing.
 * @pytorch torch.nn.CrossEntropyLoss
 */
export class CrossEntropyLoss extends Module {
  public weight: Tensor | null;
  public reduction: 'mean' | 'sum' | 'none';
  public label_smoothing: number;

  constructor(
    weight?: Tensor,
    reduction: 'mean' | 'sum' | 'none' = 'mean',
    label_smoothing: number = 0.0
  ) {
    super();
    this.weight = weight ?? null;
    this.reduction = reduction;
    this.label_smoothing = label_smoothing;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return cross_entropy(
      input, target, this.weight, this.reduction, this.label_smoothing
    );
  }
}

/**
 * Functional cross entropy loss.
 * @pytorch F.cross_entropy
 */
export function cross_entropy(
  input: Tensor,
  target: Tensor,
  weight: Tensor | null = null,
  reduction: 'mean' | 'sum' | 'none' = 'mean',
  label_smoothing: number = 0.0
): Tensor {
  if (label_smoothing > 0.0) {
    return _cross_entropy_label_smoothing(
      input, target, weight, reduction, label_smoothing
    );
  }

  const logProbs = input.log_softmax(-1);
  return nll_loss(logProbs, target, weight, reduction);
}

function _cross_entropy_label_smoothing(
  input: Tensor,
  target: Tensor,
  weight: Tensor | null,
  reduction: 'mean' | 'sum' | 'none',
  eps: number
): Tensor {
  const logProbs = input.log_softmax(-1);

  // NLL component
  const nllPart = nll_loss(logProbs, target, null, 'none');
  // Uniform distribution component
  const uniformPart = logProbs.mean(-1).neg();

  let loss = nllPart.mul(1 - eps).add(uniformPart.mul(eps));

  if (weight !== null) {
    const weightAtTarget = weight.index_select(0, target);
    loss = loss.mul(weightAtTarget);
  }

  return apply_reduction(loss, reduction);
}

// ─── BCELoss ────────────────────────────────────────────────────────────────

/**
 * Binary cross-entropy loss. Input should be probabilities (output of sigmoid).
 * @pytorch torch.nn.BCELoss
 */
export class BCELoss extends Module {
  public weight: Tensor | null;
  public reduction: 'mean' | 'sum' | 'none';

  constructor(
    weight?: Tensor,
    reduction: 'mean' | 'sum' | 'none' = 'mean'
  ) {
    super();
    this.weight = weight ?? null;
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return binary_cross_entropy(input, target, this.weight, this.reduction);
  }
}

/**
 * Functional binary cross-entropy.
 * @pytorch F.binary_cross_entropy
 */
export function binary_cross_entropy(
  input: Tensor,
  target: Tensor,
  weight: Tensor | null = null,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const eps = 1e-12;
  const clampedInput = input.clamp(eps, 1 - eps);
  const logInput = clampedInput.log();
  const logOneMinusInput = clampedInput.mul(-1).add(1).clamp(eps).log();

  let loss = target
    .mul(logInput)
    .add(target.mul(-1).add(1).mul(logOneMinusInput))
    .neg();

  if (weight !== null) {
    loss = loss.mul(weight);
  }

  return apply_reduction(loss, reduction);
}

// ─── BCEWithLogitsLoss ──────────────────────────────────────────────────────

/**
 * Binary cross-entropy with logits (sigmoid + BCE combined).
 * Numerically stable.
 * @pytorch torch.nn.BCEWithLogitsLoss
 */
export class BCEWithLogitsLoss extends Module {
  public weight: Tensor | null;
  public reduction: 'mean' | 'sum' | 'none';
  public pos_weight: Tensor | null;

  constructor(
    weight?: Tensor,
    reduction: 'mean' | 'sum' | 'none' = 'mean',
    pos_weight?: Tensor
  ) {
    super();
    this.weight = weight ?? null;
    this.reduction = reduction;
    this.pos_weight = pos_weight ?? null;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return binary_cross_entropy_with_logits(
      input, target, this.weight, this.reduction, this.pos_weight
    );
  }
}

/**
 * Functional binary cross-entropy with logits.
 * @pytorch F.binary_cross_entropy_with_logits
 */
export function binary_cross_entropy_with_logits(
  input: Tensor,
  target: Tensor,
  weight: Tensor | null = null,
  reduction: 'mean' | 'sum' | 'none' = 'mean',
  pos_weight: Tensor | null = null
): Tensor {
  // log-sum-exp trick for numerical stability
  const maxXZeros = input.clamp(0);
  const negAbs = input.abs().neg();
  let loss = maxXZeros.sub(input.mul(target)).add(negAbs.exp().add(1).log());

  if (pos_weight !== null) {
    const factor = target.mul(pos_weight.sub(1)).add(1);
    loss = loss.mul(factor);
  }

  if (weight !== null) {
    loss = loss.mul(weight);
  }

  return apply_reduction(loss, reduction);
}

// ─── HuberLoss / SmoothL1Loss ───────────────────────────────────────────────

/**
 * Huber loss. Uses squared loss when |error| < delta, L1 otherwise.
 * @pytorch torch.nn.HuberLoss
 */
export class HuberLoss extends Module {
  public delta: number;
  public reduction: 'mean' | 'sum' | 'none';

  constructor(
    delta: number = 1.0,
    reduction: 'mean' | 'sum' | 'none' = 'mean'
  ) {
    super();
    this.delta = delta;
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return huber_loss(input, target, this.delta, this.reduction);
  }
}

/**
 * Smooth L1 loss (Huber with delta=1.0).
 * @pytorch torch.nn.SmoothL1Loss
 */
export class SmoothL1Loss extends HuberLoss {
  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super(1.0, reduction);
  }
}

/**
 * Functional Huber loss.
 * @pytorch F.huber_loss
 */
export function huber_loss(
  input: Tensor,
  target: Tensor,
  delta: number = 1.0,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const diff = input.sub(target);
  const absDiff = diff.abs();
  const isSmall = absDiff.lt(delta);

  const quadPart = diff.pow(2).mul(0.5 / delta);
  const linearPart = absDiff.sub(0.5 * delta);

  const loss = isSmall.where(quadPart, linearPart);
  return apply_reduction(loss, reduction);
}

// ─── L1Loss ─────────────────────────────────────────────────────────────────

/**
 * L1 loss (mean absolute error).
 * @pytorch torch.nn.L1Loss
 */
export class L1Loss extends Module {
  public reduction: 'mean' | 'sum' | 'none';

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return l1_loss(input, target, this.reduction);
  }
}

/**
 * Functional L1 loss.
 * @pytorch F.l1_loss
 */
export function l1_loss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const absDiff = input.sub(target).abs();
  return apply_reduction(absDiff, reduction);
}

// ─── MSELoss ────────────────────────────────────────────────────────────────

/**
 * Mean squared error loss.
 * @pytorch torch.nn.MSELoss
 */
export class MSELoss extends Module {
  public reduction: 'mean' | 'sum' | 'none';

  constructor(reduction: 'mean' | 'sum' | 'none' = 'mean') {
    super();
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return mse_loss(input, target, this.reduction);
  }
}

/**
 * Functional MSE loss.
 * @pytorch F.mse_loss
 */
export function mse_loss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const squared = input.sub(target).pow(2);
  return apply_reduction(squared, reduction);
}

// ─── KLDivLoss ──────────────────────────────────────────────────────────────

/**
 * KL divergence loss. Input should be log-probabilities, target should be
 * probabilities (or log-probabilities if log_target=true).
 * @pytorch torch.nn.KLDivLoss
 */
export class KLDivLoss extends Module {
  public reduction: 'mean' | 'sum' | 'batchmean' | 'none';
  public log_target: boolean;

  constructor(
    reduction: 'mean' | 'sum' | 'batchmean' | 'none' = 'batchmean',
    log_target: boolean = false
  ) {
    super();
    this.reduction = reduction;
    this.log_target = log_target;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return kl_div_loss(input, target, this.reduction, this.log_target);
  }
}

/**
 * Functional KL divergence loss.
 * @pytorch F.kl_div
 */
export function kl_div_loss(
  input: Tensor,
  target: Tensor,
  reduction: 'mean' | 'sum' | 'batchmean' | 'none' = 'batchmean',
  log_target: boolean = false
): Tensor {
  const targetLog = log_target ? target : target.log();
  // KL = target * (log(target) - input)
  const loss = target.mul(targetLog.sub(input));
  return apply_reduction(loss, reduction);
}

// ─── HingeEmbeddingLoss ─────────────────────────────────────────────────────

/**
 * Hinge embedding loss for similarity learning.
 * @pytorch torch.nn.HingeEmbeddingLoss
 */
export class HingeEmbeddingLoss extends Module {
  public margin: number;
  public reduction: 'mean' | 'sum' | 'none';

  constructor(
    margin: number = 1.0,
    reduction: 'mean' | 'sum' | 'none' = 'mean'
  ) {
    super();
    this.margin = margin;
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return hinge_embedding_loss(input, target, this.margin, this.reduction);
  }
}

/**
 * Functional hinge embedding loss.
 * @pytorch F.hinge_embedding_loss
 */
export function hinge_embedding_loss(
  input: Tensor,
  target: Tensor,
  margin: number = 1.0,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  // target == 1: loss = input
  // target == -1: loss = max(0, margin - input)
  const marginTensor = full([...input.shape] as number[], margin, { dtype: input.dtype });
  const lossNegative = marginTensor.sub(input).clamp(0);
  const isPositive = target.eq(1);
  const loss = isPositive.where(input, lossNegative);
  return apply_reduction(loss, reduction);
}

// ─── CosineEmbeddingLoss ────────────────────────────────────────────────────

/**
 * Cosine embedding loss.
 * @pytorch torch.nn.CosineEmbeddingLoss
 */
export class CosineEmbeddingLoss extends Module {
  public margin: number;
  public reduction: 'mean' | 'sum' | 'none';

  constructor(
    margin: number = 0.0,
    reduction: 'mean' | 'sum' | 'none' = 'mean'
  ) {
    super();
    this.margin = margin;
    this.reduction = reduction;
  }

  forward(input1: Tensor, input2: Tensor, target: Tensor): Tensor {
    return cosine_embedding_loss(input1, input2, target, this.margin, this.reduction);
  }
}

/**
 * Compute cosine similarity between two tensors along the last dimension.
 */
function _cosine_similarity(a: Tensor, b: Tensor, dim: number = -1, eps: number = 1e-8): Tensor {
  const dotProduct = a.mul(b).sum(dim, true);
  const normA = a.pow(2).sum(dim, true).add(eps).sqrt();
  const normB = b.pow(2).sum(dim, true).add(eps).sqrt();
  return dotProduct.div(normA.mul(normB));
}

/**
 * Functional cosine embedding loss.
 * @pytorch F.cosine_embedding_loss
 */
export function cosine_embedding_loss(
  input1: Tensor,
  input2: Tensor,
  target: Tensor,
  margin: number = 0.0,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const cosSim = _cosine_similarity(input1, input2, -1);

  // target == 1: loss = 1 - cosine_similarity
  // target == -1: loss = max(0, cosine_similarity - margin)
  const lossPositive = cosSim.mul(-1).add(1);
  const lossNegative = cosSim.sub(margin).clamp(0);

  const isPositive = target.eq(1);
  const loss = isPositive.where(lossPositive, lossNegative);
  return apply_reduction(loss, reduction);
}

// ─── TripletMarginLoss ──────────────────────────────────────────────────────

/**
 * Triplet margin loss for metric learning.
 * @pytorch torch.nn.TripletMarginLoss
 */
export class TripletMarginLoss extends Module {
  public margin: number;
  public p: number;
  public eps: number;
  public reduction: 'mean' | 'sum' | 'none';

  constructor(
    margin: number = 1.0,
    p: number = 2,
    eps: number = 1e-6,
    reduction: 'mean' | 'sum' | 'none' = 'mean'
  ) {
    super();
    this.margin = margin;
    this.p = p;
    this.eps = eps;
    this.reduction = reduction;
  }

  forward(anchor: Tensor, positive: Tensor, negative: Tensor): Tensor {
    return triplet_margin_loss(
      anchor, positive, negative, this.margin, this.p, this.eps, this.reduction
    );
  }
}

/**
 * Compute p-norm distance along the last dimension.
 */
function _p_norm_distance(a: Tensor, b: Tensor, p: number, eps: number = 1e-6): Tensor {
  return a.sub(b).abs().pow(p).sum(-1, true).add(eps).pow(1 / p);
}

/**
 * Functional triplet margin loss.
 * @pytorch F.triplet_margin_loss
 */
export function triplet_margin_loss(
  anchor: Tensor,
  positive: Tensor,
  negative: Tensor,
  margin: number = 1.0,
  p: number = 2,
  eps: number = 1e-6,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  const distPositive = _p_norm_distance(anchor, positive, p, eps);
  const distNegative = _p_norm_distance(anchor, negative, p, eps);

  // loss = max(d(a, p) - d(a, n) + margin, 0)
  const loss = distPositive.sub(distNegative).add(margin).clamp(0);
  return apply_reduction(loss, reduction);
}

// ─── CTCLoss ────────────────────────────────────────────────────────────────

/**
 * Connectionist Temporal Classification loss (simplified).
 * Uses best-path (greedy) approximation rather than full forward-backward.
 * @pytorch torch.nn.CTCLoss
 */
export class CTCLoss extends Module {
  public blank: number;
  public reduction: 'mean' | 'sum' | 'none';
  public zero_infinity: boolean;

  constructor(
    blank: number = 0,
    reduction: 'mean' | 'sum' | 'none' = 'mean',
    zero_infinity: boolean = false
  ) {
    super();
    this.blank = blank;
    this.reduction = reduction;
    this.zero_infinity = zero_infinity;
  }

  forward(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor | number[],
    target_lengths: Tensor | number[]
  ): Tensor {
    return ctc_loss(
      log_probs, targets, input_lengths, target_lengths, this.blank, this.reduction
    );
  }
}

/**
 * Functional CTC loss (simplified, best-path approximation).
 *
 * @param log_probs - Log-probabilities of shape [T, N, C] or [T, C]
 * @param targets - Target sequences [N, max_target_length] or [max_target_length]
 * @param input_lengths - Length of each input sequence (number or 1D Tensor)
 * @param target_lengths - Length of each target sequence (number or 1D Tensor)
 * @param blank - Index of the blank token
 * @param reduction - Reduction method
 * @pytorch F.ctc_loss
 */
export function ctc_loss(
  log_probs: Tensor,
  targets: Tensor,
  input_lengths: Tensor | number[],
  target_lengths: Tensor | number[],
  blank: number = 0,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  let lp = log_probs;
  if (lp.shape.length === 2) {
    lp = lp.unsqueeze(1); // [T, C] -> [T, 1, C]
  }

  const [T, N, C] = lp.shape;
  let tgt = targets;
  if (tgt.shape.length === 1) {
    tgt = tgt.unsqueeze(0); // [L] -> [1, L]
  }

  const maxTargetLen = tgt.shape[1];
  const effectiveLen = Math.min(maxTargetLen, T);

  // Slice to effective length using narrow
  const tgtSlice = tgt.narrow(1, 0, effectiveLen); // [N, effectiveLen]
  const lpTransposed = lp.transpose(0, 1); // [N, T, C]
  const lpSlice = lpTransposed.narrow(1, 0, effectiveLen); // [N, effectiveLen, C]

  // Flatten: [N * effectiveLen, C] and [N * effectiveLen]
  const flatLp = lpSlice.reshape([-1, C]);
  const flatTgt = tgtSlice.flatten();

  // One-hot encode targets and gather log-probs
  const tgtRange = arange(0, C, 1, { dtype: tgt.dtype }).unsqueeze(0); // [1, C]
  const flatTgtExp = flatTgt.unsqueeze(-1); // [N*effectiveLen, 1]
  const oneHot = flatTgtExp.eq(tgtRange).to(lp.dtype); // [N*effectiveLen, C]
  const logProbsAtTarget = flatLp.mul(oneHot).sum(-1); // [N * effectiveLen]

  // Reshape to [N, effectiveLen], sum over time, negate
  let totalLoss = logProbsAtTarget.reshape([N, effectiveLen]).sum(-1).neg(); // [N]

  // Normalize by target lengths
  let lengthsTensor: Tensor;
  if (typeof target_lengths === 'number') {
    lengthsTensor = full([N], target_lengths, { dtype: lp.dtype });
  } else {
    // Assume it's already a Tensor
    lengthsTensor = (target_lengths as Tensor).to(lp.dtype);
  }
  totalLoss = totalLoss.div(lengthsTensor.add(1e-8));

  return apply_reduction(totalLoss, reduction);
}

// ─── MultiMarginLoss ────────────────────────────────────────────────────────

/**
 * Multi-margin loss for multi-class classification (SVM-style).
 * @pytorch torch.nn.MultiMarginLoss
 */
export class MultiMarginLoss extends Module {
  public p: number;
  public margin: number;
  public weight: Tensor | null;
  public reduction: 'mean' | 'sum' | 'none';

  constructor(
    p: number = 1,
    margin: number = 1.0,
    weight?: Tensor,
    reduction: 'mean' | 'sum' | 'none' = 'mean'
  ) {
    super();
    this.p = p;
    this.margin = margin;
    this.weight = weight ?? null;
    this.reduction = reduction;
  }

  forward(input: Tensor, target: Tensor): Tensor {
    return multi_margin_loss(
      input, target, this.p, this.margin, this.weight, this.reduction
    );
  }
}

/**
 * Functional multi-margin loss.
 * @pytorch F.multi_margin_loss
 */
export function multi_margin_loss(
  input: Tensor,
  target: Tensor,
  p: number = 1,
  margin: number = 1.0,
  weight: Tensor | null = null,
  reduction: 'mean' | 'sum' | 'none' = 'mean'
): Tensor {
  if (input.shape.length !== 2) {
    throw new Error(`multi_margin_loss: expected 2D input [N, C], got ${input.shape.length}D`);
  }
  if (target.shape.length !== 1) {
    throw new Error(`multi_margin_loss: expected 1D target [N], got ${target.shape.length}D`);
  }

  const [N, C] = input.shape;

  // Gather target scores: input[i, target[i]]
  const rowOffsets = arange(0, N, 1, { dtype: input.dtype }).mul(C);
  const linearIdx = rowOffsets.add(target);
  const flatInput = input.reshape([-1]);
  const targetScores = flatInput.index_select(0, linearIdx).unsqueeze(-1); // [N, 1]

  // margin - (target_score - input[j]) for all j
  const margins = input.sub(targetScores).add(margin);

  // Mask out the target class itself
  const colIndices = arange(0, C, 1, { dtype: target.dtype }).unsqueeze(0); // [1, C]
  const targetExpanded = target.unsqueeze(-1); // [N, 1]
  const notTargetMask = targetExpanded.ne(colIndices).to(input.dtype); // [N, C]

  const clamped = margins.clamp(0).mul(notTargetMask);
  const powered = clamped.pow(p);

  // Mean over non-target classes (C - 1 classes)
  let loss = powered.sum(-1).div(C - 1); // [N]

  if (weight !== null) {
    const weightAtTarget = weight.index_select(0, target);
    loss = loss.mul(weightAtTarget);
  }

  return apply_reduction(loss, reduction);
}
