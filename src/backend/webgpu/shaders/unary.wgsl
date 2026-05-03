@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn neg(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = -input[idx];
}

@compute @workgroup_size(256)
fn abs_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = abs(input[idx]);
}

@compute @workgroup_size(256)
fn sqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = sqrt(input[idx]);
}

@compute @workgroup_size(256)
fn exp_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = exp(input[idx]);
}

@compute @workgroup_size(256)
fn exp2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = exp2(input[idx]);
}

@compute @workgroup_size(256)
fn log_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = log(input[idx]);
}

@compute @workgroup_size(256)
fn log10(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = log(input[idx]) / 2.302585092994046;
}

@compute @workgroup_size(256)
fn log2_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = log2(input[idx]);
}

@compute @workgroup_size(256)
fn log1p(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = log(1.0 + input[idx]);
}

@compute @workgroup_size(256)
fn tanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = tanh(input[idx]);
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = max(0.0, input[idx]);
}

@compute @workgroup_size(256)
fn sin_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = sin(input[idx]);
}

@compute @workgroup_size(256)
fn cos_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = cos(input[idx]);
}

@compute @workgroup_size(256)
fn tan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = tan(input[idx]);
}

@compute @workgroup_size(256)
fn acos_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = acos(input[idx]);
}

@compute @workgroup_size(256)
fn asin_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = asin(input[idx]);
}

@compute @workgroup_size(256)
fn atan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = atan(input[idx]);
}

@compute @workgroup_size(256)
fn cosh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = cosh(input[idx]);
}

@compute @workgroup_size(256)
fn sinh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = sinh(input[idx]);
}

@compute @workgroup_size(256)
fn acosh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = log(x + sqrt(x * x - 1.0));
}

@compute @workgroup_size(256)
fn asinh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = log(x + sqrt(x * x + 1.0));
}

@compute @workgroup_size(256)
fn atanh_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = 0.5 * log((1.0 + x) / (1.0 - x));
}

@compute @workgroup_size(256)
fn ceil_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = ceil(input[idx]);
}

@compute @workgroup_size(256)
fn floor_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = floor(input[idx]);
}

@compute @workgroup_size(256)
fn round_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = round(input[idx]);
}

@compute @workgroup_size(256)
fn trunc_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = trunc(input[idx]);
}

@compute @workgroup_size(256)
fn frac_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let val = input[idx];
    result[idx] = val - trunc(val);
}

@compute @workgroup_size(256)
fn reciprocal_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = 1.0 / input[idx];
}

@compute @workgroup_size(256)
fn rsqrt_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = inverseSqrt(input[idx]);
}

@compute @workgroup_size(256)
fn square_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let val = input[idx];
    result[idx] = val * val;
}

fn erf(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}

@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    result[idx] = 0.5 * x * (1.0 + erf(x * 0.7071067811865475));
}

@compute @workgroup_size(256)
fn softplus_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = select(log(1.0 + exp(x)), x, x > 20.0);
}

@compute @workgroup_size(256)
fn silu_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = x * (1.0 / (1.0 + exp(-x)));
}

@compute @workgroup_size(256)
fn mish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    result[idx] = x * tanh(sp);
}

@compute @workgroup_size(256)
fn hardsigmoid_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = clamp(x / 6.0 + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(256)
fn hardswish_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = x * clamp(x / 6.0 + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(256)
fn softsign_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = x / (1.0 + abs(x));
}

@compute @workgroup_size(256)
fn tanhshrink_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let x = input[idx];
    result[idx] = x - tanh(x);
}

@compute @workgroup_size(256)
fn isnan_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_nan = ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) != 0u);
    result[idx] = select(0.0, 1.0, is_nan);
}

@compute @workgroup_size(256)
fn isinf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_inf = ((bits & 0x7f800000u) == 0x7f800000u) && ((bits & 0x007fffffu) == 0u);
    result[idx] = select(0.0, 1.0, is_inf);
}

@compute @workgroup_size(256)
fn isfinite_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_nan_or_inf = (bits & 0x7f800000u) == 0x7f800000u;
    result[idx] = select(1.0, 0.0, is_nan_or_inf);
}

@compute @workgroup_size(256)
fn isposinf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_pos_inf = bits == 0x7f800000u;
    result[idx] = select(0.0, 1.0, is_pos_inf);
}

@compute @workgroup_size(256)
fn isneginf_op(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&result)) { return; }
    let bits = bitcast<u32>(input[idx]);
    let is_neg_inf = bits == 0xff800000u;
    result[idx] = select(0.0, 1.0, is_neg_inf);
}