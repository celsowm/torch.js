struct Params {
    batch_size: u32,
    reduce_size: u32,
    op: u32, // 0: sum, 1: mean, 2: max, 3: min, 4: prod
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    var acc: f32;
    // Initialization based on op
    if (params.op == 0u || params.op == 1u || params.op == 5u) { 
        acc = 0.0; 
    } else if (params.op == 2u) { 
        acc = -3.4028235e+38; 
    } else if (params.op == 3u) { 
        acc = 3.4028235e+38; 
    } else if (params.op == 4u || params.op == 6u) { 
        acc = 1.0; 
    }

    let offset = batch_idx * params.reduce_size;
    for (var i = 0u; i < params.reduce_size; i++) {
        let val = input[offset + i];
        if (params.op == 0u || params.op == 1u) { 
            acc += val; 
        } else if (params.op == 2u) { 
            acc = max(acc, val); 
        } else if (params.op == 3u) { 
            acc = min(acc, val); 
        } else if (params.op == 4u) { 
            acc *= val; 
        } else if (params.op == 5u) {
            if (val != 0.0) { acc = 1.0; break; }
        } else if (params.op == 6u) {
            if (val == 0.0) { acc = 0.0; break; }
        }
    }

    if (params.op == 1u) {
        output[batch_idx] = acc / f32(params.reduce_size);
    } else {
        output[batch_idx] = acc;
    }
}
