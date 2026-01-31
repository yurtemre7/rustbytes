struct QuadUniform {
    offset: vec2<f32>,
    scale: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> u: QuadUniform;

struct VsIn {
    @location(0) pos: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(input: VsIn) -> VsOut {
    var out: VsOut;

    let p = input.pos * u.scale + u.offset;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Solid “cube” color
    return vec4<f32>(0.2, 0.85, 0.95, 1.0);
}
