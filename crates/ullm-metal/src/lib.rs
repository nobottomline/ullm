//! Metal (Apple GPU) compute backend.
//!
//! GEMV kernels for f32 and for quantized weights with **dequantization in the
//! kernel** (Q4_K, Q6_K): the weights stay quantized in GPU memory and are
//! decoded on the fly, so the GPU streams ~4-7x fewer bytes than f32 — the main
//! reason Apple-Silicon GPUs win at memory-bound decode. Buffers use shared
//! storage (unified memory: no host<->device copy). Validated against the CPU.

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use ullm_core::{DType, Error, Result};

mod forward;
pub use forward::{GpuForward, GpuLayerInput, GpuModelInput, GpuParams, GpuWeight};

pub(crate) const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matvec(
    device const float* w      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float*       y      [[buffer(2)]],
    constant uint&      in_dim [[buffer(3)]],
    uint o [[thread_position_in_grid]])
{
    device const float* row = w + (uint)o * in_dim;
    float s = 0.0f;
    for (uint i = 0; i < in_dim; ++i) s += row[i] * x[i];
    y[o] = s;
}

// Q4_K: 256 weights per 144-byte block — half d, half dmin, uchar scales[12], uchar qs[128].
kernel void matvec_q4k(
    device const uchar* w      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float*       y      [[buffer(2)]],
    constant uint&      in_dim [[buffer(3)]],
    uint o [[thread_position_in_grid]])
{
    uint blocks = in_dim / 256u;
    device const uchar* row = w + (uint)o * blocks * 144u;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = row + b * 144u;
        float d    = (float)as_type<half>((ushort)(blk[0] | (blk[1] << 8)));
        float dmin = (float)as_type<half>((ushort)(blk[2] | (blk[3] << 8)));
        device const uchar* sc = blk + 4;
        device const uchar* qs = blk + 16;
        device const float* xb = x + b * 256u;
        uint is = 0, qoff = 0, xoff = 0;
        for (uint j = 0; j < 4; ++j) {
            uchar s1, m1, s2, m2;
            if (is < 4u)  { s1 = sc[is] & 63;  m1 = sc[is+4] & 63; }
            else          { s1 = (sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4);
                            m1 = (sc[is+4] >> 4)  | ((sc[is]   >> 6) << 4); }
            uint is2 = is + 1;
            if (is2 < 4u) { s2 = sc[is2] & 63; m2 = sc[is2+4] & 63; }
            else          { s2 = (sc[is2+4] & 0xF) | ((sc[is2-4] >> 6) << 4);
                            m2 = (sc[is2+4] >> 4)  | ((sc[is2]   >> 6) << 4); }
            float d1 = d * (float)s1, mm1 = dmin * (float)m1;
            float d2 = d * (float)s2, mm2 = dmin * (float)m2;
            for (uint l = 0; l < 32; ++l)
                acc += (d1 * (float)(qs[qoff+l] & 0xF) - mm1) * xb[xoff + l];
            for (uint l = 0; l < 32; ++l)
                acc += (d2 * (float)(qs[qoff+l] >> 4) - mm2) * xb[xoff + 32 + l];
            qoff += 32; xoff += 64; is += 2;
        }
    }
    y[o] = acc;
}

// Q6_K: 256 weights per 210-byte block — uchar ql[128], uchar qh[64], char scales[16], half d.
kernel void matvec_q6k(
    device const uchar* w      [[buffer(0)]],
    device const float* x      [[buffer(1)]],
    device float*       y      [[buffer(2)]],
    constant uint&      in_dim [[buffer(3)]],
    uint o [[thread_position_in_grid]])
{
    uint blocks = in_dim / 256u;
    device const uchar* row = w + (uint)o * blocks * 210u;
    float acc = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        device const uchar* blk = row + b * 210u;
        device const uchar* ql = blk;
        device const uchar* qh = blk + 128;
        device const char*  sc = (device const char*)(blk + 192);
        float d = (float)as_type<half>((ushort)(blk[208] | (blk[209] << 8)));
        device const float* xb = x + b * 256u;
        for (uint nh = 0; nh < 2; ++nh) {
            uint qlo = nh*64u, qho = nh*32u, sco = nh*8u, yo = nh*128u;
            for (uint l = 0; l < 32; ++l) {
                uint is = l / 16u;
                int q1 = (int)((ql[qlo+l]    & 0xF) | ((int)(qh[qho+l]       & 3) << 4)) - 32;
                int q2 = (int)((ql[qlo+l+32] & 0xF) | ((int)((qh[qho+l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[qlo+l]    >> 4) | ((int)((qh[qho+l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[qlo+l+32] >> 4) | ((int)((qh[qho+l] >> 6) & 3) << 4)) - 32;
                acc += d * (float)sc[sco+is]   * (float)q1 * xb[yo + l];
                acc += d * (float)sc[sco+is+2] * (float)q2 * xb[yo + l + 32];
                acc += d * (float)sc[sco+is+4] * (float)q3 * xb[yo + l + 64];
                acc += d * (float)sc[sco+is+6] * (float)q4 * xb[yo + l + 96];
            }
        }
    }
    y[o] = acc;
}

// ---- simdgroup matvec: one simdgroup (32 lanes) cooperates on each output row,
//      reducing with simd_sum. Much higher GPU utilization than one-thread-per-row.

kernel void matvec_f32_sg(
    device const float* w       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       y       [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  sgi  [[simdgroup_index_in_threadgroup]],
    uint  sgs  [[simdgroups_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    uint o = tg * sgs + sgi;
    float acc = 0.0f;
    if (o < out_dim) {
        device const float* row = w + (uint)o * in_dim;
        for (uint i = lane; i < in_dim; i += 32u) acc += row[i] * x[i];
    }
    acc = simd_sum(acc);
    if (lane == 0 && o < out_dim) y[o] = acc;
}

kernel void matvec_q4k_sg(
    device const uchar* w       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       y       [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  sgi  [[simdgroup_index_in_threadgroup]],
    uint  sgs  [[simdgroups_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    uint o = tg * sgs + sgi;
    float acc = 0.0f;
    if (o < out_dim) {
        uint blocks = in_dim / 256u;
        device const uchar* base = w + (uint)o * blocks * 144u;
        // One work-unit = one block's j-segment (64 weights); blocks*4 units
        // spread over the 32 lanes for full utilization even at small in_dim.
        uint units = blocks * 4u;
        for (uint u = lane; u < units; u += 32u) {
            uint b = u >> 2, j = u & 3u;
            device const uchar* blk = base + b * 144u;
            float d    = (float)as_type<half>((ushort)(blk[0] | (blk[1] << 8)));
            float dmin = (float)as_type<half>((ushort)(blk[2] | (blk[3] << 8)));
            device const uchar* sc = blk + 4;
            device const uchar* qs = blk + 16;
            device const float* xb = x + b * 256u;
            uint is = j * 2u, qoff = j * 32u, xoff = j * 64u;
            uchar s1, m1, s2, m2;
            if (is < 4u)  { s1 = sc[is] & 63;  m1 = sc[is+4] & 63; }
            else          { s1 = (sc[is+4] & 0xF) | ((sc[is-4] >> 6) << 4);
                            m1 = (sc[is+4] >> 4)  | ((sc[is]   >> 6) << 4); }
            uint is2 = is + 1;
            if (is2 < 4u) { s2 = sc[is2] & 63; m2 = sc[is2+4] & 63; }
            else          { s2 = (sc[is2+4] & 0xF) | ((sc[is2-4] >> 6) << 4);
                            m2 = (sc[is2+4] >> 4)  | ((sc[is2]   >> 6) << 4); }
            float d1 = d * (float)s1, mm1 = dmin * (float)m1;
            float d2 = d * (float)s2, mm2 = dmin * (float)m2;
            for (uint l = 0; l < 32; ++l)
                acc += (d1 * (float)(qs[qoff+l] & 0xF) - mm1) * xb[xoff + l];
            for (uint l = 0; l < 32; ++l)
                acc += (d2 * (float)(qs[qoff+l] >> 4) - mm2) * xb[xoff + 32 + l];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0 && o < out_dim) y[o] = acc;
}

kernel void matvec_q6k_sg(
    device const uchar* w       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       y       [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  sgi  [[simdgroup_index_in_threadgroup]],
    uint  sgs  [[simdgroups_per_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    uint o = tg * sgs + sgi;
    float acc = 0.0f;
    if (o < out_dim) {
        uint blocks = in_dim / 256u;
        device const uchar* base = w + (uint)o * blocks * 210u;
        // One work-unit = a quarter of a block (nh-half x l-half, 64 weights);
        // blocks*4 units spread over 32 lanes for full utilization.
        uint units = blocks * 4u;
        for (uint u = lane; u < units; u += 32u) {
            uint b = u >> 2, part = u & 3u;
            uint nh = part >> 1, lhalf = part & 1u;
            device const uchar* blk = base + b * 210u;
            device const uchar* ql = blk;
            device const uchar* qh = blk + 128;
            device const char*  sc = (device const char*)(blk + 192);
            float d = (float)as_type<half>((ushort)(blk[208] | (blk[209] << 8)));
            device const float* xb = x + b * 256u;
            uint qlo = nh*64u, qho = nh*32u, sco = nh*8u, yo = nh*128u;
            uint l0 = lhalf * 16u;
            for (uint l = l0; l < l0 + 16u; ++l) {
                uint is = l / 16u;
                int q1 = (int)((ql[qlo+l]    & 0xF) | ((int)(qh[qho+l]       & 3) << 4)) - 32;
                int q2 = (int)((ql[qlo+l+32] & 0xF) | ((int)((qh[qho+l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[qlo+l]    >> 4) | ((int)((qh[qho+l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[qlo+l+32] >> 4) | ((int)((qh[qho+l] >> 6) & 3) << 4)) - 32;
                acc += d * (float)sc[sco+is]   * (float)q1 * xb[yo + l];
                acc += d * (float)sc[sco+is+2] * (float)q2 * xb[yo + l + 32];
                acc += d * (float)sc[sco+is+4] * (float)q3 * xb[yo + l + 64];
                acc += d * (float)sc[sco+is+6] * (float)q4 * xb[yo + l + 96];
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0 && o < out_dim) y[o] = acc;
}

// Multi-row Q6_K matvec (llama.cpp technique): each simdgroup computes NR0
// output rows at once, loading the activation block into registers once and
// reusing it across the rows. Each lane owns a fixed 16-element pattern across
// blocks (coalesced reads), reduced with simd_sum.
kernel void matvec_q6k_mr(
    device const uchar* src0    [[buffer(0)]],
    device const float* src1    [[buffer(1)]],
    device float*       dst     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort nsg   [[simdgroups_per_threadgroup]])
{
    const short NR0 = 2;
    uint nb   = in_dim / 256u;
    uint nb01 = nb * 210u;                       // bytes per weight row
    int  first_row = (int)((uint)tgpig * nsg + sgitg) * NR0;

    short tid = tiisg / 2, ix = tiisg % 2;
    short ip  = tid / 8,   il = tid % 8;
    short l0  = 4 * il;
    short is  = 8 * ip + l0 / 16;
    short y_offset   = 128 * ip + l0;
    short q_offset_l =  64 * ip + l0;
    short q_offset_h =  32 * ip + l0;

    float sumf[NR0] = { 0.f, 0.f };
    float yl[16];

    device const uchar* x0 = src0 + (uint)first_row * nb01;

    for (uint i = ix; i < nb; i += 2) {
        device const float* y = src1 + i * 256u + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }
        for (short row = 0; row < NR0; ++row) {
            if (first_row + row >= (int)out_dim) break;
            device const uchar* blk = x0 + (uint)row * nb01 + i * 210u;
            device const uchar* q1 = blk + q_offset_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = blk + 128 + q_offset_h;
            device const char*  sc = (device const char*)(blk + 192) + is;
            float d = (float)as_type<half>((ushort)(blk[208] | (blk[209] << 8)));
            float4 sums = { 0.f, 0.f, 0.f, 0.f };
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l+0] * (float)((int)((q1[l] & 0xF) | ((qh[l] & 0x03) << 4)) - 32);
                sums[1] += yl[4*l+1] * (float)((int)((q2[l] & 0xF) | ((qh[l] & 0x0C) << 2)) - 32);
                sums[2] += yl[4*l+2] * (float)((int)((q1[l]  >> 4) | ((qh[l] & 0x30) << 0)) - 32);
                sums[3] += yl[4*l+3] * (float)((int)((q2[l]  >> 4) | ((qh[l] & 0xC0) >> 2)) - 32);
            }
            sumf[row] += d * (sums[0]*sc[0] + sums[1]*sc[2] + sums[2]*sc[4] + sums[3]*sc[6]);
        }
    }
    for (short row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < (int)out_dim) dst[first_row + row] = s;
    }
}

// Multi-row BF16 matvec (for SafeTensors / HF weights). bf16 is the top 16 bits
// of an f32, so dequant is a 16-bit left shift. Lanes stride the in_dim with
// coalesced reads; each simdgroup does NR0 rows, reusing each activation.
kernel void matvec_bf16_mr(
    device const ushort* src0    [[buffer(0)]],
    device const float*  src1    [[buffer(1)]],
    device float*        dst     [[buffer(2)]],
    constant uint&       in_dim  [[buffer(3)]],
    constant uint&       out_dim [[buffer(4)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort nsg   [[simdgroups_per_threadgroup]])
{
    const short NR0 = 2;
    int first_row = (int)((uint)tgpig * nsg + sgitg) * NR0;
    float sumf[NR0] = { 0.f, 0.f };
    for (uint i = tiisg; i < in_dim; i += 32u) {
        float xi = src1[i];
        for (short row = 0; row < NR0; ++row) {
            if (first_row + row >= (int)out_dim) break;
            ushort wb = src0[(uint)(first_row + row) * in_dim + i];
            sumf[row] += as_type<float>((uint)wb << 16) * xi;
        }
    }
    for (short row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < (int)out_dim) dst[first_row + row] = s;
    }
}

// Multi-row F16 matvec (same layout, half -> float).
kernel void matvec_f16_mr(
    device const half*  src0    [[buffer(0)]],
    device const float* src1    [[buffer(1)]],
    device float*       dst     [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort nsg   [[simdgroups_per_threadgroup]])
{
    const short NR0 = 2;
    int first_row = (int)((uint)tgpig * nsg + sgitg) * NR0;
    float sumf[NR0] = { 0.f, 0.f };
    for (uint i = tiisg; i < in_dim; i += 32u) {
        float xi = src1[i];
        for (short row = 0; row < NR0; ++row) {
            if (first_row + row >= (int)out_dim) break;
            sumf[row] += (float)src0[(uint)(first_row + row) * in_dim + i] * xi;
        }
    }
    for (short row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < (int)out_dim) dst[first_row + row] = s;
    }
}

// ---- Full-forward kernels (activations resident on the GPU) ----

// RMSNorm over `n` elements with a per-channel gain. `y` may alias `x`.
kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float*       y [[buffer(2)]],
    constant uint&      n [[buffer(3)]],
    constant float&     eps [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt  [[threads_per_threadgroup]])
{
    threadgroup float sh[1024];
    float local = 0.0f;
    for (uint i = tid; i < n; i += nt) local += x[i] * x[i];
    sh[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0u; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(sh[0] / (float)n + eps);
    for (uint i = tid; i < n; i += nt) y[i] = x[i] * inv * w[i];
}

// Per-head RMSNorm: one threadgroup per head, normalizing a head_dim slice in
// place with a shared weight. Used for Gemma/Qwen3 Q/K-norm.
kernel void rmsnorm_heads(
    device float*       x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    constant uint&      head_dim [[buffer(2)]],
    constant float&     eps [[buffer(3)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt  [[threads_per_threadgroup]])
{
    device float* xh = x + h * head_dim;
    threadgroup float sh[1024];
    float local = 0.0f;
    for (uint i = tid; i < head_dim; i += nt) local += xh[i] * xh[i];
    sh[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0u; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(sh[0] / (float)head_dim + eps);
    for (uint i = tid; i < head_dim; i += nt) xh[i] = xh[i] * inv * w[i];
}

// NeoX (rotate-half) RoPE, in place; one thread per rotated pair.
kernel void rope_neox(
    device float*    v [[buffer(0)]],
    constant uint&   n_heads  [[buffer(1)]],
    constant uint&   head_dim [[buffer(2)]],
    constant uint&   pos      [[buffer(3)]],
    constant float&  theta    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint halfd = head_dim / 2u;
    if (gid >= n_heads * halfd) return;
    uint h = gid / halfd, i = gid % halfd;
    uint off = h * head_dim;
    float freq = pow(theta, -2.0f * (float)i / (float)head_dim);
    float ang = (float)pos * freq;
    float c = cos(ang), s = sin(ang);
    float a = v[off + i], b = v[off + i + halfd];
    v[off + i]         = a * c - b * s;
    v[off + i + halfd] = a * s + b * c;
}

// Interleaved (ggml "NORM") RoPE, in place; one thread per rotated pair.
kernel void rope_norm(
    device float*    v [[buffer(0)]],
    constant uint&   n_heads  [[buffer(1)]],
    constant uint&   head_dim [[buffer(2)]],
    constant uint&   pos      [[buffer(3)]],
    constant float&  theta    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint halfd = head_dim / 2u;
    if (gid >= n_heads * halfd) return;
    uint h = gid / halfd, i = gid % halfd;
    uint off = h * head_dim + 2u * i;
    float freq = pow(theta, (float)(2u * i) / (float)head_dim);
    float ang = (float)pos / freq;     // theta^(-2i/hd) == 1/theta^(2i/hd)
    float c = cos(ang), s = sin(ang);
    float a = v[off], b = v[off + 1u];
    v[off]      = a * c - b * s;
    v[off + 1u] = a * s + b * c;
}

// Attention scores: scores[h,t] = scale * dot(q_h, K[t, kvh]).
kernel void attn_scores(
    device const float* q       [[buffer(0)]],
    device const float* kcache  [[buffer(1)]],  // bound at the layer's offset
    device float*       scores  [[buffer(2)]],
    constant uint&      head_dim [[buffer(3)]],
    constant uint&      kv_dim   [[buffer(4)]],
    constant uint&      kv_mul   [[buffer(5)]],
    constant uint&      stride   [[buffer(6)]],  // scores row stride (n_ctx)
    constant float&     scale    [[buffer(7)]],
    constant uint&      seqlen   [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])       // x = t, y = h
{
    uint t = gid.x, h = gid.y;
    if (t >= seqlen) return;
    uint kvh = h / kv_mul;
    device const float* qh = q + h * head_dim;
    device const float* kt = kcache + t * kv_dim + kvh * head_dim;
    float s = 0.0f;
    for (uint d = 0; d < head_dim; ++d) s += qh[d] * kt[d];
    scores[h * stride + t] = s * scale;
}

// In-place softmax over each head's score row.
kernel void attn_softmax(
    device float*  scores [[buffer(0)]],
    constant uint& stride [[buffer(1)]],
    constant uint& seqlen [[buffer(2)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt  [[threads_per_threadgroup]])
{
    device float* row = scores + h * stride;
    threadgroup float sh[1024];
    float m = -INFINITY;
    for (uint t = tid; t < seqlen; t += nt) m = max(m, row[t]);
    sh[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0u; s >>= 1) {
        if (tid < s) sh[tid] = max(sh[tid], sh[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mx = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint t = tid; t < seqlen; t += nt) { float e = exp(row[t] - mx); row[t] = e; sum += e; }
    sh[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0u; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sh[0];
    for (uint t = tid; t < seqlen; t += nt) row[t] *= inv;
}

// Attention output: out[h,d] = sum_t scores[h,t] * V[t, kvh, d].
kernel void attn_output(
    device const float* scores [[buffer(0)]],
    device const float* vcache [[buffer(1)]],  // bound at the layer's offset
    device float*       out    [[buffer(2)]],
    constant uint&      head_dim [[buffer(3)]],
    constant uint&      kv_dim   [[buffer(4)]],
    constant uint&      kv_mul   [[buffer(5)]],
    constant uint&      stride   [[buffer(6)]],
    constant uint&      seqlen   [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])       // x = d, y = h
{
    uint d = gid.x, h = gid.y;
    if (d >= head_dim) return;
    uint kvh = h / kv_mul;
    device const float* row = scores + h * stride;
    float acc = 0.0f;
    for (uint t = 0; t < seqlen; ++t) acc += row[t] * vcache[t * kv_dim + kvh * head_dim + d];
    out[h * head_dim + d] = acc;
}

// SwiGLU: hidden[i] = silu(gate[i]) * up[i].
kernel void silu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

// GeGLU (tanh approx): hidden[i] = gelu(gate[i]) * up[i].
kernel void gelu_mul(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float g = gate[i];
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    // Clamp the tanh argument: Metal's tanh computes exp(2x), which overflows to
    // inf (-> inf/inf = NaN) well before tanh saturates. tanh(+-30) is already
    // +-1 in f32, so clamping is exact.
    float arg = clamp(c * (g + 0.044715f * g * g * g), -30.0f, 30.0f);
    float gelu = 0.5f * g * (1.0f + tanh(arg));
    out[i] = gelu * up[i];
}

// Residual add: x[i] += y[i].
kernel void add_inplace(
    device float*       x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    constant uint&      n [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    x[i] += y[i];
}
"#;

/// A Metal device with compiled GEMV pipelines, ready to dispatch work.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    matvec_pso: ComputePipelineState,
    q4k_pso: ComputePipelineState,
    q6k_pso: ComputePipelineState,
}

impl MetalContext {
    /// Create a context on the system default GPU, compiling the kernels.
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or_else(|| Error::Unsupported("no Metal device available".into()))?;
        let queue = device.new_command_queue();
        let library = device
            .new_library_with_source(SHADER, &metal::CompileOptions::new())
            .map_err(|e| Error::Format(format!("metal shader compile failed: {e}")))?;
        let pso = |name: &str| -> Result<ComputePipelineState> {
            let func = library
                .get_function(name, None)
                .map_err(|e| Error::Format(format!("metal function '{name}' missing: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| Error::Format(format!("metal pipeline '{name}' failed: {e}")))
        };
        Ok(Self {
            matvec_pso: pso("matvec")?,
            q4k_pso: pso("matvec_q4k")?,
            q6k_pso: pso("matvec_q6k")?,
            queue,
            device,
        })
    }

    /// Human-readable name of the GPU.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Copy `bytes` into a resident, GPU-addressable (shared) buffer.
    pub fn upload(&self, bytes: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// f32 GEMV: `y[o] = sum_i w[o*in + i] * x[i]`, `w` row-major `[out, in]`.
    pub fn matvec(&self, w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let wbuf = self.device.new_buffer_with_data(
            w.as_ptr().cast(),
            (w.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.dispatch(&self.matvec_pso, &wbuf, x, out_dim, in_dim)
    }

    /// Quantized GEMV from raw GGUF block bytes, dequantizing in the kernel.
    pub fn matvec_quant(
        &self,
        dtype: DType,
        w_bytes: &[u8],
        x: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let pso = self.pso_for(dtype)?;
        let wbuf = self.upload(w_bytes);
        Ok(self.dispatch(pso, &wbuf, x, out_dim, in_dim))
    }

    /// Quantized GEMV against weights already resident in a GPU buffer.
    pub fn matvec_resident(
        &self,
        dtype: DType,
        wbuf: &Buffer,
        x: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let pso = self.pso_for(dtype)?;
        Ok(self.dispatch(pso, wbuf, x, out_dim, in_dim))
    }

    fn pso_for(&self, dtype: DType) -> Result<&ComputePipelineState> {
        match dtype {
            DType::Q4K => Ok(&self.q4k_pso),
            DType::Q6K => Ok(&self.q6k_pso),
            other => Err(Error::Unsupported(format!(
                "no Metal kernel for {other:?} yet"
            ))),
        }
    }

    fn dispatch(
        &self,
        pso: &ComputePipelineState,
        wbuf: &Buffer,
        x: &[f32],
        out_dim: usize,
        in_dim: usize,
    ) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let xbuf =
            self.device
                .new_buffer_with_data(x.as_ptr().cast(), (x.len() * 4) as u64, shared);
        let ybuf = self.device.new_buffer((out_dim * 4) as u64, shared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(wbuf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        let in_dim_u32 = in_dim as u32;
        enc.set_bytes(3, 4, (&in_dim_u32 as *const u32).cast());

        let tpt = pso
            .max_total_threads_per_threadgroup()
            .min(out_dim as u64)
            .max(1);
        enc.dispatch_threads(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(tpt, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut out = vec![0.0f32; out_dim];
        // SAFETY: `ybuf` holds `out_dim` f32 written by the kernel; shared storage
        // makes them visible to the CPU after `wait_until_completed`.
        unsafe {
            std::ptr::copy_nonoverlapping(ybuf.contents().cast::<f32>(), out.as_mut_ptr(), out_dim);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        (0..out_dim)
            .map(|o| {
                w[o * in_dim..o * in_dim + in_dim]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    fn rel_err(gpu: &[f32], cpu: &[f32]) -> f32 {
        let scale = cpu.iter().map(|c| c.abs()).fold(0.0f32, f32::max).max(1e-6);
        gpu.iter()
            .zip(cpu)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max)
            / scale
    }

    #[test]
    fn f32_gemv_matches_cpu() {
        let Ok(ctx) = MetalContext::new() else {
            return; // no GPU (e.g. CI) — skip
        };
        let (o, i) = (512usize, 1024usize);
        let w: Vec<f32> = (0..o * i).map(|k| ((k % 17) as f32 - 8.0) * 0.01).collect();
        let x: Vec<f32> = (0..i).map(|k| ((k % 13) as f32 - 6.0) * 0.1).collect();
        assert!(rel_err(&ctx.matvec(&w, &x, o, i), &cpu_matvec(&w, &x, o, i)) < 1e-3);
    }

    /// Validate a quantized kernel against `ullm_core`'s CPU dequantizer on the
    /// same bytes (scales pinned to a sane half so values stay finite).
    fn check_quant(dtype: DType, d_offsets: &[usize]) {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (o, i) = (256usize, 512usize);
        let ts = dtype.type_size();
        let total = o * (i / 256) * ts;
        let half = 0x3000u16.to_le_bytes(); // 0.125
        let mut w: Vec<u8> = (0..total)
            .map(|k| (k.wrapping_mul(131).wrapping_add(7) % 251) as u8)
            .collect();
        for blk in w.chunks_mut(ts) {
            for &off in d_offsets {
                blk[off] = half[0];
                blk[off + 1] = half[1];
            }
        }
        let x: Vec<f32> = (0..i).map(|k| ((k % 13) as f32 - 6.0) * 0.1).collect();

        let cpu_w = ullm_core::dequant::dequantize(dtype, &w, o * i).unwrap();
        let cpu = cpu_matvec(&cpu_w, &x, o, i);
        let gpu = ctx.matvec_quant(dtype, &w, &x, o, i).unwrap();
        assert!(rel_err(&gpu, &cpu) < 1e-3, "{dtype:?} kernel mismatch");
    }

    #[test]
    fn q4k_gemv_matches_cpu() {
        check_quant(DType::Q4K, &[0, 2]); // d, dmin halves
    }

    #[test]
    fn q6k_gemv_matches_cpu() {
        check_quant(DType::Q6K, &[208]); // d half
    }
}
