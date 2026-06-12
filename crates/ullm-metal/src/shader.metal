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

// Multi-row Q4_K matvec (ported from ggml-metal kernel_mul_mv_q4_K). Each
// simdgroup does NR0 rows; the 32 lanes cooperatively cover the block with
// uint16 reads and a packed-scale trick. Activations loaded once per block.
kernel void matvec_q4k_mr(
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
    const ushort kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    short ix = tiisg / 8, it = tiisg % 8, iq = it / 4, ir = it % 4;
    uint nb   = in_dim / 256u;
    uint nb01 = nb * 144u;
    int  first_row = (int)((uint)tgpig * nsg + sgitg) * NR0;
    device const uchar* x0 = src0 + (uint)first_row * nb01;

    float yl[16], yh[16];
    float sumf[NR0] = { 0.f, 0.f };
    device const float* y4 = src1 + (uint)ix * 256u + 64u * (uint)iq + 8u * (uint)ir;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = { 0.f, 0.f, 0.f, 0.f };
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }
        for (short row = 0; row < NR0; ++row) {
            if (first_row + row >= (int)out_dim) break;
            device const uchar*  blk = x0 + (uint)row * nb01 + ib * 144u;
            device const half*   dh  = (device const half*)(blk);
            device const ushort* sc  = (device const ushort*)(blk + 4) + iq;
            device const ushort* q1  = (device const ushort*)(blk + 16) + 16u*(uint)iq + 4u*(uint)ir;
            device const ushort* q2  = q1 + 32;
            ushort sc16[4];
            thread const uchar* sc8 = (thread const uchar*)sc16;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            float4 acc1 = { 0.f, 0.f, 0.f, 0.f };
            float4 acc2 = { 0.f, 0.f, 0.f, 0.f };
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (float)(q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (float)(q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (float)(q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (float)(q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (float)(q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (float)(q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (float)(q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (float)(q2[i] & 0xF000);
            }
            float dall = (float)dh[0], dmin = (float)dh[1];
            sumf[row] += dall * ((acc1[0] + (1.f/256.f)*acc1[1]) * (float)sc8[0] +
                                 (acc1[2] + (1.f/256.f)*acc1[3]) * (float)sc8[1] * (1.f/16.f) +
                                 (acc2[0] + (1.f/256.f)*acc2[1]) * (float)sc8[4] +
                                 (acc2[2] + (1.f/256.f)*acc2[3]) * (float)sc8[5] * (1.f/16.f))
                       - dmin * (sumy[0]*(float)sc8[2] + sumy[1]*(float)sc8[3] +
                                 sumy[2]*(float)sc8[6] + sumy[3]*(float)sc8[7]);
        }
        y4 += 4u * 256u;
    }
    for (short row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < (int)out_dim) dst[first_row + row] = s;
    }
}

// MLX 4-bit matvec: weight is u32-packed (8 nibbles/word, LSB first); each group
// of `group_size` weights has one scale + bias (value = q*scale + bias). One
// simdgroup per output row, lanes stride the in_dim, reduced with simd_sum.
kernel void matvec_mlx4(
    device const uint*  w          [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device float*       y          [[buffer(2)]],
    device const float* scales     [[buffer(3)]],
    device const float* biases     [[buffer(4)]],
    constant uint&      in_dim     [[buffer(5)]],
    constant uint&      out_dim    [[buffer(6)]],
    constant uint&      group_size [[buffer(7)]],
    uint   tg   [[threadgroup_position_in_grid]],
    ushort sgi  [[simdgroup_index_in_threadgroup]],
    ushort sgs  [[simdgroups_per_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    uint o = (uint)tg * sgs + sgi;
    float acc = 0.0f;
    if (o < out_dim) {
        uint words  = in_dim / 8u;
        uint wpg    = group_size / 8u;          // words per scale/bias group
        device const uint*  row  = w + o * words;
        device const float* srow = scales + o * (in_dim / group_size);
        device const float* brow = biases + o * (in_dim / group_size);
        // Each lane consumes whole u32 words (8 nibbles), so the word and its
        // group scale/bias load once and feed 8 multiply-adds.
        for (uint wi = lane; wi < words; wi += 32u) {
            uint word = row[wi];
            uint g = wi / wpg;
            float sc = srow[g], bi = brow[g];
            device const float* xb = x + wi * 8u;
            for (uint n = 0; n < 8u; ++n) {
                uint q = (word >> (n * 4u)) & 0xFu;
                acc += ((float)q * sc + bi) * xb[n];
            }
        }
    }
    acc = simd_sum(acc);
    if (lane == 0 && o < out_dim) y[o] = acc;
}

// Top-k expert selection + renormalized softmax over the selected logits.
// Router widths are tiny (e.g. 128), so a single thread suffices.
kernel void moe_topk(
    device const float* logits [[buffer(0)]],
    device uint*        idx    [[buffer(1)]],
    device float*       wts    [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    constant uint&      kk     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;
    const uint KMAX = 16u;
    uint k = min(kk, KMAX);
    float best[KMAX];
    uint  bidx[KMAX];
    for (uint j = 0; j < k; ++j) { best[j] = -INFINITY; bidx[j] = 0u; }
    for (uint i = 0; i < n; ++i) {
        float v = logits[i];
        if (v > best[k - 1u]) {
            uint j = k - 1u;
            while (j > 0u && v > best[j - 1u]) {
                best[j] = best[j - 1u]; bidx[j] = bidx[j - 1u]; --j;
            }
            best[j] = v; bidx[j] = i;
        }
    }
    float mx = best[0];
    float sum = 0.0f;
    for (uint j = 0; j < k; ++j) { float e = exp(best[j] - mx); wts[j] = e; sum += e; }
    float inv = 1.0f / sum;
    for (uint j = 0; j < k; ++j) { wts[j] *= inv; idx[j] = bidx[j]; }
}

// All-experts fused gate+up+activation. 2D grid (rows x k): tgpig.y = slot,
// computing hidden[slot*out_dim + o] for every selected expert in one dispatch.
kernel void moe_gate_up_all(
    device const uint*  wg         [[buffer(0)]],
    device const uint*  wu         [[buffer(1)]],
    device const float* x          [[buffer(2)]],
    device float*       hidden     [[buffer(3)]],
    device const float* sg         [[buffer(4)]],
    device const float* bg         [[buffer(5)]],
    device const float* su         [[buffer(6)]],
    device const float* bu         [[buffer(7)]],
    constant uint&      in_dim     [[buffer(8)]],
    constant uint&      out_dim    [[buffer(9)]],
    constant uint&      group_size [[buffer(10)]],
    device const uint*  eidx       [[buffer(11)]],
    constant uint&      geglu      [[buffer(12)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort sgi   [[simdgroup_index_in_threadgroup]],
    ushort sgs   [[simdgroups_per_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]])
{
    uint o = tgpig.x * sgs + sgi;
    if (o >= out_dim) return;
    uint slot = tgpig.y;
    uint e = eidx[slot];
    uint words = in_dim / 8u, groups = in_dim / group_size, wpg = group_size / 8u;
    ulong base = ((ulong)e * out_dim + o);
    device const uint*  rg  = wg + base * words;
    device const uint*  ru  = wu + base * words;
    device const float* sgr = sg + base * groups;
    device const float* bgr = bg + base * groups;
    device const float* sur = su + base * groups;
    device const float* bur = bu + base * groups;
    float ag = 0.0f, au = 0.0f;
    for (uint wi = lane; wi < words; wi += 32u) {
        uint gw = rg[wi], uw = ru[wi], g = wi / wpg;
        float sgv = sgr[g], bgv = bgr[g], suv = sur[g], buv = bur[g];
        device const float* xb = x + wi * 8u;
        for (uint n = 0; n < 8u; ++n) {
            float xi = xb[n];
            ag += ((float)((gw >> (n * 4u)) & 0xFu) * sgv + bgv) * xi;
            au += ((float)((uw >> (n * 4u)) & 0xFu) * suv + buv) * xi;
        }
    }
    ag = simd_sum(ag);
    au = simd_sum(au);
    if (lane == 0) {
        float act;
        if (geglu != 0u) {
            const float c = 0.7978845608028654f;
            float arg = clamp(c * (ag + 0.044715f * ag * ag * ag), -30.0f, 30.0f);
            act = 0.5f * ag * (1.0f + tanh(arg));
        } else {
            act = ag / (1.0f + exp(-ag));
        }
        hidden[(ulong)slot * out_dim + o] = act * au;
    }
}

// All-experts down projection. 2D grid (rows x k): out[slot*out_dim + o] =
// W_down[eidx[slot]][o] · hidden[slot].
kernel void moe_down_all(
    device const uint*  wd         [[buffer(0)]],
    device const float* hidden     [[buffer(1)]],
    device float*       out        [[buffer(2)]],
    device const float* sd         [[buffer(3)]],
    device const float* bd         [[buffer(4)]],
    constant uint&      in_dim     [[buffer(5)]],
    constant uint&      out_dim    [[buffer(6)]],
    constant uint&      group_size [[buffer(7)]],
    device const uint*  eidx       [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort sgi   [[simdgroup_index_in_threadgroup]],
    ushort sgs   [[simdgroups_per_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]])
{
    uint o = tgpig.x * sgs + sgi;
    if (o >= out_dim) return;
    uint slot = tgpig.y;
    uint e = eidx[slot];
    uint words = in_dim / 8u, groups = in_dim / group_size, wpg = group_size / 8u;
    ulong base = ((ulong)e * out_dim + o);
    device const uint*  row  = wd + base * words;
    device const float* srow = sd + base * groups;
    device const float* brow = bd + base * groups;
    device const float* h    = hidden + (ulong)slot * in_dim;
    float acc = 0.0f;
    for (uint wi = lane; wi < words; wi += 32u) {
        uint word = row[wi], g = wi / wpg;
        float sc = srow[g], bi = brow[g];
        device const float* hb = h + wi * 8u;
        for (uint n = 0; n < 8u; ++n) {
            acc += ((float)((word >> (n * 4u)) & 0xFu) * sc + bi) * hb[n];
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) out[(ulong)slot * out_dim + o] = acc;
}

// Combine the experts' down outputs: xb2[o] = sum_slot wts[slot] * down[slot,o].
kernel void moe_combine(
    device const float* down [[buffer(0)]],
    device const float* wts  [[buffer(1)]],
    device float*       xb2  [[buffer(2)]],
    constant uint&      n    [[buffer(3)]],
    constant uint&      k    [[buffer(4)]],
    uint o [[thread_position_in_grid]])
{
    if (o >= n) return;
    float acc = 0.0f;
    for (uint s = 0; s < k; ++s) acc += wts[s] * down[(ulong)s * n + o];
    xb2[o] = acc;
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

// Batched BF16 matmul (prompt prefill): W[out,in] x X[S,in] -> Y[S,out], both
// activation matrices row-major (token-major). Each simdgroup computes one output
// row for a tile of T columns, reading W[o] ONCE and reusing it across the tile —
// the key to fast prefill (vs S separate matvecs each re-reading the weight).
kernel void matmul_bf16(
    device const ushort* w       [[buffer(0)]],
    device const float*  x       [[buffer(1)]],
    device float*        y       [[buffer(2)]],
    constant uint&       in_dim  [[buffer(3)]],
    constant uint&       out_dim [[buffer(4)]],
    constant uint&       n_cols  [[buffer(5)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort nsg   [[simdgroups_per_threadgroup]])
{
    const uint T = 8;
    uint o = (uint)tgpig.x * nsg + sgitg;
    uint col0 = (uint)tgpig.y * T;
    if (o >= out_dim) return;
    device const ushort* wrow = w + (uint)o * in_dim;
    float acc[T]; for (uint t = 0; t < T; ++t) acc[t] = 0.f;
    for (uint i = tiisg; i < in_dim; i += 32u) {
        float wv = as_type<float>((uint)wrow[i] << 16);
        for (uint t = 0; t < T; ++t) {
            uint s = col0 + t;
            if (s < n_cols) acc[t] += wv * x[s * in_dim + i];
        }
    }
    for (uint t = 0; t < T; ++t) {
        float r = simd_sum(acc[t]);
        uint s = col0 + t;
        if (tiisg == 0 && s < n_cols) y[s * out_dim + o] = r;
    }
}

// Batched MLX 4-bit matmul (prompt prefill): packed-u32 weights dequantized in
// the kernel (q*scale+bias, 8 nibbles/word LSB-first), W[out,in] x X[S,in] ->
// Y[S,out]. One simdgroup per output row computes a tile of T columns, reading
// and dequantizing W[o] ONCE per word and reusing it across the tile.
kernel void matmul_mlx4(
    device const uint*  w          [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device float*       y          [[buffer(2)]],
    device const float* scales     [[buffer(3)]],
    device const float* biases     [[buffer(4)]],
    constant uint&      in_dim     [[buffer(5)]],
    constant uint&      out_dim    [[buffer(6)]],
    constant uint&      group_size [[buffer(7)]],
    constant uint&      n_cols     [[buffer(8)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort nsg   [[simdgroups_per_threadgroup]])
{
    const uint T = 8;
    uint o = (uint)tgpig.x * nsg + sgitg;
    uint col0 = (uint)tgpig.y * T;
    if (o >= out_dim) return;
    uint words = in_dim / 8u;
    uint wpg = group_size / 8u;
    uint groups = in_dim / group_size;
    device const uint*  wrow = w + (uint)o * words;
    device const float* srow = scales + (uint)o * groups;
    device const float* brow = biases + (uint)o * groups;
    float acc[T]; for (uint t = 0; t < T; ++t) acc[t] = 0.f;
    for (uint wi = tiisg; wi < words; wi += 32u) {
        uint word = wrow[wi];
        uint g = wi / wpg;
        float sc = srow[g], bi = brow[g];
        for (uint n = 0; n < 8u; ++n) {
            float wv = (float)((word >> (n * 4u)) & 0xFu) * sc + bi;
            uint i = wi * 8u + n;
            for (uint t = 0; t < T; ++t) {
                uint s = col0 + t;
                if (s < n_cols) acc[t] += wv * x[s * in_dim + i];
            }
        }
    }
    for (uint t = 0; t < T; ++t) {
        float r = simd_sum(acc[t]);
        uint s = col0 + t;
        if (tiisg == 0 && s < n_cols) y[s * out_dim + o] = r;
    }
}

// Batched Q4_K matmul (prompt prefill): W[out,in] x X[S,in] -> Y[S,out]. One
// simdgroup per output row, tile of T token columns. The row's 32-weight
// sub-blocks (8 per 256-block) are spread one-per-lane so all 32 lanes stay
// busy; each lane dequantizes its sub-block ONCE and reuses every weight across
// the T columns (amortizing both the weight read AND the dequant). Mirrors
// matvec_q4k's scale unpacking; reduced with simd_sum.
kernel void matmul_q4k(
    device const uchar* w       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       y       [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    constant uint&      n_cols  [[buffer(5)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort nsg   [[simdgroups_per_threadgroup]])
{
    const uint T = 8;
    uint o = (uint)tgpig.x * nsg + sgitg;
    uint col0 = (uint)tgpig.y * T;
    if (o >= out_dim) return;
    uint blocks = in_dim / 256u;
    uint subs = blocks * 8u; // 32-weight sub-blocks in this row
    device const uchar* row = w + (uint)o * blocks * 144u;
    float acc[T]; for (uint t = 0; t < T; ++t) acc[t] = 0.f;
    for (uint sb = tiisg; sb < subs; sb += 32u) {
        uint b = sb / 8u, si = sb % 8u;
        device const uchar* blk = row + b * 144u;
        float d    = (float)as_type<half>((ushort)(blk[0] | (blk[1] << 8)));
        float dmin = (float)as_type<half>((ushort)(blk[2] | (blk[3] << 8)));
        device const uchar* sc = blk + 4;
        device const uchar* qs = blk + 16;
        uchar sd, sm;
        if (si < 4u) { sd = sc[si] & 63; sm = sc[si+4] & 63; }
        else         { sd = (sc[si+4] & 0xF) | ((sc[si-4] >> 6) << 4);
                       sm = (sc[si+4] >> 4)  | ((sc[si]   >> 6) << 4); }
        float dscale = d * (float)sd, dm = dmin * (float)sm;
        uint qoff = (si / 2u) * 32u;     // qs byte offset for this sub-block
        bool low = (si & 1u) == 0u;      // even sub-block = low nibble
        uint col = b * 256u + si * 32u;  // first activation column
        // Dequantize the 32 weights ONCE into registers, then sweep each token
        // column with 32 consecutive activation reads (better x locality).
        float wv[32];
        for (uint l = 0; l < 32u; ++l) {
            uchar qb = qs[qoff + l];
            float q = low ? (float)(qb & 0xF) : (float)(qb >> 4);
            wv[l] = dscale * q - dm;
        }
        for (uint t = 0; t < T; ++t) {
            uint s = col0 + t;
            if (s >= n_cols) break;
            device const float* xr = x + s * in_dim + col;
            float a = 0.f;
            for (uint l = 0; l < 32u; ++l) a += wv[l] * xr[l];
            acc[t] += a;
        }
    }
    for (uint t = 0; t < T; ++t) {
        float r = simd_sum(acc[t]);
        uint s = col0 + t;
        if (tiisg == 0 && s < n_cols) y[s * out_dim + o] = r;
    }
}

// Batched Q6_K matmul (prompt prefill): the row's 16-weight sub-blocks (16 per
// 256-block, one signed scale each) are spread one-per-lane for full 32-lane
// occupancy; each lane dequantizes its 16 weights ONCE into registers, then
// sweeps the T token columns with 16 consecutive activation reads. Mirrors
// matvec_q6k's ql/qh bit layout; reduced with simd_sum.
kernel void matmul_q6k(
    device const uchar* w       [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device float*       y       [[buffer(2)]],
    constant uint&      in_dim  [[buffer(3)]],
    constant uint&      out_dim [[buffer(4)]],
    constant uint&      n_cols  [[buffer(5)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort nsg   [[simdgroups_per_threadgroup]])
{
    const uint T = 4; // q6k stays at 4 — acc[8]+wv[16] spills registers and regresses
    uint o = (uint)tgpig.x * nsg + sgitg;
    uint col0 = (uint)tgpig.y * T;
    if (o >= out_dim) return;
    uint blocks = in_dim / 256u;
    uint subs = blocks * 16u; // 16-weight sub-blocks in this row
    device const uchar* row = w + (uint)o * blocks * 210u;
    float acc[T]; for (uint t = 0; t < T; ++t) acc[t] = 0.f;
    for (uint sb = tiisg; sb < subs; sb += 32u) {
        uint b = sb / 16u, k = sb % 16u;       // block, scale index 0..15
        device const uchar* blk = row + b * 210u;
        device const uchar* ql = blk;
        device const uchar* qh = blk + 128;
        device const char*  sc = (device const char*)(blk + 192);
        float d = (float)as_type<half>((ushort)(blk[208] | (blk[209] << 8)));
        uint nh = k / 8u, rem = k % 8u, g = rem / 2u, is = rem % 2u;
        uint qlo = nh * 64u, qho = nh * 32u;
        uint ql_off = (g & 1u) * 32u;          // q2/q4 read the +32 half
        bool hi = g >= 2u;                      // q3/q4 use the high nibble
        uint qhs = g * 2u;                      // qh bit pair for this group
        float sk = d * (float)sc[k];
        uint base = b * 256u + nh * 128u + g * 32u + is * 16u;
        float wv[16];
        for (uint ll = 0; ll < 16u; ++ll) {
            uint idx = is * 16u + ll;
            uchar qlb = ql[qlo + idx + ql_off];
            uint qn = hi ? (uint)(qlb >> 4) : (uint)(qlb & 0xF);
            uint qhb = ((uint)qh[qho + idx] >> qhs) & 3u;
            wv[ll] = sk * (float)((int)(qn | (qhb << 4)) - 32);
        }
        for (uint t = 0; t < T; ++t) {
            uint s = col0 + t;
            if (s >= n_cols) break;
            device const float* xr = x + s * in_dim + base;
            float a = 0.f;
            for (uint ll = 0; ll < 16u; ++ll) a += wv[ll] * xr[ll];
            acc[t] += a;
        }
    }
    for (uint t = 0; t < T; ++t) {
        float r = simd_sum(acc[t]);
        uint s = col0 + t;
        if (tiisg == 0 && s < n_cols) y[s * out_dim + o] = r;
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
    constant uint&      start    [[buffer(9)]],  // sliding-window first key
    uint2 gid [[thread_position_in_grid]])       // x = t, y = h
{
    uint t = gid.x, h = gid.y;
    if (t < start || t >= seqlen) return;
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
    constant uint& start  [[buffer(3)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nt  [[threads_per_threadgroup]])
{
    device float* row = scores + h * stride;
    threadgroup float sh[1024];
    float m = -INFINITY;
    for (uint t = start + tid; t < seqlen; t += nt) m = max(m, row[t]);
    sh[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0u; s >>= 1) {
        if (tid < s) sh[tid] = max(sh[tid], sh[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mx = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint t = start + tid; t < seqlen; t += nt) { float e = exp(row[t] - mx); row[t] = e; sum += e; }
    sh[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nt / 2u; s > 0u; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sh[0];
    for (uint t = start + tid; t < seqlen; t += nt) row[t] *= inv;
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
    constant uint&      start    [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])       // x = d, y = h
{
    uint d = gid.x, h = gid.y;
    if (d >= head_dim) return;
    uint kvh = h / kv_mul;
    device const float* row = scores + h * stride;
    float acc = 0.0f;
    for (uint t = start; t < seqlen; ++t) acc += row[t] * vcache[t * kv_dim + kvh * head_dim + d];
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
