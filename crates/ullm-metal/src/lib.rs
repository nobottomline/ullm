//! Metal (Apple GPU) compute backend.
//!
//! GEMV kernels for f32 and for quantized weights with **dequantization in the
//! kernel** (Q4_K, Q6_K): the weights stay quantized in GPU memory and are
//! decoded on the fly, so the GPU streams ~4-7x fewer bytes than f32 — the main
//! reason Apple-Silicon GPUs win at memory-bound decode. Buffers use shared
//! storage (unified memory: no host<->device copy). Validated against the CPU.

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use ullm_core::{DType, Error, Result};

const SHADER: &str = r#"
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
