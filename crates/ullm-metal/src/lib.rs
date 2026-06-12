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
pub use forward::{GpuExperts, GpuForward, GpuLayerInput, GpuModelInput, GpuParams, GpuWeight};

pub(crate) const SHADER: &str = include_str!("shader.metal");

/// A Metal device with compiled GEMV pipelines, ready to dispatch work.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    matvec_pso: ComputePipelineState,
    q4k_pso: ComputePipelineState,
    q6k_pso: ComputePipelineState,
    mlx4_pso: ComputePipelineState,
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
            mlx4_pso: pso("matvec_mlx4")?,
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

    /// MLX 4-bit GEMV: `w` is the packed u32 bytes, `scales`/`biases` the group
    /// tables. Validates the kernel; the full forward keeps weights resident.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_mlx4(
        &self,
        w: &[u8],
        scales: &[f32],
        biases: &[f32],
        x: &[f32],
        out_dim: usize,
        in_dim: usize,
        group_size: usize,
    ) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let f32buf = |v: &[f32]| {
            self.device
                .new_buffer_with_data(v.as_ptr().cast(), (v.len() * 4) as u64, shared)
        };
        let wbuf = self.upload(w);
        let sbuf = f32buf(scales);
        let bbuf = f32buf(biases);
        let xbuf = f32buf(x);
        let ybuf = self.device.new_buffer((out_dim * 4) as u64, shared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.mlx4_pso);
        enc.set_buffer(0, Some(&wbuf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        enc.set_buffer(3, Some(&sbuf), 0);
        enc.set_buffer(4, Some(&bbuf), 0);
        let (in_u, out_u, gs) = (in_dim as u32, out_dim as u32, group_size as u32);
        enc.set_bytes(5, 4, (&in_u as *const u32).cast());
        enc.set_bytes(6, 4, (&out_u as *const u32).cast());
        enc.set_bytes(7, 4, (&gs as *const u32).cast());
        let threads = 256u64;
        let groups = (out_dim as u64).div_ceil(threads / 32);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(threads, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut out = vec![0.0f32; out_dim];
        // SAFETY: shared storage; `ybuf` holds `out_dim` f32 after completion.
        unsafe {
            std::ptr::copy_nonoverlapping(ybuf.contents().cast::<f32>(), out.as_mut_ptr(), out_dim);
        }
        out
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

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn mlx4_gemv_matches_cpu() {
        let Ok(ctx) = MetalContext::new() else {
            return; // no GPU (e.g. CI) — skip
        };
        let (out, inn, gs) = (96usize, 256usize, 64usize);
        let words = inn / 8;
        let groups = inn / gs;
        // Deterministic pseudo-random packed weights / scales / biases / x.
        let w: Vec<u8> = (0..out * words * 4)
            .map(|k| (k.wrapping_mul(131).wrapping_add(7) % 251) as u8)
            .collect();
        let scales: Vec<f32> = (0..out * groups)
            .map(|k| ((k % 7) as f32 - 3.0) * 0.01)
            .collect();
        let biases: Vec<f32> = (0..out * groups)
            .map(|k| ((k % 5) as f32 - 2.0) * 0.02)
            .collect();
        let x: Vec<f32> = (0..inn).map(|k| ((k % 13) as f32 - 6.0) * 0.1).collect();

        // CPU reference: value = q*scale + bias, then dot.
        let cpu: Vec<f32> = (0..out)
            .map(|o| {
                let mut acc = 0.0f32;
                for i in 0..inn {
                    let wb = (o * words + i / 8) * 4;
                    let word = u32::from_le_bytes([w[wb], w[wb + 1], w[wb + 2], w[wb + 3]]);
                    let q = (word >> ((i % 8) * 4)) & 0xF;
                    let g = o * groups + i / gs;
                    acc += (q as f32 * scales[g] + biases[g]) * x[i];
                }
                acc
            })
            .collect();
        let gpu = ctx.matvec_mlx4(&w, &scales, &biases, &x, out, inn, gs);
        assert!(rel_err(&gpu, &cpu) < 1e-3, "MLX4 kernel mismatch");
    }
}
