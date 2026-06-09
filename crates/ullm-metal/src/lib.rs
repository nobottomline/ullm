//! Metal (Apple GPU) compute backend.
//!
//! Phase 1 foundation: a validated GEMV (matrix-vector) kernel. Buffers use
//! shared storage, so on Apple Silicon's unified memory there is no host<->device
//! copy — the CPU and GPU address the same bytes. This is the seed the rest of
//! the GPU forward pass grows from; it is validated against the CPU reference.

use metal::{CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use ullm_core::{Error, Result};

/// The GEMV kernel: `y[o] = sum_i w[o*in + i] * x[i]`, one thread per output.
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
    for (uint i = 0; i < in_dim; ++i) {
        s += row[i] * x[i];
    }
    y[o] = s;
}
"#;

/// A Metal device with a compiled GEMV pipeline, ready to dispatch work.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    matvec_pso: ComputePipelineState,
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
        let func = library
            .get_function("matvec", None)
            .map_err(|e| Error::Format(format!("metal function 'matvec' missing: {e}")))?;
        let matvec_pso = device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| Error::Format(format!("metal pipeline creation failed: {e}")))?;
        Ok(Self {
            device,
            queue,
            matvec_pso,
        })
    }

    /// Human-readable name of the GPU.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Compute `y[o] = sum_i w[o*in + i] * x[i]` on the GPU. `w` is row-major
    /// `[out_dim, in_dim]`.
    pub fn matvec(&self, w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let wbuf =
            self.device
                .new_buffer_with_data(w.as_ptr().cast(), (w.len() * 4) as u64, shared);
        let xbuf =
            self.device
                .new_buffer_with_data(x.as_ptr().cast(), (x.len() * 4) as u64, shared);
        let ybuf = self.device.new_buffer((out_dim * 4) as u64, shared);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.matvec_pso);
        enc.set_buffer(0, Some(&wbuf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&ybuf), 0);
        let in_dim_u32 = in_dim as u32;
        enc.set_bytes(3, 4, (&in_dim_u32 as *const u32).cast());

        let tpt = self
            .matvec_pso
            .max_total_threads_per_threadgroup()
            .min(out_dim as u64)
            .max(1);
        enc.dispatch_threads(MTLSize::new(out_dim as u64, 1, 1), MTLSize::new(tpt, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut out = vec![0.0f32; out_dim];
        // SAFETY: `ybuf` holds `out_dim` f32 written by the kernel; shared storage
        // means the bytes are visible to the CPU after `wait_until_completed`.
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

    #[test]
    fn gemv_matches_cpu_when_gpu_present() {
        // Skip gracefully on machines / CI without a usable Metal device.
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (out_dim, in_dim) = (512usize, 1024usize);
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

        let gpu = ctx.matvec(&w, &x, out_dim, in_dim);
        let cpu = cpu_matvec(&w, &x, out_dim, in_dim);

        let max_err = gpu
            .iter()
            .zip(&cpu)
            .map(|(g, c)| (g - c).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-2, "gpu/cpu mismatch: {max_err}");
    }
}
