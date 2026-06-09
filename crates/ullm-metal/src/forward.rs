//! Full transformer forward pass on the GPU.
//!
//! Weights, activations, and the KV cache all live in GPU buffers; every layer's
//! operations (matvec, RMSNorm, RoPE, attention, GeGLU/SwiGLU, residual add) are
//! encoded into a *single* command buffer per token and committed once. The only
//! host<->device traffic per token is uploading the input embedding and reading
//! back the logits, so the per-dispatch sync overhead that makes single GEMVs
//! pointless on the GPU is amortized across the whole forward.

use metal::{
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use ullm_core::{DType, Error, Result};

use crate::SHADER;

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;
/// Threads per threadgroup for the reduction kernels (power of two).
const REDUCE_NT: u64 = 256;

/// Scalar hyperparameters and architecture toggles for the GPU forward.
#[derive(Debug, Clone, Copy)]
pub struct GpuParams {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// NeoX (rotate-half) RoPE when true, interleaved otherwise.
    pub rope_neox: bool,
    /// Apply per-head Q/K RMSNorm before RoPE (Gemma / Qwen3).
    pub qk_norm: bool,
    /// Apply post-attention / post-FFN RMSNorms before the residual add (Gemma).
    pub sandwich_norm: bool,
    /// GeGLU feed-forward when true, SwiGLU otherwise.
    pub geglu: bool,
}

/// A weight matrix to upload: raw (possibly quantized) bytes plus its shape.
pub struct GpuWeight<'a> {
    pub dtype: DType,
    pub bytes: &'a [u8],
    pub out: usize,
    pub cols: usize,
}

/// Per-layer weights handed to the GPU loader (norms as f32, projections raw).
pub struct GpuLayerInput<'a> {
    pub attn_norm: &'a [f32],
    pub wq: GpuWeight<'a>,
    pub wk: GpuWeight<'a>,
    pub wv: GpuWeight<'a>,
    pub wo: GpuWeight<'a>,
    pub q_bias: Option<&'a [f32]>,
    pub k_bias: Option<&'a [f32]>,
    pub v_bias: Option<&'a [f32]>,
    pub q_norm: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
    pub post_attn_norm: Option<&'a [f32]>,
    pub post_ffn_norm: Option<&'a [f32]>,
    pub ffn_norm: &'a [f32],
    pub w_gate: GpuWeight<'a>,
    pub w_up: GpuWeight<'a>,
    pub w_down: GpuWeight<'a>,
}

/// The whole model handed to the GPU loader. The input embedding is looked up
/// on the host (one dequantized row per token), so only the output projection is
/// uploaded here.
pub struct GpuModelInput<'a> {
    pub params: GpuParams,
    pub output: GpuWeight<'a>,
    pub final_norm: &'a [f32],
    pub layers: Vec<GpuLayerInput<'a>>,
}

/// A resident weight matrix on the GPU.
struct WBuf {
    buf: Buffer,
    dtype: DType,
    out: usize,
    cols: usize,
}

struct GpuLayer {
    attn_norm: Buffer,
    wq: WBuf,
    wk: WBuf,
    wv: WBuf,
    wo: WBuf,
    q_bias: Option<Buffer>,
    k_bias: Option<Buffer>,
    v_bias: Option<Buffer>,
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
    post_attn_norm: Option<Buffer>,
    post_ffn_norm: Option<Buffer>,
    ffn_norm: Buffer,
    w_gate: WBuf,
    w_up: WBuf,
    w_down: WBuf,
}

/// A model resident on the GPU, ready to decode tokens.
pub struct GpuForward {
    queue: CommandQueue,
    // pipelines
    p_matvec_f32: ComputePipelineState,
    p_matvec_q4k: ComputePipelineState,
    p_matvec_q6k: ComputePipelineState,
    p_rmsnorm: ComputePipelineState,
    p_rmsnorm_heads: ComputePipelineState,
    p_rope_neox: ComputePipelineState,
    p_rope_norm: ComputePipelineState,
    p_attn_scores: ComputePipelineState,
    p_attn_softmax: ComputePipelineState,
    p_attn_output: ComputePipelineState,
    p_silu_mul: ComputePipelineState,
    p_gelu_mul: ComputePipelineState,
    p_add: ComputePipelineState,
    // config
    p: GpuParams,
    // weights
    output: WBuf,
    final_norm: Buffer,
    layers: Vec<GpuLayer>,
    // activation scratch (reused every token)
    x: Buffer,
    xb: Buffer,
    xb2: Buffer,
    q: Buffer,
    attn: Buffer,
    gate: Buffer,
    up: Buffer,
    hidden: Buffer,
    scores: Buffer,
    logits: Buffer,
    key_cache: Buffer,
    val_cache: Buffer,
}

// SAFETY: the Metal handles are only ever touched while decoding a single token,
// and the runtime serializes inference on a model (the server holds it behind a
// Mutex; the CLI is single-threaded). We never issue concurrent GPU work against
// one `GpuForward`, so moving it between threads is sound.
unsafe impl Send for GpuForward {}

impl GpuForward {
    /// Upload a model to the GPU and compile the forward kernels.
    pub fn new(input: &GpuModelInput) -> Result<Self> {
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

        let p = input.params;
        let upload_bytes = |b: &[u8]| -> Buffer {
            device.new_buffer_with_data(b.as_ptr().cast(), b.len() as u64, SHARED)
        };
        let upload_f32 = |v: &[f32]| -> Buffer {
            device.new_buffer_with_data(v.as_ptr().cast(), (v.len() * 4) as u64, SHARED)
        };
        let wbuf = |w: &GpuWeight| -> WBuf {
            WBuf {
                buf: upload_bytes(w.bytes),
                dtype: w.dtype,
                out: w.out,
                cols: w.cols,
            }
        };
        let opt = |o: Option<&[f32]>| o.map(upload_f32);

        let layers = input
            .layers
            .iter()
            .map(|l| GpuLayer {
                attn_norm: upload_f32(l.attn_norm),
                wq: wbuf(&l.wq),
                wk: wbuf(&l.wk),
                wv: wbuf(&l.wv),
                wo: wbuf(&l.wo),
                q_bias: opt(l.q_bias),
                k_bias: opt(l.k_bias),
                v_bias: opt(l.v_bias),
                q_norm: opt(l.q_norm),
                k_norm: opt(l.k_norm),
                post_attn_norm: opt(l.post_attn_norm),
                post_ffn_norm: opt(l.post_ffn_norm),
                ffn_norm: upload_f32(l.ffn_norm),
                w_gate: wbuf(&l.w_gate),
                w_up: wbuf(&l.w_up),
                w_down: wbuf(&l.w_down),
            })
            .collect();

        let kv_dim = p.n_kv_head * p.head_dim;
        let q_dim = p.n_head * p.head_dim;
        let alloc = |n: usize| device.new_buffer((n * 4) as u64, SHARED);

        Ok(Self {
            p_matvec_f32: pso("matvec")?,
            p_matvec_q4k: pso("matvec_q4k")?,
            p_matvec_q6k: pso("matvec_q6k")?,
            p_rmsnorm: pso("rmsnorm")?,
            p_rmsnorm_heads: pso("rmsnorm_heads")?,
            p_rope_neox: pso("rope_neox")?,
            p_rope_norm: pso("rope_norm")?,
            p_attn_scores: pso("attn_scores")?,
            p_attn_softmax: pso("attn_softmax")?,
            p_attn_output: pso("attn_output")?,
            p_silu_mul: pso("silu_mul")?,
            p_gelu_mul: pso("gelu_mul")?,
            p_add: pso("add_inplace")?,
            output: wbuf(&input.output),
            final_norm: upload_f32(input.final_norm),
            layers,
            x: alloc(p.n_embd),
            xb: alloc(p.n_embd),
            xb2: alloc(p.n_embd),
            q: alloc(q_dim),
            attn: alloc(q_dim),
            gate: alloc(p.n_ff),
            up: alloc(p.n_ff),
            hidden: alloc(p.n_ff),
            scores: alloc(p.n_head * p.n_ctx),
            logits: alloc(p.vocab),
            key_cache: alloc(p.n_layer * p.n_ctx * kv_dim),
            val_cache: alloc(p.n_layer * p.n_ctx * kv_dim),
            queue,
            p,
        })
    }

    /// The matvec pipeline for a weight's dtype.
    fn pso_matvec(&self, dtype: DType) -> Result<&ComputePipelineState> {
        match dtype {
            DType::F32 => Ok(&self.p_matvec_f32),
            DType::Q4K => Ok(&self.p_matvec_q4k),
            DType::Q6K => Ok(&self.p_matvec_q6k),
            other => Err(Error::Unsupported(format!(
                "no Metal forward kernel for {other:?}"
            ))),
        }
    }

    /// Decode one token: `x_init` is the (already scaled) input embedding for
    /// `token` at sequence position `pos`. Returns the logits over the vocab.
    pub fn forward(&self, x_init: &[f32], pos: usize) -> Result<Vec<f32>> {
        // Upload the input embedding into the residual-stream buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                x_init.as_ptr(),
                self.x.contents().cast::<f32>(),
                self.p.n_embd,
            );
        }

        let p = self.p;
        let kv_dim = p.n_kv_head * p.head_dim;
        let q_dim = p.n_head * p.head_dim;
        let kv_mul = (p.n_head / p.n_kv_head) as u32;
        let seqlen = (pos + 1) as u32;
        let scale = (p.head_dim as f32).sqrt().recip();
        let rope_pso = if p.rope_neox {
            &self.p_rope_neox
        } else {
            &self.p_rope_norm
        };

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        for (l, lw) in self.layers.iter().enumerate() {
            let kv_layer_off = (l * p.n_ctx * kv_dim * 4) as u64;
            let kv_pos_off = ((l * p.n_ctx + pos) * kv_dim * 4) as u64;

            // attention pre-norm
            self.rmsnorm(enc, &self.x, 0, &lw.attn_norm, &self.xb, 0, p.n_embd);
            self.matvec(enc, &lw.wq, &self.xb, &self.q, 0);
            self.matvec(enc, &lw.wk, &self.xb, &self.key_cache, kv_pos_off);
            self.matvec(enc, &lw.wv, &self.xb, &self.val_cache, kv_pos_off);

            // Q/K/V attention biases (Qwen2).
            if let Some(b) = &lw.q_bias {
                self.add(enc, &self.q, b, q_dim);
            }
            if let Some(b) = &lw.k_bias {
                self.add_off(enc, &self.key_cache, kv_pos_off, b, kv_dim);
            }
            if let Some(b) = &lw.v_bias {
                self.add_off(enc, &self.val_cache, kv_pos_off, b, kv_dim);
            }

            if p.qk_norm {
                if let Some(qn) = &lw.q_norm {
                    self.rmsnorm_heads(enc, &self.q, 0, qn, p.n_head as u64);
                }
                if let Some(kn) = &lw.k_norm {
                    self.rmsnorm_heads(enc, &self.key_cache, kv_pos_off, kn, p.n_kv_head as u64);
                }
            }

            self.rope(enc, rope_pso, &self.q, 0, p.n_head as u32, pos as u32);
            self.rope(enc, rope_pso, &self.key_cache, kv_pos_off, p.n_kv_head as u32, pos as u32);

            // attention
            self.attn_scores(enc, kv_layer_off, kv_mul, scale, seqlen);
            self.attn_softmax(enc, seqlen);
            self.attn_output(enc, kv_layer_off, kv_mul, seqlen);

            // output projection into xb (n_embd), optional post-norm, residual
            self.matvec(enc, &lw.wo, &self.attn, &self.xb, 0);
            if p.sandwich_norm {
                if let Some(w) = &lw.post_attn_norm {
                    self.rmsnorm(enc, &self.xb, 0, w, &self.xb, 0, p.n_embd);
                }
            }
            self.add(enc, &self.x, &self.xb, p.n_embd);

            // feed-forward
            self.rmsnorm(enc, &self.x, 0, &lw.ffn_norm, &self.xb, 0, p.n_embd);
            self.matvec(enc, &lw.w_gate, &self.xb, &self.gate, 0);
            self.matvec(enc, &lw.w_up, &self.xb, &self.up, 0);
            let ffn_pso = if p.geglu { &self.p_gelu_mul } else { &self.p_silu_mul };
            self.glu(enc, ffn_pso, p.n_ff);
            self.matvec(enc, &lw.w_down, &self.hidden, &self.xb2, 0);
            if p.sandwich_norm {
                if let Some(w) = &lw.post_ffn_norm {
                    self.rmsnorm(enc, &self.xb2, 0, w, &self.xb2, 0, p.n_embd);
                }
            }
            self.add(enc, &self.x, &self.xb2, p.n_embd);
        }

        // final norm + output projection
        self.rmsnorm(enc, &self.x, 0, &self.final_norm, &self.xb, 0, p.n_embd);
        self.matvec(enc, &self.output, &self.xb, &self.logits, 0);

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let mut out = vec![0.0f32; p.vocab];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.logits.contents().cast::<f32>(),
                out.as_mut_ptr(),
                p.vocab,
            );
        }
        Ok(out)
    }

    /// Step through layer 0 one op at a time, reporting NaN/inf/max for each
    /// intermediate. Used to localize numerical breakage in the GPU forward.
    pub fn forward_debug(&self, x_init: &[f32], pos: usize) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                x_init.as_ptr(),
                self.x.contents().cast::<f32>(),
                self.p.n_embd,
            );
        }
        let p = self.p;
        let kv_dim = p.n_kv_head * p.head_dim;
        let q_dim = p.n_head * p.head_dim;
        let kv_mul = (p.n_head / p.n_kv_head) as u32;
        let seqlen = (pos + 1) as u32;
        let scale = (p.head_dim as f32).sqrt().recip();
        let rope_pso = if p.rope_neox { &self.p_rope_neox } else { &self.p_rope_norm };
        let lw = &self.layers[0];
        let kv_pos_off = (pos * kv_dim * 4) as u64;

        let run = |f: &dyn Fn(&ComputeCommandEncoderRef)| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            f(enc);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        };
        let check = |buf: &Buffer, off: u64, n: usize, label: &str| {
            let mut v = vec![0f32; n];
            unsafe {
                let base = buf.contents().cast::<u8>().add(off as usize).cast::<f32>();
                std::ptr::copy_nonoverlapping(base, v.as_mut_ptr(), n);
            }
            let nan = v.iter().filter(|x| x.is_nan()).count();
            let inf = v.iter().filter(|x| x.is_infinite()).count();
            let mx = v.iter().copied().filter(|x| x.is_finite()).fold(0f32, |a, b| a.max(b.abs()));
            eprintln!("[dbg pos{pos}] {label:<14} nan={nan} inf={inf} maxabs={mx:.4}");
        };

        check(&self.x, 0, p.n_embd, "x_init");
        run(&|e| self.rmsnorm(e, &self.x, 0, &lw.attn_norm, &self.xb, 0, p.n_embd));
        check(&self.xb, 0, p.n_embd, "attn_norm");
        run(&|e| self.matvec(e, &lw.wq, &self.xb, &self.q, 0));
        check(&self.q, 0, q_dim, "wq");
        run(&|e| self.matvec(e, &lw.wk, &self.xb, &self.key_cache, kv_pos_off));
        check(&self.key_cache, kv_pos_off, kv_dim, "wk");
        if let Some(qn) = &lw.q_norm {
            run(&|e| self.rmsnorm_heads(e, &self.q, 0, qn, p.n_head as u64));
            check(&self.q, 0, q_dim, "q_norm");
        }
        if let Some(kn) = &lw.k_norm {
            run(&|e| self.rmsnorm_heads(e, &self.key_cache, kv_pos_off, kn, p.n_kv_head as u64));
            check(&self.key_cache, kv_pos_off, kv_dim, "k_norm");
        }
        run(&|e| self.rope(e, rope_pso, &self.q, 0, p.n_head as u32, pos as u32));
        check(&self.q, 0, q_dim, "rope_q");
        run(&|e| self.rope(e, rope_pso, &self.key_cache, kv_pos_off, p.n_kv_head as u32, pos as u32));
        check(&self.key_cache, kv_pos_off, kv_dim, "rope_k");
        run(&|e| self.attn_scores(e, 0, kv_mul, scale, seqlen));
        check(&self.scores, 0, (p.n_head * seqlen as usize).min(p.n_head * p.n_ctx), "attn_scores");
        run(&|e| self.attn_softmax(e, seqlen));
        run(&|e| self.attn_output(e, 0, kv_mul, seqlen));
        check(&self.attn, 0, q_dim, "attn_out");
        run(&|e| self.matvec(e, &lw.wo, &self.attn, &self.xb, 0));
        check(&self.xb, 0, p.n_embd, "wo");
        if let Some(w) = &lw.post_attn_norm {
            run(&|e| self.rmsnorm(e, &self.xb, 0, w, &self.xb, 0, p.n_embd));
            check(&self.xb, 0, p.n_embd, "post_attn");
        }
        run(&|e| self.rmsnorm(e, &self.x, 0, &lw.ffn_norm, &self.xb, 0, p.n_embd));
        check(&self.xb, 0, p.n_embd, "ffn_norm");
        run(&|e| self.matvec(e, &lw.w_gate, &self.xb, &self.gate, 0));
        check(&self.gate, 0, p.n_ff, "gate");
        run(&|e| self.matvec(e, &lw.w_up, &self.xb, &self.up, 0));
        check(&self.up, 0, p.n_ff, "up");
        let ffn_pso = if p.geglu { &self.p_gelu_mul } else { &self.p_silu_mul };
        run(&|e| self.glu(e, ffn_pso, p.n_ff));
        check(&self.hidden, 0, p.n_ff, "glu");
        run(&|e| self.matvec(e, &lw.w_down, &self.hidden, &self.xb2, 0));
        check(&self.xb2, 0, p.n_embd, "w_down");
    }

    // ---- op encoders (each appends one dispatch to `enc`) ----

    fn matvec(&self, enc: &ComputeCommandEncoderRef, w: &WBuf, x: &Buffer, y: &Buffer, y_off: u64) {
        let pso = self.pso_matvec(w.dtype).expect("matvec dtype");
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), y_off);
        let in_dim = w.cols as u32;
        enc.set_bytes(3, 4, (&in_dim as *const u32).cast());
        dispatch_1d(enc, pso, w.out as u64);
    }

    #[allow(clippy::too_many_arguments)]
    fn rmsnorm(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        w: &Buffer,
        y: &Buffer,
        y_off: u64,
        n: usize,
    ) {
        enc.set_compute_pipeline_state(&self.p_rmsnorm);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(w), 0);
        enc.set_buffer(2, Some(y), y_off);
        let n_u = n as u32;
        enc.set_bytes(3, 4, (&n_u as *const u32).cast());
        enc.set_bytes(4, 4, (&self.p.eps as *const f32).cast());
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(REDUCE_NT, 1, 1));
    }

    fn rmsnorm_heads(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        w: &Buffer,
        n_heads: u64,
    ) {
        enc.set_compute_pipeline_state(&self.p_rmsnorm_heads);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(w), 0);
        let hd = self.p.head_dim as u32;
        enc.set_bytes(2, 4, (&hd as *const u32).cast());
        enc.set_bytes(3, 4, (&self.p.eps as *const f32).cast());
        enc.dispatch_thread_groups(MTLSize::new(n_heads, 1, 1), MTLSize::new(REDUCE_NT, 1, 1));
    }

    fn rope(
        &self,
        enc: &ComputeCommandEncoderRef,
        pso: &ComputePipelineState,
        v: &Buffer,
        v_off: u64,
        n_heads: u32,
        pos: u32,
    ) {
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(v), v_off);
        let hd = self.p.head_dim as u32;
        enc.set_bytes(1, 4, (&n_heads as *const u32).cast());
        enc.set_bytes(2, 4, (&hd as *const u32).cast());
        enc.set_bytes(3, 4, (&pos as *const u32).cast());
        enc.set_bytes(4, 4, (&self.p.rope_theta as *const f32).cast());
        let pairs = (n_heads * hd / 2) as u64;
        dispatch_1d(enc, pso, pairs);
    }

    fn attn_scores(
        &self,
        enc: &ComputeCommandEncoderRef,
        kv_layer_off: u64,
        kv_mul: u32,
        scale: f32,
        seqlen: u32,
    ) {
        enc.set_compute_pipeline_state(&self.p_attn_scores);
        enc.set_buffer(0, Some(&self.q), 0);
        enc.set_buffer(1, Some(&self.key_cache), kv_layer_off);
        enc.set_buffer(2, Some(&self.scores), 0);
        let hd = self.p.head_dim as u32;
        let kv_dim = (self.p.n_kv_head * self.p.head_dim) as u32;
        let stride = self.p.n_ctx as u32;
        enc.set_bytes(3, 4, (&hd as *const u32).cast());
        enc.set_bytes(4, 4, (&kv_dim as *const u32).cast());
        enc.set_bytes(5, 4, (&kv_mul as *const u32).cast());
        enc.set_bytes(6, 4, (&stride as *const u32).cast());
        enc.set_bytes(7, 4, (&scale as *const f32).cast());
        enc.set_bytes(8, 4, (&seqlen as *const u32).cast());
        let nh = self.p.n_head as u64;
        let tg = (seqlen as u64).clamp(1, 64);
        enc.dispatch_threads(MTLSize::new(seqlen as u64, nh, 1), MTLSize::new(tg, 1, 1));
    }

    fn attn_softmax(&self, enc: &ComputeCommandEncoderRef, seqlen: u32) {
        enc.set_compute_pipeline_state(&self.p_attn_softmax);
        enc.set_buffer(0, Some(&self.scores), 0);
        let stride = self.p.n_ctx as u32;
        enc.set_bytes(1, 4, (&stride as *const u32).cast());
        enc.set_bytes(2, 4, (&seqlen as *const u32).cast());
        let nh = self.p.n_head as u64;
        enc.dispatch_thread_groups(MTLSize::new(nh, 1, 1), MTLSize::new(REDUCE_NT, 1, 1));
    }

    fn attn_output(
        &self,
        enc: &ComputeCommandEncoderRef,
        kv_layer_off: u64,
        kv_mul: u32,
        seqlen: u32,
    ) {
        enc.set_compute_pipeline_state(&self.p_attn_output);
        enc.set_buffer(0, Some(&self.scores), 0);
        enc.set_buffer(1, Some(&self.val_cache), kv_layer_off);
        enc.set_buffer(2, Some(&self.attn), 0);
        let hd = self.p.head_dim as u32;
        let kv_dim = (self.p.n_kv_head * self.p.head_dim) as u32;
        let stride = self.p.n_ctx as u32;
        enc.set_bytes(3, 4, (&hd as *const u32).cast());
        enc.set_bytes(4, 4, (&kv_dim as *const u32).cast());
        enc.set_bytes(5, 4, (&kv_mul as *const u32).cast());
        enc.set_bytes(6, 4, (&stride as *const u32).cast());
        enc.set_bytes(7, 4, (&seqlen as *const u32).cast());
        let nh = self.p.n_head as u64;
        let hd64 = self.p.head_dim as u64;
        let tg = hd64.clamp(1, 256);
        enc.dispatch_threads(MTLSize::new(hd64, nh, 1), MTLSize::new(tg, 1, 1));
    }

    fn glu(&self, enc: &ComputeCommandEncoderRef, pso: &ComputePipelineState, n: usize) {
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&self.gate), 0);
        enc.set_buffer(1, Some(&self.up), 0);
        enc.set_buffer(2, Some(&self.hidden), 0);
        let n_u = n as u32;
        enc.set_bytes(3, 4, (&n_u as *const u32).cast());
        dispatch_1d(enc, pso, n as u64);
    }

    fn add(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, y: &Buffer, n: usize) {
        self.add_off(enc, x, 0, y, n);
    }

    fn add_off(&self, enc: &ComputeCommandEncoderRef, x: &Buffer, x_off: u64, y: &Buffer, n: usize) {
        enc.set_compute_pipeline_state(&self.p_add);
        enc.set_buffer(0, Some(x), x_off);
        enc.set_buffer(1, Some(y), 0);
        let n_u = n as u32;
        enc.set_bytes(2, 4, (&n_u as *const u32).cast());
        dispatch_1d(enc, &self.p_add, n as u64);
    }
}

/// Dispatch `n` threads (one per element/row), threadgroup sized to the kernel.
fn dispatch_1d(enc: &ComputeCommandEncoderRef, pso: &ComputePipelineState, n: u64) {
    // n is a grid dimension (>= 1), so clamping the threadgroup to [1, n] is safe.
    let tg = pso.max_total_threads_per_threadgroup().clamp(1, n.max(1));
    enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
}
