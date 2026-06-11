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
    /// Mixture-of-experts: number of experts (0 = dense FFN).
    pub n_experts: usize,
    /// Experts selected per token (top-k routing).
    pub n_experts_used: usize,
    /// Per-expert feed-forward width.
    pub moe_inter: usize,
    /// Sliding-window span (0 = full attention); Gemma local layers use it.
    pub sliding_window: usize,
}

/// A weight matrix to upload: raw (possibly quantized) bytes plus its shape.
/// For MLX 4-bit weights, `mlx_scales`/`mlx_biases` (the per-group tables) are
/// also given and `dtype` is `U32`.
pub struct GpuWeight<'a> {
    pub dtype: DType,
    pub bytes: &'a [u8],
    pub out: usize,
    pub cols: usize,
    pub mlx_scales: Option<&'a [f32]>,
    pub mlx_biases: Option<&'a [f32]>,
    pub mlx_group: usize,
}

/// Stacked per-expert MLX 4-bit weights (`n_experts` matrices concatenated).
pub struct GpuExperts {
    pub bytes: Vec<u8>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    pub n_experts: usize,
    pub out: usize,
    pub cols: usize,
    pub group: usize,
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
    /// MoE router + stacked experts (set on MoE layers; dense `w_*` then unused).
    pub moe_gate: Option<GpuWeight<'a>>,
    pub experts_gate: Option<GpuExperts>,
    pub experts_up: Option<GpuExperts>,
    pub experts_down: Option<GpuExperts>,
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

/// The per-group scale/bias buffers for a resident MLX 4-bit weight.
struct MlxBufs {
    scales: Buffer,
    biases: Buffer,
    group: u32,
}

/// A resident weight matrix on the GPU.
struct WBuf {
    buf: Buffer,
    dtype: DType,
    out: usize,
    cols: usize,
    mlx: Option<MlxBufs>,
}

/// Stacked per-expert MLX 4-bit weights resident on the GPU.
struct EBuf {
    buf: Buffer,
    scales: Buffer,
    biases: Buffer,
    out: usize,
    cols: usize,
    group: u32,
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
    moe_gate: Option<WBuf>,
    experts_gate: Option<EBuf>,
    experts_up: Option<EBuf>,
    experts_down: Option<EBuf>,
}

/// A model resident on the GPU, ready to decode tokens.
pub struct GpuForward {
    queue: CommandQueue,
    // pipelines
    p_matvec_f32: ComputePipelineState,
    p_matvec_f16: ComputePipelineState,
    p_matvec_bf16: ComputePipelineState,
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
    p_matvec_mlx4: ComputePipelineState,
    p_moe_topk: ComputePipelineState,
    p_moe_gate_up_all: ComputePipelineState,
    p_moe_down_all: ComputePipelineState,
    p_moe_combine: ComputePipelineState,
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
    // MoE scratch (allocated even for dense models)
    moe_logits: Buffer,
    moe_idx: Buffer,
    moe_wts: Buffer,
    moe_hidden: Buffer, // [n_experts_used * moe_inter]
    moe_down: Buffer,   // [n_experts_used * n_embd]
}

// SAFETY: the Metal handles are only ever touched while decoding a single token,
// and the runtime serializes inference on a model (the server holds it behind a
// Mutex; the CLI is single-threaded). We never issue concurrent GPU work against
// one `GpuForward`, so moving it between threads is sound.
unsafe impl Send for GpuForward {}

impl GpuForward {
    /// Upload a model to the GPU and compile the forward kernels.
    #[allow(clippy::redundant_closure)] // upload closures are reused, can't be moved
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
                mlx: match (w.mlx_scales, w.mlx_biases) {
                    (Some(s), Some(b)) => Some(MlxBufs {
                        scales: upload_f32(s),
                        biases: upload_f32(b),
                        group: w.mlx_group as u32,
                    }),
                    _ => None,
                },
            }
        };
        let ebuf = |e: &GpuExperts| -> EBuf {
            EBuf {
                buf: upload_bytes(&e.bytes),
                scales: upload_f32(&e.scales),
                biases: upload_f32(&e.biases),
                out: e.out,
                cols: e.cols,
                group: e.group as u32,
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
                moe_gate: l.moe_gate.as_ref().map(|w| wbuf(w)),
                experts_gate: l.experts_gate.as_ref().map(|e| ebuf(e)),
                experts_up: l.experts_up.as_ref().map(|e| ebuf(e)),
                experts_down: l.experts_down.as_ref().map(|e| ebuf(e)),
            })
            .collect();

        let kv_dim = p.n_kv_head * p.head_dim;
        let q_dim = p.n_head * p.head_dim;
        let ffn_dim = p.n_ff.max(p.moe_inter).max(1);
        let alloc = |n: usize| device.new_buffer((n * 4) as u64, SHARED);

        Ok(Self {
            p_matvec_f32: pso("matvec_f32_sg")?,
            p_matvec_f16: pso("matvec_f16_mr")?,
            p_matvec_bf16: pso("matvec_bf16_mr")?,
            p_matvec_q4k: pso("matvec_q4k_mr")?,
            p_matvec_q6k: pso("matvec_q6k_mr")?,
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
            p_matvec_mlx4: pso("matvec_mlx4")?,
            p_moe_topk: pso("moe_topk")?,
            p_moe_gate_up_all: pso("moe_gate_up_all")?,
            p_moe_down_all: pso("moe_down_all")?,
            p_moe_combine: pso("moe_combine")?,
            output: wbuf(&input.output),
            final_norm: upload_f32(input.final_norm),
            layers,
            x: alloc(p.n_embd),
            xb: alloc(p.n_embd),
            xb2: alloc(p.n_embd),
            q: alloc(q_dim),
            attn: alloc(q_dim),
            gate: alloc(ffn_dim),
            up: alloc(ffn_dim),
            hidden: alloc(ffn_dim),
            scores: alloc(p.n_head * p.n_ctx),
            logits: alloc(p.vocab),
            key_cache: alloc(p.n_layer * p.n_ctx * kv_dim),
            val_cache: alloc(p.n_layer * p.n_ctx * kv_dim),
            moe_logits: alloc(p.n_experts.max(1)),
            moe_idx: alloc(p.n_experts_used.max(1)),
            moe_wts: alloc(p.n_experts_used.max(1)),
            moe_hidden: alloc((p.n_experts_used * p.moe_inter).max(1)),
            moe_down: alloc((p.n_experts_used * p.n_embd).max(1)),
            queue,
            p,
        })
    }

    /// The matvec pipeline for a weight's dtype.
    fn pso_matvec(&self, dtype: DType) -> Result<&ComputePipelineState> {
        match dtype {
            DType::F32 => Ok(&self.p_matvec_f32),
            DType::F16 => Ok(&self.p_matvec_f16),
            DType::BF16 => Ok(&self.p_matvec_bf16),
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
            self.rope(
                enc,
                rope_pso,
                &self.key_cache,
                kv_pos_off,
                p.n_kv_head as u32,
                pos as u32,
            );

            // attention (Gemma local layers use a sliding window; every 6th is full)
            let attn_start = if p.sliding_window > 0 && l % 6 != 5 && pos + 1 > p.sliding_window {
                (pos + 1 - p.sliding_window) as u32
            } else {
                0
            };
            self.attn_scores(enc, kv_layer_off, kv_mul, scale, seqlen, attn_start);
            self.attn_softmax(enc, seqlen, attn_start);
            self.attn_output(enc, kv_layer_off, kv_mul, seqlen, attn_start);

            // output projection into xb (n_embd), optional post-norm, residual
            self.matvec(enc, &lw.wo, &self.attn, &self.xb, 0);
            if p.sandwich_norm {
                if let Some(w) = &lw.post_attn_norm {
                    self.rmsnorm(enc, &self.xb, 0, w, &self.xb, 0, p.n_embd);
                }
            }
            self.add(enc, &self.x, &self.xb, p.n_embd);

            // feed-forward: dense SwiGLU/GeGLU, or mixture-of-experts
            self.rmsnorm(enc, &self.x, 0, &lw.ffn_norm, &self.xb, 0, p.n_embd);
            let ffn_pso = if p.geglu {
                &self.p_gelu_mul
            } else {
                &self.p_silu_mul
            };
            if let Some(gate_w) = &lw.moe_gate {
                // router logits -> top-k -> weighted sum of selected experts in xb2
                self.matvec(enc, gate_w, &self.xb, &self.moe_logits, 0);
                self.moe_topk(enc);
                // All k experts in 3 dispatches: gate+up+act -> down -> combine.
                let eg = lw.experts_gate.as_ref().unwrap();
                let eu = lw.experts_up.as_ref().unwrap();
                let ed = lw.experts_down.as_ref().unwrap();
                self.moe_gate_up_all(enc, eg, eu, &self.xb);
                self.moe_down_all(enc, ed);
                self.moe_combine(enc);
            } else {
                self.matvec(enc, &lw.w_gate, &self.xb, &self.gate, 0);
                self.matvec(enc, &lw.w_up, &self.xb, &self.up, 0);
                self.glu(enc, ffn_pso, p.n_ff);
                self.matvec(enc, &lw.w_down, &self.hidden, &self.xb2, 0);
            }
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
        let rope_pso = if p.rope_neox {
            &self.p_rope_neox
        } else {
            &self.p_rope_norm
        };
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
            let mx = v
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .fold(0f32, |a, b| a.max(b.abs()));
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
        run(&|e| {
            self.rope(
                e,
                rope_pso,
                &self.key_cache,
                kv_pos_off,
                p.n_kv_head as u32,
                pos as u32,
            )
        });
        check(&self.key_cache, kv_pos_off, kv_dim, "rope_k");
        run(&|e| self.attn_scores(e, 0, kv_mul, scale, seqlen, 0));
        check(
            &self.scores,
            0,
            (p.n_head * seqlen as usize).min(p.n_head * p.n_ctx),
            "attn_scores",
        );
        run(&|e| self.attn_softmax(e, seqlen, 0));
        run(&|e| self.attn_output(e, 0, kv_mul, seqlen, 0));
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
        let ffn_pso = if p.geglu {
            &self.p_gelu_mul
        } else {
            &self.p_silu_mul
        };
        run(&|e| self.glu(e, ffn_pso, p.n_ff));
        check(&self.hidden, 0, p.n_ff, "glu");
        run(&|e| self.matvec(e, &lw.w_down, &self.hidden, &self.xb2, 0));
        check(&self.xb2, 0, p.n_embd, "w_down");
    }

    // ---- op encoders (each appends one dispatch to `enc`) ----

    fn matvec(&self, enc: &ComputeCommandEncoderRef, w: &WBuf, x: &Buffer, y: &Buffer, y_off: u64) {
        let (in_dim, out_dim) = (w.cols as u32, w.out as u32);
        if let Some(m) = &w.mlx {
            enc.set_compute_pipeline_state(&self.p_matvec_mlx4);
            enc.set_buffer(0, Some(&w.buf), 0);
            enc.set_buffer(1, Some(x), 0);
            enc.set_buffer(2, Some(y), y_off);
            enc.set_buffer(3, Some(&m.scales), 0);
            enc.set_buffer(4, Some(&m.biases), 0);
            enc.set_bytes(5, 4, (&in_dim as *const u32).cast());
            enc.set_bytes(6, 4, (&out_dim as *const u32).cast());
            enc.set_bytes(7, 4, (&m.group as *const u32).cast());
            const THREADS: u64 = 256;
            let groups = (w.out as u64).div_ceil(THREADS / 32);
            enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(THREADS, 1, 1));
            return;
        }
        let pso = self.pso_matvec(w.dtype).expect("matvec dtype");
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&w.buf), 0);
        enc.set_buffer(1, Some(x), 0);
        enc.set_buffer(2, Some(y), y_off);
        enc.set_bytes(3, 4, (&in_dim as *const u32).cast());
        enc.set_bytes(4, 4, (&out_dim as *const u32).cast());
        // 8 simdgroups per threadgroup; the Q6_K kernel computes NR0=4 rows per
        // simdgroup (activation reuse), the others 1 row per simdgroup.
        const THREADS: u64 = 256;
        let sgs = THREADS / 32;
        let nr0 = match w.dtype {
            DType::Q6K | DType::Q4K | DType::BF16 | DType::F16 => 2,
            _ => 1,
        };
        let groups = (w.out as u64).div_ceil(sgs * nr0);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// All-experts fused gate+up+activation: one 2D dispatch over (rows x k)
    /// fills `moe_hidden[slot*moe_inter + o]` for every selected expert.
    fn moe_gate_up_all(&self, enc: &ComputeCommandEncoderRef, eg: &EBuf, eu: &EBuf, x: &Buffer) {
        enc.set_compute_pipeline_state(&self.p_moe_gate_up_all);
        enc.set_buffer(0, Some(&eg.buf), 0);
        enc.set_buffer(1, Some(&eu.buf), 0);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(&self.moe_hidden), 0);
        enc.set_buffer(4, Some(&eg.scales), 0);
        enc.set_buffer(5, Some(&eg.biases), 0);
        enc.set_buffer(6, Some(&eu.scales), 0);
        enc.set_buffer(7, Some(&eu.biases), 0);
        let (in_dim, out_dim) = (eg.cols as u32, eg.out as u32);
        enc.set_bytes(8, 4, (&in_dim as *const u32).cast());
        enc.set_bytes(9, 4, (&out_dim as *const u32).cast());
        enc.set_bytes(10, 4, (&eg.group as *const u32).cast());
        enc.set_buffer(11, Some(&self.moe_idx), 0);
        let geglu = u32::from(self.p.geglu);
        enc.set_bytes(12, 4, (&geglu as *const u32).cast());
        const THREADS: u64 = 256;
        let gx = (eg.out as u64).div_ceil(THREADS / 32);
        let k = self.p.n_experts_used as u64;
        enc.dispatch_thread_groups(MTLSize::new(gx, k, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// All-experts down projection: `moe_down[slot*n_embd + o] = W_down · hidden`.
    fn moe_down_all(&self, enc: &ComputeCommandEncoderRef, ed: &EBuf) {
        enc.set_compute_pipeline_state(&self.p_moe_down_all);
        enc.set_buffer(0, Some(&ed.buf), 0);
        enc.set_buffer(1, Some(&self.moe_hidden), 0);
        enc.set_buffer(2, Some(&self.moe_down), 0);
        enc.set_buffer(3, Some(&ed.scales), 0);
        enc.set_buffer(4, Some(&ed.biases), 0);
        let (in_dim, out_dim) = (ed.cols as u32, ed.out as u32);
        enc.set_bytes(5, 4, (&in_dim as *const u32).cast());
        enc.set_bytes(6, 4, (&out_dim as *const u32).cast());
        enc.set_bytes(7, 4, (&ed.group as *const u32).cast());
        enc.set_buffer(8, Some(&self.moe_idx), 0);
        const THREADS: u64 = 256;
        let gx = (ed.out as u64).div_ceil(THREADS / 32);
        let k = self.p.n_experts_used as u64;
        enc.dispatch_thread_groups(MTLSize::new(gx, k, 1), MTLSize::new(THREADS, 1, 1));
    }

    /// Combine the experts' down outputs into `xb2` with the router weights.
    fn moe_combine(&self, enc: &ComputeCommandEncoderRef) {
        enc.set_compute_pipeline_state(&self.p_moe_combine);
        enc.set_buffer(0, Some(&self.moe_down), 0);
        enc.set_buffer(1, Some(&self.moe_wts), 0);
        enc.set_buffer(2, Some(&self.xb2), 0);
        let n = self.p.n_embd as u32;
        let k = self.p.n_experts_used as u32;
        enc.set_bytes(3, 4, (&n as *const u32).cast());
        enc.set_bytes(4, 4, (&k as *const u32).cast());
        dispatch_1d(enc, &self.p_moe_combine, self.p.n_embd as u64);
    }

    /// Router top-k + softmax: fills `moe_idx`/`moe_wts` from `moe_logits`.
    fn moe_topk(&self, enc: &ComputeCommandEncoderRef) {
        enc.set_compute_pipeline_state(&self.p_moe_topk);
        enc.set_buffer(0, Some(&self.moe_logits), 0);
        enc.set_buffer(1, Some(&self.moe_idx), 0);
        enc.set_buffer(2, Some(&self.moe_wts), 0);
        let n = self.p.n_experts as u32;
        let k = self.p.n_experts_used as u32;
        enc.set_bytes(3, 4, (&n as *const u32).cast());
        enc.set_bytes(4, 4, (&k as *const u32).cast());
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
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
        start: u32,
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
        enc.set_bytes(9, 4, (&start as *const u32).cast());
        let nh = self.p.n_head as u64;
        let tg = (seqlen as u64).clamp(1, 64);
        enc.dispatch_threads(MTLSize::new(seqlen as u64, nh, 1), MTLSize::new(tg, 1, 1));
    }

    fn attn_softmax(&self, enc: &ComputeCommandEncoderRef, seqlen: u32, start: u32) {
        enc.set_compute_pipeline_state(&self.p_attn_softmax);
        enc.set_buffer(0, Some(&self.scores), 0);
        let stride = self.p.n_ctx as u32;
        enc.set_bytes(1, 4, (&stride as *const u32).cast());
        enc.set_bytes(2, 4, (&seqlen as *const u32).cast());
        enc.set_bytes(3, 4, (&start as *const u32).cast());
        let nh = self.p.n_head as u64;
        enc.dispatch_thread_groups(MTLSize::new(nh, 1, 1), MTLSize::new(REDUCE_NT, 1, 1));
    }

    fn attn_output(
        &self,
        enc: &ComputeCommandEncoderRef,
        kv_layer_off: u64,
        kv_mul: u32,
        seqlen: u32,
        start: u32,
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
        enc.set_bytes(8, 4, (&start as *const u32).cast());
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

    fn add_off(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_off: u64,
        y: &Buffer,
        n: usize,
    ) {
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
