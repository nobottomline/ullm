//! Qwen3.5 / Qwen3-Next linear attention — the Gated DeltaNet (state-space)
//! block used by ~3/4 of the layers in the Qwen3.5 hybrid architecture, in
//! place of softmax attention. Ported to match the reference
//! `transformers` implementation (`Qwen3_5GatedDeltaNet` /
//! `torch_recurrent_gated_delta_rule`), validated numerically in the tests.
//!
//! The per-layer flow is: project the input to `qkv` (one matrix) plus the
//! gate `z` and the scalar streams `a`, `b`; run a causal depthwise conv1d +
//! SiLU over `qkv`; split into per-head query/key/value; then a gated delta
//! recurrence carries a `[head_k, head_v]` state per value-head across the
//! sequence; finally a gated RMSNorm and the output projection. The
//! projections are plain matmuls done by the caller (quantized in the model,
//! f32 in the test); [`deltanet_core`] is the validated recurrence in between.
// Validated against transformers here; wired into the hybrid forward next.
#![allow(dead_code)]
// Index loops mirror the reference math (per-head/per-dim); clearer than iterators.
#![allow(clippy::needless_range_loop)]

/// Geometry of a Gated-DeltaNet block.
#[derive(Clone, Copy, Debug)]
pub struct DeltaNetDims {
    pub hidden: usize,
    pub n_v_heads: usize,
    pub n_k_heads: usize,
    pub head_k: usize,
    pub head_v: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

impl DeltaNetDims {
    pub fn key_dim(&self) -> usize {
        self.n_k_heads * self.head_k
    }
    pub fn value_dim(&self) -> usize {
        self.n_v_heads * self.head_v
    }
    /// Channels carried through the conv (q, k, v concatenated).
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn softplus(x: f32) -> f32 {
    // ln(1 + e^x), stable for large x (matches torch F.softplus, threshold 20).
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// The conv1d + gated delta-rule recurrence + gated RMSNorm, i.e. everything
/// between the input projections and the output projection. Inputs are already
/// projected: `qkv` `[seq, conv_dim]`, `z` `[seq, value_dim]`, `b`/`a`
/// `[seq, n_v_heads]`. Returns the normed core output `[seq, value_dim]`, ready
/// for `out_proj`. Pure f32, sequential over the sequence (correctness-first).
#[allow(clippy::too_many_arguments)]
pub fn deltanet_core(
    qkv: &[f32],
    z: &[f32],
    b: &[f32],
    a: &[f32],
    conv1d: &[f32], // [conv_dim, conv_kernel]
    a_log: &[f32],  // [n_v_heads]
    dt_bias: &[f32],
    norm: &[f32], // [head_v]
    d: DeltaNetDims,
    seq: usize,
) -> Vec<f32> {
    let (hk, hv, dk, dv, k) = (d.n_k_heads, d.n_v_heads, d.head_k, d.head_v, d.conv_kernel);
    let key_dim = d.key_dim();
    let value_dim = d.value_dim();
    let conv_dim = d.conv_dim();
    let rep = hv / hk; // grouped-query expansion factor
    let scale = 1.0 / (dk as f32).sqrt();

    // 1. Causal depthwise conv1d over `qkv` along time, per channel, then SiLU.
    let mut conv = vec![0f32; seq * conv_dim];
    for t in 0..seq {
        for c in 0..conv_dim {
            let mut acc = 0f32;
            for kk in 0..k {
                let src = t as isize - (k as isize - 1) + kk as isize;
                if src >= 0 {
                    acc += conv1d[c * k + kk] * qkv[src as usize * conv_dim + c];
                }
            }
            conv[t * conv_dim + c] = silu(acc);
        }
    }

    // 2. Gated delta-rule recurrence: a [head_k, head_v] state per value head,
    //    carried across the sequence. q/k are L2-normalized; q is scaled.
    let mut core = vec![0f32; seq * value_dim]; // [seq, n_v_heads, head_v]
    let mut state = vec![0f32; dk * dv]; // reused per head
    let mut q = vec![0f32; dk];
    let mut key = vec![0f32; dk];
    for h in 0..hv {
        state.iter_mut().for_each(|s| *s = 0.0);
        let kh = h / rep; // grouped key/query head feeding this value head
        for t in 0..seq {
            // L2-normalize q (then scale) and k from the conv output.
            let qoff = t * conv_dim + kh * dk;
            let koff = t * conv_dim + key_dim + kh * dk;
            let mut qn = 0f32;
            let mut kn = 0f32;
            for i in 0..dk {
                q[i] = conv[qoff + i];
                key[i] = conv[koff + i];
                qn += q[i] * q[i];
                kn += key[i] * key[i];
            }
            let qinv = (qn + 1e-6).sqrt().recip() * scale;
            let kinv = (kn + 1e-6).sqrt().recip();
            for i in 0..dk {
                q[i] *= qinv;
                key[i] *= kinv;
            }
            // Decay g_t = exp(-exp(A_log) * softplus(a + dt_bias)); gate beta.
            let g_t = (-a_log[h].exp() * softplus(a[t * hv + h] + dt_bias[h])).exp();
            let beta_t = sigmoid(b[t * hv + h]);
            for s in state.iter_mut() {
                *s *= g_t;
            }
            let voff = t * conv_dim + 2 * key_dim + h * dv;
            let coff = t * value_dim + h * dv;
            // Each value column j is independent: read decayed state, apply the
            // delta update, then read it back for the query projection.
            for j in 0..dv {
                let mut kv_mem = 0f32;
                for i in 0..dk {
                    kv_mem += state[i * dv + j] * key[i];
                }
                let delta = (conv[voff + j] - kv_mem) * beta_t;
                let mut out = 0f32;
                for i in 0..dk {
                    let sij = state[i * dv + j] + key[i] * delta;
                    state[i * dv + j] = sij;
                    out += sij * q[i];
                }
                core[coff + j] = out;
            }
        }
    }

    // 3. Gated RMSNorm per (token, value head): normalize over head_v, scale by
    //    `norm`, then multiply by silu(z) (the gate).
    let mut normed = vec![0f32; seq * value_dim];
    for t in 0..seq {
        for h in 0..hv {
            let off = t * value_dim + h * dv;
            let mut var = 0f32;
            for j in 0..dv {
                var += core[off + j] * core[off + j];
            }
            let inv = (var / dv as f32 + d.eps).sqrt().recip();
            for j in 0..dv {
                normed[off + j] = core[off + j] * inv * norm[j] * silu(z[off + j]);
            }
        }
    }
    normed
}

/// Persistent recurrent state for one linear-attention layer during decoding —
/// the analogue of the KV cache for softmax attention. `recur` is the
/// `[head_k, head_v]` delta-rule state per value head; `conv` is the ring of the
/// last `conv_kernel-1` projected `qkv` vectors feeding the causal conv.
pub struct DeltaNetState {
    pub recur: Vec<f32>, // [n_v_heads * head_k * head_v]
    pub conv: Vec<f32>,  // [(conv_kernel - 1) * conv_dim]
}

impl DeltaNetState {
    pub fn new(d: &DeltaNetDims) -> Self {
        Self {
            recur: vec![0.0; d.n_v_heads * d.head_k * d.head_v],
            conv: vec![0.0; (d.conv_kernel - 1) * d.conv_dim()],
        }
    }

    pub fn reset(&mut self) {
        self.recur.iter_mut().for_each(|x| *x = 0.0);
        self.conv.iter_mut().for_each(|x| *x = 0.0);
    }
}

/// One decode step: advance `state` with the current token's projected
/// `(qkv, z, b, a)` and return its normed core output `[value_dim]` (ready for
/// `out_proj`). Running this token-by-token reproduces [`deltanet_core`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_step(
    state: &mut DeltaNetState,
    qkv: &[f32], // [conv_dim]
    z: &[f32],   // [value_dim]
    b: &[f32],   // [n_v_heads]
    a: &[f32],   // [n_v_heads]
    conv1d: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm: &[f32],
    d: DeltaNetDims,
) -> Vec<f32> {
    let (hk, hv, dk, dv, k) = (d.n_k_heads, d.n_v_heads, d.head_k, d.head_v, d.conv_kernel);
    let key_dim = d.key_dim();
    let value_dim = d.value_dim();
    let conv_dim = d.conv_dim();
    let kk1 = k - 1; // conv history length

    // Causal conv1d over [history | current] + SiLU, then slide the history.
    let mut conv = vec![0f32; conv_dim];
    for c in 0..conv_dim {
        let mut acc = conv1d[c * k + kk1] * qkv[c];
        for w in 0..kk1 {
            acc += conv1d[c * k + w] * state.conv[w * conv_dim + c];
        }
        conv[c] = silu(acc);
    }
    if kk1 > 0 {
        for s in 0..kk1 - 1 {
            for c in 0..conv_dim {
                state.conv[s * conv_dim + c] = state.conv[(s + 1) * conv_dim + c];
            }
        }
        state.conv[(kk1 - 1) * conv_dim..].copy_from_slice(&qkv[..conv_dim]);
    }

    // One gated delta-rule recurrence step per value head.
    let rep = hv / hk;
    let scale = 1.0 / (dk as f32).sqrt();
    let mut core = vec![0f32; value_dim];
    let mut q = vec![0f32; dk];
    let mut key = vec![0f32; dk];
    for h in 0..hv {
        let kh = h / rep;
        let mut qn = 0f32;
        let mut kn = 0f32;
        for i in 0..dk {
            q[i] = conv[kh * dk + i];
            key[i] = conv[key_dim + kh * dk + i];
            qn += q[i] * q[i];
            kn += key[i] * key[i];
        }
        let qinv = (qn + 1e-6).sqrt().recip() * scale;
        let kinv = (kn + 1e-6).sqrt().recip();
        for i in 0..dk {
            q[i] *= qinv;
            key[i] *= kinv;
        }
        let g_t = (-a_log[h].exp() * softplus(a[h] + dt_bias[h])).exp();
        let beta_t = sigmoid(b[h]);
        let s0 = h * dk * dv;
        for x in &mut state.recur[s0..s0 + dk * dv] {
            *x *= g_t;
        }
        let voff = 2 * key_dim + h * dv;
        for j in 0..dv {
            let mut kv_mem = 0f32;
            for i in 0..dk {
                kv_mem += state.recur[s0 + i * dv + j] * key[i];
            }
            let delta = (conv[voff + j] - kv_mem) * beta_t;
            let mut out = 0f32;
            for i in 0..dk {
                let sij = state.recur[s0 + i * dv + j] + key[i] * delta;
                state.recur[s0 + i * dv + j] = sij;
                out += sij * q[i];
            }
            core[h * dv + j] = out;
        }
    }

    // Gated RMSNorm per value head.
    let mut normed = vec![0f32; value_dim];
    for h in 0..hv {
        let off = h * dv;
        let mut var = 0f32;
        for j in 0..dv {
            var += core[off + j] * core[off + j];
        }
        let inv = (var / dv as f32 + d.eps).sqrt().recip();
        for j in 0..dv {
            normed[off + j] = core[off + j] * inv * norm[j] * silu(z[off + j]);
        }
    }
    normed
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Plain f32 `y[t,o] = sum_i w[o,i] * x[t,i]` (row-major weight `[out, in]`).
    fn matmul(x: &[f32], w: &[f32], seq: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut y = vec![0f32; seq * out_dim];
        for t in 0..seq {
            for o in 0..out_dim {
                let mut acc = 0f32;
                for i in 0..in_dim {
                    acc += w[o * in_dim + i] * x[t * in_dim + i];
                }
                y[t * out_dim + o] = acc;
            }
        }
        y
    }

    fn arr(v: &Value, k: &str) -> Vec<f32> {
        v["weights"][k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    #[test]
    fn matches_transformers_reference() {
        // Reference generated by tools/gen_deltanet_ref.py against the real
        // transformers `Qwen3_5GatedDeltaNet` (torch fallback kernels).
        let raw = include_str!("testdata/deltanet_ref.json");
        let v: Value = serde_json::from_str(raw).unwrap();
        let c = &v["config"];
        let g = |k: &str| c[k].as_u64().unwrap() as usize;
        let d = DeltaNetDims {
            hidden: g("hidden_size"),
            n_v_heads: g("num_v_heads"),
            n_k_heads: g("num_k_heads"),
            head_k: g("head_k_dim"),
            head_v: g("head_v_dim"),
            conv_kernel: g("conv_kernel"),
            eps: c["eps"].as_f64().unwrap() as f32,
        };
        let seq = g("seq");
        let input: Vec<f32> = v["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let expected: Vec<f32> = v["output"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();

        // Input projections (the model uses quantized matvecs; here plain f32).
        let qkv = matmul(
            &input,
            &arr(&v, "in_proj_qkv.weight"),
            seq,
            d.hidden,
            d.conv_dim(),
        );
        let z = matmul(
            &input,
            &arr(&v, "in_proj_z.weight"),
            seq,
            d.hidden,
            d.value_dim(),
        );
        let b = matmul(
            &input,
            &arr(&v, "in_proj_b.weight"),
            seq,
            d.hidden,
            d.n_v_heads,
        );
        let a = matmul(
            &input,
            &arr(&v, "in_proj_a.weight"),
            seq,
            d.hidden,
            d.n_v_heads,
        );

        let core = deltanet_core(
            &qkv,
            &z,
            &b,
            &a,
            &arr(&v, "conv1d.weight"),
            &arr(&v, "A_log"),
            &arr(&v, "dt_bias"),
            &arr(&v, "norm.weight"),
            d,
            seq,
        );
        let out = matmul(
            &core,
            &arr(&v, "out_proj.weight"),
            seq,
            d.value_dim(),
            d.hidden,
        );

        let max_diff = out
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "Gated-DeltaNet forward differs from transformers: max|Δ| = {max_diff}"
        );
    }

    #[test]
    fn step_matches_core() {
        // The token-by-token decode path (deltanet_step + persistent state) must
        // reproduce the whole-sequence deltanet_core bit-for-bit (same math).
        let v: Value = serde_json::from_str(include_str!("testdata/deltanet_ref.json")).unwrap();
        let c = &v["config"];
        let g = |k: &str| c[k].as_u64().unwrap() as usize;
        let d = DeltaNetDims {
            hidden: g("hidden_size"),
            n_v_heads: g("num_v_heads"),
            n_k_heads: g("num_k_heads"),
            head_k: g("head_k_dim"),
            head_v: g("head_v_dim"),
            conv_kernel: g("conv_kernel"),
            eps: c["eps"].as_f64().unwrap() as f32,
        };
        let seq = g("seq");
        let input: Vec<f32> = v["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let qkv = matmul(
            &input,
            &arr(&v, "in_proj_qkv.weight"),
            seq,
            d.hidden,
            d.conv_dim(),
        );
        let z = matmul(
            &input,
            &arr(&v, "in_proj_z.weight"),
            seq,
            d.hidden,
            d.value_dim(),
        );
        let b = matmul(
            &input,
            &arr(&v, "in_proj_b.weight"),
            seq,
            d.hidden,
            d.n_v_heads,
        );
        let a = matmul(
            &input,
            &arr(&v, "in_proj_a.weight"),
            seq,
            d.hidden,
            d.n_v_heads,
        );
        let (conv1d, alog, dtb, nrm) = (
            arr(&v, "conv1d.weight"),
            arr(&v, "A_log"),
            arr(&v, "dt_bias"),
            arr(&v, "norm.weight"),
        );
        let core = deltanet_core(&qkv, &z, &b, &a, &conv1d, &alog, &dtb, &nrm, d, seq);

        let (cd, vd, hv) = (d.conv_dim(), d.value_dim(), d.n_v_heads);
        let mut st = DeltaNetState::new(&d);
        let mut max_diff = 0f32;
        for t in 0..seq {
            let out = deltanet_step(
                &mut st,
                &qkv[t * cd..(t + 1) * cd],
                &z[t * vd..(t + 1) * vd],
                &b[t * hv..(t + 1) * hv],
                &a[t * hv..(t + 1) * hv],
                &conv1d,
                &alog,
                &dtb,
                &nrm,
                d,
            );
            for j in 0..vd {
                max_diff = max_diff.max((out[j] - core[t * vd + j]).abs());
            }
        }
        assert!(max_diff < 1e-5, "step vs core differ: max|Δ| = {max_diff}");
    }
}
