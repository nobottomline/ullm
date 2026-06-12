//! Numeric kernels for the CPU forward pass (the reference implementation).

use rayon::prelude::*;

use crate::{MlxQuant, QWeight};

/// `y[o] = sum_i W[o, i] * x[i]`, dequantizing each row of `w` on the fly into a
/// per-thread reused buffer. Memory-bound work, parallel over output rows.
pub(crate) fn matvec_q(w: &QWeight, x: &[f32]) -> Vec<f32> {
    let cols = w.cols;
    (0..w.out)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; cols],
            |buf, o| {
                dequant_row(w, o, buf);
                buf.iter().zip(x).map(|(a, b)| a * b).sum()
            },
        )
        .collect()
}

/// Dequantize weight row `o` into `buf` (GGUF block quants or MLX 4-bit).
fn dequant_row(w: &QWeight, o: usize, buf: &mut [f32]) {
    if let Some(mlx) = &w.mlx {
        dequant_mlx_row(&w.data, mlx, o, w.cols, buf);
    } else {
        let rb = w.row_bytes();
        ullm_core::dequant::dequantize_into(w.dtype, &w.data[o * rb..o * rb + rb], buf)
            .expect("dequantize weight row");
    }
}

/// Batched matmul: `xs` is `[s_len, cols]` (token-major), returns `[s_len, out]`.
/// Each weight row is dequantized ONCE and reused across all `s_len` columns —
/// the win for prompt prefill (vs `s_len` separate `matvec_q` calls, each of
/// which re-reads and re-dequantizes the whole weight).
pub(crate) fn matmul_q(w: &QWeight, xs: &[f32], s_len: usize) -> Vec<f32> {
    let (cols, out) = (w.cols, w.out);
    let columns: Vec<Vec<f32>> = (0..out)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; cols],
            |buf, o| {
                dequant_row(w, o, buf);
                (0..s_len)
                    .map(|s| {
                        let xrow = &xs[s * cols..s * cols + cols];
                        buf.iter().zip(xrow).map(|(a, b)| a * b).sum::<f32>()
                    })
                    .collect()
            },
        )
        .collect();
    // Transpose [out, s_len] -> token-major [s_len, out].
    let mut y = vec![0.0f32; s_len * out];
    for (o, col) in columns.iter().enumerate() {
        for s in 0..s_len {
            y[s * out + o] = col[s];
        }
    }
    y
}

/// Dequantize one row of an MLX 4-bit weight (`[out, cols]`, eight 4-bit values
/// per u32, LSB first; one scale/bias per `group_size`) into `buf`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn dequant_mlx_row(data: &[u8], mlx: &MlxQuant, o: usize, cols: usize, buf: &mut [f32]) {
    let words = cols / 8;
    let groups = cols / mlx.group_size;
    let base = o * words * 4;
    for i in 0..cols {
        let wb = base + (i / 8) * 4;
        let word = u32::from_le_bytes([data[wb], data[wb + 1], data[wb + 2], data[wb + 3]]);
        let q = (word >> ((i % 8) * 4)) & 0xF;
        let g = o * groups + i / mlx.group_size;
        buf[i] = q as f32 * mlx.scales[g] + mlx.biases[g];
    }
}

/// RMS normalization with a learned per-channel gain.
pub(crate) fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps;
    let inv = ms.sqrt().recip();
    x.iter().zip(weight).map(|(xi, wi)| xi * inv * wi).collect()
}

/// In-place numerically-stable softmax.
pub(crate) fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// SiLU / swish activation: `x * sigmoid(x)`.
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Add a bias vector elementwise: `x[i] += b[i]`.
pub(crate) fn add_bias(x: &mut [f32], b: &[f32]) {
    for (xi, bi) in x.iter_mut().zip(b) {
        *xi += bi;
    }
}

/// Rotary position embedding (interleaved / ggml "NORM" convention), applied to
/// each head independently in place.
pub(crate) fn rope(vec: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    for h in 0..n_heads {
        let off = h * head_dim;
        let mut i = 0;
        while i + 1 < head_dim {
            let freq = theta.powf(i as f32 / head_dim as f32).recip();
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            let (a, b) = (vec[off + i], vec[off + i + 1]);
            vec[off + i] = a * cos - b * sin;
            vec[off + i + 1] = a * sin + b * cos;
            i += 2;
        }
    }
}

/// NeoX / "rotate-half" RoPE: rotates `(x[i], x[i+d/2])`. Used for Gemma and for
/// HF/SafeTensors weights, which are not permuted into the interleaved layout.
pub(crate) fn rope_neox(vec: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..half {
            let freq = theta.powf(2.0 * i as f32 / head_dim as f32).recip();
            let (sin, cos) = (pos as f32 * freq).sin_cos();
            let (a, b) = (vec[off + i], vec[off + i + half]);
            vec[off + i] = a * cos - b * sin;
            vec[off + i + half] = a * sin + b * cos;
        }
    }
}

/// GELU activation (tanh approximation, as used by Gemma's GeGLU).
pub(crate) fn gelu(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt(); // sqrt(2 / pi)
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ullm_core::DType;

    #[test]
    fn rmsnorm_normalizes_to_unit_rms() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let y = rmsnorm(&x, &w, 0.0);
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_is_a_distribution() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn matvec_q_f32_identity() {
        let data: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let w = QWeight {
            data,
            dtype: DType::F32,
            out: 2,
            cols: 2,
            mlx: None,
        };
        assert_eq!(matvec_q(&w, &[3.0, 5.0]), vec![3.0, 5.0]);
    }

    #[test]
    fn matmul_q_matches_stacked_matvec() {
        // A 3x2 weight (out=3, cols=2), F32.
        let rows: [[f32; 2]; 3] = [[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]];
        let data: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let w = QWeight {
            data,
            dtype: DType::F32,
            out: 3,
            cols: 2,
            mlx: None,
        };
        // Three token rows, token-major [s_len, cols].
        let xs = [1.0f32, 0.0, 0.5, -2.0, 3.0, 3.0];
        let s_len = 3;
        let batched = matmul_q(&w, &xs, s_len);
        // Reference: run each token through matvec_q and concatenate.
        let mut reference = Vec::new();
        for s in 0..s_len {
            reference.extend(matvec_q(&w, &xs[s * 2..s * 2 + 2]));
        }
        assert_eq!(batched, reference);
    }
}
