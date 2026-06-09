//! Numeric kernels for the CPU forward pass (the reference implementation).

use rayon::prelude::*;

use crate::QWeight;

/// `y[o] = sum_i W[o, i] * x[i]`, dequantizing each row of `w` on the fly into a
/// per-thread reused buffer. Memory-bound work, parallel over output rows.
pub(crate) fn matvec_q(w: &QWeight, x: &[f32]) -> Vec<f32> {
    let rb = w.row_bytes();
    let cols = w.cols;
    (0..w.out)
        .into_par_iter()
        .map_init(
            || vec![0.0f32; cols],
            |buf, o| {
                ullm_core::dequant::dequantize_into(w.dtype, &w.data[o * rb..o * rb + rb], buf)
                    .expect("dequantize weight row");
                buf.iter().zip(x).map(|(a, b)| a * b).sum()
            },
        )
        .collect()
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
        };
        assert_eq!(matvec_q(&w, &[3.0, 5.0]), vec![3.0, 5.0]);
    }
}
