//! Token sampling: greedy, temperature, top-k, and nucleus (top-p) — with a
//! small dependency-free SplitMix64 RNG.

use std::cmp::Ordering;

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SampleParams {
    /// Softmax temperature. `<= 0` means greedy (argmax).
    pub temperature: f32,
    /// Keep only the top-k highest logits (`0` disables).
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set with cumulative prob >= `top_p`.
    pub top_p: f32,
    /// RNG seed (`0` uses a fixed default for reproducibility).
    pub seed: u64,
    /// Repetition penalty (`1.0` disables). Discourages loops on recent tokens.
    pub repeat_penalty: f32,
    /// How many recent tokens the repetition penalty looks back over.
    pub repeat_last_n: usize,
    /// Forbid repeating any `n`-gram already produced (`0` disables). A hard stop
    /// against verbatim loops, applied only when no grammar constraint is active.
    pub no_repeat_ngram: usize,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            no_repeat_ngram: 0,
        }
    }
}

/// Block any token that would complete an `n`-gram already present in `history`
/// — a hard guarantee against verbatim loops (cf. transformers' no_repeat_ngram).
pub(crate) fn apply_no_repeat_ngram(logits: &mut [f32], history: &[u32], n: usize) {
    if n < 2 || history.len() < n {
        return;
    }
    let suffix = &history[history.len() - (n - 1)..];
    for i in 0..=history.len() - n {
        if &history[i..i + n - 1] == suffix {
            let banned = history[i + n - 1] as usize;
            if banned < logits.len() {
                logits[banned] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Penalize tokens that appear in `recent` so greedy/low-temperature decoding
/// doesn't loop (llama.cpp style: a positive logit is divided by the penalty, a
/// negative one multiplied). `-inf` (grammar-masked) tokens are left untouched.
pub(crate) fn apply_repetition_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for &t in recent {
        let i = t as usize;
        if i < logits.len() && logits[i].is_finite() {
            logits[i] = if logits[i] > 0.0 {
                logits[i] / penalty
            } else {
                logits[i] * penalty
            };
        }
    }
}

/// Sample a token id from `logits` according to `params`.
pub(crate) fn sample_token(logits: &[f32], params: &SampleParams, rng: &mut u64) -> u32 {
    if params.temperature <= 0.0 {
        return argmax(logits) as u32;
    }

    let mut cand: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / params.temperature))
        .collect();
    cand.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    if params.top_k > 0 && params.top_k < cand.len() {
        cand.truncate(params.top_k);
    }

    let max = cand[0].1;
    let mut probs: Vec<f32> = cand.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let mut cutoff = probs.len();
    if params.top_p < 1.0 {
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= params.top_p {
                cutoff = i + 1;
                break;
            }
        }
    }

    let total: f32 = probs[..cutoff].iter().sum();
    let r = next_f32(rng) * total;
    let mut acc = 0.0;
    for (&p, c) in probs[..cutoff].iter().zip(&cand[..cutoff]) {
        acc += p;
        if r < acc {
            return c.0 as u32;
        }
    }
    cand[cutoff - 1].0 as u32
}

/// Index of the largest element (first on ties).
fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

/// One step of a SplitMix64 RNG.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform `f32` in `[0, 1)`.
fn next_f32(state: &mut u64) -> f32 {
    (next_u64(state) >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    }
}
