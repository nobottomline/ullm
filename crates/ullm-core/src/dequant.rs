//! Dequantization of GGUF block formats to `f32`.
//!
//! These are faithful, readable ports of the ggml reference dequantizers — they
//! favor clarity over speed (no SIMD). The Metal backend will dequantize
//! in-kernel later; this module is the numerical reference.
//!
//! Index-based loops mirror the reference layout closely, so the
//! `needless_range_loop` lint is allowed for the whole module.
#![allow(clippy::needless_range_loop)]

use crate::{DType, Error, Result};

/// Convert an IEEE-754 half-precision value to `f32`.
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let v = match exp {
        0 => mant as f32 * 2.0f32.powi(-24), // zero or subnormal
        0x1f => {
            if mant == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15),
    };
    if sign == 1 { -v } else { v }
}

/// Read a little-endian `f16` at byte offset `off` and widen it to `f32`.
#[inline]
fn rd_f16(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}

/// The 6-bit packed (scale, min) pair `j` of a k-quant super-block.
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize `n` elements of `dtype` from `data` into a fresh `f32` vector.
pub fn dequantize(dtype: DType, data: &[u8], n: usize) -> Result<Vec<f32>> {
    let block = dtype.block_size();
    let ts = dtype.type_size();
    if block == 0 || n % block != 0 {
        return Err(Error::Format(format!(
            "element count {n} is not a multiple of block size {block} for {dtype:?}"
        )));
    }
    let nblocks = n / block;
    let need = nblocks * ts;
    if data.len() < need {
        return Err(Error::Format(format!(
            "{dtype:?}: need {need} bytes, have {}",
            data.len()
        )));
    }

    let mut out = vec![0.0f32; n];
    match dtype {
        DType::F32 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        DType::F16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        DType::BF16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        DType::Q8_0 => q8_0(data, nblocks, &mut out),
        DType::Q4_0 => q4_0(data, nblocks, &mut out),
        DType::Q4_1 => q4_1(data, nblocks, &mut out),
        DType::Q5_0 => q5_0(data, nblocks, &mut out),
        DType::Q5_1 => q5_1(data, nblocks, &mut out),
        DType::Q4K => q4_k(data, nblocks, &mut out),
        DType::Q5K => q5_k(data, nblocks, &mut out),
        DType::Q6K => q6_k(data, nblocks, &mut out),
        other => {
            return Err(Error::Unsupported(format!(
                "dequantization of {other:?} is not implemented yet"
            )));
        }
    }
    Ok(out)
}

fn q8_0(data: &[u8], nblocks: usize, out: &mut [f32]) {
    const TS: usize = 34;
    for b in 0..nblocks {
        let blk = &data[b * TS..b * TS + TS];
        let d = rd_f16(blk, 0);
        let base = b * 32;
        for j in 0..32 {
            out[base + j] = d * (blk[2 + j] as i8) as f32;
        }
    }
}

fn q4_0(data: &[u8], nblocks: usize, out: &mut [f32]) {
    const TS: usize = 18;
    for b in 0..nblocks {
        let blk = &data[b * TS..b * TS + TS];
        let d = rd_f16(blk, 0);
        let qs = &blk[2..18];
        let base = b * 32;
        for j in 0..16 {
            out[base + j] = d * ((qs[j] & 0x0F) as i32 - 8) as f32;
            out[base + j + 16] = d * ((qs[j] >> 4) as i32 - 8) as f32;
        }
    }
}

fn q4_1(data: &[u8], nblocks: usize, out: &mut [f32]) {
    const TS: usize = 20;
    for b in 0..nblocks {
        let blk = &data[b * TS..b * TS + TS];
        let d = rd_f16(blk, 0);
        let m = rd_f16(blk, 2);
        let qs = &blk[4..20];
        let base = b * 32;
        for j in 0..16 {
            out[base + j] = d * (qs[j] & 0x0F) as f32 + m;
            out[base + j + 16] = d * (qs[j] >> 4) as f32 + m;
        }
    }
}

fn q5_0(data: &[u8], nblocks: usize, out: &mut [f32]) {
    const TS: usize = 22;
    for b in 0..nblocks {
        let blk = &data[b * TS..b * TS + TS];
        let d = rd_f16(blk, 0);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let base = b * 32;
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
            out[base + j] = d * (((qs[j] & 0x0F) | xh0) as i32 - 16) as f32;
            out[base + j + 16] = d * (((qs[j] >> 4) | xh1) as i32 - 16) as f32;
        }
    }
}

fn q5_1(data: &[u8], nblocks: usize, out: &mut [f32]) {
    const TS: usize = 24;
    for b in 0..nblocks {
        let blk = &data[b * TS..b * TS + TS];
        let d = rd_f16(blk, 0);
        let m = rd_f16(blk, 2);
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let qs = &blk[8..24];
        let base = b * 32;
        for j in 0..16 {
            let xh0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
            out[base + j] = d * ((qs[j] & 0x0F) | xh0) as f32 + m;
            out[base + j + 16] = d * ((qs[j] >> 4) | xh1) as f32 + m;
        }
    }
}

fn q4_k(data: &[u8], nblocks: usize, out: &mut [f32]) {
    const TS: usize = 144;
    for b in 0..nblocks {
        let blk = &data[b * TS..b * TS + TS];
        let d = rd_f16(blk, 0);
        let dmin = rd_f16(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let base = b * 256;
        let mut oi = 0;
        let mut is = 0;
        let mut qoff = 0;
        for _ in 0..4 {
            let (s1, m1) = scale_min_k4(is, scales);
            let (s2, m2) = scale_min_k4(is + 1, scales);
            let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * s2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                out[base + oi] = d1 * (qs[qoff + l] & 0x0F) as f32 - mm1;
                oi += 1;
            }
            for l in 0..32 {
                out[base + oi] = d2 * (qs[qoff + l] >> 4) as f32 - mm2;
                oi += 1;
            }
            qoff += 32;
            is += 2;
        }
    }
}

fn q5_k(data: &[u8], nblocks: usize, out: &mut [f32]) {
    const TS: usize = 176;
    for b in 0..nblocks {
        let blk = &data[b * TS..b * TS + TS];
        let d = rd_f16(blk, 0);
        let dmin = rd_f16(blk, 2);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let base = b * 256;
        let mut oi = 0;
        let mut is = 0;
        let mut qoff = 0;
        for jj in 0..4 {
            let u1 = 1u8 << (2 * jj);
            let u2 = 2u8 << (2 * jj);
            let (s1, m1) = scale_min_k4(is, scales);
            let (s2, m2) = scale_min_k4(is + 1, scales);
            let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * s2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                out[base + oi] = d1 * ((qs[qoff + l] & 0x0F) as f32 + hi) - mm1;
                oi += 1;
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                out[base + oi] = d2 * ((qs[qoff + l] >> 4) as f32 + hi) - mm2;
                oi += 1;
            }
            qoff += 32;
            is += 2;
        }
    }
}

fn q6_k(data: &[u8], nblocks: usize, out: &mut [f32]) {
    const TS: usize = 210;
    for b in 0..nblocks {
        let blk = &data[b * TS..b * TS + TS];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d = rd_f16(blk, 208);
        let base = b * 256;
        for nh in 0..2 {
            let (qlo, qho, sco, yo) = (nh * 64, nh * 32, nh * 8, base + nh * 128);
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[qlo + l] & 0x0F) as i32 | ((qh[qho + l] & 3) as i32) << 4) - 32;
                let q2 = ((ql[qlo + l + 32] & 0x0F) as i32
                    | (((qh[qho + l] >> 2) & 3) as i32) << 4)
                    - 32;
                let q3 = ((ql[qlo + l] >> 4) as i32 | (((qh[qho + l] >> 4) & 3) as i32) << 4) - 32;
                let q4 =
                    ((ql[qlo + l + 32] >> 4) as i32 | (((qh[qho + l] >> 6) & 3) as i32) << 4) - 32;
                out[yo + l] = d * (sc[sco + is] as i8) as f32 * q1 as f32;
                out[yo + l + 32] = d * (sc[sco + is + 2] as i8) as f32 * q2 as f32;
                out[yo + l + 64] = d * (sc[sco + is + 4] as i8) as f32 * q3 as f32;
                out[yo + l + 96] = d * (sc[sco + is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrips_simple_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
    }

    #[test]
    fn q8_0_scales_int8() {
        // d = 2.0 (f16 0x4000), qs = 0..32
        let mut data = vec![0x00u8, 0x40];
        data.extend(0..32u8);
        let out = dequantize(DType::Q8_0, &data, 32).unwrap();
        for j in 0..32 {
            assert!((out[j] - 2.0 * j as f32).abs() < 1e-4);
        }
    }

    #[test]
    fn q4_0_centers_on_eight() {
        // d = 1.0 (f16 0x3C00), qs = 0x80 -> low nibble 0 (->-8), high nibble 8 (->0)
        let mut data = vec![0x00u8, 0x3C];
        data.extend(std::iter::repeat_n(0x80u8, 16));
        let out = dequantize(DType::Q4_0, &data, 32).unwrap();
        for j in 0..16 {
            assert!((out[j] - (-8.0)).abs() < 1e-4);
        }
        for j in 16..32 {
            assert!(out[j].abs() < 1e-4);
        }
    }

    #[test]
    fn rejects_short_buffer() {
        let err = dequantize(DType::Q8_0, &[0u8; 10], 32).unwrap_err();
        assert!(matches!(err, Error::Format(_)));
    }
}
