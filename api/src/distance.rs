use crate::label::N_DIMS_PADDED;

pub fn l2_sq(q: &[f32; N_DIMS_PADDED], c: &[f32]) -> f32 {
    let mut sum_sq = 0.0_f32;
    for i in 0..N_DIMS_PADDED {
        let diff = q[i] - c[i];
        sum_sq += diff * diff;
    }
    sum_sq
}

pub fn l1(q: &[f32; N_DIMS_PADDED], c: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..N_DIMS_PADDED {
        sum += (q[i] - c[i]).abs()
    }
    sum
}

pub fn l_inf(q: &[f32; N_DIMS_PADDED], c: &[f32]) -> f32 {
    let mut max_diff = 0.0_f32;
    for i in 0..N_DIMS_PADDED {
        max_diff = f32::max(max_diff, (q[i] - c[i]).abs())
    }
    max_diff
}

pub fn cosine_dist(q: &[f32; N_DIMS_PADDED], c: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_q_sq = 0.0_f32;
    let mut norm_c_sq = 0.0_f32;

    for i in 0..N_DIMS_PADDED {
        dot += q[i] * c[i];
        norm_q_sq += q[i] * q[i];
        norm_c_sq += c[i] * c[i];
    }

    1.0_f32 - dot / (norm_q_sq.sqrt() * norm_c_sq.sqrt())
}

pub fn weighted_l2_sq(q: &[f32; N_DIMS_PADDED], c: &[f32], w: &[f32; N_DIMS_PADDED]) -> f32 {
    let mut sum = 0.0_f32;
    for i in 0..N_DIMS_PADDED {
        let diff = q[i] - c[i];
        sum += w[i] * diff * diff
    }
    sum
}

pub fn l_inf_best(q: &[f32; N_DIMS_PADDED], c: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return l_inf_neon(q, c);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { l_inf_avx2(q, c) };
        }
    }
    #[allow(unused)]
    l_inf(q, c)
}

#[cfg(target_arch = "aarch64")]
pub fn l_inf_neon(q: &[f32; N_DIMS_PADDED], c: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..4 {
            let qv = vld1q_f32(q.as_ptr().add(i * 4));
            let cv = vld1q_f32(c.as_ptr().add(i * 4));
            let diff = vsubq_f32(qv, cv);
            let abs_diff = vabsq_f32(diff);
            acc = vmaxq_f32(acc, abs_diff);
        }

        vmaxvq_f32(acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l_inf_avx2(q: &[f32; N_DIMS_PADDED], c: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: caller must guarantee AVX2 is available at runtime
    // (verify with `is_x86_feature_detected!("avx2")` before calling).
    // q has 16 floats by type. c comes from chunks_exact(16). Unaligned loads are ok.

    // Mask to clear the sign bit of each f32: abs(x) = x & 0x7FFF_FFFF.
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));

    let mut acc = _mm256_setzero_ps();

    // 2 iterations of 8 floats each, covering all 16 dims.
    for i in 0..2 {
        let qv = _mm256_loadu_ps(q.as_ptr().add(i * 8));
        let cv = _mm256_loadu_ps(c.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(qv, cv);
        let abs_diff = _mm256_and_ps(diff, abs_mask);
        acc = _mm256_max_ps(acc, abs_diff);
    }

    // Horizontal max reduction: 8 floats -> 1 scalar.
    // Step 1: split the 256-bit register into two 128-bit halves and max them.
    let lo = _mm256_castps256_ps128(acc);
    let hi = _mm256_extractf128_ps(acc, 1);
    let v4 = _mm_max_ps(lo, hi); // [m0, m1, m2, m3]

    // Step 2: bring high pair on top of low pair, max -> [max(m0,m2), max(m1,m3), _, _].
    let v4_high = _mm_movehl_ps(v4, v4);
    let v2 = _mm_max_ps(v4, v4_high);

    // Step 3: broadcast element 1 to position 0, scalar max of element 0.
    let v2_high = _mm_shuffle_ps(v2, v2, 0b01_01_01_01);
    let v1 = _mm_max_ss(v2, v2_high);

    _mm_cvtss_f32(v1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn l_inf_neon_matches_scalar() {
        let q = [
            0.1, -0.5, 0.7, 0.2, 0.0, 0.3, 0.9, -0.1, 0.4, 0.6, -0.8, 0.05, 0.15, 0.25, 0.0, 0.0,
        ];
        let c = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        // Esperado: max |q_i - c_i| = max |q_i| = 0.9
        let scalar = l_inf(&q, &c);
        let simd = l_inf_neon(&q, &c);
        assert!(
            (scalar - simd).abs() < 1e-6,
            "scalar={} simd={}",
            scalar,
            simd
        );
        assert!((scalar - 0.9).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn l_inf_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not detected on this CPU; skipping test");
            return;
        }
        let q = [
            0.1, -0.5, 0.7, 0.2, 0.0, 0.3, 0.9, -0.1, 0.4, 0.6, -0.8, 0.05, 0.15, 0.25, 0.0, 0.0,
        ];
        let c = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let scalar = l_inf(&q, &c);
        let simd = unsafe { l_inf_avx2(&q, &c) };
        assert!(
            (scalar - simd).abs() < 1e-6,
            "scalar={} simd={}",
            scalar,
            simd
        );
        assert!((scalar - 0.9).abs() < 1e-6);
    }
}
