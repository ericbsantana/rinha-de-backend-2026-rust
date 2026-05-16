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
    }
