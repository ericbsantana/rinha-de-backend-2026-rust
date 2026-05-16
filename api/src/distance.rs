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
