use crate::dataset::Dataset;
use crate::label::N_DIMS_PADDED;
use std::collections::BinaryHeap;

fn l2_sq(a: &[f32; N_DIMS_PADDED], b: &[f32]) -> f32 {
    let mut sum_sq = 0.0_f32;
    for i in 0..N_DIMS_PADDED {
        let diff = a[i] - b[i];
        sum_sq += diff * diff
    }
    sum_sq
}

struct Candidate {
    dist_sq: f32,
    label: u8,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq.total_cmp(&other.dist_sq).is_eq()
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist_sq.total_cmp(&other.dist_sq)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub fn knn(query: &[f32; N_DIMS_PADDED], dataset: &Dataset, k: usize) -> Vec<(f32, u8)> {
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k);

    for (chunk, &label) in dataset
        .vectors()
        .chunks_exact(N_DIMS_PADDED)
        .zip(dataset.labels().iter())
    {
        let dist_sq = l2_sq(query, chunk);

        if heap.len() < k {
            heap.push(Candidate { dist_sq, label });
        } else if dist_sq <= heap.peek().unwrap().dist_sq {
            let _ = heap.pop().unwrap();
            heap.push(Candidate { dist_sq, label });
        }
    }

    heap.into_sorted_vec()
        .into_iter()
        .map(|c| (c.dist_sq, c.label))
        .collect()
}
