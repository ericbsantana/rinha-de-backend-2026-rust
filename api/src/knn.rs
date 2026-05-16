use crate::dataset::Dataset;
use crate::label::N_DIMS_PADDED;
use std::collections::BinaryHeap;

struct Candidate {
    score: f32,
    label: u8,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score).is_eq()
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.total_cmp(&other.score)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub fn knn<D>(query: &[f32; N_DIMS_PADDED], dataset: &Dataset, k: usize, dist: D) -> Vec<(f32, u8)>
where
    D: Fn(&[f32; N_DIMS_PADDED], &[f32]) -> f32,
{
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k);

    for (chunk, &label) in dataset
        .vectors()
        .chunks_exact(N_DIMS_PADDED)
        .zip(dataset.labels().iter())
    {
        let score = dist(query, chunk);

        if heap.len() < k {
            heap.push(Candidate { score, label });
        } else if score <= heap.peek().unwrap().score {
            let _ = heap.pop().unwrap();
            heap.push(Candidate { score, label });
        }
    }

    heap.into_sorted_vec()
        .into_iter()
        .map(|c| (c.score, c.label))
        .collect()
}
