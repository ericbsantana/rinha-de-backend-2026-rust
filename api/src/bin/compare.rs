use api::dataset::Dataset;
#[cfg(target_arch = "aarch64")]
use api::distance::l_inf_neon;
use api::distance::{cosine_dist, l_inf, l1, l2_sq, weighted_l2_sq};
use api::knn::knn;
use api::label::N_DIMS_PADDED;
use std::path::Path;
use std::time::Instant;

const K: usize = 5;
const FRAUD_THRESHOLD: f32 = 0.6;
const TEST_STEP: usize = 20;
const TEST_SAMPLE_SIZE: usize = 5000;

#[derive(Default, Debug)]
struct Metrics {
    tp: u32,
    fp: u32,
    tn: u32,
    fn_: u32,
}

impl Metrics {
    fn n(&self) -> u32 {
        self.tp + self.fp + self.tn + self.fn_
    }
    fn errors(&self) -> u32 {
        self.fp + self.fn_
    }
}

fn score_det(m: &Metrics) -> f64 {
    let n = m.n() as f64;
    let e = m.fp as f64 + 3.0 * m.fn_ as f64;
    let failures = m.errors() as f64;
    if failures / n > 0.15 {
        return -3000.0;
    }
    let epsilon = (e / n).max(0.001);
    1000.0 * (1.0_f64 / epsilon).log10() - 300.0 * (1.0 + e).log10()
}

fn build_split(dataset: &Dataset) -> (Vec<usize>, Vec<usize>) {
    let mut legit_train = Vec::new();
    let mut legit_test = Vec::new();
    let mut fraud_train = Vec::new();
    let mut fraud_test = Vec::new();

    for (i, &label) in dataset.labels().iter().enumerate() {
        let goes_to_test = i % TEST_STEP == 0;
        match (goes_to_test, label) {
            (true, 0) => legit_test.push(i),
            (true, 1) => fraud_test.push(i),
            (false, 0) => legit_train.push(i),
            (false, 1) => fraud_train.push(i),
            _ => panic!("unknown label: {label}"),
        }
    }

    let test_pool_size = legit_test.len() + fraud_test.len();
    let n_fraud_test = (TEST_SAMPLE_SIZE * fraud_test.len() / test_pool_size).max(1);
    let n_legit_test = TEST_SAMPLE_SIZE - n_fraud_test;

    let mut test_indices = Vec::with_capacity(TEST_SAMPLE_SIZE);
    test_indices.extend(fraud_test.iter().take(n_fraud_test));
    test_indices.extend(legit_test.iter().take(n_legit_test));

    let mut train_indices = legit_train;
    train_indices.extend(fraud_train);

    (train_indices, test_indices)
}

fn build_train_dataset(full: &Dataset, train_indices: &[usize]) -> Dataset {
    let n_train = train_indices.len();
    let bytes_per_vec = N_DIMS_PADDED * std::mem::size_of::<f32>();

    let mut vectors_bytes: Vec<u8> = Vec::with_capacity(n_train * bytes_per_vec);
    let mut labels: Vec<u8> = Vec::with_capacity(n_train);

    let full_bytes = bytemuck::cast_slice::<f32, u8>(full.vectors());
    let full_labels = full.labels();

    for &i in train_indices {
        let start = i * bytes_per_vec;
        let end = start + bytes_per_vec;
        vectors_bytes.extend_from_slice(&full_bytes[start..end]);
        labels.push(full_labels[i]);
    }

    Dataset::from_parts(vectors_bytes, labels)
}

fn evaluate<D>(
    name: &str,
    train: &Dataset,
    test_indices: &[usize],
    full: &Dataset,
    dist: D,
) -> Metrics
where
    D: Fn(&[f32; N_DIMS_PADDED], &[f32]) -> f32 + Copy,
{
    let mut m = Metrics::default();
    let start = Instant::now();

    for &i in test_indices {
        let mut query = [0.0_f32; N_DIMS_PADDED];
        let bytes_per_vec = N_DIMS_PADDED * std::mem::size_of::<f32>();
        let full_bytes = bytemuck::cast_slice::<f32, u8>(full.vectors());
        let q_bytes = &full_bytes[i * bytes_per_vec..(i + 1) * bytes_per_vec];
        let q_floats: &[f32] = bytemuck::cast_slice(q_bytes);
        query.copy_from_slice(q_floats);

        let true_is_fraud = full.labels()[i] == 1;

        let neighbors = knn(&query, train, K, dist);
        let frauds = neighbors.iter().filter(|(_, l)| *l == 1).count();
        let fraud_score = frauds as f32 / K as f32;
        let predicted_fraud = fraud_score >= FRAUD_THRESHOLD;

        match (predicted_fraud, true_is_fraud) {
            (true, true) => m.tp += 1,
            (true, false) => m.fp += 1,
            (false, true) => m.fn_ += 1,
            (false, false) => m.tn += 1,
        }
    }

    let elapsed = start.elapsed();
    println!("{:<12} took {:.2?}", name, elapsed);
    m
}

fn main() -> std::io::Result<()> {
    let dataset = Dataset::load(Path::new("out"))?;
    println!("loaded {} vectors", dataset.len());

    println!("splitting...");
    let (train_indices, test_indices) = build_split(&dataset);
    println!("  train: {}", train_indices.len());
    println!("  test:  {}", test_indices.len());

    println!("building train dataset...");
    let train = build_train_dataset(&dataset, &train_indices);

    println!(
        "\n{:<12} {:>5} {:>5} {:>5} {:>5}   {:>10}",
        "distance", "TP", "FP", "FN", "TN", "score_det"
    );

    let weights: [f32; N_DIMS_PADDED] = [1.0; N_DIMS_PADDED];

    let m_l2 = evaluate("l2_sq", &train, &test_indices, &dataset, l2_sq);
    print_row("l2_sq", &m_l2);

    let m_l1 = evaluate("l1", &train, &test_indices, &dataset, l1);
    print_row("l1", &m_l1);

    let m_linf = evaluate("l_inf", &train, &test_indices, &dataset, l_inf);
    print_row("l_inf", &m_linf);

    let m_cos = evaluate("cosine", &train, &test_indices, &dataset, cosine_dist);
    print_row("cosine", &m_cos);

    let weighted_fn = |q: &[f32; N_DIMS_PADDED], c: &[f32]| weighted_l2_sq(q, c, &weights);
    let m_w = evaluate("weighted", &train, &test_indices, &dataset, weighted_fn);
    print_row("weighted", &m_w);

    println!("\nlatency bench (5000 queries):");

    let bytes_per_vec = N_DIMS_PADDED * std::mem::size_of::<f32>();
    let full_bytes = bytemuck::cast_slice::<f32, u8>(dataset.vectors());

    let mut queries: Vec<[f32; N_DIMS_PADDED]> = Vec::with_capacity(test_indices.len());
    for &i in &test_indices {
        let mut q = [0.0_f32; N_DIMS_PADDED];
        let q_bytes = &full_bytes[i * bytes_per_vec..(i + 1) * bytes_per_vec];
        q.copy_from_slice(bytemuck::cast_slice(q_bytes));
        queries.push(q);
    }

    let scalar_t = {
        let start = Instant::now();
        for q in &queries {
            let _ = knn(q, &train, K, l_inf);
        }
        start.elapsed()
    };
    println!("  l_inf  scalar: {:.2?}", scalar_t);

    #[cfg(target_arch = "aarch64")]
    {
        let simd_t = {
            let start = Instant::now();
            for q in &queries {
                let _ = knn(q, &train, K, l_inf_neon);
            }
            start.elapsed()
        };
        println!("  l_inf  neon:   {:.2?}", simd_t);
        println!(
            "  speedup: {:.2}x",
            scalar_t.as_secs_f64() / simd_t.as_secs_f64()
        );
    }

    Ok(())
}

fn print_row(name: &str, m: &Metrics) {
    println!(
        "{:<12} {:>5} {:>5} {:>5} {:>5}   {:>+10.0}",
        name,
        m.tp,
        m.fp,
        m.fn_,
        m.tn,
        score_det(m)
    );
}
