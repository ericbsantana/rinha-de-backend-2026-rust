use api::dataset::Dataset;
use api::knn::knn;
use api::label::N_DIMS_PADDED;
use std::path::Path;
use std::sync::Arc;

fn main() -> std::io::Result<()> {
    let dataset = Arc::new(Dataset::load(Path::new("out"))?);
    println!(
        "loaded {} vectors, {} labels",
        dataset.vectors().len() / 16,
        dataset.labels().len()
    );

    let mut query = [0.0_f32; N_DIMS_PADDED];

    query.copy_from_slice(&dataset.vectors()[0..16]);

    let start = std::time::Instant::now();
    let neighbors = knn(&query, &dataset, 5);
    let elapsed = start.elapsed();

    println!("knn took {:?}", elapsed);
    for (dist_sq, label) in &neighbors {
        println!("  dist_sq={:.6} label={}", dist_sq, label);
    }
    Ok(())
}
