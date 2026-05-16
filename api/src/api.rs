use api::dataset::Dataset;
use std::path::Path;
use std::sync::Arc;

fn main() -> std::io::Result<()> {
    let dataset = Arc::new(Dataset::load(Path::new("out"))?);
    println!(
        "loaded {} vectors, {} labels",
        dataset.vectors().len() / 16,
        dataset.labels().len()
    );
    Ok(())
}
