use std::{
    fs::{self, File},
    io::{BufReader, Read},
};

use api::label::{Label, N_DIMS_PADDED};
use flate2::read::GzDecoder;
use serde::Deserialize;

const N_DIMS_RAW: usize = 14;

#[derive(Deserialize)]
struct Reference {
    vector: [f32; N_DIMS_RAW],
    label: String,
}

fn main() -> std::io::Result<()> {
    let input_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "resources/example-references.json".to_string());
    let out_dir = "out";

    let file = File::open(&input_path)?;
    let reader: Box<dyn Read> = if input_path.ends_with(".gz") {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let reader = BufReader::new(reader);

    let references: Vec<Reference> = serde_json::from_reader(reader)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let n = references.len();
    println!("parsed {n} references");

    let mut flat_vectors: Vec<f32> = Vec::with_capacity(n * N_DIMS_PADDED);
    let mut labels: Vec<u8> = Vec::with_capacity(n);

    for r in &references {
        flat_vectors.extend_from_slice(&r.vector);
        flat_vectors.extend_from_slice(&[0.0, 0.0]);
        let label = match r.label.as_str() {
            "legit" => Label::Legit,
            "fraud" => Label::Fraud,
            other => panic!("unknown label: {other:?}"),
        };

        labels.push(label as u8);
    }

    fs::create_dir_all(out_dir)?;
    fs::write(
        format!("{out_dir}/vectors.bin"),
        bytemuck::cast_slice::<f32, u8>(&flat_vectors),
    )?;
    fs::write(format!("{out_dir}/labels.bin"), &labels)?;
    let vec_file = fs::metadata(format!("{out_dir}/vectors.bin"))?.len();
    let lbl_file = fs::metadata(format!("{out_dir}/labels.bin"))?.len();

    println!("n_vectors    = {n}");
    println!(
        "vectors.bin  = {vec_file} bytes (expected {} = n*{N_DIMS_PADDED}*4)",
        n * N_DIMS_PADDED * 4
    );
    println!("labels.bin   = {lbl_file} bytes (expected {n})");

    let vec_bytes = fs::read(format!("{out_dir}/vectors.bin"))?;
    let lbl_bytes = fs::read(format!("{out_dir}/labels.bin"))?;

    assert_eq!(
        vec_bytes.len(),
        n * N_DIMS_PADDED * std::mem::size_of::<f32>()
    );
    assert_eq!(lbl_bytes.len(), n);

    let vec_back: &[f32] = bytemuck::cast_slice::<u8, f32>(&vec_bytes);
    assert_eq!(
        vec_back,
        flat_vectors.as_slice(),
        "vectors roundtrip mismatch"
    );

    assert_eq!(
        lbl_bytes.as_slice(),
        labels.as_slice(),
        "labels roundtrip mismatch"
    );

    println!("roundtrip ok");

    Ok(())
}
