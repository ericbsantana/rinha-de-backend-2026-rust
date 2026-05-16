use std::{fs, io, path::Path};

use crate::label::N_DIMS_PADDED;

pub struct Dataset {
    vectors_bytes: Vec<u8>,
    labels: Vec<u8>,
}

impl Dataset {
    pub fn load(dir: &Path) -> io::Result<Self> {
        let vectors_bytes = fs::read(dir.join("vectors.bin"))?;
        let labels = fs::read(dir.join("labels.bin"))?;

        assert_eq!(
            vectors_bytes.len(),
            labels.len() * N_DIMS_PADDED * size_of::<f32>(),
            "vectors.bin and labels.bin disagree: {} vs {}",
            vectors_bytes.len(),
            labels.len() * N_DIMS_PADDED * size_of::<f32>()
        );

        Ok(Self {
            vectors_bytes,
            labels,
        })
    }

    pub fn from_parts(vectors_bytes: Vec<u8>, labels: Vec<u8>) -> Self {
        assert_eq!(
            vectors_bytes.len(),
            labels.len() * N_DIMS_PADDED * size_of::<f32>(),
            "vectors.bin and labels.bin disagree: {} vs {}",
            vectors_bytes.len(),
            labels.len() * N_DIMS_PADDED * size_of::<f32>()
        );

        Self {
            vectors_bytes,
            labels,
        }
    }

    pub fn vectors(&self) -> &[f32] {
        bytemuck::cast_slice(&self.vectors_bytes)
    }

    pub fn labels(&self) -> &[u8] {
        &self.labels
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }
}
