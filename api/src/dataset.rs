use std::{fs::File, io, path::Path};

use memmap2::Mmap;

use crate::label::N_DIMS_PADDED;

enum Storage {
    Owned(Vec<u8>),
    Mapped(Mmap),
}

impl Storage {
    fn as_bytes(&self) -> &[u8] {
        match self {
            Storage::Owned(v) => v.as_slice(),
            Storage::Mapped(m) => &m[..],
        }
    }
}

pub struct Dataset {
    vectors: Storage,
    labels: Storage,
}

impl Dataset {
    pub fn load(dir: &Path) -> io::Result<Self> {
        let vec_file = File::open(dir.join("vectors.bin"))?;
        let lbl_file = File::open(dir.join("labels.bin"))?;

        // SAFETY: out/vectors.bin and out/labels.bin are produced at build time by
        // the preprocess binary and never modified at runtime. If another process
        // truncated or wrote to these files while mapped, accessing the mapping
        // could SIGBUS. The deployment contract guarantees they are immutable.
        let vectors_mmap = unsafe { Mmap::map(&vec_file)? };
        let labels_mmap = unsafe { Mmap::map(&lbl_file)? };

        let dataset = Self {
            vectors: Storage::Mapped(vectors_mmap),
            labels: Storage::Mapped(labels_mmap),
        };
        dataset.check_invariant();
        Ok(dataset)
    }

    pub fn from_parts(vectors_bytes: Vec<u8>, labels: Vec<u8>) -> Self {
        let dataset = Self {
            vectors: Storage::Owned(vectors_bytes),
            labels: Storage::Owned(labels),
        };
        dataset.check_invariant();
        dataset
    }

    fn check_invariant(&self) {
        let vl = self.vectors.as_bytes().len();
        let ll = self.labels.as_bytes().len();
        let expected = ll * N_DIMS_PADDED * size_of::<f32>();
        assert_eq!(
            vl, expected,
            "vectors and labels disagree: {vl} vs {expected}"
        );
    }

    pub fn vectors(&self) -> &[f32] {
        bytemuck::cast_slice(self.vectors.as_bytes())
    }

    pub fn labels(&self) -> &[u8] {
        self.labels.as_bytes()
    }

    pub fn len(&self) -> usize {
        self.labels.as_bytes().len()
    }
}
