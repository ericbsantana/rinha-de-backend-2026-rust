#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Label {
    Legit = 0,
    Fraud = 1,
}
pub const N_DIMS_PADDED: usize = 16;
