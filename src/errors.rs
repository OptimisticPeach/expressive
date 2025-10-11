pub type Result<T> = std::result::Result<T, MathError>;

pub enum MathError {
    NonSquareDeterminant,
    MatrixOutOfRange,
    AugmentShapeMismatch,
    MatrixMultShapeMismatch,
    EmptyMatrix,
    MatrixAddShapeMismatch,
    MatrixHadamardMismatch,
}
