pub type Result<T> = std::result::Result<T, MathError>;

#[derive(Debug)]
pub enum MathError {
    NonSquareDeterminant,
    MatrixOutOfRange,
    AugmentShapeMismatch,
    MatrixShapeMismatch,
    EmptyMatrix,
    AddScalarMatrix,
    DivideByZero,
    MatrixNotSquare,
    MatrixNotInvertible,
    NotImplemented,
    UnsupportedMatrix,
    NotColumnVector,
    NoIdentityMatrix,
    ZeroSizeMatrix,
    WrongNumberOfArgs,
    UnsupportedMatrixOp,
    CannotDeduceDimensions,
}
