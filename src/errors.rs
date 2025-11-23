use crate::eval::Ident;

use std::{
    convert::Infallible,
    ops::{FromResidual, Try},
};

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
    UnknownVariable,
    CannotOverwriteRuntime,
}

#[derive(Debug)]
pub enum EvalStatus<T> {
    // Eventually going to mean no reducible expressions.
    //
    // But currently just means that there are
    // no free variables left.
    Yielded(T),

    // The first param is the partially-reduced expression.
    //
    // The second param is a list of free variables
    // we encountered.
    Halted(T, Vec<Ident>),

    // There was a mathematical error somewhere, like
    // dividing by zero or multiply matrices of the
    // wrong size.
    Errored(MathError),
}

impl<T> FromResidual for EvalStatus<T> {
    fn from_residual(residual: Result<(T, Vec<Ident>), MathError>) -> Self {
        match residual {
            Ok((partially_reduced, idents)) => Self::Halted(partially_reduced, idents),
            Err(e) => Self::Errored(e),
        }
    }
}

impl<T> FromResidual<EvalStatus<Infallible>> for EvalStatus<T> {
    fn from_residual(residual: EvalStatus<Infallible>) -> Self {
        match residual {
            EvalStatus::Errored(e) => Self::Errored(e),
        }
    }
}

impl<T> Try for EvalStatus<T> {
    type Output = T;

    type Residual = Result<(T, Vec<Ident>), MathError>;

    fn from_output(output: T) -> Self {
        Self::Yielded(output)
    }

    fn branch(self) -> std::ops::ControlFlow<Self::Residual, Self::Output> {
        match self {
            EvalStatus::Yielded(x) => std::ops::ControlFlow::Continue(x),
            EvalStatus::Halted(partially_reduced, idents) => {
                std::ops::ControlFlow::Break(Ok((partially_reduced, idents)))
            }
            EvalStatus::Errored(e) => std::ops::ControlFlow::Break(Err(e)),
        }
    }
}
