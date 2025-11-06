pub mod linalg;

use crate::{Floatify, Value, errors::Result, scalar::Scalar};
use linalg::ConcreteMatrix;

#[derive(Clone, Debug)]
pub enum Matrix {
    /// A matrix whose size is well known.
    ///
    /// The dimensions, w, h, of this matrix are:
    /// - w = concrete.w
    /// - h = concrete.h
    Concrete(ConcreteMatrix),

    /// An identity matrix with possibly more
    /// stuff added on. For example:
    ///
    /// ```
    /// 1 0 1 0 0 0
    /// 1 1 0 0 0 0
    /// 0 1 1 0 0 0
    /// 0 0 0 1 0 0
    /// 0 0 0 0 1 0
    /// 0 0 0 0 0 1
    /// ```
    /// This can be split into the block matrix:
    /// ```
    /// M O
    /// O I
    /// ```
    /// `concrete` stores `M`, `I` is `one`
    /// repeated indefinitely.
    ///
    /// For a fresh identity, `concrete` will be
    /// empty.
    ///
    /// We always keep `concrete` as square.
    ///
    /// The dimensions, w, h, of this matrix are:
    /// - w >= max(concrete.w, concrete.h)
    /// - h >= max(concrete.w, concrete.h)
    /// - w = h.
    Identity {
        concrete: ConcreteMatrix,

        // boxed to not be a recursive def
        scale: Option<Box<Value>>,
    },

    /// Matrix with a known number of columns.
    ///
    /// The dimensions, w, h, of this matrix are:
    /// - w = concrete.w
    /// - h >= concrete.h
    UnboundedRows(ConcreteMatrix),

    /// Matrix with a known number of rows.
    ///
    /// The dimensions, w, h, of this matrix are:
    /// - w >= concrete.w
    /// - h = concrete.h
    UnboundedCols(ConcreteMatrix),

    /// Matrix whose size is unbounded.
    ///
    /// The dimensions, w, h, of this matrix are:
    /// - w >= concrete.w
    /// - h >= concrete.h
    UnboundedSize(ConcreteMatrix),
}

impl Floatify for Matrix {
    type Floated = Self;

    fn floatify(mut self) -> Self::Floated {
        match self {
            Matrix::Concrete(ref mut concrete)
            | Matrix::UnboundedRows(ref mut concrete)
            | Matrix::UnboundedCols(ref mut concrete)
            | Matrix::UnboundedSize(ref mut concrete) => {
                let mut temp = ConcreteMatrix::EMPTY;
                std::mem::swap(&mut temp, concrete);
                *concrete = temp.floatify();

                self
            }
            Matrix::Identity { concrete, scale } => Matrix::Identity {
                concrete: concrete.floatify(),
                scale: scale.map(|x| Box::new(x.floatify())),
            },
        }
    }
}

impl Matrix {
    // These three probably don't make sense?
    // pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &Value)> {}

    // pub fn iter_col(&self, col: usize) -> Result<impl Iterator<Item = &Value>> {}

    // pub fn iter_row(&self, row: usize) -> Result<impl Iterator<Item = &Value>> {}

    pub fn remove_col(&self, col: usize) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete) => concrete.remove_col(col).map(Matrix::Concrete),

            Matrix::UnboundedRows(concrete) => concrete.remove_col(col).map(Matrix::UnboundedRows),

            Matrix::UnboundedCols(concrete) => {
                let result = if col >= concrete.width {
                    Ok(concrete.clone())
                } else {
                    concrete.remove_col(col)
                };

                result.map(Matrix::UnboundedRows)
            }

            Matrix::UnboundedSize(concrete) => {
                let result = if col >= concrete.width {
                    Ok(concrete.clone())
                } else {
                    concrete.remove_col(col)
                };

                result.map(Matrix::UnboundedSize)
            }

            Matrix::Identity { .. } => Err(crate::errors::MathError::UnsupportedMatrixOp),
        }
    }

    pub fn remove_row(&self, row: usize) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete) => concrete.remove_row(row).map(Matrix::Concrete),

            Matrix::UnboundedRows(concrete) => {
                let result = if row >= concrete.height {
                    Ok(concrete.clone())
                } else {
                    concrete.remove_row(row)
                };

                result.map(Matrix::UnboundedRows)
            }

            Matrix::UnboundedCols(concrete) => concrete.remove_row(row).map(Matrix::UnboundedRows),

            Matrix::UnboundedSize(concrete) => {
                let result = if row >= concrete.height {
                    Ok(concrete.clone())
                } else {
                    concrete.remove_row(row)
                };

                result.map(Matrix::UnboundedSize)
            }

            Matrix::Identity { .. } => Err(crate::errors::MathError::UnsupportedMatrixOp),
        }
    }

    pub fn remove_col_row(&self, col: usize, row: usize) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix
                .remove_col_row(col, row)
                .map(Matrix::Concrete),

            Matrix::UnboundedRows(concrete) => {
                if row >= concrete.height {
                    concrete.remove_col(col).map(Matrix::UnboundedRows)
                } else {
                    concrete.remove_col_row(col, row).map(Matrix::UnboundedRows)
                }
            }

            Matrix::UnboundedCols(concrete) => {
                if col >= concrete.width {
                    concrete.remove_row(row).map(Matrix::UnboundedCols)
                } else {
                    concrete.remove_col_row(col, row).map(Matrix::UnboundedCols)
                }
            }

            Matrix::UnboundedSize(concrete) => {
                let result = if col >= concrete.width {
                    if row >= concrete.height {
                        Ok(concrete.clone())
                    } else {
                        concrete.remove_row(row)
                    }
                } else {
                    if row >= concrete.height {
                        concrete.remove_col(col)
                    } else {
                        concrete.remove_col_row(col, row)
                    }
                };

                result.map(Matrix::UnboundedSize)
            }

            Matrix::Identity { .. } => Err(crate::errors::MathError::UnsupportedMatrixOp),
        }
    }

    pub fn concrete_rows(&self, height: usize) -> Option<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => {
                if height != concrete_matrix.height {
                    None
                } else {
                    Some(self.clone())
                }
            }

            Matrix::Identity { concrete, scale } => {
                if height < concrete.height || height < concrete.width {
                    None
                } else {
                    if height == concrete.height {
                        Some(Matrix::Concrete(concrete.clone()))
                    } else {
                        let delta = height - concrete.height;

                        let mut result = concrete.ext_diag(&Value::ZERO, delta, delta);

                        let one = scale.clone().map(|x| *x).unwrap_or(Value::ONE);

                        for i in height - delta..height {
                            result[(i, i)] = (**one).clone();
                        }

                        Some(Matrix::Concrete(result))
                    }
                }
            }

            Matrix::UnboundedRows(concrete_matrix) => {
                if height >= concrete_matrix.height {
                    Some(Self::Concrete(concrete_matrix.ext_vert(
                        &Value::Scalar(Scalar::ZERO),
                        height - concrete_matrix.height,
                    )))
                } else {
                    None
                }
            }

            Matrix::UnboundedCols(concrete_matrix) => {
                if height == concrete_matrix.height {
                    Some(self.clone())
                } else {
                    None
                }
            }

            Matrix::UnboundedSize(concrete_matrix) => {
                if height >= concrete_matrix.height {
                    Some(Self::UnboundedCols(concrete_matrix.ext_vert(
                        &Value::Scalar(Scalar::ZERO),
                        height - concrete_matrix.height,
                    )))
                } else {
                    None
                }
            }
        }
    }

    pub fn concrete_cols(&self, width: usize) -> Option<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => {
                if width != concrete_matrix.width {
                    None
                } else {
                    Some(self.clone())
                }
            }

            Matrix::Identity { concrete, one } => {
                if width < concrete.height || width < concrete.width {
                    None
                } else {
                    if width == concrete.width {
                        Some(Matrix::Concrete(concrete.clone()))
                    } else {
                        let delta = width - concrete.width;

                        let mut result =
                            concrete.ext_diag(&Value::Scalar(Scalar::ZERO), delta, delta);

                        for i in width - delta..width {
                            result[(i, i)] = (**one).clone();
                        }

                        Some(Matrix::Concrete(result))
                    }
                }
            }

            Matrix::UnboundedRows(concrete_matrix) => {
                if width == concrete_matrix.width {
                    Some(self.clone())
                } else {
                    None
                }
            }

            Matrix::UnboundedCols(concrete_matrix) => {
                if width >= concrete_matrix.width {
                    Some(Self::Concrete(concrete_matrix.ext_hor(
                        &Value::Scalar(Scalar::ZERO),
                        width - concrete_matrix.width,
                    )))
                } else {
                    None
                }
            }

            Matrix::UnboundedSize(concrete_matrix) => {
                if width >= concrete_matrix.width {
                    Some(Self::UnboundedCols(concrete_matrix.ext_hor(
                        &Value::Scalar(Scalar::ZERO),
                        width - concrete_matrix.width,
                    )))
                } else {
                    None
                }
            }
        }
    }

    pub fn det(&self) -> Result<Value> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix.det(),
            Matrix::Identity { concrete, one } => concrete.det().and_then(|x| x.mul(one)),
            Matrix::ColumnBasisElems { .. } => Err(crate::errors::MathError::NonSquareDeterminant),
            Matrix::RowBasisElems { .. } => Err(crate::errors::MathError::NonSquareDeterminant),
            Matrix::MatrixBasisElems { concrete } => todo!(),
        }
    }

    pub fn get_col(&self, col: usize) -> Result<Self> {}

    pub fn get_row(&self, row: usize) -> Result<Self> {}

    pub fn aug_vert(&self, bottom: &Self) -> Result<Self> {}

    pub fn aug_hor(&self, right: &Self) -> Result<Self> {}

    // Fills in off-diagonals with zeros
    pub fn aug_diag(&self, bottom_right: &Self) -> Self {}

    pub fn mul(&self, rhs: &Self) -> Result<Self> {}

    pub fn hadamard_op(
        &self,
        rhs: &Self,
        mut op: impl FnMut(&Value, &Value) -> Result<Value>,
    ) -> Result<Self> {
    }

    pub fn scalar_mul(&self, rhs: &Value) -> Result<Self> {}

    pub fn select_cols(&self, cols: impl Iterator<Item = usize>) -> Result<Self> {}

    pub fn select_rows(&self, rows: impl Iterator<Item = usize>) -> Result<Self> {}

    pub fn from_size_slice_rowmaj(width: usize, height: usize, values: &[Value]) -> Result<Self> {}

    pub fn from_size_slice_colmaj(width: usize, height: usize, values: &[Value]) -> Result<Self> {}

    pub fn add(&self, rhs: &Self) -> Result<Self> {}

    pub fn sub(&self, rhs: &Self) -> Result<Self> {}

    pub fn mul_componentwise(&self, rhs: &ConcreteMatrix) -> Result<Self> {}

    pub fn neg(&self) -> Result<Self> {}

    fn swap_rows(&mut self, r1: usize, r2: usize) -> Result<()> {}

    fn scale_row(&mut self, row: usize, value: &Value) -> Result<()> {}

    // Rdest -= scale * src
    fn sub_row(&mut self, src: usize, dest: usize, scale: &Value) -> Result<()> {}

    fn row_echelon_form_internal(&self, reduced: bool) -> Result<(Self, usize)> {}

    #[doc(alias = "rref")]
    pub fn row_echelon_form(&self, reduced: bool) -> Result<Self> {}

    pub fn identity(width: usize) -> Result<Self> {}

    pub fn my_identity(&self) -> Result<Self> {}

    pub fn col_from_iter(iter: impl IntoIterator<Item = Value>) -> Self {}

    pub fn row_from_iter(iter: impl IntoIterator<Item = Value>) -> Self {}

    pub fn my_basis_elem(&self, elem: usize) -> Result<Self> {}

    pub fn zero_matrix(&self) -> Self {}

    pub fn invert(&self) -> Result<Self> {}

    pub fn is_zero(&self) -> bool {}

    pub fn transpose(&self) -> Result<Self> {}

    pub fn conj(&self) -> Result<Self> {}

    pub fn norm_sq(&self) -> Result<Value> {}

    pub fn sin(&self) -> Result<Self> {}

    pub fn cos(&self) -> Result<Self> {}

    pub fn tan(&self) -> Result<Self> {}

    pub fn sinh(&self) -> Result<Self> {}

    pub fn cosh(&self) -> Result<Self> {}

    pub fn tanh(&self) -> Result<Self> {}

    // Matrix exponentiation through Sylvester's Formula
    fn exp_inner(&self, eigenvalues: &[(Scalar, usize)]) -> Result<Self> {}

    // Householder matrix for given column vector
    pub fn householder_matrix(&self) -> Result<Self> {}

    pub fn lower_hessenberg(&self) -> Result<Self> {}

    // todo: finish a proper exp impl instead of this
    // taylor series eval
    pub fn exp(&self) -> Result<Self> {}
}
