pub mod linalg;

use crate::{
    Floatify, Value,
    errors::{MathError, Result},
};
use linalg::ConcreteMatrix;

#[derive(Clone, Debug, PartialEq)]
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
    UnboundedHeight(ConcreteMatrix),

    /// Matrix with a known number of rows.
    ///
    /// The dimensions, w, h, of this matrix are:
    /// - w >= concrete.w
    /// - h = concrete.h
    UnboundedWidth(ConcreteMatrix),

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
            | Matrix::UnboundedHeight(ref mut concrete)
            | Matrix::UnboundedWidth(ref mut concrete)
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

            Matrix::UnboundedHeight(concrete) => {
                concrete.remove_col(col).map(Matrix::UnboundedHeight)
            }

            Matrix::UnboundedWidth(concrete) => {
                let result = if col >= concrete.width {
                    Ok(concrete.clone())
                } else {
                    concrete.remove_col(col)
                };

                result.map(Matrix::UnboundedHeight)
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

            Matrix::UnboundedHeight(concrete) => {
                let result = if row >= concrete.height {
                    Ok(concrete.clone())
                } else {
                    concrete.remove_row(row)
                };

                result.map(Matrix::UnboundedHeight)
            }

            Matrix::UnboundedWidth(concrete) => {
                concrete.remove_row(row).map(Matrix::UnboundedHeight)
            }

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

            Matrix::UnboundedHeight(concrete) => {
                if row >= concrete.height {
                    concrete.remove_col(col).map(Matrix::UnboundedHeight)
                } else {
                    concrete
                        .remove_col_row(col, row)
                        .map(Matrix::UnboundedHeight)
                }
            }

            Matrix::UnboundedWidth(concrete) => {
                if col >= concrete.width {
                    concrete.remove_row(row).map(Matrix::UnboundedWidth)
                } else {
                    concrete
                        .remove_col_row(col, row)
                        .map(Matrix::UnboundedWidth)
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

    fn concrete_rows(&self, height: usize) -> Option<Self> {
        let result = match self.extend_rows(height)? {
            x @ Matrix::UnboundedWidth(_) => x,
            Matrix::UnboundedSize(concrete_matrix) => Matrix::UnboundedWidth(concrete_matrix),

            x @ (Matrix::Concrete(_) | Matrix::Identity { .. } | Matrix::UnboundedHeight(_)) => {
                Self::Concrete(x.unwrap_best_guess())
            }
        };

        Some(result)
    }

    fn concrete_cols(&self, width: usize) -> Option<Self> {
        let result = match self.extend_cols(width)? {
            x @ Matrix::UnboundedHeight(_) => x,
            Matrix::UnboundedSize(concrete_matrix) => Matrix::UnboundedHeight(concrete_matrix),

            x @ (Matrix::Concrete(_) | Matrix::Identity { .. } | Matrix::UnboundedWidth(_)) => {
                Self::Concrete(x.unwrap_best_guess())
            }
        };

        Some(result)
    }

    fn concrete_row_cols(&self, width: usize, height: usize) -> Option<ConcreteMatrix> {
        self.extend_cols_rows(width, height)
            .map(Matrix::unwrap_best_guess)
    }

    fn extend_rows(&self, height: usize) -> Option<Self> {
        let cur_height = self.min_height();

        if cur_height == height {
            return Some(self.clone());
        }

        if cur_height > height {
            return None;
        }

        // We need to extend...

        match self {
            // can't.
            Matrix::Concrete(_) => None,
            Matrix::UnboundedWidth(_) => None,

            Matrix::Identity { concrete, scale } => {
                let delta = height - concrete.height;

                let mut result = concrete.ext_diag(&Value::ZERO, delta, delta);

                let one = scale.clone().map(|x| *x).unwrap_or(Value::ONE);

                for i in height - delta..height {
                    result[(i, i)] = one.clone();
                }

                Some(Matrix::Identity {
                    concrete: result,
                    scale: scale.clone(),
                })
            }
            Matrix::UnboundedHeight(concrete_matrix) => Some(Matrix::UnboundedHeight(
                concrete_matrix.ext_vert(&Value::ZERO, height - concrete_matrix.height),
            )),

            Matrix::UnboundedSize(concrete_matrix) => Some(Matrix::UnboundedSize(
                concrete_matrix.ext_vert(&Value::ZERO, height - concrete_matrix.height),
            )),
        }
    }

    fn extend_cols(&self, width: usize) -> Option<Self> {
        let cur_width = self.min_width();

        if cur_width == width {
            return Some(self.clone());
        }

        if cur_width > width {
            return None;
        }

        // We need to extend...

        match self {
            // can't.
            Matrix::Concrete(_) => None,
            Matrix::UnboundedHeight(_) => None,

            Matrix::Identity { concrete, scale } => {
                let delta = width - concrete.width;

                let mut result = concrete.ext_diag(&Value::ZERO, delta, delta);

                let one = scale.clone().map(|x| *x).unwrap_or(Value::ONE);

                for i in width - delta..width {
                    result[(i, i)] = one.clone();
                }

                Some(Matrix::Identity {
                    concrete: result,
                    scale: scale.clone(),
                })
            }
            Matrix::UnboundedWidth(concrete_matrix) => Some(Matrix::UnboundedWidth(
                concrete_matrix.ext_hor(&Value::ZERO, width - concrete_matrix.width),
            )),

            Matrix::UnboundedSize(concrete_matrix) => Some(Matrix::UnboundedSize(
                concrete_matrix.ext_hor(&Value::ZERO, width - concrete_matrix.width),
            )),
        }
    }

    fn extend_cols_rows(&self, width: usize, height: usize) -> Option<Self> {
        let cur_width = self.min_width();
        let cur_height = self.min_height();

        if cur_width > width || cur_height > height {
            return None;
        }

        if cur_width == width && cur_height == height {
            return Some(self.clone());
        } else if cur_width == width {
            return self.extend_rows(height);
        } else if cur_height == height {
            return self.extend_cols(width);
        }

        let delta_x = width - cur_width;
        let delta_y = height - cur_height;

        // We need to extend both...
        match self {
            // can't.
            Matrix::Concrete(_) => None,
            Matrix::UnboundedHeight(_) => None,
            Matrix::UnboundedWidth(_) => None,

            Matrix::Identity { concrete, scale } => {
                if height != width {
                    return None;
                }

                let mut result = concrete.ext_diag(&Value::ZERO, delta_x, delta_x);

                let one = scale.clone().map(|x| *x).unwrap_or(Value::ONE);

                for i in width - delta_x..width {
                    result[(i, i)] = one.clone();
                }

                Some(Matrix::Identity {
                    concrete: result,
                    scale: scale.clone(),
                })
            }
            Matrix::UnboundedSize(concrete_matrix) => Some(Matrix::UnboundedSize(
                concrete_matrix.ext_diag(&Value::ZERO, delta_x, delta_y),
            )),
        }
    }

    pub fn det(&self) -> Result<Value> {
        let width = self.min_width();
        let height = self.min_height();

        if width == 0 || height == 0 {
            assert!(width == 0 && height == 0);

            let Matrix::Identity { scale, .. } = self else {
                unreachable!();
            };

            return Ok(scale.clone().map(|x| *x).unwrap_or(Value::ONE));
        }

        let result = self
            .concrete_row_cols(width, height)
            .ok_or(MathError::NonSquareDeterminant)?;

        result.det()
    }

    pub fn get_col(&self, col: usize) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix.get_col(col).map(Self::Concrete),
            Matrix::Identity { concrete, scale } => {
                if col >= concrete.width {
                    Self::vector_basis_elem(col, scale.clone().map(|x| *x).unwrap_or(Value::ZERO))
                } else {
                    concrete.get_col(col).map(Self::UnboundedHeight)
                }
            }
            Matrix::UnboundedHeight(concrete_matrix) => {
                concrete_matrix.get_col(col).map(Self::UnboundedHeight)
            }
            Matrix::UnboundedWidth(concrete_matrix) => {
                if col > concrete_matrix.width {
                    Ok(Self::Concrete(ConcreteMatrix::col_from_iter(
                        std::iter::repeat_n(Value::ZERO, concrete_matrix.height),
                    )))
                } else {
                    concrete_matrix.get_col(col).map(Self::Concrete)
                }
            }
            Matrix::UnboundedSize(concrete_matrix) => {
                if col > concrete_matrix.width {
                    Ok(Self::UnboundedHeight(ConcreteMatrix::col_from_iter(
                        std::iter::repeat_n(Value::ZERO, concrete_matrix.height),
                    )))
                } else {
                    concrete_matrix.get_col(col).map(Self::UnboundedHeight)
                }
            }
        }
    }

    pub fn get_row(&self, row: usize) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix.get_row(row).map(Self::Concrete),
            Matrix::Identity { concrete, scale } => {
                if row >= concrete.width {
                    Self::transpose_basis_elem(
                        row,
                        scale.clone().map(|x| *x).unwrap_or(Value::ZERO),
                    )
                } else {
                    concrete.get_row(row).map(Self::UnboundedWidth)
                }
            }
            Matrix::UnboundedHeight(concrete_matrix) => {
                if row > concrete_matrix.height {
                    Ok(Self::Concrete(ConcreteMatrix::col_from_iter(
                        std::iter::repeat_n(Value::ZERO, concrete_matrix.width),
                    )))
                } else {
                    concrete_matrix.get_row(row).map(Self::Concrete)
                }
            }
            Matrix::UnboundedWidth(concrete_matrix) => {
                concrete_matrix.get_row(row).map(Self::UnboundedWidth)
            }
            Matrix::UnboundedSize(concrete_matrix) => {
                if row > concrete_matrix.height {
                    Ok(Self::UnboundedHeight(ConcreteMatrix::row_from_iter(
                        std::iter::repeat_n(Value::ZERO, concrete_matrix.width),
                    )))
                } else {
                    concrete_matrix.get_row(row).map(Self::UnboundedWidth)
                }
            }
        }
    }

    fn is_concrete_width(&self) -> bool {
        matches!(self, Matrix::Concrete(_) | Matrix::UnboundedHeight(_))
    }

    fn is_concrete_height(&self) -> bool {
        matches!(self, Matrix::Concrete(_) | Matrix::UnboundedWidth(_))
    }

    fn best_guess(&self) -> &ConcreteMatrix {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix,
            Matrix::Identity { concrete, .. } => concrete,
            Matrix::UnboundedHeight(concrete_matrix) => concrete_matrix,
            Matrix::UnboundedWidth(concrete_matrix) => concrete_matrix,
            Matrix::UnboundedSize(concrete_matrix) => concrete_matrix,
        }
    }

    fn unwrap_best_guess(self) -> ConcreteMatrix {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix,
            Matrix::Identity { concrete, .. } => concrete,
            Matrix::UnboundedHeight(concrete_matrix) => concrete_matrix,
            Matrix::UnboundedWidth(concrete_matrix) => concrete_matrix,
            Matrix::UnboundedSize(concrete_matrix) => concrete_matrix,
        }
    }

    fn min_width(&self) -> usize {
        self.best_guess().width
    }

    fn min_height(&self) -> usize {
        self.best_guess().height
    }

    // augment functions will just take the best guess we have in
    // the unbounded cases and identity case.
    pub fn aug_vert(&self, bottom: &Self) -> Result<Self> {
        if self.is_concrete_width() {
            let reshaped = bottom
                .concrete_cols(self.min_width())
                .ok_or(MathError::AugmentShapeMismatch)?;

            let result = self.best_guess().aug_vert(reshaped.best_guess())?;

            if reshaped.is_concrete_height() {
                return Ok(Matrix::Concrete(result));
            } else {
                return Ok(Matrix::UnboundedHeight(result));
            }
        }

        if bottom.is_concrete_width() {
            let reshaped = self
                .concrete_cols(bottom.min_width())
                .ok_or(MathError::AugmentShapeMismatch)?;

            let result = reshaped.best_guess().aug_vert(bottom.best_guess())?;

            if bottom.is_concrete_height() || matches!(bottom, Matrix::Identity { .. }) {
                return Ok(Matrix::Concrete(result));
            } else {
                return Ok(Matrix::UnboundedHeight(result));
            }
        }

        let result;
        let concrete_height;

        if self.min_width() == bottom.min_width() {
            result = self.best_guess().aug_vert(bottom.best_guess());
            concrete_height =
                bottom.is_concrete_height() || matches!(bottom, Matrix::Identity { .. });
        } else if self.min_width() > bottom.min_width() {
            let bottom = bottom
                .concrete_cols(self.min_width())
                .ok_or(MathError::AugmentShapeMismatch)?;

            result = self.best_guess().aug_vert(bottom.best_guess());
            concrete_height = bottom.is_concrete_height();
        } else {
            let top = self
                .concrete_cols(bottom.min_width())
                .ok_or(MathError::AugmentShapeMismatch)?;

            result = top.best_guess().aug_vert(bottom.best_guess());
            concrete_height =
                bottom.is_concrete_height() || matches!(bottom, Matrix::Identity { .. });
        }

        if concrete_height {
            result.map(Matrix::UnboundedWidth)
        } else {
            result.map(Matrix::UnboundedSize)
        }
    }

    pub fn aug_hor(&self, right: &Self) -> Result<Self> {
        if self.is_concrete_height() {
            let reshaped = right
                .concrete_rows(self.min_height())
                .ok_or(MathError::AugmentShapeMismatch)?;

            let result = self.best_guess().aug_hor(reshaped.best_guess())?;

            if reshaped.is_concrete_width() {
                return Ok(Matrix::Concrete(result));
            } else {
                return Ok(Matrix::UnboundedHeight(result));
            }
        }

        if right.is_concrete_height() {
            let reshaped = self
                .concrete_rows(right.min_height())
                .ok_or(MathError::AugmentShapeMismatch)?;

            let result = reshaped.best_guess().aug_hor(right.best_guess())?;

            if right.is_concrete_width() || matches!(right, Matrix::Identity { .. }) {
                return Ok(Matrix::Concrete(result));
            } else {
                return Ok(Matrix::UnboundedHeight(result));
            }
        }

        let result;
        let concrete_width;

        if self.min_height() == right.min_height() {
            result = self.best_guess().aug_hor(right.best_guess());
            concrete_width = right.is_concrete_width() || matches!(right, Matrix::Identity { .. });
        } else if self.min_height() > right.min_height() {
            let right = right
                .concrete_rows(self.min_height())
                .ok_or(MathError::AugmentShapeMismatch)?;

            result = self.best_guess().aug_hor(right.best_guess());
            concrete_width = right.is_concrete_width();
        } else {
            let top = self
                .concrete_rows(right.min_height())
                .ok_or(MathError::AugmentShapeMismatch)?;

            result = top.best_guess().aug_hor(right.best_guess());
            concrete_width = right.is_concrete_width() || matches!(right, Matrix::Identity { .. });
        }

        if concrete_width {
            result.map(Matrix::UnboundedHeight)
        } else {
            result.map(Matrix::UnboundedSize)
        }
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        // the size upon which both must agree
        let same_size = self.min_width().max(rhs.min_height());

        let lhs = self
            .concrete_cols(same_size)
            .ok_or(MathError::MatrixShapeMismatch)?;
        let rhs = rhs
            .concrete_rows(same_size)
            .ok_or(MathError::MatrixShapeMismatch)?;

        let result = lhs.best_guess().mul(rhs.best_guess())?;

        let result = match (lhs, rhs) {
            (Matrix::Concrete(_), Matrix::Concrete(_)) => Matrix::Concrete(result),
            (Matrix::Concrete(_), Matrix::UnboundedWidth(_)) => Matrix::UnboundedWidth(result),
            (Matrix::UnboundedHeight(_), Matrix::Concrete(_)) => Matrix::UnboundedHeight(result),
            (Matrix::UnboundedHeight(_), Matrix::UnboundedWidth(_)) => {
                Matrix::UnboundedSize(result)
            }

            // doing concrete_rows/cols will always turn an Identity into a Concrete
            (Matrix::Identity { .. }, _) | (_, Matrix::Identity { .. }) => unreachable!(),

            // we just picked the number of columns for lhs
            (Matrix::UnboundedWidth(_) | Matrix::UnboundedSize(_), _) => unreachable!(),

            // we just picked the number of rows for rhs
            (_, Matrix::UnboundedHeight(_) | Matrix::UnboundedSize(_)) => unreachable!(),
        };

        Ok(result)
    }

    pub fn scalar_mul(&self, rhs: &Value) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => {
                concrete_matrix.scalar_mul(rhs).map(Matrix::Concrete)
            }
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.scalar_mul(rhs)?,
                scale: scale
                    .as_ref()
                    .map(|x| x.mul(rhs))
                    .transpose()?
                    .map(Box::new),
            }),
            Matrix::UnboundedHeight(concrete_matrix) => {
                concrete_matrix.scalar_mul(rhs).map(Matrix::UnboundedHeight)
            }
            Matrix::UnboundedWidth(concrete_matrix) => {
                concrete_matrix.scalar_mul(rhs).map(Matrix::UnboundedWidth)
            }
            Matrix::UnboundedSize(concrete_matrix) => {
                concrete_matrix.scalar_mul(rhs).map(Matrix::UnboundedSize)
            }
        }
    }

    pub fn from_size_slice_rowmaj(width: usize, height: usize, values: &[Value]) -> Result<Self> {
        ConcreteMatrix::from_size_slice_rowmaj(width, height, values).map(Matrix::Concrete)
    }

    pub fn from_size_slice_colmaj(width: usize, height: usize, values: &[Value]) -> Result<Self> {
        ConcreteMatrix::from_size_slice_colmaj(width, height, values).map(Matrix::Concrete)
    }

    fn componentwise_binop(
        &self,
        rhs: &Self,
        op: impl FnOnce(&ConcreteMatrix, &ConcreteMatrix) -> Result<ConcreteMatrix>,
        id_op: impl FnOnce(&Value, &Value) -> Result<Value>,
    ) -> Result<Self> {
        // we need to get `self`'s dimensions and `rhs`'s dimensions to agree.

        let width = self.min_width().max(rhs.min_width());
        let height = self.min_height().max(rhs.min_height());

        let temp_me;
        let me;
        let temp_rh;
        let rh;

        if width == self.min_width() && height == self.min_height() {
            me = self.best_guess();
        } else {
            temp_me = self
                .concrete_row_cols(width, height)
                .ok_or(MathError::MatrixShapeMismatch)?;

            me = &temp_me;
        }

        if width == rhs.min_width() && height == rhs.min_height() {
            rh = rhs.best_guess();
        } else {
            temp_rh = rhs
                .concrete_row_cols(width, height)
                .ok_or(MathError::MatrixShapeMismatch)?;

            rh = &temp_rh;
        }

        let matrix_result = op(me, rh)?;

        let result =
            if matches!(self, Matrix::Identity { .. }) || matches!(rhs, Matrix::Identity { .. }) {
                if let (
                    Matrix::Identity {
                        scale: scale_me, ..
                    },
                    Matrix::Identity {
                        scale: scale_rh, ..
                    },
                ) = (self, rhs)
                {
                    let scale = id_op(
                        scale_me.as_deref().unwrap_or(&Value::ONE),
                        scale_rh.as_deref().unwrap_or(&Value::ONE),
                    )?;

                    if scale == Value::ONE {
                        Matrix::Identity {
                            concrete: matrix_result,
                            scale: None,
                        }
                    } else if scale == Value::ZERO {
                        Matrix::UnboundedSize(matrix_result)
                    } else {
                        Matrix::Identity {
                            concrete: matrix_result,
                            scale: Some(Box::new(scale)),
                        }
                    }
                } else {
                    Matrix::Concrete(matrix_result)
                }
            } else if self.is_concrete_width() || rhs.is_concrete_width() {
                if self.is_concrete_height() || rhs.is_concrete_height() {
                    Matrix::Concrete(matrix_result)
                } else {
                    Matrix::UnboundedHeight(matrix_result)
                }
            } else if self.is_concrete_height() || rhs.is_concrete_height() {
                Matrix::UnboundedWidth(matrix_result)
            } else {
                Matrix::UnboundedSize(matrix_result)
            };

        Ok(result)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.componentwise_binop(rhs, |x, y| x.add(y), |x, y| x.add(y))
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.componentwise_binop(rhs, |x, y| x.sub(y), |x, y| x.sub(y))
    }

    pub fn mul_componentwise(&self, rhs: &Self) -> Result<Self> {
        self.componentwise_binop(rhs, |x, y| x.mul_componentwise(y), |x, y| x.mul(y))
    }

    pub fn neg(&self) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix.neg().map(Matrix::Concrete),
            Matrix::UnboundedHeight(concrete_matrix) => {
                concrete_matrix.neg().map(Matrix::UnboundedHeight)
            }
            Matrix::UnboundedWidth(concrete_matrix) => {
                concrete_matrix.neg().map(Matrix::UnboundedWidth)
            }
            Matrix::UnboundedSize(concrete_matrix) => {
                concrete_matrix.neg().map(Matrix::UnboundedSize)
            }
            Matrix::Identity { concrete, scale } => {
                let scale = scale.as_deref().unwrap_or(&Value::ONE).neg()?;
                let scale = if scale == Value::ONE {
                    None
                } else {
                    Some(Box::new(scale))
                };

                Ok(Matrix::Identity {
                    concrete: concrete.neg()?,
                    scale,
                })
            }
        }
    }

    #[doc(alias = "rref")]
    pub fn row_echelon_form(&self, reduced: bool) -> Result<Self> {
        self.best_guess()
            .row_echelon_form(reduced)
            .map(Self::Concrete)
    }

    pub fn identity(dimension: Option<usize>) -> Result<Self> {
        match dimension {
            None => Ok(Self::Identity {
                concrete: ConcreteMatrix::EMPTY,
                scale: None,
            }),
            Some(width) => ConcreteMatrix::identity(width).map(Self::Concrete),
        }
    }

    pub fn vector_basis_elem(row: usize, one: Value) -> Result<Self> {
        let mut elems = vec![Value::ZERO; row];

        *elems.last_mut().ok_or(MathError::EmptyMatrix)? = one;

        Ok(Self::UnboundedHeight(ConcreteMatrix {
            elems,
            width: 1,
            height: row,
        }))
    }

    pub fn transpose_basis_elem(col: usize, one: Value) -> Result<Self> {
        let mut elems = vec![Value::ZERO; col];

        *elems.last_mut().ok_or(MathError::EmptyMatrix)? = one;

        Ok(Self::UnboundedWidth(ConcreteMatrix {
            elems,
            width: col,
            height: 1,
        }))
    }

    pub fn matrix_basis_elem(col: usize, row: usize, one: Option<Value>) -> Result<Self> {
        // the 1 will always be in the last place
        let mut elems = vec![Value::ZERO; col * row];

        *elems.last_mut().ok_or(MathError::EmptyMatrix)? = one.unwrap_or(Value::ONE);

        Ok(Self::UnboundedWidth(ConcreteMatrix {
            elems,
            width: col,
            height: row,
        }))
    }

    pub fn zero_matrix(&self) -> Self {
        Matrix::UnboundedSize(ConcreteMatrix::EMPTY)
    }

    pub fn invert(&self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.invert()?,
                scale: scale
                    .as_ref()
                    .map(|x| x.invert().map(Box::new))
                    .transpose()?,
            }),

            _ => self.best_guess().invert().map(Self::Concrete),
        }
    }

    pub fn is_zero(&self) -> bool {
        if let Matrix::Identity { scale, .. } = self {
            match scale {
                None => return false,
                Some(x) => {
                    if !x.is_zero() {
                        return false;
                    }
                }
            }
        }
        self.best_guess().is_zero()
    }

    pub fn transpose(&self) -> Self {
        match self {
            Matrix::Concrete(concrete_matrix) => Self::Concrete(concrete_matrix.transpose()),
            Matrix::Identity { concrete, scale } => Matrix::Identity {
                concrete: concrete.transpose(),
                scale: scale.clone(),
            },
            Matrix::UnboundedHeight(concrete_matrix) => {
                Matrix::UnboundedWidth(concrete_matrix.transpose())
            }
            Matrix::UnboundedWidth(concrete_matrix) => {
                Matrix::UnboundedHeight(concrete_matrix.transpose())
            }
            Matrix::UnboundedSize(concrete_matrix) => {
                Matrix::UnboundedSize(concrete_matrix.transpose())
            }
        }
    }

    pub fn conj(&self) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix.conj().map(Self::Concrete),
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.conj()?,
                scale: scale.as_ref().map(|x| x.conj().map(Box::new)).transpose()?,
            }),
            Matrix::UnboundedHeight(concrete_matrix) => {
                concrete_matrix.conj().map(Matrix::UnboundedHeight)
            }
            Matrix::UnboundedWidth(concrete_matrix) => {
                concrete_matrix.conj().map(Matrix::UnboundedWidth)
            }
            Matrix::UnboundedSize(concrete_matrix) => {
                concrete_matrix.conj().map(Matrix::UnboundedSize)
            }
        }
    }

    pub fn norm_sq(&self) -> Result<Value> {
        match self {
            Matrix::Identity { concrete, scale } => {
                if concrete.width == 0 {
                    Ok(scale.as_ref().map(|x| (&**x).clone()).unwrap_or(Value::ONE))
                } else {
                    concrete.norm_sq()
                }
            }

            _ => self.best_guess().norm_sq(),
        }
    }

    pub fn sin(&self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.sin()?,
                scale: scale.as_ref().map(|x| x.sin().map(Box::new)).transpose()?,
            }),

            _ => self.best_guess().sin().map(Self::Concrete),
        }
    }

    pub fn cos(&self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.cos()?,
                scale: scale.as_ref().map(|x| x.cos().map(Box::new)).transpose()?,
            }),

            _ => self.best_guess().cos().map(Self::Concrete),
        }
    }

    pub fn tan(&self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.tan()?,
                scale: scale.as_ref().map(|x| x.tan().map(Box::new)).transpose()?,
            }),

            _ => self.best_guess().tan().map(Self::Concrete),
        }
    }

    pub fn sinh(&self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.sinh()?,
                scale: scale.as_ref().map(|x| x.sinh().map(Box::new)).transpose()?,
            }),

            _ => self.best_guess().sinh().map(Self::Concrete),
        }
    }

    pub fn cosh(&self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.cosh()?,
                scale: scale.as_ref().map(|x| x.cosh().map(Box::new)).transpose()?,
            }),

            _ => self.best_guess().cosh().map(Self::Concrete),
        }
    }

    pub fn tanh(&self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.tanh()?,
                scale: scale.as_ref().map(|x| x.tanh().map(Box::new)).transpose()?,
            }),

            _ => self.best_guess().tanh().map(Self::Concrete),
        }
    }

    // Householder matrix for given column vector
    pub fn householder_matrix(&self) -> Result<Self> {
        self.best_guess().householder_matrix().map(Self::Concrete)
    }

    pub fn lower_hessenberg(&self) -> Result<Self> {
        self.best_guess().lower_hessenberg().map(Self::Concrete)
    }

    // todo: finish a proper exp impl instead of this
    // taylor series eval
    pub fn exp(&self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.exp()?,
                scale: scale.as_ref().map(|x| x.exp().map(Box::new)).transpose()?,
            }),

            _ => self.best_guess().exp().map(Self::Concrete),
        }
    }
}
