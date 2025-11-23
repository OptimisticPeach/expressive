pub mod linalg;

use crate::{
    errors::{MathError, Result},
    eval::EvalAst,
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
        scale: Option<Box<EvalAst>>,
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

impl Matrix {
    pub const IDENTITY: Self = Matrix::Identity {
        concrete: ConcreteMatrix::EMPTY,
        scale: None,
    };
    pub const ZERO: Self = Matrix::UnboundedSize(ConcreteMatrix::EMPTY);

    // These three probably don't make sense?
    // pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &EvalAst)> {}

    // pub fn iter_col(&self, col: usize) -> Result<impl Iterator<Item = &EvalAst>> {}

    // pub fn iter_row(&self, row: usize) -> Result<impl Iterator<Item = &EvalAst>> {}

    pub fn remove_col(&mut self, col: usize) -> Result<Self> {
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

    pub fn remove_row(&mut self, row: usize) -> Result<Self> {
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

    pub fn remove_col_row(&mut self, col: usize, row: usize) -> Result<()> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix.remove_col_row(col, row),

            Matrix::UnboundedHeight(concrete) => {
                if row >= concrete.height {
                    concrete.remove_col(col).map(|_| {})
                } else {
                    concrete.remove_col_row(col, row)
                }
            }

            Matrix::UnboundedWidth(concrete) => {
                if col >= concrete.width {
                    concrete.remove_row(row).map(|_| {})
                } else {
                    concrete.remove_col_row(col, row)
                }
            }

            Matrix::UnboundedSize(concrete) => {
                if col >= concrete.width {
                    if row >= concrete.height {
                        Ok(())
                    } else {
                        concrete.remove_row(row).map(|_| {})
                    }
                } else {
                    if row >= concrete.height {
                        concrete.remove_col(col).map(|_| {})
                    } else {
                        concrete.remove_col_row(col, row)
                    }
                }
            }

            Matrix::Identity { .. } => Err(crate::errors::MathError::UnsupportedMatrixOp),
        }
    }

    fn concrete_rows(self, height: usize) -> Option<Self> {
        let result = match self.extend_rows(height)? {
            x @ Matrix::UnboundedWidth(_) => x,
            Matrix::UnboundedSize(concrete_matrix) => Matrix::UnboundedWidth(concrete_matrix),

            x @ (Matrix::Concrete(_) | Matrix::Identity { .. } | Matrix::UnboundedHeight(_)) => {
                Self::Concrete(x.unwrap_best_guess())
            }
        };

        Some(result)
    }

    fn concrete_cols(self, width: usize) -> Option<Self> {
        let result = match self.extend_cols(width)? {
            x @ Matrix::UnboundedHeight(_) => x,
            Matrix::UnboundedSize(concrete_matrix) => Matrix::UnboundedHeight(concrete_matrix),

            x @ (Matrix::Concrete(_) | Matrix::Identity { .. } | Matrix::UnboundedWidth(_)) => {
                Self::Concrete(x.unwrap_best_guess())
            }
        };

        Some(result)
    }

    fn concrete_row_cols(self, width: usize, height: usize) -> Option<ConcreteMatrix> {
        self.extend_cols_rows(width, height)
            .map(Matrix::unwrap_best_guess)
    }

    fn extend_rows(self, height: usize) -> Option<Self> {
        let cur_height = self.min_height();

        if cur_height == height {
            return Some(self);
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

                let mut result = concrete.ext_diag(EvalAst::ZERO, delta, delta);

                let one = scale.clone().map(|x| *x).unwrap_or(EvalAst::ONE);

                for i in height - delta..height {
                    result[(i, i)] = one.clone();
                }

                Some(Matrix::Identity {
                    concrete: result,
                    scale: scale.clone(),
                })
            }
            Matrix::UnboundedHeight(concrete_matrix) => Some(Matrix::UnboundedHeight(
                concrete_matrix.ext_vert(EvalAst::ZERO, height - cur_height),
            )),

            Matrix::UnboundedSize(concrete_matrix) => Some(Matrix::UnboundedSize(
                concrete_matrix.ext_vert(EvalAst::ZERO, height - cur_height),
            )),
        }
    }

    fn extend_cols(self, width: usize) -> Option<Self> {
        let cur_width = self.min_width();

        if cur_width == width {
            return Some(self);
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

                let mut result = concrete.ext_diag(EvalAst::ZERO, delta, delta);

                let one = scale.clone().map(|x| *x).unwrap_or(EvalAst::ONE);

                for i in width - delta..width {
                    result[(i, i)] = one.clone();
                }

                Some(Matrix::Identity {
                    concrete: result,
                    scale: scale.clone(),
                })
            }
            Matrix::UnboundedWidth(concrete_matrix) => Some(Matrix::UnboundedWidth(
                concrete_matrix.ext_hor(EvalAst::ZERO, width - cur_width),
            )),

            Matrix::UnboundedSize(concrete_matrix) => Some(Matrix::UnboundedSize(
                concrete_matrix.ext_hor(EvalAst::ZERO, width - cur_width),
            )),
        }
    }

    fn extend_cols_rows(self, width: usize, height: usize) -> Option<Self> {
        let cur_width = self.min_width();
        let cur_height = self.min_height();

        if cur_width > width || cur_height > height {
            return None;
        }

        if cur_width == width && cur_height == height {
            return Some(self);
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

                let mut result = concrete.ext_diag(EvalAst::ZERO, delta_x, delta_x);

                let one = scale.clone().map(|x| *x).unwrap_or(EvalAst::ONE);

                for i in width - delta_x..width {
                    result[(i, i)] = one.clone();
                }

                Some(Matrix::Identity {
                    concrete: result,
                    scale: scale.clone(),
                })
            }
            Matrix::UnboundedSize(concrete_matrix) => Some(Matrix::UnboundedSize(
                concrete_matrix.ext_diag(EvalAst::ZERO, delta_x, delta_y),
            )),
        }
    }

    pub fn det(self) -> Result<EvalAst> {
        let width = self.min_width();
        let height = self.min_height();

        if width == 0 || height == 0 {
            assert!(width == 0 && height == 0);

            let Matrix::Identity { scale, .. } = self else {
                unreachable!();
            };

            return Ok(scale.clone().map(|x| *x).unwrap_or(EvalAst::ONE));
        }

        let dim = width.max(height);

        self.concrete_row_cols(dim, dim)
            .ok_or(MathError::NonSquareDeterminant)?
            .det()
    }

    pub fn get_col(&self, col: usize) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix.get_col(col).map(Self::Concrete),
            Matrix::Identity { concrete, scale } => {
                if col >= concrete.width {
                    Self::vector_basis_elem(col, scale.clone().map(|x| *x).unwrap_or(EvalAst::ZERO))
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
                        std::iter::repeat_n(EvalAst::ZERO, concrete_matrix.height),
                    )))
                } else {
                    concrete_matrix.get_col(col).map(Self::Concrete)
                }
            }
            Matrix::UnboundedSize(concrete_matrix) => {
                if col > concrete_matrix.width {
                    Ok(Self::UnboundedHeight(ConcreteMatrix::col_from_iter(
                        std::iter::repeat_n(EvalAst::ZERO, concrete_matrix.height),
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
                        scale.clone().map(|x| *x).unwrap_or(EvalAst::ZERO),
                    )
                } else {
                    concrete.get_row(row).map(Self::UnboundedWidth)
                }
            }
            Matrix::UnboundedHeight(concrete_matrix) => {
                if row > concrete_matrix.height {
                    Ok(Self::Concrete(ConcreteMatrix::col_from_iter(
                        std::iter::repeat_n(EvalAst::ZERO, concrete_matrix.width),
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
                        std::iter::repeat_n(EvalAst::ZERO, concrete_matrix.width),
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

    fn best_guess_mut(&mut self) -> &mut ConcreteMatrix {
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

    fn take_best_guess(&mut self) -> ConcreteMatrix {
        std::mem::replace(self.best_guess_mut(), ConcreteMatrix::EMPTY)
    }

    fn min_width(&self) -> usize {
        self.best_guess().width
    }

    fn min_height(&self) -> usize {
        self.best_guess().height
    }

    // augment functions will just take the best guess we have in
    // the unbounded cases and identity case.
    pub fn aug_vert(self, bottom: Self) -> Result<Self> {
        if self.is_concrete_width() {
            let reshaped = bottom
                .concrete_cols(self.min_width())
                .ok_or(MathError::AugmentShapeMismatch)?;

            let concrete_height = reshaped.is_concrete_height();

            let result = self
                .unwrap_best_guess()
                .aug_vert(reshaped.unwrap_best_guess())?;

            if concrete_height {
                return Ok(Matrix::Concrete(result));
            } else {
                return Ok(Matrix::UnboundedHeight(result));
            }
        }

        if bottom.is_concrete_width() {
            let reshaped = self
                .concrete_cols(bottom.min_width())
                .ok_or(MathError::AugmentShapeMismatch)?;

            let concrete_height =
                bottom.is_concrete_height() || matches!(bottom, Matrix::Identity { .. });

            let result = reshaped
                .unwrap_best_guess()
                .aug_vert(bottom.unwrap_best_guess())?;

            if concrete_height {
                return Ok(Matrix::Concrete(result));
            } else {
                return Ok(Matrix::UnboundedHeight(result));
            }
        }

        let result;
        let concrete_height;

        if self.min_width() == bottom.min_width() {
            concrete_height =
                bottom.is_concrete_height() || matches!(bottom, Matrix::Identity { .. });
            result = self
                .unwrap_best_guess()
                .aug_vert(bottom.unwrap_best_guess());
        } else if self.min_width() > bottom.min_width() {
            let bottom = bottom
                .concrete_cols(self.min_width())
                .ok_or(MathError::AugmentShapeMismatch)?;

            concrete_height = bottom.is_concrete_height();

            result = self
                .unwrap_best_guess()
                .aug_vert(bottom.unwrap_best_guess());
        } else {
            let top = self
                .concrete_cols(bottom.min_width())
                .ok_or(MathError::AugmentShapeMismatch)?;

            concrete_height =
                bottom.is_concrete_height() || matches!(bottom, Matrix::Identity { .. });

            result = top.unwrap_best_guess().aug_vert(bottom.unwrap_best_guess());
        }

        if concrete_height {
            result.map(Matrix::UnboundedWidth)
        } else {
            result.map(Matrix::UnboundedSize)
        }
    }

    pub fn aug_hor(self, right: Self) -> Result<Self> {
        if self.is_concrete_height() {
            let reshaped = right
                .concrete_rows(self.min_height())
                .ok_or(MathError::AugmentShapeMismatch)?;

            let concrete_width = reshaped.is_concrete_width();

            let result = self
                .unwrap_best_guess()
                .aug_hor(reshaped.unwrap_best_guess())?;

            if concrete_width {
                return Ok(Matrix::Concrete(result));
            } else {
                return Ok(Matrix::UnboundedHeight(result));
            }
        }

        if right.is_concrete_height() {
            let reshaped = self
                .concrete_rows(right.min_height())
                .ok_or(MathError::AugmentShapeMismatch)?;

            let concrete_width =
                right.is_concrete_width() || matches!(right, Matrix::Identity { .. });

            let result = reshaped
                .unwrap_best_guess()
                .aug_hor(right.unwrap_best_guess())?;

            if concrete_width {
                return Ok(Matrix::Concrete(result));
            } else {
                return Ok(Matrix::UnboundedHeight(result));
            }
        }

        let result;
        let concrete_width;

        if self.min_height() == right.min_height() {
            concrete_width = right.is_concrete_width() || matches!(right, Matrix::Identity { .. });
            result = self.unwrap_best_guess().aug_hor(right.unwrap_best_guess());
        } else if self.min_height() > right.min_height() {
            let right = right
                .concrete_rows(self.min_height())
                .ok_or(MathError::AugmentShapeMismatch)?;

            concrete_width = right.is_concrete_width();
            result = self.unwrap_best_guess().aug_hor(right.unwrap_best_guess());
        } else {
            let top = self
                .concrete_rows(right.min_height())
                .ok_or(MathError::AugmentShapeMismatch)?;

            concrete_width = right.is_concrete_width() || matches!(right, Matrix::Identity { .. });
            result = top.unwrap_best_guess().aug_hor(right.unwrap_best_guess());
        }

        if concrete_width {
            result.map(Matrix::UnboundedHeight)
        } else {
            result.map(Matrix::UnboundedSize)
        }
    }

    pub fn mul(self, rhs: Self) -> Result<Self> {
        // the size upon which both must agree
        let same_size = self.min_width().max(rhs.min_height());

        let lhs = self
            .concrete_cols(same_size)
            .ok_or(MathError::MatrixShapeMismatch)?;
        let rhs = rhs
            .concrete_rows(same_size)
            .ok_or(MathError::MatrixShapeMismatch)?;

        let wrapper = match (&lhs, &rhs) {
            (Matrix::Concrete(_), Matrix::Concrete(_)) => Matrix::Concrete,
            (Matrix::Concrete(_), Matrix::UnboundedWidth(_)) => Matrix::UnboundedWidth,
            (Matrix::UnboundedHeight(_), Matrix::Concrete(_)) => Matrix::UnboundedHeight,
            (Matrix::UnboundedHeight(_), Matrix::UnboundedWidth(_)) => Matrix::UnboundedSize,

            // doing concrete_rows/cols will always turn an Identity into a Concrete
            (Matrix::Identity { .. }, _) | (_, Matrix::Identity { .. }) => unreachable!(),

            // we just picked the number of columns for lhs
            (Matrix::UnboundedWidth(_) | Matrix::UnboundedSize(_), _) => unreachable!(),

            // we just picked the number of rows for rhs
            (_, Matrix::UnboundedHeight(_) | Matrix::UnboundedSize(_)) => unreachable!(),
        };

        let result = lhs.unwrap_best_guess().mul(rhs.unwrap_best_guess())?;

        Ok(wrapper(result))
    }

    pub fn scalar_mul(mut self, rhs: EvalAst) -> Result<Self> {
        let mat = std::mem::replace(self.best_guess_mut(), ConcreteMatrix::EMPTY);

        let result = mat.scalar_mul(rhs.clone())?;

        let result = match self {
            Matrix::Concrete(_) => Matrix::Concrete(result),
            Matrix::Identity { scale, .. } => Matrix::Identity {
                concrete: result,
                scale: scale.map(|x| x.mul(rhs)).transpose()?.map(Box::new),
            },
            Matrix::UnboundedHeight(_) => Matrix::UnboundedHeight(result),
            Matrix::UnboundedWidth(_) => Matrix::UnboundedWidth(result),
            Matrix::UnboundedSize(_) => Matrix::UnboundedSize(result),
        };

        Ok(result)
    }

    pub fn from_size_slice_rowmaj(width: usize, height: usize, values: &[EvalAst]) -> Result<Self> {
        ConcreteMatrix::from_size_slice_rowmaj(width, height, values).map(Matrix::Concrete)
    }

    pub fn from_size_slice_colmaj(width: usize, height: usize, values: &[EvalAst]) -> Result<Self> {
        ConcreteMatrix::from_size_slice_colmaj(width, height, values).map(Matrix::Concrete)
    }

    fn componentwise_binop(
        mut self,
        mut rhs: Self,
        op: impl FnOnce(ConcreteMatrix, ConcreteMatrix) -> Result<ConcreteMatrix>,
        id_op: impl FnOnce(EvalAst, EvalAst) -> Result<EvalAst>,
    ) -> Result<Self> {
        // we need to get `self`'s dimensions and `rhs`'s dimensions to agree.

        let width = self.min_width().max(rhs.min_width());
        let height = self.min_height().max(rhs.min_height());

        let me;
        let rh;

        if width == self.min_width() && height == self.min_height() {
            me = self.take_best_guess();
        } else {
            let temp = self.take_best_guess();
            let mut other = self.clone();

            *other.best_guess_mut() = temp;

            me = other
                .concrete_row_cols(width, height)
                .ok_or(MathError::MatrixShapeMismatch)?;
        }

        if width == rhs.min_width() && height == rhs.min_height() {
            rh = rhs.take_best_guess();
        } else {
            let temp = rhs.take_best_guess();
            let mut other = rhs.clone();

            *other.best_guess_mut() = temp;

            rh = other
                .concrete_row_cols(width, height)
                .ok_or(MathError::MatrixShapeMismatch)?;
        }

        let matrix_result = op(me, rh)?;

        let result = if matches!(self, Matrix::Identity { .. })
            || matches!(rhs, Matrix::Identity { .. })
        {
            match (self, rhs) {
                (
                    Matrix::Identity {
                        scale: scale_me, ..
                    },
                    Matrix::Identity {
                        scale: scale_rh, ..
                    },
                ) => {
                    let scale = id_op(
                        scale_me.map(|x| *x).unwrap_or(EvalAst::ONE),
                        scale_rh.map(|x| *x).unwrap_or(EvalAst::ONE),
                    )?;

                    if scale == EvalAst::ONE {
                        Matrix::Identity {
                            concrete: matrix_result,
                            scale: None,
                        }
                    } else if scale == EvalAst::ZERO {
                        Matrix::UnboundedSize(matrix_result)
                    } else {
                        Matrix::Identity {
                            concrete: matrix_result,
                            scale: Some(Box::new(scale)),
                        }
                    }
                }

                (Matrix::Identity { scale, .. }, Matrix::UnboundedSize(_))
                | (Matrix::UnboundedSize(_), Matrix::Identity { scale, .. }) => Matrix::Identity {
                    concrete: matrix_result,
                    scale: scale.clone(),
                },

                _ => Matrix::Concrete(matrix_result),
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

    pub fn add(self, rhs: Self) -> Result<Self> {
        self.componentwise_binop(rhs, |x, y| x.add(y), |x, y| x.add(y))
    }

    pub fn sub(self, rhs: Self) -> Result<Self> {
        self.componentwise_binop(rhs, |x, y| x.sub(y), |x, y| x.sub(y))
    }

    pub fn mul_componentwise(self, rhs: Self) -> Result<Self> {
        self.componentwise_binop(rhs, |x, y| x.mul_componentwise(y), |x, y| x.mul(y))
    }

    pub fn neg(self) -> Result<Self> {
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
                let scale = scale.map(|x| *x).unwrap_or(EvalAst::ONE).neg()?;
                let scale = if scale == EvalAst::ONE {
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

    pub fn vector_basis_elem(row: usize, one: EvalAst) -> Result<Self> {
        let mut elems = vec![EvalAst::ZERO; row];

        *elems.last_mut().ok_or(MathError::EmptyMatrix)? = one;

        Ok(Self::UnboundedHeight(ConcreteMatrix {
            elems,
            width: 1,
            height: row,
        }))
    }

    pub fn transpose_basis_elem(col: usize, one: EvalAst) -> Result<Self> {
        let mut elems = vec![EvalAst::ZERO; col];

        *elems.last_mut().ok_or(MathError::EmptyMatrix)? = one;

        Ok(Self::UnboundedWidth(ConcreteMatrix {
            elems,
            width: col,
            height: 1,
        }))
    }

    pub fn matrix_basis_elem(col: usize, row: usize, one: Option<EvalAst>) -> Result<Self> {
        // the 1 will always be in the last place
        let mut elems = vec![EvalAst::ZERO; col * row];

        *elems.last_mut().ok_or(MathError::EmptyMatrix)? = one.unwrap_or(EvalAst::ONE);

        Ok(Self::UnboundedWidth(ConcreteMatrix {
            elems,
            width: col,
            height: row,
        }))
    }

    pub fn zero_matrix(&self) -> Self {
        Matrix::UnboundedSize(ConcreteMatrix::EMPTY)
    }

    pub fn invert(self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.invert()?,
                scale: scale.map(|x| x.invert().map(Box::new)).transpose()?,
            }),

            _ => {
                let size = self.min_width().max(self.min_height());

                self.concrete_row_cols(size, size)
                    .ok_or(MathError::MatrixNotSquare)
                    .and_then(|x| x.invert())
                    .map(Self::Concrete)
            }
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

    pub fn conj(self) -> Result<Self> {
        match self {
            Matrix::Concrete(concrete_matrix) => concrete_matrix.conj().map(Self::Concrete),
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.conj()?,
                scale: scale.map(|x| x.conj().map(Box::new)).transpose()?,
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

    pub fn norm_sq(self) -> Result<EvalAst> {
        match self {
            Matrix::Identity { concrete, scale } => {
                if concrete.width == 0 {
                    Ok(scale
                        .map(|x| x.norm_sq())
                        .transpose()?
                        .unwrap_or(EvalAst::ONE))
                } else {
                    concrete.norm_sq()
                }
            }

            _ => self.unwrap_best_guess().norm_sq(),
        }
    }

    pub fn sin(self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.sin()?,
                scale: scale.map(|x| x.sin().map(Box::new)).transpose()?,
            }),

            _ => self.unwrap_best_guess().sin().map(Self::Concrete),
        }
    }

    pub fn cos(self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.cos()?,
                scale: scale.map(|x| x.cos().map(Box::new)).transpose()?,
            }),

            _ => self.unwrap_best_guess().cos().map(Self::Concrete),
        }
    }

    pub fn tan(self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.tan()?,
                scale: scale.map(|x| x.tan().map(Box::new)).transpose()?,
            }),

            _ => self.unwrap_best_guess().tan().map(Self::Concrete),
        }
    }

    pub fn sinh(self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.sinh()?,
                scale: scale.map(|x| x.sinh().map(Box::new)).transpose()?,
            }),

            _ => self.unwrap_best_guess().sinh().map(Self::Concrete),
        }
    }

    pub fn cosh(self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.cosh()?,
                scale: scale.map(|x| x.cosh().map(Box::new)).transpose()?,
            }),

            _ => self.unwrap_best_guess().cosh().map(Self::Concrete),
        }
    }

    pub fn tanh(self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.tanh()?,
                scale: scale.map(|x| x.tanh().map(Box::new)).transpose()?,
            }),

            _ => self.unwrap_best_guess().tanh().map(Self::Concrete),
        }
    }

    // Householder matrix for given column vector
    pub fn householder_matrix(self) -> Result<Self> {
        self.unwrap_best_guess()
            .householder_matrix()
            .map(Self::Concrete)
    }

    pub fn lower_hessenberg(self) -> Result<Self> {
        self.unwrap_best_guess()
            .lower_hessenberg()
            .map(Self::Concrete)
    }

    // todo: finish a proper exp impl instead of this
    // taylor series eval
    pub fn exp(self) -> Result<Self> {
        match self {
            Matrix::Identity { concrete, scale } => Ok(Matrix::Identity {
                concrete: concrete.exp()?,
                scale: scale.map(|x| x.exp().map(Box::new)).transpose()?,
            }),

            _ => self.best_guess().exp().map(Self::Concrete),
        }
    }
}
