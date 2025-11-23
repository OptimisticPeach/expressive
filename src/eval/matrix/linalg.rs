// todo: handle empty cases in all functions

use std::ops::{Index, IndexMut};

use crate::errors::Result;
use crate::eval::scalar::Scalar;
use crate::eval::value::EvalAst;

/// Matrix stored in column-major format:
/// [ 0 3 6 ]
/// [ 1 4 7 ]
/// [ 2 5 8 ]
#[derive(Clone, Debug, PartialEq)]
pub struct ConcreteMatrix {
    pub(crate) elems: Vec<EvalAst>,
    pub width: usize,
    pub height: usize,
}

impl Index<(usize, usize)> for ConcreteMatrix {
    type Output = EvalAst;

    fn index(&self, (col, row): (usize, usize)) -> &EvalAst {
        &self.elems[row + col * self.height]
    }
}

impl IndexMut<(usize, usize)> for ConcreteMatrix {
    fn index_mut(&mut self, (col, row): (usize, usize)) -> &mut EvalAst {
        &mut self.elems[row + col * self.height]
    }
}

impl ConcreteMatrix {
    pub const EMPTY: Self = ConcreteMatrix {
        elems: Vec::new(),
        width: 0,
        height: 0,
    };

    pub fn get_idx(&self, col: usize, row: usize) -> usize {
        row + col * self.height
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &EvalAst)> {
        self.elems
            .iter()
            .enumerate()
            .map(|(idx, val)| (idx / self.height, idx % self.height, val))
    }

    pub fn into_iter(self) -> impl Iterator<Item = (usize, usize, EvalAst)> {
        self.elems
            .into_iter()
            .enumerate()
            .map(move |(idx, val)| (idx / self.height, idx % self.height, val))
    }

    pub fn iter_col(&self, col: usize) -> Result<impl Iterator<Item = &EvalAst>> {
        if col >= self.width {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        Ok((0..self.height).map(move |y| &self[(col, y)]))
    }

    pub fn iter_row(&self, row: usize) -> Result<impl Iterator<Item = &EvalAst>> {
        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        Ok((0..self.width).map(move |x| &self[(x, row)]))
    }

    pub fn remove_col(&mut self, col: usize) -> Result<Self> {
        if col >= self.width {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        let mut this = Self {
            elems: Vec::new(),
            width: self.width - 1,
            height: self.height,
        };

        std::mem::swap(self, &mut this);

        let mut result = Self {
            elems: Vec::with_capacity(self.width),
            width: self.width,
            height: 1,
        };

        for (x, _, elem) in this.into_iter() {
            if x == col {
                result.elems.push(elem);
            } else {
                self.elems.push(elem);
            }
        }

        Ok(result)
    }

    pub fn remove_row(&mut self, row: usize) -> Result<Self> {
        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        let mut this = Self {
            elems: Vec::new(),
            width: self.width,
            height: self.height - 1,
        };

        std::mem::swap(self, &mut this);

        let mut result = Self {
            elems: Vec::with_capacity(self.width),
            width: self.width,
            height: 1,
        };

        for (_, y, elem) in this.into_iter() {
            if y == row {
                result.elems.push(elem);
            } else {
                self.elems.push(elem);
            }
        }

        Ok(result)
    }

    pub fn remove_col_row(&mut self, col: usize, row: usize) -> Result<()> {
        if col >= self.width {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        let mut this = Self {
            elems: Vec::new(),
            width: self.width - 1,
            height: self.height - 1,
        };

        std::mem::swap(self, &mut this);

        self.elems.extend(
            this.into_iter()
                .filter(|(x, y, _)| *x != col && *y != row)
                .map(|(_, _, v)| v),
        );

        Ok(())
    }

    pub fn det(mut self) -> Result<EvalAst> {
        if self.width != self.height {
            return Err(crate::errors::MathError::NonSquareDeterminant);
        }

        match &mut *self.elems {
            [] => Err(crate::errors::MathError::EmptyMatrix),
            [x] => Ok(std::mem::take(x)),
            [m11, m12, m21, m22] => std::mem::take(m11)
                .mul(std::mem::take(m22))?
                .sub(std::mem::take(m12).mul(std::mem::take(m21))?),
            _ => self.gaussian_det(),
        }
    }

    fn gaussian_det(self) -> Result<EvalAst> {
        let (mut val, _pivots, det_scl) = self.row_echelon_form_internal(false, true)?;

        let mut det = det_scl;

        for i in 0..val.width {
            let value = std::mem::replace(&mut val[(i, i)], EvalAst::ZERO);

            det = det.mul(value)?;
        }

        Ok(det)
    }

    pub fn get_col(&self, col: usize) -> Result<Self> {
        if col >= self.width {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        let mut this = Self {
            elems: Vec::with_capacity(self.height),
            width: 1,
            height: self.height,
        };

        this.elems
            .extend((0..self.height).map(|i| self[(col, i)].clone()));

        Ok(this)
    }

    pub fn get_row(&self, row: usize) -> Result<Self> {
        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        let mut this = Self {
            elems: Vec::with_capacity(self.width),
            width: self.width,
            height: 1,
        };

        this.elems
            .extend((0..self.width).map(|i| self[(i, row)].clone()));

        Ok(this)
    }

    pub fn aug_vert(self, bottom: Self) -> Result<Self> {
        if self.width != bottom.width {
            return Err(crate::errors::MathError::AugmentShapeMismatch);
        }

        let mut this = Self {
            elems: Vec::with_capacity((self.height + bottom.height) * self.width),
            width: self.width,
            height: self.height + bottom.height,
        };

        let mut me = self.elems.into_iter();
        let mut other = bottom.elems.into_iter();

        for _ in 0..self.width {
            this.elems.extend((&mut me).take(self.height));
            this.elems.extend((&mut other).take(bottom.height));
        }

        Ok(this)
    }

    pub fn ext_vert(self, value: EvalAst, extra_rows: usize) -> Self {
        let mut this = Self {
            elems: Vec::with_capacity((self.height + extra_rows) * self.width),
            width: self.width,
            height: self.height + extra_rows,
        };

        let mut me = self.elems.into_iter();

        for _ in 0..self.width {
            this.elems.extend((&mut me).take(self.height));
            this.elems
                .extend(std::iter::repeat_n(value.clone(), extra_rows));
        }

        this
    }

    pub fn aug_hor(self, right: Self) -> Result<Self> {
        if self.height != right.height {
            return Err(crate::errors::MathError::AugmentShapeMismatch);
        }

        let mut this = Self {
            elems: self.elems,
            width: self.width + right.width,
            height: self.height,
        };

        this.elems.extend(right.elems.into_iter());

        Ok(this)
    }

    pub fn ext_hor(self, value: EvalAst, extra_cols: usize) -> Self {
        let mut this = Self {
            elems: self.elems,
            width: self.width + extra_cols,
            height: self.height,
        };

        this.elems
            .extend(std::iter::repeat_n(value, extra_cols * self.height));

        this
    }

    // Fills in off-diagonals with zeros
    pub fn aug_diag(self, bottom_right: Self) -> Self {
        let width = self.width + bottom_right.width;
        let height = self.height + bottom_right.height;

        let mut this = Self {
            elems: Vec::with_capacity(width * height),
            width,
            height,
        };

        let mut me = self.elems.into_iter();

        for _ in 0..self.width {
            this.elems.extend((&mut me).take(self.height));

            this.elems
                .extend(std::iter::repeat_n(EvalAst::ZERO, bottom_right.height));
        }

        let mut other = bottom_right.elems.into_iter();

        for _ in 0..bottom_right.width {
            this.elems
                .extend(std::iter::repeat_n(EvalAst::ZERO, self.height));

            this.elems.extend((&mut other).take(bottom_right.height));
        }

        this
    }

    pub fn ext_diag(self, value: EvalAst, extra_cols: usize, extra_rows: usize) -> Self {
        let width = self.width + extra_cols;
        let height = self.height + extra_rows;

        let mut this = Self {
            elems: Vec::with_capacity(width * height),
            width,
            height,
        };

        let mut me = self.elems.into_iter();

        for _ in 0..self.width {
            this.elems.extend((&mut me).take(self.height));

            this.elems
                .extend(std::iter::repeat_n(value.clone(), extra_rows));
        }

        this.elems
            .extend(std::iter::repeat_n(value, extra_cols * height));

        this
    }

    pub fn mul(self, rhs: Self) -> Result<Self> {
        if self.width != rhs.height {
            return Err(crate::errors::MathError::MatrixShapeMismatch);
        }

        let mut this = Self {
            elems: Vec::with_capacity(self.height * rhs.width),
            height: self.height,
            width: rhs.width,
        };

        for col in 0..this.width {
            for row in 0..this.height {
                let result = self
                    .iter_row(row)?
                    .zip(rhs.iter_col(col)?)
                    .map(|(a, b)| a.clone().mul(b.clone()))
                    .reduce(|acc, rhs| acc.and_then(|x| x.add(rhs?)))
                    .unwrap();

                this.elems.push(result?)
            }
        }

        Ok(this)
    }

    pub fn hadamard_op(
        self,
        rhs: Self,
        mut op: impl FnMut(EvalAst, EvalAst) -> Result<EvalAst>,
    ) -> Result<Self> {
        if self.width != rhs.width || self.height != rhs.height {
            return Err(crate::errors::MathError::MatrixShapeMismatch);
        }

        Ok(Self {
            elems: self
                .elems
                .into_iter()
                .zip(rhs.elems.into_iter())
                .map(move |(x, y)| op(x, y))
                .collect::<Result<Vec<_>>>()?,
            ..self
        })
    }

    pub fn scalar_mul(self, rhs: EvalAst) -> Result<Self> {
        Ok(Self {
            elems: self
                .elems
                .into_iter()
                .map(|x| x.mul(rhs.clone()))
                .collect::<Result<Vec<_>>>()?,
            ..self
        })
    }

    pub fn select_cols(&self, cols: impl Iterator<Item = usize>) -> Result<Self> {
        let mut this = Self {
            elems: Vec::with_capacity(self.height),
            width: 0,
            height: self.height,
        };

        for col in cols {
            if col >= self.width {
                return Err(crate::errors::MathError::MatrixOutOfRange);
            }

            this.width += 1;
            this.elems
                .extend((0..self.height).map(|i| self[(col, i)].clone()));
        }

        if this.width == 0 {
            return Err(crate::errors::MathError::EmptyMatrix);
        }

        Ok(this)
    }

    pub fn select_rows(&self, rows: impl Iterator<Item = usize>) -> Result<Self> {
        let mut this = Self {
            elems: Vec::with_capacity(self.width),
            width: self.width,
            height: 0,
        };

        for row in rows {
            if row >= self.height {
                return Err(crate::errors::MathError::MatrixOutOfRange);
            }

            this.height += 1;
            let mut acc = 0;

            for i in 0..this.width {
                acc += this.height;

                this.elems.insert(acc, this[(i, row)].clone());

                acc += 1;
            }
        }

        if this.height == 0 {
            return Err(crate::errors::MathError::EmptyMatrix);
        }

        Ok(this)
    }

    pub fn from_size_slice_rowmaj(width: usize, height: usize, values: &[EvalAst]) -> Result<Self> {
        if values.len() != width * height {
            Err(crate::errors::MathError::WrongNumberOfArgs)?;
        }

        let mut this = Self {
            elems: Vec::with_capacity(width * height),
            width,
            height,
        };

        for col in 0..width {
            for row in 0..height {
                this.elems.push(values[row * width + col].clone());
            }
        }

        Ok(this)
    }

    pub fn from_size_slice_colmaj(width: usize, height: usize, values: &[EvalAst]) -> Result<Self> {
        if values.len() != width * height {
            Err(crate::errors::MathError::WrongNumberOfArgs)?;
        }

        let this = Self {
            elems: values.into(),
            width,
            height,
        };

        Ok(this)
    }

    pub fn add(self, rhs: Self) -> Result<Self> {
        self.hadamard_op(rhs, EvalAst::add)
    }

    pub fn sub(self, rhs: Self) -> Result<Self> {
        self.hadamard_op(rhs, EvalAst::sub)
    }

    pub fn mul_componentwise(self, rhs: ConcreteMatrix) -> Result<Self> {
        self.hadamard_op(rhs, EvalAst::mul)
    }

    pub fn neg(self) -> Result<Self> {
        Ok(Self {
            elems: self
                .elems
                .into_iter()
                .map(|x| x.neg())
                .collect::<Result<Vec<_>>>()?,
            ..self
        })
    }

    fn swap_rows(&mut self, r1: usize, r2: usize) -> Result<()> {
        if r1 >= self.height || r2 >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        if r1 == r2 {
            return Ok(());
        }

        for i in 0..self.width {
            let idx_1 = self.get_idx(i, r1);
            let idx_2 = self.get_idx(i, r2);

            self.elems.swap(idx_1, idx_2);
        }

        Ok(())
    }

    fn scale_row(&mut self, row: usize, value: &EvalAst) -> Result<()> {
        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        for i in 0..self.width {
            let mut elem = EvalAst::ZERO;

            std::mem::swap(&mut self[(i, row)], &mut elem);

            self[(i, row)] = value.clone().mul(elem)?;
        }

        Ok(())
    }

    // Rdest -= scale * src
    fn sub_row(&mut self, src: usize, dest: usize, scale: &EvalAst) -> Result<()> {
        if src >= self.height || dest >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        for i in 0..self.width {
            let mut elem = EvalAst::ZERO;

            std::mem::swap(&mut self[(i, dest)], &mut elem);

            self[(i, dest)] = elem.sub(scale.clone().mul(self[(i, src)].clone())?)?;
        }

        Ok(())
    }

    fn row_echelon_form_internal(
        &self,
        reduced: bool,
        track_determinant: bool,
    ) -> Result<(Self, usize, EvalAst)> {
        let mut det = EvalAst::ONE;

        let mut this = self.clone();

        let mut row = 0;
        let mut end_col = this.width - 1;

        for col in 0..this.width {
            if row >= this.height {
                end_col = col;
                break;
            }

            let mut pivot_row = None;

            for search in row..this.height {
                // todo: should be is_invertible
                if !this[(col, search)].is_zero() {
                    pivot_row = Some(search);
                    break;
                }
            }

            let Some(pivot_row) = pivot_row else { continue };

            if pivot_row != row {
                this.swap_rows(row, pivot_row)?;

                if track_determinant {
                    det = det.neg()?;
                }
            }

            let pivot_scale = this[(col, row)].clone().invert()?;

            for clear in (row + 1)..this.height {
                this.sub_row(
                    row,
                    clear,
                    &this[(col, clear)].clone().mul(pivot_scale.clone())?,
                )?;

                // determinant not affected by this.
            }

            if reduced {
                for clear in 0..row {
                    this.sub_row(
                        row,
                        clear,
                        &this[(col, clear)].clone().mul(pivot_scale.clone())?,
                    )?;
                }

                this.scale_row(row, &pivot_scale)?;

                if track_determinant {
                    det = det.mul(pivot_scale)?;
                }
            }

            row += 1;
        }

        Ok((this, end_col, det))
    }

    #[doc(alias = "rref")]
    pub fn row_echelon_form(&self, reduced: bool) -> Result<Self> {
        self.row_echelon_form_internal(reduced, false).map(|x| x.0)
    }

    pub fn identity(width: usize) -> Result<Self> {
        if width == 0 {
            return Err(crate::errors::MathError::ZeroSizeMatrix);
        }

        let this = Self {
            elems: (0..width)
                .map(move |x| (0..width).map(move |y| (x, y)))
                .flatten()
                .map(|(x, y)| if x == y { Scalar::ONE } else { Scalar::ZERO })
                .map(EvalAst::Scalar)
                .collect(),
            width,
            height: width,
        };

        Ok(this)
    }

    pub fn my_identity(&self) -> Result<Self> {
        if !(self.width == self.height || self.width == 1 || self.height == 1) {
            return Err(crate::errors::MathError::NoIdentityMatrix);
        }

        let this = Self::identity(self.width.max(self.height));

        this
    }

    pub fn col_from_iter(iter: impl IntoIterator<Item = EvalAst>) -> Self {
        let elems = Vec::from_iter(iter);

        let height = elems.len();

        Self {
            elems,
            width: 1,
            height,
        }
    }

    pub fn row_from_iter(iter: impl IntoIterator<Item = EvalAst>) -> Self {
        let elems = Vec::from_iter(iter);

        let width = elems.len();

        Self {
            elems,
            width,
            height: 1,
        }
    }

    pub fn my_basis_elem(&self, elem: usize) -> Result<Self> {
        // defaults to column vector
        if self.height == self.width || self.width == 1 {
            if elem >= self.height {
                return Err(crate::errors::MathError::MatrixOutOfRange);
            }

            let mut result = Self::col_from_iter(std::iter::repeat_n(
                EvalAst::Scalar(Scalar::ZERO),
                self.height,
            ));

            result.elems[elem] = EvalAst::Scalar(Scalar::ONE);

            Ok(result)
        } else if self.height == 1 {
            if elem >= self.width {
                return Err(crate::errors::MathError::MatrixOutOfRange);
            }

            let mut result = Self::row_from_iter(std::iter::repeat_n(
                EvalAst::Scalar(Scalar::ZERO),
                self.height,
            ));

            result.elems[elem] = EvalAst::Scalar(Scalar::ONE);

            Ok(result)
        } else {
            Err(crate::errors::MathError::NoIdentityMatrix)
        }
    }

    pub fn zero_matrix(&self) -> Self {
        Self {
            elems: std::iter::repeat_n(Scalar::ZERO, self.elems.len())
                .map(EvalAst::Scalar)
                .collect(),
            ..*self
        }
    }

    pub fn invert(self) -> Result<Self> {
        if self.width != self.height {
            return Err(crate::errors::MathError::MatrixNotSquare);
        }

        let width = self.width;
        let id = self.my_identity()?;

        let aug = self.aug_hor(id)?;

        let (rref, last_col, _) = aug.row_echelon_form_internal(true, false)?;

        if last_col >= width {
            return Err(crate::errors::MathError::MatrixNotInvertible);
        }

        rref.select_cols(width..width * 2)
    }

    pub fn is_zero(&self) -> bool {
        self.elems.iter().all(|x| x.is_zero())
    }

    pub fn transpose(&self) -> Self {
        let mut this = Self {
            elems: Vec::with_capacity(self.elems.len()),
            width: self.height,
            height: self.width,
        };

        for row in 0..self.height {
            this.elems.extend(self.iter_row(row).unwrap().cloned());
        }

        this
    }

    pub fn conj(self) -> Result<Self> {
        Ok(Self {
            elems: self
                .elems
                .into_iter()
                .map(|x| x.conj())
                .collect::<Result<Vec<_>>>()?,
            ..self
        })
    }

    pub fn norm_sq(self) -> Result<EvalAst> {
        self.elems
            .into_iter()
            .map(|x| x.norm_sq())
            .reduce(|x, y| x.and_then(|x| y.and_then(|y| x.add(y))))
            .unwrap() // Option unwrap
    }

    pub fn sin(self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn cos(self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn tan(self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn sinh(self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn cosh(self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn tanh(self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    // Matrix exponentiation through Sylvester's Formula
    fn exp_inner(self, eigenvalues: &[(Scalar, usize)]) -> Result<Self> {
        let height = self.height;
        let width = self.width;
        let zero = self.zero_matrix();

        let mut coeff_matrix = Self {
            elems: Vec::with_capacity(self.elems.len()),
            ..self
        };

        let mut lambdas = Vec::with_capacity(self.width);
        let mut exponents = Vec::with_capacity(self.width);

        for (lambda, mult) in eigenvalues {
            for i in 0..*mult {
                lambdas.push(*lambda);
                exponents.push(i);
            }
        }

        fn make_coeff(n: usize, k: usize, lambda: Scalar) -> Scalar {
            let n = n as u128;
            let k = k as u128;
            // result is nCr(n, k) * k! * lambda^(n-k)
            if n < k {
                return Scalar::ZERO;
            }

            let power = Scalar::from_integer((n - k) as i128);

            let power = lambda.pow(power);

            let coeff = binomial(n, k) * factorial(k);

            if coeff > i128::MAX as u128 {
                Scalar::from_float(coeff as f64).mul(power)
            } else {
                Scalar::from_integer(coeff as i128).mul(power)
            }
        }

        for col in 0..self.width {
            let lambda = lambdas[col];
            let power = exponents[col];

            // row is derivative
            for row in 0..self.height {
                coeff_matrix
                    .elems
                    .push(EvalAst::Scalar(make_coeff(row, power, lambda)));
            }
        }

        let inverted = coeff_matrix.invert()?;

        let mut powers = Vec::with_capacity(self.width);

        powers.push(self.my_identity()?);

        if self.width > 1 {
            powers.push(self.clone());
        }

        let mut acc = self.clone();
        for _ in 2..width {
            acc = acc.mul(self.clone())?;
            powers.push(acc.clone());
        }

        // We solved M B = A for B where both B and A are vectors of matrices.
        // `powers` is `A`, and it contains `I, A, A^2, A^3, ...`.
        // We now compute:
        let mut result = zero.clone();

        for row in 0..height {
            let lambda_coeff = EvalAst::Scalar(lambdas[row].exp());

            let mut acc = zero.clone();

            for (power, coeff) in powers.iter().zip(inverted.iter_row(row)?) {
                acc = acc.add(power.clone().scalar_mul(coeff.clone())?)?;
            }

            result = result.add(acc.scalar_mul(lambda_coeff)?)?;
        }

        Ok(result)
    }

    // Householder matrix for given column vector
    pub fn householder_matrix(self) -> Result<Self> {
        if self.width != 1 {
            return Err(crate::errors::MathError::NotColumnVector);
        }

        let star = self.clone().conj()?.transpose();

        let numerator = self.clone().mul(star)?;

        let denominator = self.norm_sq()?;

        numerator
            .my_identity()?
            .sub(numerator.scalar_mul(EvalAst::Scalar(Scalar::from_integer(2)).div(denominator)?)?)
    }

    pub fn lower_hessenberg(mut self) -> Result<Self> {
        if self.width != self.height {
            return Err(crate::errors::MathError::MatrixNotSquare);
        }

        if self.width < 3 {
            return Ok(self.clone());
        }

        let mut prod_acc = self.my_identity()?;

        let mut rectangle = self.remove_row(0)?;
        for i in 0..self.width - 2 {
            let col = rectangle.get_col(0)?;

            let vector = col.my_basis_elem(0)?.scalar_mul(col.clone().norm_sq()?)?;
            let vector = vector.sub(col)?;

            let householder = vector.householder_matrix()?;

            let matrix = Self::identity(i + 1)?.aug_diag(householder);

            prod_acc = matrix.mul(prod_acc)?;

            rectangle.remove_col_row(0, 0)?;
        }

        prod_acc.mul(self)
    }

    // todo: finish a proper exp impl instead of this
    // taylor series eval
    pub fn exp(&self) -> Result<Self> {
        let mut acc = self.my_identity()?;

        let mut prod_acc = self.clone();
        let mut factorial_acc = 1;
        for i in 2..12 {
            acc = acc.add(
                prod_acc
                    .clone()
                    .scalar_mul(EvalAst::Scalar(Scalar::from_num_denom(1, factorial_acc)))?,
            )?;

            prod_acc = prod_acc.mul(self.clone())?;
            factorial_acc *= i;
        }

        Ok(acc)
    }
}

fn factorial(n: u128) -> u128 {
    let mut result = 1;

    for i in 2..n {
        result *= i;
    }

    result
}

fn binomial(n: u128, k: u128) -> u128 {
    if n < k {
        return 0;
    }

    if k == 0 || k == n {
        return 1;
    }

    let min = k.min(n - k);
    let max = k.max(n - k);

    let mut result = 1;

    for i in (max - 1..n + 1).rev() {
        result *= i;
    }

    result /= factorial(min);

    result
}
