// todo: handle empty cases in all functions
#![allow(dead_code)]

use std::ops::{Index, IndexMut};

use crate::Floatify;

use crate::errors::Result;
use crate::scalar::Scalar;

use super::Value;

/// Matrix stored in column-major format:
/// [ 0 3 6 ]
/// [ 1 4 7 ]
/// [ 2 5 8 ]
#[derive(Clone, Debug, PartialEq)]
pub struct ConcreteMatrix {
    pub(crate) elems: Vec<Value>,
    pub width: usize,
    pub height: usize,
}

impl Floatify for ConcreteMatrix {
    type Floated = Self;

    fn floatify(mut self) -> Self::Floated {
        self.elems
            .iter_mut()
            .for_each(|x| *x = x.clone().floatify());

        self
    }
}

impl Index<(usize, usize)> for ConcreteMatrix {
    type Output = Value;

    fn index(&self, (col, row): (usize, usize)) -> &Value {
        &self.elems[row + col * self.height]
    }
}

impl IndexMut<(usize, usize)> for ConcreteMatrix {
    fn index_mut(&mut self, (col, row): (usize, usize)) -> &mut Value {
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

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &Value)> {
        self.elems
            .iter()
            .enumerate()
            .map(|(idx, val)| (idx / self.height, idx % self.height, val))
    }

    pub fn iter_col(&self, col: usize) -> Result<impl Iterator<Item = &Value>> {
        if col >= self.width {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        Ok((0..self.height).map(move |y| &self[(col, y)]))
    }

    pub fn iter_row(&self, row: usize) -> Result<impl Iterator<Item = &Value>> {
        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        Ok((0..self.width).map(move |x| &self[(x, row)]))
    }

    pub fn remove_col(&self, col: usize) -> Result<Self> {
        if col >= self.width {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        let mut this = Self {
            elems: Vec::with_capacity((self.width - 1) * self.height),
            width: self.width - 1,
            height: self.height,
        };

        this.elems.extend(
            self.iter()
                .filter(|(x, _, _)| *x != col)
                .map(|(_, _, v)| v)
                .cloned(),
        );

        Ok(this)
    }

    pub fn remove_row(&self, row: usize) -> Result<Self> {
        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        let mut this = Self {
            elems: Vec::with_capacity(self.width * (self.height - 1)),
            width: self.width,
            height: self.height - 1,
        };

        this.elems.extend(
            self.iter()
                .filter(|(_, y, _)| *y != row)
                .map(|(_, _, v)| v)
                .cloned(),
        );

        Ok(this)
    }

    pub fn remove_col_row(&self, col: usize, row: usize) -> Result<Self> {
        if col >= self.width {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        let mut this = Self {
            elems: Vec::with_capacity((self.width - 1) * (self.height - 1)),
            width: self.width - 1,
            height: self.height - 1,
        };

        this.elems.extend(
            self.iter()
                .filter(|(x, y, _)| *x != col && *y != row)
                .map(|(_, _, v)| v)
                .cloned(),
        );

        Ok(this)
    }

    pub fn det(&self) -> Result<Value> {
        if self.width != self.height {
            return Err(crate::errors::MathError::NonSquareDeterminant);
        }

        if self.width == 0 {
            return Err(crate::errors::MathError::EmptyMatrix);
        }

        if self.width == 1 {
            return Ok(self.elems[0].clone());
        }

        if self.width == 2 {
            let result = self[(0, 0)]
                .mul(&self[(1, 1)])?
                .sub(&self[(0, 1)].mul(&self[(1, 0)])?)?;

            return Ok(result);
        }

        let make_val = |i: usize| -> Result<Value> {
            let pivot = self[(i, 0)].clone();
            let inner = self.remove_col_row(i, 0).and_then(|x| x.det())?;

            if i % 2 == 0 {
                pivot.mul(&inner)
            } else {
                pivot.mul(&inner).and_then(|x| x.neg())
            }
        };

        let mut acc = make_val(0)?;

        for i in 1..self.width {
            acc = acc.add(&make_val(i)?)?;
        }

        Ok(acc)
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

    pub fn aug_vert(&self, bottom: &Self) -> Result<Self> {
        if self.width != bottom.width {
            return Err(crate::errors::MathError::AugmentShapeMismatch);
        }

        let mut this = Self {
            elems: Vec::with_capacity((self.height + bottom.height) * self.width),
            width: self.width,
            height: self.height + bottom.height,
        };

        for col in 0..self.width {
            this.elems
                .extend_from_slice(&self.elems[self.height * col..(self.height + 1) * col]);
            this.elems
                .extend_from_slice(&bottom.elems[bottom.height * col..(bottom.height + 1) * col]);
        }

        Ok(this)
    }

    pub fn ext_vert(&self, value: &Value, extra_rows: usize) -> Self {
        let mut this = Self {
            elems: Vec::with_capacity((self.height + extra_rows) * self.width),
            width: self.width,
            height: self.height + extra_rows,
        };

        for col in 0..self.width {
            this.elems
                .extend_from_slice(&self.elems[self.height * col..(self.height + 1) * col]);
            this.elems.extend((0..extra_rows).map(|_| value.clone()));
        }

        this
    }

    pub fn aug_hor(&self, right: &Self) -> Result<Self> {
        if self.height != right.height {
            return Err(crate::errors::MathError::AugmentShapeMismatch);
        }

        let mut this = Self {
            elems: Vec::with_capacity(self.height * (self.width + right.width)),
            width: self.width + right.width,
            height: self.height,
        };

        this.elems.extend_from_slice(&self.elems);
        this.elems.extend_from_slice(&right.elems);

        Ok(this)
    }

    pub fn ext_hor(&self, value: &Value, extra_cols: usize) -> Self {
        let mut this = Self {
            elems: Vec::with_capacity(self.height * (self.width + extra_cols)),
            width: self.width + extra_cols,
            height: self.height,
        };

        this.elems.extend_from_slice(&self.elems);
        this.elems
            .extend((0..extra_cols * self.height).map(|_| value.clone()));

        this
    }

    // Fills in off-diagonals with zeros
    pub fn aug_diag(&self, bottom_right: &Self) -> Self {
        let width = self.width + bottom_right.width;
        let height = self.height + bottom_right.height;

        let mut this = Self {
            elems: Vec::with_capacity(width * height),
            width,
            height,
        };

        for col in 0..self.width {
            this.elems
                .extend_from_slice(&self.elems[col * self.height..(col + 1) * self.height]);

            this.elems
                .extend(std::iter::repeat_n(Scalar::ZERO, bottom_right.height).map(Value::Scalar));
        }

        for col in 0..bottom_right.width {
            this.elems
                .extend(std::iter::repeat_n(Scalar::ZERO, bottom_right.height).map(Value::Scalar));

            this.elems.extend_from_slice(
                &bottom_right.elems[col * bottom_right.height..(col + 1) * bottom_right.height],
            );
        }

        this
    }

    pub fn ext_diag(&self, value: &Value, extra_cols: usize, extra_rows: usize) -> Self {
        let width = self.width + extra_cols;
        let height = self.height + extra_rows;

        let mut this = Self {
            elems: Vec::with_capacity(width * height),
            width,
            height,
        };

        for col in 0..self.width {
            this.elems
                .extend_from_slice(&self.elems[col * self.height..(col + 1) * self.height]);

            this.elems
                .extend(std::iter::repeat_n(value.clone(), extra_rows));
        }

        for _ in 0..extra_cols {
            this.elems
                .extend(std::iter::repeat_n(value.clone(), height));
        }

        this
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
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
                    .map(|(a, b)| a.mul(b))
                    .reduce(|acc, rhs| acc.and_then(|x| x.add(&rhs?)))
                    .unwrap();

                this.elems.push(result?)
            }
        }

        Ok(this)
    }

    pub fn hadamard_op(
        &self,
        rhs: &Self,
        mut op: impl FnMut(&Value, &Value) -> Result<Value>,
    ) -> Result<Self> {
        if self.width != rhs.width || self.height != rhs.height {
            return Err(crate::errors::MathError::MatrixShapeMismatch);
        }

        Ok(Self {
            elems: self
                .elems
                .iter()
                .zip(rhs.elems.iter())
                .map(move |(x, y)| op(x, y))
                .collect::<Result<Vec<_>>>()?,
            ..*self
        })
    }

    pub fn scalar_mul(&self, rhs: &Value) -> Result<Self> {
        Ok(Self {
            elems: self
                .elems
                .iter()
                .map(|x| x.mul(rhs))
                .collect::<Result<Vec<_>>>()?,
            ..*self
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

    pub fn from_size_slice_rowmaj(width: usize, height: usize, values: &[Value]) -> Result<Self> {
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

    pub fn from_size_slice_colmaj(width: usize, height: usize, values: &[Value]) -> Result<Self> {
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

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.hadamard_op(rhs, Value::add)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.hadamard_op(rhs, Value::sub)
    }

    pub fn mul_componentwise(&self, rhs: &ConcreteMatrix) -> Result<Self> {
        self.hadamard_op(rhs, Value::mul)
    }

    pub fn neg(&self) -> Result<Self> {
        Ok(Self {
            elems: self
                .elems
                .iter()
                .map(|x| x.neg())
                .collect::<Result<Vec<_>>>()?,
            ..*self
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

    fn scale_row(&mut self, row: usize, value: &Value) -> Result<()> {
        if row >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        for i in 0..self.width {
            self[(i, row)] = value.mul(&self[(i, row)])?;
        }

        Ok(())
    }

    // Rdest -= scale * src
    fn sub_row(&mut self, src: usize, dest: usize, scale: &Value) -> Result<()> {
        if src >= self.height || dest >= self.height {
            return Err(crate::errors::MathError::MatrixOutOfRange);
        }

        for i in 0..self.width {
            self[(i, dest)] = self[(i, dest)].sub(&scale.mul(&self[(i, src)])?)?;
        }

        Ok(())
    }

    fn row_echelon_form_internal(&self, reduced: bool) -> Result<(Self, usize)> {
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
            }

            let pivot_scale = this[(col, row)].invert()?;

            for clear in (row + 1)..this.height {
                this.sub_row(row, clear, &this[(col, clear)].mul(&pivot_scale)?)?;
            }

            if reduced {
                for clear in 0..row {
                    this.sub_row(row, clear, &this[(col, clear)].mul(&pivot_scale)?)?;
                }

                this.scale_row(row, &pivot_scale)?;
            }

            row += 1;
        }

        Ok((this, end_col))
    }

    #[doc(alias = "rref")]
    pub fn row_echelon_form(&self, reduced: bool) -> Result<Self> {
        self.row_echelon_form_internal(reduced).map(|x| x.0)
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
                .map(Value::Scalar)
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

    pub fn col_from_iter(iter: impl IntoIterator<Item = Value>) -> Self {
        let elems = Vec::from_iter(iter);

        let height = elems.len();

        Self {
            elems,
            width: 1,
            height,
        }
    }

    pub fn row_from_iter(iter: impl IntoIterator<Item = Value>) -> Self {
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
                Value::Scalar(Scalar::ZERO),
                self.height,
            ));

            result.elems[elem] = Value::Scalar(Scalar::ONE);

            Ok(result)
        } else if self.height == 1 {
            if elem >= self.width {
                return Err(crate::errors::MathError::MatrixOutOfRange);
            }

            let mut result = Self::row_from_iter(std::iter::repeat_n(
                Value::Scalar(Scalar::ZERO),
                self.height,
            ));

            result.elems[elem] = Value::Scalar(Scalar::ONE);

            Ok(result)
        } else {
            Err(crate::errors::MathError::NoIdentityMatrix)
        }
    }

    pub fn zero_matrix(&self) -> Self {
        Self {
            elems: std::iter::repeat_n(Scalar::ZERO, self.elems.len())
                .map(Value::Scalar)
                .collect(),
            ..*self
        }
    }

    pub fn invert(&self) -> Result<Self> {
        let aug = self.aug_hor(&self.my_identity()?)?;

        let (rref, last_col) = aug.row_echelon_form_internal(true)?;

        if last_col >= self.width {
            return Err(crate::errors::MathError::MatrixNotInvertible);
        }

        rref.select_cols(self.width..self.width * 2)
    }

    pub fn is_zero(&self) -> bool {
        self.elems.iter().all(|x| x.is_zero())
    }

    pub fn transpose(&self) -> Result<Self> {
        let mut this = Self {
            elems: Vec::with_capacity(self.elems.len()),
            width: self.height,
            height: self.width,
        };

        for row in 0..self.height {
            this.elems.extend(self.iter_row(row)?.cloned());
        }

        Ok(this)
    }

    pub fn conj(&self) -> Result<Self> {
        Ok(Self {
            elems: self
                .elems
                .iter()
                .map(|x| x.conj())
                .collect::<Result<Vec<_>>>()?,
            ..*self
        })
    }

    pub fn norm_sq(&self) -> Result<Value> {
        self.elems
            .iter()
            .map(|x| x.norm_sq())
            .reduce(|x, y| x.and_then(|x| y.and_then(|y| x.add(&y))))
            .unwrap() // Option unwrap
    }

    pub fn sin(&self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn cos(&self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn tan(&self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn sinh(&self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn cosh(&self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    pub fn tanh(&self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
    }

    // Matrix exponentiation through Sylvester's Formula
    fn exp_inner(&self, eigenvalues: &[(Scalar, usize)]) -> Result<Self> {
        let mut coeff_matrix = Self {
            elems: Vec::with_capacity(self.elems.len()),
            ..*self
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

            let power = lambda.pow(&power);

            let coeff = binomial(n, k) * factorial(k);

            if coeff > i128::MAX as u128 {
                Scalar::from_float(coeff as f64).mul(&power)
            } else {
                Scalar::from_integer(coeff as i128).mul(&power)
            }
        }

        for col in 0..self.width {
            let lambda = lambdas[col];
            let power = exponents[col];

            // row is derivative
            for row in 0..self.height {
                coeff_matrix
                    .elems
                    .push(Value::Scalar(make_coeff(row, power, lambda)));
            }
        }

        let inverted = coeff_matrix.invert()?;

        let mut powers = Vec::with_capacity(self.width);

        powers.push(self.my_identity()?);

        if self.width > 1 {
            powers.push(self.clone());
        }

        let mut acc = self.clone();
        for _ in 2..self.width {
            acc = acc.mul(self)?;
            powers.push(acc.clone());
        }

        // We solved M B = A for B where both B and A are vectors of matrices.
        // `powers` is `A`, and it contains `I, A, A^2, A^3, ...`.
        // We now compute:
        let mut result = self.zero_matrix();

        for row in 0..self.height {
            let lambda_coeff = Value::Scalar(lambdas[row].exp());

            let mut acc = self.zero_matrix();

            for (power, coeff) in powers.iter().zip(inverted.iter_row(row)?) {
                acc = acc.add(&power.scalar_mul(coeff)?)?;
            }

            result = result.add(&acc.scalar_mul(&lambda_coeff)?)?;
        }

        Ok(result)
    }

    // Householder matrix for given column vector
    pub fn householder_matrix(&self) -> Result<Self> {
        if self.width != 1 {
            return Err(crate::errors::MathError::NotColumnVector);
        }

        let star = self.conj()?.transpose()?;

        let numerator = self.mul(&star)?;

        let denominator = self.norm_sq()?;

        numerator.my_identity()?.sub(
            &numerator.scalar_mul(&Value::Scalar(Scalar::from_integer(2)).div(&denominator)?)?,
        )
    }

    pub fn lower_hessenberg(&self) -> Result<Self> {
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

            let vector = col.my_basis_elem(0)?.scalar_mul(&col.norm_sq()?)?;
            let vector = vector.sub(&col)?;

            let householder = vector.householder_matrix()?;

            let matrix = Self::identity(i + 1)?.aug_diag(&householder);

            prod_acc = matrix.mul(&prod_acc)?;

            rectangle = rectangle.remove_col_row(0, 0)?;
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
                &prod_acc.scalar_mul(&Value::Scalar(Scalar::from_num_denom(1, factorial_acc)))?,
            )?;

            prod_acc = prod_acc.mul(self)?;
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
