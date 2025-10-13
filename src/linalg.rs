use std::ops::{Index, IndexMut};

use crate::Floatify;
use crate::errors::Result;
use crate::scalar::Scalar;

use super::Value;

/// Matrix stored in column-major format:
/// [ 0 3 6 ]
/// [ 1 4 7 ]
/// [ 2 5 8 ]
#[derive(Clone, Debug)]
pub struct Matrix {
    elems: Vec<Value>,
    width: usize,
    height: usize,
}

impl Index<(usize, usize)> for Matrix {
    type Output = Value;

    fn index(&self, (col, row): (usize, usize)) -> &Value {
        &self.elems[row + col * self.height]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (col, row): (usize, usize)) -> &mut Value {
        &mut self.elems[row + col * self.height]
    }
}

impl Matrix {
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

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        if self.width != rhs.height {
            return Err(crate::errors::MathError::MatrixMultShapeMismatch);
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
            return Err(crate::errors::MathError::MatrixHadamardMismatch);
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

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.hadamard_op(rhs, Value::add)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.hadamard_op(rhs, Value::sub)
    }

    pub fn mul_componentwise(&self, rhs: &Matrix) -> Result<Self> {
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

    pub fn square_identity(&self) -> Result<Self> {
        if self.width != self.height {
            return Err(crate::errors::MathError::MatrixNotSquare);
        }

        let this = Self {
            elems: self
                .iter()
                .map(|(x, y, _)| if x == y { Scalar::ONE } else { Scalar::ZERO })
                .map(Value::Scalar)
                .collect(),
            ..*self
        };

        Ok(this)
    }

    pub fn invert(&self) -> Result<Self> {
        let aug = self.aug_hor(&self.square_identity()?)?;

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
            elems: Vec::with_capacity(self.elems.capacity()),
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

    pub fn exp(&self) -> Result<Self> {
        Err(crate::errors::MathError::NotImplemented)
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
}
