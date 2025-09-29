use nalgebra::{Matrix, RawStorage};

use crate::Floatify;

use super::Value;

impl Value {
    pub fn det(self) -> Value {
        match self {
            x @ (Value::RationalComplex(_) | Value::FloatComplex(_)) => x,
            Value::RationalMatrix(matrix) => Value::FloatComplex(matrix.floatify().determinant()),
            Value::FloatMatrix(matrix) => Value::FloatComplex(matrix.determinant()),
        }
    }

    pub fn index_row(mat: Value, which_row: Value) -> Value {
        let idx = which_row.into_integer().unwrap();

        assert!(idx >= 0);

        let idx = idx as usize;

        match mat {
            mat @ (Value::RationalComplex(_) | Value::FloatComplex(_)) => {
                if idx != 0 {
                    panic!("Cannot take row {idx} of a single scalar!");
                }

                mat
            }
            Value::RationalMatrix(matrix) => Value::RationalMatrix(to_dmat(&matrix.row(idx))),
            Value::FloatMatrix(matrix) => Value::FloatMatrix(to_dmat(&matrix.row(idx))),
        }
    }

    pub fn index_col(mat: Value, which_row: Value) -> Value {
        let idx = which_row.into_integer().unwrap();

        assert!(idx >= 0);

        let idx = idx as usize;

        match mat {
            mat @ (Value::RationalComplex(_) | Value::FloatComplex(_)) => {
                if idx != 0 {
                    panic!("Cannot take row {idx} of a single scalar!");
                }

                mat
            }
            Value::RationalMatrix(matrix) => Value::RationalMatrix(to_dmat(&matrix.row(idx))),
            Value::FloatMatrix(matrix) => Value::FloatMatrix(to_dmat(&matrix.row(idx))),
        }
    }

    pub fn aug_vert(top: Value, bottom: Value) -> Value {
        match (top, bottom) {
            (Value::RationalComplex(top), Value::RationalComplex(bottom)) => Value::RationalMatrix(
                Matrix::from_iterator_generic(nalgebra::Dyn(2), nalgebra::Dyn(1), [top, bottom]),
            ),
            (Value::FloatComplex(top), Value::FloatComplex(bottom)) => Value::FloatMatrix(
                Matrix::from_iterator_generic(nalgebra::Dyn(2), nalgebra::Dyn(1), [top, bottom]),
            ),
            (Value::RationalComplex(top), Value::FloatComplex(bottom)) => {
                Value::FloatMatrix(Matrix::from_iterator_generic(
                    nalgebra::Dyn(2),
                    nalgebra::Dyn(1),
                    [top.floatify(), bottom],
                ))
            }
            (Value::FloatComplex(top), Value::RationalComplex(bottom)) => {
                Value::FloatMatrix(Matrix::from_iterator_generic(
                    nalgebra::Dyn(2),
                    nalgebra::Dyn(1),
                    [top, bottom.floatify()],
                ))
            }
            (Value::RationalComplex(_) | Value::FloatComplex(_), _)
            | (_, Value::RationalComplex(_) | Value::FloatComplex(_)) => {
                panic!("Cannot glue together a scalar with a matrix!")
            }
            (Value::RationalMatrix(top), Value::RationalMatrix(bottom)) => Value::RationalMatrix({
                assert_eq!(top.shape().1, bottom.shape().1);

                let rows = top.row_iter().chain(bottom.row_iter()).collect::<Vec<_>>();

                Matrix::from_rows(&rows)
            }),
            (Value::RationalMatrix(top), Value::FloatMatrix(bottom)) => Value::FloatMatrix({
                assert_eq!(top.shape().1, bottom.shape().1);

                let rows = top
                    .row_iter()
                    .map(|x| x.map(|v| v.floatify()))
                    .chain(bottom.row_iter().map(|x| x.map(|v| v)))
                    .collect::<Vec<_>>();

                Matrix::from_rows(&rows[..])
            }),
            (Value::FloatMatrix(top), Value::RationalMatrix(bottom)) => Value::FloatMatrix({
                assert_eq!(top.shape().1, bottom.shape().1);

                let rows = top
                    .row_iter()
                    .map(|x| x.map(|v| v))
                    .chain(bottom.row_iter().map(|x| x.map(|v| v.floatify())))
                    .collect::<Vec<_>>();

                Matrix::from_rows(&rows[..])
            }),
            (Value::FloatMatrix(top), Value::FloatMatrix(bottom)) => Value::FloatMatrix({
                assert_eq!(top.shape().1, bottom.shape().1);

                let rows = top.row_iter().chain(bottom.row_iter()).collect::<Vec<_>>();

                Matrix::from_rows(&rows)
            }),
        }
    }

    pub fn aug_horizontal(left: Value, right: Value) -> Value {
        match (left, right) {
            (Value::RationalComplex(left), Value::RationalComplex(right)) => Value::RationalMatrix(
                Matrix::from_iterator_generic(nalgebra::Dyn(1), nalgebra::Dyn(2), [left, right]),
            ),
            (Value::FloatComplex(left), Value::FloatComplex(right)) => Value::FloatMatrix(
                Matrix::from_iterator_generic(nalgebra::Dyn(1), nalgebra::Dyn(2), [left, right]),
            ),
            (Value::RationalComplex(left), Value::FloatComplex(right)) => {
                Value::FloatMatrix(Matrix::from_iterator_generic(
                    nalgebra::Dyn(1),
                    nalgebra::Dyn(2),
                    [left.floatify(), right],
                ))
            }
            (Value::FloatComplex(left), Value::RationalComplex(right)) => {
                Value::FloatMatrix(Matrix::from_iterator_generic(
                    nalgebra::Dyn(1),
                    nalgebra::Dyn(2),
                    [left, right.floatify()],
                ))
            }
            (Value::RationalComplex(_) | Value::FloatComplex(_), _)
            | (_, Value::RationalComplex(_) | Value::FloatComplex(_)) => {
                panic!("Cannot glue together a scalar with a matrix!")
            }
            (Value::RationalMatrix(left), Value::RationalMatrix(right)) => Value::RationalMatrix({
                assert_eq!(left.shape().0, right.shape().0);

                let columns = left
                    .column_iter()
                    .chain(right.column_iter())
                    .collect::<Vec<_>>();

                Matrix::from_columns(&columns)
            }),
            (Value::RationalMatrix(left), Value::FloatMatrix(right)) => Value::FloatMatrix({
                assert_eq!(left.shape().1, right.shape().1);

                let columns = left
                    .column_iter()
                    .map(|x| x.map(|v| v.floatify()))
                    .chain(right.column_iter().map(|x| x.map(|v| v)))
                    .collect::<Vec<_>>();

                Matrix::from_columns(&columns[..])
            }),
            (Value::FloatMatrix(left), Value::RationalMatrix(right)) => Value::FloatMatrix({
                assert_eq!(left.shape().1, right.shape().1);

                let columns = left
                    .column_iter()
                    .map(|x| x.map(|v| v))
                    .chain(right.column_iter().map(|x| x.map(|v| v.floatify())))
                    .collect::<Vec<_>>();

                Matrix::from_columns(&columns[..])
            }),
            (Value::FloatMatrix(left), Value::FloatMatrix(right)) => Value::FloatMatrix({
                assert_eq!(left.shape().1, right.shape().1);

                let columns = left
                    .column_iter()
                    .chain(right.column_iter())
                    .collect::<Vec<_>>();

                Matrix::from_columns(&columns)
            }),
        }
    }
}

fn to_dmat<T, R, C, S>(matrix: &Matrix<T, R, C, S>) -> nalgebra::DMatrix<T>
where
    T: Clone + PartialEq + std::fmt::Debug + num_traits::Zero + 'static,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: RawStorage<T, R, C>,
{
    let mut dynamic_matrix = nalgebra::DMatrix::<T>::zeros(matrix.nrows(), matrix.ncols());
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            dynamic_matrix[(i, j)] = matrix[(i, j)].clone();
        }
    }
    dynamic_matrix
}
