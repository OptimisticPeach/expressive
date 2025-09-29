use nalgebra::{DMatrix, DefaultAllocator, Dyn};
use num_complex::Complex;
use num_rational::Ratio;
use num_traits::ToPrimitive;

mod linalg;
mod trig;

type Storage<T> = <DefaultAllocator as nalgebra::allocator::Allocator<Dyn, Dyn>>::Buffer<T>;

pub type Rational = Ratio<i128>;
pub type RationalComplex = Complex<Rational>;
pub type RationalMatrix = DMatrix<RationalComplex>;

pub type FloatComplex = Complex<f64>;
pub type FloatMatrix = DMatrix<FloatComplex>;

trait Floatify {
    type Floated;

    fn floatify(self) -> Self::Floated;
}

impl Floatify for Rational {
    type Floated = f64;

    fn floatify(self) -> Self::Floated {
        self.to_f64().unwrap()
    }
}

impl Floatify for RationalComplex {
    type Floated = FloatComplex;

    fn floatify(self) -> Self::Floated {
        FloatComplex {
            re: self.re.floatify(),
            im: self.im.floatify(),
        }
    }
}

impl Floatify for RationalMatrix {
    type Floated = FloatMatrix;

    fn floatify(self) -> Self::Floated {
        self.map(RationalComplex::floatify)
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    RationalComplex(RationalComplex),
    FloatComplex(FloatComplex),
    RationalMatrix(RationalMatrix),
    FloatMatrix(FloatMatrix),
}

impl Value {
    pub fn add(lhs: Value, rhs: Value) -> Value {
        use Value::*;

        match (lhs, rhs) {
            (RationalComplex(x), RationalComplex(y)) => RationalComplex(x + y),
            (FloatComplex(x), FloatComplex(y)) => FloatComplex(x + y),
            (RationalMatrix(x), RationalMatrix(y)) => RationalMatrix(x + y),
            (FloatMatrix(x), FloatMatrix(y)) => FloatMatrix(x + y),

            (RationalComplex(x), FloatComplex(y)) => FloatComplex(x.floatify() + y),
            (FloatComplex(x), RationalComplex(y)) => FloatComplex(x + y.floatify()),
            (RationalMatrix(x), FloatMatrix(y)) => FloatMatrix(x.floatify() + y),
            (FloatMatrix(x), RationalMatrix(y)) => FloatMatrix(x + y.floatify()),

            (RationalComplex(_) | FloatComplex(_), RationalMatrix(_) | FloatMatrix(_)) => {
                panic!("Unsupported addition between scalar and matrix!")
            }

            (RationalMatrix(_) | FloatMatrix(_), RationalComplex(_) | FloatComplex(_)) => {
                panic!("Unsupported addition between matrix and scalar!")
            }
        }
    }

    pub fn mul(lhs: Value, rhs: Value) -> Value {
        use Value::*;

        match (lhs, rhs) {
            (RationalComplex(x), RationalComplex(y)) => RationalComplex(x * y),
            (FloatComplex(x), FloatComplex(y)) => FloatComplex(x * y),
            (RationalMatrix(x), RationalMatrix(y)) => RationalMatrix(x * y),
            (FloatMatrix(x), FloatMatrix(y)) => FloatMatrix(x * y),

            (RationalComplex(x), FloatComplex(y)) => FloatComplex(x.floatify() * y),
            (FloatComplex(x), RationalComplex(y)) => FloatComplex(x * y.floatify()),
            (RationalMatrix(x), FloatMatrix(y)) => FloatMatrix(x.floatify() * y),
            (FloatMatrix(x), RationalMatrix(y)) => FloatMatrix(x * y.floatify()),

            (RationalComplex(x), RationalMatrix(y)) => RationalMatrix(y * x),
            (RationalMatrix(x), RationalComplex(y)) => RationalMatrix(x * y),
            (FloatComplex(x), FloatMatrix(y)) => FloatMatrix(y * x),
            (FloatMatrix(x), FloatComplex(y)) => FloatMatrix(x * y),

            (FloatComplex(x), RationalMatrix(y)) => FloatMatrix(y.floatify() * x),
            (FloatMatrix(x), RationalComplex(y)) => FloatMatrix(x * y.floatify()),
            (RationalComplex(x), FloatMatrix(y)) => FloatMatrix(y * x.floatify()),
            (RationalMatrix(x), FloatComplex(y)) => FloatMatrix(x.floatify() * y),
        }
    }

    pub fn neg(self) -> Value {
        use Value::*;

        match self {
            RationalComplex(x) => RationalComplex(-x),
            FloatComplex(x) => FloatComplex(-x),
            RationalMatrix(x) => RationalMatrix(-x),
            FloatMatrix(x) => FloatMatrix(-x),
        }
    }

    pub fn invert(self) -> Value {
        use Value::*;

        match self {
            RationalComplex(x) => RationalComplex(x.inv()),
            FloatComplex(x) => FloatComplex(x.inv()),
            // todo: make a rational inversion.
            RationalMatrix(x) => FloatMatrix(x.floatify().try_inverse().unwrap()),
            FloatMatrix(x) => FloatMatrix(x.try_inverse().unwrap()),
        }
    }

    pub fn into_integer(&self) -> Option<i128> {
        use Value::*;

        match self {
            RationalComplex(x) => Some(*x.re.floor().numer()),
            FloatComplex(x) => Some(x.re as i128),
            _ => None,
        }
    }

    pub fn identity(&self) -> Value {
        let size = match self {
            Value::RationalComplex(_) | Value::FloatComplex(_) => {
                let num = self.into_integer().unwrap();

                if num <= 0 {
                    return Value::RationalComplex(Complex::ONE);
                }

                num as usize
            }

            Value::RationalMatrix(x) => x.shape().0,
            Value::FloatMatrix(x) => x.shape().0,
        };

        return Value::RationalMatrix(RationalMatrix::identity(size, size));
    }

    pub fn not(self) -> Value {
        Value::add(self.identity(), self.neg())
    }

    pub fn and(lhs: Value, rhs: Value) -> Value {
        Value::mul(lhs, rhs)
    }

    pub fn or(lhs: Value, rhs: Value) -> Value {
        Value::and(lhs.neg(), rhs.neg()).neg()
    }
}
