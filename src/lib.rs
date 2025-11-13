mod errors;
mod function;
mod matrix;
mod scalar;

use errors::Result;
use matrix::Matrix;
use scalar::Scalar;

trait Floatify {
    type Floated;

    fn floatify(self) -> Self::Floated;
}

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Scalar(Scalar),
    Matrix(Matrix),
}

impl Floatify for Value {
    type Floated = Self;

    fn floatify(self) -> Self::Floated {
        match self {
            Self::Scalar(x) => Self::Scalar(x.floatify()),
            Self::Matrix(x) => Self::Matrix(x.floatify()),
        }
    }
}

impl Value {
    pub const ONE: Self = Value::Scalar(Scalar::ONE);
    pub const ZERO: Self = Value::Scalar(Scalar::ZERO);

    pub fn add(&self, rhs: &Value) -> Result<Value> {
        match (self, rhs) {
            (Value::Scalar(_), Value::Matrix(_)) | (Value::Matrix(_), Value::Scalar(_)) => {
                Err(errors::MathError::AddScalarMatrix)
            }
            (Value::Scalar(x), Value::Scalar(y)) => Ok(Value::Scalar(x.add(y))),
            (Value::Matrix(x), Value::Matrix(y)) => x.add(y).map(Value::Matrix),
        }
    }

    pub fn sub(&self, rhs: &Value) -> Result<Value> {
        match (self, rhs) {
            (Value::Scalar(_), Value::Matrix(_)) | (Value::Matrix(_), Value::Scalar(_)) => {
                Err(errors::MathError::AddScalarMatrix)
            }
            (Value::Scalar(x), Value::Scalar(y)) => Ok(Value::Scalar(x.sub(y))),
            (Value::Matrix(x), Value::Matrix(y)) => x.sub(y).map(Value::Matrix),
        }
    }

    pub fn mul(&self, rhs: &Value) -> Result<Value> {
        match (self, rhs) {
            (x @ Value::Scalar(_), Value::Matrix(m)) | (Value::Matrix(m), x @ Value::Scalar(_)) => {
                Ok(Value::Matrix(m.scalar_mul(x)?))
            }
            (Value::Scalar(x), Value::Scalar(y)) => Ok(Value::Scalar(x.mul(y))),
            (Value::Matrix(x), Value::Matrix(y)) => x.mul(y).map(Value::Matrix),
        }
    }

    pub fn div(&self, rhs: &Value) -> Result<Value> {
        let result = match (self, rhs) {
            (Value::Scalar(x), Value::Scalar(y)) => Value::Scalar(x.div(y)?),
            (Value::Matrix(x), Value::Scalar(y)) => {
                Value::Matrix(x.scalar_mul(&Value::Scalar(y.invert()?))?)
            }
            (x @ Value::Scalar(_), Value::Matrix(y)) => Value::Matrix(y.invert()?.scalar_mul(x)?),
            (Value::Matrix(x), Value::Matrix(y)) => Value::Matrix(x.mul(&y.invert()?)?),
        };

        Ok(result)
    }

    pub fn neg(&self) -> Result<Value> {
        match self {
            Self::Scalar(x) => Ok(Self::Scalar(x.neg())),
            Self::Matrix(x) => x.neg().map(Self::Matrix),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Self::Scalar(x) => x.is_zero(),
            Self::Matrix(x) => x.is_zero(),
        }
    }

    pub fn invert(&self) -> Result<Value> {
        match self {
            Self::Scalar(x) => x.invert().map(Value::Scalar),
            Self::Matrix(x) => x.invert().map(Value::Matrix),
        }
    }

    pub fn identity(&self) -> Result<Value> {
        match self {
            Value::Scalar(_) => Ok(Value::Scalar(Scalar::ONE)),
            Value::Matrix(_) => Ok(Value::Matrix(Matrix::identity(None)?)),
        }
    }

    pub fn not(&self) -> Result<Value> {
        Value::add(&self.identity()?, &self.neg()?)
    }

    pub fn and(&self, rhs: &Value) -> Result<Value> {
        self.mul(rhs)
    }

    pub fn or(&self, rhs: &Value) -> Result<Value> {
        self.neg()?.and(&rhs.neg()?)?.neg()
    }

    pub fn xor(&self, rhs: &Value) -> Result<Value> {
        // xor = |a, b| { (a && !b) || (!a && b) }
        // == |a, b| { (a || b) && (!a || !b) }
        self.or(rhs)?.and(&self.not()?.or(&rhs.not()?)?)
    }

    pub fn conj(&self) -> Result<Value> {
        match self {
            Self::Scalar(x) => Ok(Self::Scalar(x.conj())),
            Self::Matrix(x) => x.conj().map(Self::Matrix),
        }
    }

    pub fn transpose(&self) -> Value {
        match self {
            Self::Matrix(x) => Self::Matrix(x.transpose()),
            x @ Self::Scalar(_) => x.clone(),
        }
    }

    pub fn norm_sq(&self) -> Result<Value> {
        match self {
            Self::Scalar(x) => Ok(Self::Scalar(x.mul(&x.conj()))),
            Self::Matrix(x) => x.norm_sq(),
        }
    }
}

macro_rules! trig_ops {
    (($($both:ident),+), ($($scalar:ident),+)) => {
        $(
            pub fn $both(&self) -> Result<Value> {
                match self {
                    Self::Scalar(x) => Ok(Self::Scalar(x.$both())),
                    Self::Matrix(m) => m.$both().map(Self::Matrix),
                }
            }
        )+

        $(
            pub fn $scalar(&self) -> Result<Value> {
                match self {
                    Self::Scalar(x) => Ok(Self::Scalar(x.$scalar())),
                    Self::Matrix(_) => Err(errors::MathError::UnsupportedMatrix),
                }
            }
        )+
    }
}

impl Value {
    trig_ops! {
        (exp, sin, cos, tan, sinh, cosh, tanh),
        (ln, sqrt, asin, acos, atan, asinh, acosh, atanh)
    }
}
