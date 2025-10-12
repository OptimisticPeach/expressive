mod errors;
mod linalg;
mod scalar;
// mod trig;

use errors::Result;
use linalg::Matrix;
use scalar::Scalar;

trait Floatify {
    type Floated;

    fn floatify(self) -> Self::Floated;
}

#[derive(Clone, Debug)]
pub enum Value {
    Scalar(Scalar),
    Matrix(Matrix),
}

impl Value {
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

    pub fn neg(&self) -> Result<Value> {
        match self {
            Self::Scalar(x) => Ok(Self::Scalar(x.neg())),
            Self::Matrix(x) => x.neg().map(Self::Matrix),
        }
    }

    pub fn invert(&self) -> Result<Value> {
        todo!()
    }

    pub fn into_integer(&self) -> Option<i128> {
        todo!()
    }

    pub fn identity(&self) -> Result<Value> {
        todo!()
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
}
