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
    AdditiveId,
    MultiplicativeId,
    Scalar(Scalar),
    Matrix(Matrix),
}

impl Value {
    pub fn add(&self, rhs: &Value) -> Result<Value> {
        todo!()
    }

    pub fn sub(&self, rhs: &Value) -> Result<Value> {
        todo!()
    }

    pub fn mul(&self, rhs: &Value) -> Result<Value> {
        todo!()
    }

    pub fn neg(&self) -> Result<Value> {
        todo!()
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
