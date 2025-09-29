use nalgebra::Complex;

use crate::{Floatify, RationalComplex};

use super::Value;

impl Value {
    pub fn exp(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().exp()),
            Value::FloatComplex(x) => Value::FloatComplex(x.exp()),
            Value::RationalMatrix(x) => Value::FloatMatrix(x.floatify().exp()),
            Value::FloatMatrix(x) => Value::FloatMatrix(x.exp()),
        }
    }

    pub fn sin(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().sin()),
            Value::FloatComplex(x) => Value::FloatComplex(x.sin()),
            x => {
                let ix = Value::mul(Value::RationalComplex(RationalComplex::I), x);

                Value::mul(
                    Value::FloatComplex(Complex { re: 0.0, im: -0.5 }),
                    Value::add(ix.clone().exp(), ix.neg().exp().neg()),
                )
            }
        }
    }

    pub fn cos(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().cos()),
            Value::FloatComplex(x) => Value::FloatComplex(x.cos()),
            x => {
                let ix = Value::mul(Value::RationalComplex(RationalComplex::I), x);

                Value::mul(
                    Value::FloatComplex(Complex { re: 0.5, im: 0.0 }),
                    Value::add(ix.clone().exp(), ix.neg().exp()),
                )
            }
        }
    }

    pub fn tan(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().tan()),
            Value::FloatComplex(x) => Value::FloatComplex(x.tan()),
            x => {
                let ix = Value::mul(Value::RationalComplex(RationalComplex::I), x);
                let exp = ix.clone().exp();
                let iexp = ix.neg().exp();

                let num = Value::add(exp.clone(), iexp.clone().neg());
                let denom = Value::add(exp.clone(), iexp.clone());

                Value::mul(
                    Value::FloatComplex(Complex { re: 0.0, im: -1.0 }),
                    Value::mul(num, denom.invert()),
                )
            }
        }
    }

    pub fn sinh(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().sinh()),
            Value::FloatComplex(x) => Value::FloatComplex(x.sinh()),
            x => Value::mul(
                Value::FloatComplex(Complex { re: 0.5, im: 0.0 }),
                Value::add(x.clone().exp(), x.neg().exp().neg()),
            ),
        }
    }

    pub fn cosh(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().cosh()),
            Value::FloatComplex(x) => Value::FloatComplex(x.cosh()),
            x => Value::mul(
                Value::FloatComplex(Complex { re: 0.5, im: 0.0 }),
                Value::add(x.clone().exp(), x.neg().exp()),
            ),
        }
    }

    pub fn tanh(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().tanh()),
            Value::FloatComplex(x) => Value::FloatComplex(x.tanh()),
            x => {
                let exp = x.clone().exp();
                let iexp = x.neg().exp();

                let num = Value::add(exp.clone(), iexp.clone().neg());
                let denom = Value::add(exp.clone(), iexp.clone());

                Value::mul(num, denom.invert())
            }
        }
    }

    pub fn asin(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().asin()),
            Value::FloatComplex(x) => Value::FloatComplex(x.asin()),
            _ => unimplemented!("Advanced trig functions are not implemented for matrices!"),
        }
    }

    pub fn acos(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().acos()),
            Value::FloatComplex(x) => Value::FloatComplex(x.acos()),
            _ => unimplemented!("Advanced trig functions are not implemented for matrices!"),
        }
    }

    pub fn atan(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().atan()),
            Value::FloatComplex(x) => Value::FloatComplex(x.atan()),
            _ => unimplemented!("Advanced trig functions are not implemented for matrices!"),
        }
    }

    pub fn asinh(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().asinh()),
            Value::FloatComplex(x) => Value::FloatComplex(x.asinh()),
            _ => unimplemented!("Advanced trig functions are not implemented for matrices!"),
        }
    }

    pub fn acosh(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().acosh()),
            Value::FloatComplex(x) => Value::FloatComplex(x.acosh()),
            _ => unimplemented!("Advanced trig functions are not implemented for matrices!"),
        }
    }

    pub fn atanh(self) -> Value {
        match self {
            Value::RationalComplex(x) => Value::FloatComplex(x.floatify().atanh()),
            Value::FloatComplex(x) => Value::FloatComplex(x.atanh()),
            _ => unimplemented!("Advanced trig functions are not implemented for matrices!"),
        }
    }
}
