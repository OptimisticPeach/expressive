use crate::{Floatify, errors::Result};
use num_complex::{Complex, ComplexFloat};
use num_rational::Ratio;
use num_traits::ToPrimitive;

pub type Rational = Ratio<i128>;
pub type RationalComplex = Complex<Rational>;

pub type FloatComplex = Complex<f64>;

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

#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    Rational(RationalComplex),
    Float(FloatComplex),
}

impl Floatify for Scalar {
    type Floated = Self;

    fn floatify(self) -> Self::Floated {
        match self {
            Self::Rational(x) => Self::Float(x.floatify()),
            x => x,
        }
    }
}

macro_rules! impl_scalar_ops {
    ($name:ident, ($op:tt)) => {
        pub fn $name(&self, other: &Self) -> Self {
            match (self, other) {
                (Self::Rational(x), Self::Rational(y)) => Self::Rational(x $op y),
                (Self::Rational(x), Self::Float(y)) => Self::Float(x.floatify() $op y),
                (Self::Float(x), Self::Rational(y)) => Self::Float(x $op y.floatify()),
                (Self::Float(x), Self::Float(y)) => Self::Float(x $op y),
            }
        }
    }
}

macro_rules! impl_float_ops {
    ($($name:ident),+$(,)?) => {
        $(
            pub fn $name(&self) -> Self {
                Self::Float(self.float().$name())
            }
        )+
    }
}

impl Scalar {
    pub const ONE: Self = Scalar::Rational(RationalComplex::ONE);
    pub const ZERO: Self = Scalar::Rational(RationalComplex::ZERO);
    pub const IMAG: Self = Scalar::Rational(RationalComplex::I);

    fn float(&self) -> FloatComplex {
        match self {
            Scalar::Rational(complex) => complex.floatify(),
            Scalar::Float(complex) => *complex,
        }
    }

    // todo: failsafe to go to f64 if we overflow i128
    impl_scalar_ops!(add, (+));
    impl_scalar_ops!(sub, (-));
    impl_scalar_ops!(mul, (*));

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        Ok(self.mul(&rhs.invert()?))
    }

    pub fn neg(&self) -> Self {
        match self {
            Self::Rational(x) => Self::Rational(-x),
            Self::Float(x) => Self::Float(-x),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Self::Rational(x) => *x.norm_sqr().numer() == 0,
            Self::Float(x) => x.norm_sqr() == 0.0,
        }
    }

    pub fn invert(&self) -> Result<Self> {
        if self.is_zero() {
            return Err(crate::errors::MathError::DivideByZero);
        }

        match self {
            Self::Rational(x) => Ok(Self::Rational(x.inv())),
            Self::Float(x) => Ok(Self::Float(x.finv())),
        }
    }

    pub fn conj(&self) -> Self {
        match self {
            Self::Rational(x) => Self::Rational(x.conj()),
            Self::Float(x) => Self::Float(x.conj()),
        }
    }

    fn faithful_integer(&self) -> Option<i128> {
        match self {
            Scalar::Rational(complex) => {
                if complex.im.reduced() != Ratio::ZERO {
                    return None;
                }

                // docs say not necessary, but just in case
                let reduced = complex.re.reduced();

                if reduced.is_integer() {
                    Some(reduced.to_integer())
                } else {
                    None
                }
            }
            Scalar::Float(complex) => {
                if complex.im != 0.0 {
                    return None;
                }

                if complex.im.fract() == 0.0 {
                    Some(complex.im as i128)
                } else {
                    None
                }
            }
        }
    }

    pub fn pow(&self, power: &Self) -> Self {
        let floated = match self {
            Scalar::Rational(complex) => {
                if let Some(pow) = power.faithful_integer() {
                    let result =
                        complex.powi(pow.min(i32::MAX as i128).max(i32::MIN as i128) as i32);
                    return Self::Rational(result);
                }

                complex.floatify()
            }
            Scalar::Float(complex) => *complex,
        };

        Self::Float(floated.powc(power.float()))
    }

    impl_float_ops!(
        exp, ln, sqrt, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh,
    );

    pub fn from_integer(integer: i128) -> Self {
        Self::Rational(Complex {
            re: Ratio::from_integer(integer),
            im: Ratio::ZERO,
        })
    }

    pub fn from_float(float: f64) -> Self {
        Self::Float(Complex { re: float, im: 0.0 })
    }

    pub fn from_num_denom(num: i128, denom: i128) -> Self {
        Self::Rational(Complex {
            re: Ratio::new(num, denom),
            im: Ratio::ZERO,
        })
    }
}
