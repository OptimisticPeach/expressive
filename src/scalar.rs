use crate::{Floatify, errors::Result};
use num_complex::Complex;
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

impl Scalar {
    pub const ONE: Self = Scalar::Rational(RationalComplex::ONE);
    pub const ZERO: Self = Scalar::Rational(RationalComplex::ZERO);

    impl_scalar_ops!(add, (+));
    impl_scalar_ops!(sub, (-));
    impl_scalar_ops!(mul, (*));
    impl_scalar_ops!(div, (/));

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
}
