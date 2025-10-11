use crate::Floatify;
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

#[derive(Clone, Debug)]
pub enum Scalar {
    Rational(RationalComplex),
    Float(FloatComplex),
}

impl Scalar {}
