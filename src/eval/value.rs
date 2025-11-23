use crate::{
    ast::Ast,
    errors::{EvalStatus, MathError},
    eval::{Ident, function::Function, matrix::Matrix, scalar::Scalar},
};

#[derive(Clone, Debug, PartialEq)]
pub enum EvalAst {
    Scalar(Scalar),
    Matrix(Matrix),
    Lambda(Function),
    Symbol(Ident),

    // todo: replace w/ arena
    Application(Box<EvalAst>, Vec<EvalAst>),
}

impl Default for EvalAst {
    fn default() -> Self {
        Self::ZERO
    }
}

impl EvalAst {
    pub const ONE: Self = EvalAst::Scalar(Scalar::ONE);
    pub const ZERO: Self = EvalAst::Scalar(Scalar::ZERO);

    pub fn to_ast(self) -> Ast {
        todo!()
    }

    pub fn add(self, rhs: EvalAst) -> Result<EvalAst> {
        match (self, rhs) {
            (EvalAst::Scalar(_), EvalAst::Matrix(_)) | (EvalAst::Matrix(_), EvalAst::Scalar(_)) => {
                Err(MathError::AddScalarMatrix)
            }
            (EvalAst::Scalar(x), EvalAst::Scalar(y)) => Ok(EvalAst::Scalar(x.add(y))),
            (EvalAst::Matrix(x), EvalAst::Matrix(y)) => x.add(y).map(EvalAst::Matrix),
        }
    }

    pub fn sub(self, rhs: EvalAst) -> Result<EvalAst> {
        match (self, rhs) {
            (EvalAst::Scalar(_), EvalAst::Matrix(_)) | (EvalAst::Matrix(_), EvalAst::Scalar(_)) => {
                Err(MathError::AddScalarMatrix)
            }
            (EvalAst::Scalar(x), EvalAst::Scalar(y)) => Ok(EvalAst::Scalar(x.sub(y))),
            (EvalAst::Matrix(x), EvalAst::Matrix(y)) => x.sub(y).map(EvalAst::Matrix),
        }
    }

    pub fn mul(self, rhs: EvalAst) -> Result<EvalAst> {
        match (self, rhs) {
            (x @ EvalAst::Scalar(_), EvalAst::Matrix(m))
            | (EvalAst::Matrix(m), x @ EvalAst::Scalar(_)) => Ok(EvalAst::Matrix(m.scalar_mul(x)?)),
            (EvalAst::Scalar(x), EvalAst::Scalar(y)) => Ok(EvalAst::Scalar(x.mul(y))),
            (EvalAst::Matrix(x), EvalAst::Matrix(y)) => x.mul(y).map(EvalAst::Matrix),
        }
    }

    pub fn div(self, rhs: EvalAst) -> Result<EvalAst> {
        let result = match (self, rhs) {
            (EvalAst::Scalar(x), EvalAst::Scalar(y)) => EvalAst::Scalar(x.div(y)?),
            (EvalAst::Matrix(x), EvalAst::Scalar(y)) => {
                EvalAst::Matrix(x.scalar_mul(EvalAst::Scalar(y.invert()?))?)
            }
            (x @ EvalAst::Scalar(_), EvalAst::Matrix(y)) => {
                EvalAst::Matrix(y.invert()?.scalar_mul(x)?)
            }
            (EvalAst::Matrix(x), EvalAst::Matrix(y)) => EvalAst::Matrix(x.mul(y.invert()?)?),
        };

        Ok(result)
    }

    pub fn neg(self) -> Result<EvalAst> {
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

    pub fn invert(self) -> Result<EvalAst> {
        match self {
            Self::Scalar(x) => x.invert().map(EvalAst::Scalar),
            Self::Matrix(x) => x.invert().map(EvalAst::Matrix),
        }
    }

    pub fn identity(&self) -> Result<EvalAst> {
        match self {
            EvalAst::Scalar(_) => Ok(EvalAst::Scalar(Scalar::ONE)),
            EvalAst::Matrix(_) => Ok(EvalAst::Matrix(Matrix::identity(None)?)),
        }
    }

    pub fn not(self) -> Result<EvalAst> {
        self.identity()?.sub(self.neg()?)
    }

    pub fn and(self, rhs: EvalAst) -> Result<EvalAst> {
        self.mul(rhs)
    }

    pub fn or(self, rhs: EvalAst) -> Result<EvalAst> {
        self.neg()?.and(rhs.neg()?)?.neg()
    }

    pub fn xor(self, rhs: EvalAst) -> Result<EvalAst> {
        // xor = |a, b| { (a && !b) || (!a && b) }
        // == |a, b| { (a || b) && (!a || !b) }
        self.clone()
            .or(rhs.clone())?
            .and(self.not()?.or(rhs.not()?)?)
    }

    pub fn conj(self) -> Result<EvalAst> {
        match self {
            Self::Scalar(x) => Ok(Self::Scalar(x.conj())),
            Self::Matrix(x) => x.conj().map(Self::Matrix),
        }
    }

    pub fn transpose(self) -> EvalAst {
        match self {
            Self::Matrix(x) => Self::Matrix(x.transpose()),
            x @ Self::Scalar(_) => x.clone(),
        }
    }

    pub fn norm_sq(self) -> Result<EvalAst> {
        match self {
            Self::Scalar(x) => Ok(Self::Scalar(x.mul(x.conj()))),
            Self::Matrix(x) => x.norm_sq(),
        }
    }
}

// macro_rules! trig_ops {
//     (($($both:ident),+), ($($scalar:ident),+)) => {
//         $(
//             pub fn $both(self) -> Result<EvalAst> {
//                 match self {
//                     Self::Scalar(x) => Ok(Self::Scalar(x.$both())),
//                     Self::Matrix(m) => m.$both().map(Self::Matrix),
//                 }
//             }
//         )+

//         $(
//             pub fn $scalar(self) -> Result<EvalAst> {
//                 match self {
//                     Self::Scalar(x) => Ok(Self::Scalar(x.$scalar())),
//                     Self::Matrix(_) => Err(MathError::UnsupportedMatrix),
//                 }
//             }
//         )+
//     }
// }

// impl EvalAst {
//     trig_ops! {
//         (exp, sin, cos, tan, sinh, cosh, tanh),
//         (ln, sqrt, asin, acos, atan, asinh, acosh, atanh)
//     }
// }
