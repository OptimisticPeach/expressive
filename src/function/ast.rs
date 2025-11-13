use crate::{Matrix, Value, errors::*, scalar::Scalar};

pub enum Nilary {
    Integer(i128),
    ImaginaryUnit,
    Identity { size: Option<usize> },
    Variable,
}

impl Nilary {
    pub fn eval(&self) -> Result<Value> {
        match self {
            Nilary::Integer(i) => Ok(Value::Scalar(Scalar::from_integer(*i))),
            Nilary::ImaginaryUnit => Ok(Value::Scalar(Scalar::IMAG)),
            &Nilary::Identity { size } => {
                if let Some(size) = size {
                    Matrix::identity(Some(size)).map(Value::Matrix)
                } else {
                    todo!()
                }
            }
            Nilary::Variable => todo!(),
        }
    }
}

pub enum Unary {
    Neg,
    Exp,
    Sin,
    Cos,
    Tan,
    // Sinh,
    // Cosh,
    // Tanh,
    // etc.
    Transpose,
    Conjugate,
    Not,
    Invert,
}

impl Unary {
    pub fn eval(&self, arg: &Value) -> Result<Value> {
        match *self {
            Unary::Neg => arg.neg(),
            Unary::Exp => arg.exp(),
            Unary::Sin => arg.sin(),
            Unary::Cos => arg.cos(),
            Unary::Tan => arg.tan(),
            Unary::Transpose => Ok(arg.transpose()),
            Unary::Conjugate => arg.conj(),
            Unary::Not => arg.not(),
            Unary::Invert => arg.invert(),
        }
    }
}

pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Xor,
    // powers?
}

impl Binary {
    pub fn eval(&self, a: &Value, b: &Value) -> Result<Value> {
        match *self {
            Binary::Add => a.add(b),
            Binary::Sub => a.sub(b),
            Binary::Mul => a.mul(b),
            Binary::Div => a.div(b),
            Binary::And => a.and(b),
            Binary::Or => a.or(b),
            Binary::Xor => a.xor(b),
        }
    }
}

pub enum Variadic {
    MatrixCtor { width: usize, height: usize },
}

impl Variadic {
    pub fn eval(&self, args: &[Value]) -> Result<Value> {
        match *self {
            Variadic::MatrixCtor { width, height } => {
                Matrix::from_size_slice_rowmaj(width, height, args).map(Value::Matrix)
            }
        }
    }
}

pub enum Function {
    Nilary(Nilary),
    Unary(Unary),
    Binary(Binary),
    Variadic(Variadic),
}

pub struct Ast {
    pub function: Function,
    pub args: Vec<Ast>,
}

impl Ast {
    pub fn check_arity(&self) -> Result<()> {
        match (&self.function, self.args.len()) {
            (Function::Nilary(_), 0) => {}
            (Function::Unary(_), 1) => {}
            (Function::Binary(_), 2) => {}
            (Function::Variadic(Variadic::MatrixCtor { width, height }), l)
                if l == width * height => {}
            _ => return Err(MathError::WrongNumberOfArgs),
        }

        Ok(())
    }

    pub fn eval(&self) -> Result<Value> {
        match (&self.function, &self.args[..]) {
            (Function::Nilary(nil), []) => nil.eval(),
            (Function::Unary(unary), [val]) => unary.eval(&val.eval()?),
            (Function::Binary(binary), [lhs, rhs]) => binary.eval(&lhs.eval()?, &rhs.eval()?),
            (Function::Variadic(variadic), vals) => {
                let vals = vals.iter().map(|x| x.eval()).collect::<Result<Vec<_>>>()?;
                variadic.eval(&vals)
            }

            _ => Err(MathError::WrongNumberOfArgs),
        }
    }
}
