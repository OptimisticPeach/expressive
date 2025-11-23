#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum RuntimeIdent {
    Add,
    Sub,
    Mul,
    Div,

    Invert,
    Neg,

    Identity,

    Not,
    And,
    Or,
    Xor,

    Conj,
    Transpose,

    NormSq,

    Exp,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,

    Ln,
    Sqrt,
    ArcSin,
    ArcCos,
    ArcTan,
    ArSinh,
    ArCosh,
    ArTanh,

    Matrix,
    ConcreteMatrix,
    UnknownWidthMatrix,
    UnknownHeightMatrix,
    UnknownSizeMatrix,
    IdentityMatrix,
}

impl RuntimeIdent {
    pub const fn to_str(self) -> &'static str {
        match self {
            RuntimeIdent::Add => "add",
            RuntimeIdent::Sub => "sub",
            RuntimeIdent::Mul => "mul",
            RuntimeIdent::Div => "div",
            RuntimeIdent::Invert => "invert",
            RuntimeIdent::Neg => "neg",
            RuntimeIdent::Identity => "identity",
            RuntimeIdent::Not => "not",
            RuntimeIdent::And => "and",
            RuntimeIdent::Or => "or",
            RuntimeIdent::Xor => "xor",
            RuntimeIdent::Conj => "conj",
            RuntimeIdent::Transpose => "transpose",
            RuntimeIdent::NormSq => "norm-sq",
            RuntimeIdent::Exp => "exp",
            RuntimeIdent::Sin => "sin",
            RuntimeIdent::Cos => "cos",
            RuntimeIdent::Tan => "tan",
            RuntimeIdent::Sinh => "sinh",
            RuntimeIdent::Cosh => "cosh",
            RuntimeIdent::Tanh => "tanh",
            RuntimeIdent::Ln => "ln",
            RuntimeIdent::Sqrt => "sqrt",
            RuntimeIdent::ArcSin => "arcsin",
            RuntimeIdent::ArcCos => "arccos",
            RuntimeIdent::ArcTan => "arctan",
            RuntimeIdent::ArSinh => "arsinh",
            RuntimeIdent::ArCosh => "arcosh",
            RuntimeIdent::ArTanh => "artanh",

            RuntimeIdent::Matrix => "matrix",
            RuntimeIdent::ConcreteMatrix => "concrete",
            RuntimeIdent::UnknownWidthMatrix => "?width",
            RuntimeIdent::UnknownHeightMatrix => "?height",
            RuntimeIdent::UnknownSizeMatrix => "?size",
            RuntimeIdent::IdentityMatrix => "id-matrix",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "add" => Some(RuntimeIdent::Add),
            "sub" => Some(RuntimeIdent::Sub),
            "mul" => Some(RuntimeIdent::Mul),
            "div" => Some(RuntimeIdent::Div),
            "invert" => Some(RuntimeIdent::Invert),
            "neg" => Some(RuntimeIdent::Neg),
            "identity" => Some(RuntimeIdent::Identity),
            "not" => Some(RuntimeIdent::Not),
            "and" => Some(RuntimeIdent::And),
            "or" => Some(RuntimeIdent::Or),
            "xor" => Some(RuntimeIdent::Xor),
            "conj" => Some(RuntimeIdent::Conj),
            "transpose" => Some(RuntimeIdent::Transpose),
            "norm-sq" => Some(RuntimeIdent::NormSq),
            "exp" => Some(RuntimeIdent::Exp),
            "sin" => Some(RuntimeIdent::Sin),
            "cos" => Some(RuntimeIdent::Cos),
            "tan" => Some(RuntimeIdent::Tan),
            "sinh" => Some(RuntimeIdent::Sinh),
            "cosh" => Some(RuntimeIdent::Cosh),
            "tanh" => Some(RuntimeIdent::Tanh),
            "ln" => Some(RuntimeIdent::Ln),
            "sqrt" => Some(RuntimeIdent::Sqrt),
            "arcsin" => Some(RuntimeIdent::ArcSin),
            "arccos" => Some(RuntimeIdent::ArcCos),
            "arctan" => Some(RuntimeIdent::ArcTan),
            "arsinh" => Some(RuntimeIdent::ArSinh),
            "arcosh" => Some(RuntimeIdent::ArCosh),
            "artanh" => Some(RuntimeIdent::ArTanh),

            "matrix" => Some(RuntimeIdent::Matrix),
            "concrete" => Some(RuntimeIdent::ConcreteMatrix),
            "?width" => Some(RuntimeIdent::UnknownWidthMatrix),
            "?height" => Some(RuntimeIdent::UnknownHeightMatrix),
            "?size" => Some(RuntimeIdent::UnknownSizeMatrix),
            "id-matrix" => Some(RuntimeIdent::IdentityMatrix),

            _ => None,
        }
    }
}
