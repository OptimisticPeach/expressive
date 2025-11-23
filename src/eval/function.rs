use crate::eval::runtime_symbols::RuntimeIdent;

#[derive(Clone, Debug, PartialEq)]
pub enum Function {
    Builtin(RuntimeIdent),
    // Lambda
}
