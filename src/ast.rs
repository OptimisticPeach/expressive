use crate::eval::scalar::Scalar;

/// User-facing AST
///
/// This is meant to be very very similar to S-Expressions.
///
/// Note that during evaluation, sexprs aren't the only representation.
pub enum Ast {
    /// 1, 2 i, etc.
    Literal(Scalar),

    /// Symbols which we haven't tried to resolve yet.
    Symbol(String),

    /// Applications
    FunctionApplication(Vec<Ast>),
}

impl Ast {
    pub(crate) fn to_inner_ast(self) -> (EvalAst, Env) {}
}
