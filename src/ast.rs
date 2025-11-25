use crate::eval::{Env, NodeId, scalar::Scalar, value::EvalAst};

/// User-facing AST
///
/// This is meant to be very very similar to S-Expressions.
///
/// Note that during evaluation, sexprs aren't the only representation.
pub enum Ast {
    /// 1, 2 i, etc.
    Literal(Scalar),

    /// Symbols.
    Symbol(String),

    /// Applications
    FunctionApplication(Box<Ast>, Vec<Ast>),
}

impl Ast {
    pub(crate) fn into_eval_ast(self) -> (NodeId, Env) {
        let mut env = Env::default();

        let result = self.eval_ast_inner(&mut env);

        (result, env)
    }

    fn eval_ast_inner(self, env: &mut Env) -> NodeId {
        let node = match self {
            Ast::Literal(scalar) => EvalAst::Scalar(scalar),
            Ast::Symbol(sym) => EvalAst::Symbol(env.ident(sym)),
            Ast::FunctionApplication(ast, asts) => EvalAst::Application(
                ast.eval_ast_inner(env),
                asts.into_iter().map(|x| x.eval_ast_inner(env)).collect(),
            ),
        };

        env.add_node(node)
    }
}
