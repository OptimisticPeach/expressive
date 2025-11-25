#![feature(try_trait_v2, try_trait_v2_residual)]

use crate::eval::{NodeId, scalar::Scalar};

pub mod ast;
pub mod errors;
pub mod eval;

enum EvalResult {
    // 1 -> 1
    // a -> a
    // x -> 3
    // (+ a b) -> (+ a b)
    Unchanged,

    // For example, (mul (matrix ...) (matrix ...))
    // can become (matrix ...) with expanded contents.
    Become(NodeId),

    DependsOn(Vec<NodeId>),
}

pub fn eval(program: ast::Ast) -> Result<ast::Ast, errors::MathError> {
    let (head, mut env) = program.into_eval_ast();

    let mut stack = vec![head];

    // how many items after ourself do we depend on
    let mut dependents = vec![];

    let mut free_vars = vec![];

    Ok(ast::Ast::Literal(Scalar::ZERO))
}
