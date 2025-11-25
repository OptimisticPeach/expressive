use std::collections::HashMap;

use crate::errors::Result;

pub(crate) mod function;
pub(crate) mod matrix;
pub(crate) mod runtime_symbols;
pub mod scalar;
pub(crate) mod value;

use runtime_symbols::RuntimeIdent;
use value::EvalAst;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NodeId(usize);

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct UserIdent(usize);

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Ident {
    Builtin(RuntimeIdent),
    User(UserIdent),
}

pub struct Env {
    pub str_to_id: HashMap<String, Ident>,

    pub user_id_to_str: Vec<String>,

    pub scopes: Vec<HashMap<Ident, EvalAst>>,

    pub nodes: HashMap<NodeId, EvalAst>,
    node_generation: usize,
}

impl Default for Env {
    fn default() -> Self {
        todo!()
    }
}

impl Env {
    pub fn empty() -> Self {
        Self {
            str_to_id: HashMap::new(),
            user_id_to_str: Vec::new(),

            scopes: Vec::new(),

            nodes: HashMap::new(),
            node_generation: 0,
        }
    }

    pub fn ident(&mut self, s: String) -> Ident {
        *self.str_to_id.entry(s.clone()).or_insert_with(|| {
            let ident = Ident::User(UserIdent(self.user_id_to_str.len()));

            self.user_id_to_str.push(s);

            ident
        })
    }

    pub fn scope(&'_ mut self) -> EnvScope<'_> {
        EnvScope::new(self)
    }

    pub fn add_node(&mut self, node: EvalAst) -> NodeId {
        let id = self.node_generation;

        self.node_generation += 1;

        self.nodes.insert(NodeId(id), node);

        NodeId(id)
    }
}

pub struct EnvScope<'a>(&'a mut Env);

impl<'a> EnvScope<'a> {
    fn new(scope: &'a mut Env) -> Self {
        scope.scopes.push(HashMap::new());

        Self(scope)
    }

    pub fn scope<'b: 'a>(&'b mut self) -> EnvScope<'b> {
        Self::new(self.0)
    }

    pub fn insert(&mut self, ident: Ident, ast: EvalAst) -> Result<()> {
        match ident {
            Ident::Builtin(_) => Err(crate::errors::MathError::CannotOverwriteRuntime)?,
            Ident::User(_) => self
                .0
                .scopes
                .last_mut()
                .expect("at least once scope should always be present")
                .insert(ident, ast),
        };

        Ok(())
    }

    pub fn retrieve(&mut self, ident: Ident) -> Result<EvalAst> {
        for scope in self.0.scopes.iter().rev() {
            if let Some(x) = scope.get(&ident) {
                return Ok(x.clone());
            }
        }

        Err(crate::errors::MathError::UnknownVariable)
    }
}

impl<'a> Drop for EnvScope<'a> {
    fn drop(&mut self) {
        self.0.scopes.pop();
    }
}
