use std::collections::HashMap;

use crate::Value;

pub mod ast;

use ast::Ast;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Ident(usize);

pub struct Env {
    pub str_to_id: HashMap<String, Ident>,
    pub id_to_str: Vec<String>,

    pub globals: Vec<HashMap<Ident, Ast>>,

    pub scopes: Vec<HashMap<Ident, Value>>,
}

impl Env {
    pub fn ident(&mut self, s: String) -> Ident {
        *self
            .str_to_id
            .entry(s)
            .or_insert_with(|| Ident(self.id_to_str.len()))
    }

    pub fn scope(&'_ mut self) -> EnvScope<'_> {
        EnvScope::new(&mut self.scopes, &mut self.globals)
    }
}

pub struct EnvScope<'a> {
    scopes: &'a mut Vec<HashMap<Ident, Value>>,

    globals: &'a mut Vec<HashMap<Ident, ast::Ast>>,
}

impl<'a> EnvScope<'a> {
    fn new(
        scopes: &'a mut Vec<HashMap<Ident, Value>>,
        globals: &'a mut Vec<HashMap<Ident, Ast>>,
    ) -> Self {
        scopes.push(HashMap::new());

        Self { scopes, globals }
    }

    pub fn scope<'b: 'a>(&'b mut self) -> EnvScope<'b> {
        self.scopes.push(HashMap::new());

        Self::new(self.scopes, self.globals)
    }

    pub fn retrieve(&self, ident: Ident) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            let result = scope.get(&ident);

            if result.is_some() {
                return result.cloned();
            }
        }

        None
    }
}

impl<'a> Drop for EnvScope<'a> {
    fn drop(&mut self) {
        self.scopes.pop();
    }
}
