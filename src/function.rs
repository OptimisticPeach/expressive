use std::collections::HashMap;

use crate::Value;
use crate::errors::Result;
use crate::function::ast::Function;

pub mod ast;

use ast::Ast;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Ident(usize);

pub struct Env {
    pub str_to_id: HashMap<String, Ident>,
    pub id_to_str: Vec<String>,

    pub globals: HashMap<Ident, Global>,

    pub scopes: Vec<HashMap<Ident, Value>>,
}

impl Env {
    pub fn empty() -> Self {
        Self {
            str_to_id: HashMap::new(),
            id_to_str: Vec::new(),

            globals: HashMap::new(),

            scopes: Vec::new(),
        }
    }

    pub fn define_variable(&mut self, name: String, ast: Ast) {
        let ident = self.ident(name);
        self.globals.insert(ident, Global::Variable(ast));
    }

    pub fn define_builtin(&mut self, name: String, function: Function) {
        let ident = self.ident(name);
        self.globals.insert(ident, Global::Function(function));
    }

    pub fn ident(&mut self, s: String) -> Ident {
        *self
            .str_to_id
            .entry(s)
            .or_insert_with(|| Ident(self.id_to_str.len()))
    }

    pub fn scope(&'_ mut self) -> EnvScope<'_> {
        EnvScope::new(&mut self.scopes, &self.globals)
    }
}

pub struct EnvScope<'a> {
    scopes: &'a mut Vec<HashMap<Ident, Value>>,

    globals: &'a HashMap<Ident, Global>,
}

impl<'a> EnvScope<'a> {
    fn new(
        scopes: &'a mut Vec<HashMap<Ident, Value>>,
        globals: &'a HashMap<Ident, Global>,
    ) -> Self {
        scopes.push(HashMap::new());

        Self { scopes, globals }
    }

    pub fn scope<'b: 'a>(&'b mut self) -> EnvScope<'b> {
        self.scopes.push(HashMap::new());

        Self::new(self.scopes, self.globals)
    }

    pub fn retrieve(&mut self, ident: Ident) -> Result<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(x) = scope.get(&ident) {
                return Ok(x.clone());
            }
        }

        if let Some(ast) = self.globals.get(&ident) {
            ast.eval(self)
        } else {
            Err(crate::errors::MathError::UnknownVariable)
        }
    }
}

impl<'a> Drop for EnvScope<'a> {
    fn drop(&mut self) {
        self.scopes.pop();
    }
}

pub enum Global {
    Variable(Ast),
    Function(Function),
}

impl Global {
    pub fn eval(&self, scope: &mut EnvScope<'_>) -> Result<Value> {
        match self {
            Global::Variable(ast) => ast.eval(scope),
            Global::Function(function) => Ok(Value::Lambda(function.clone())),
        }
    }
}
