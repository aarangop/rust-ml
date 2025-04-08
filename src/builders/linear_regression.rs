use crate::builders::builder::Builder;
use crate::core::error::ModelError;
use crate::core::types::{Matrix, Vector};
use crate::model::linear_regression::LinearRegression;

pub struct LinearRegressionBuilder {
    n_x: usize,
}

impl Builder<LinearRegression, Matrix, Vector> for LinearRegressionBuilder {
    fn build(&self) -> Result<LinearRegression, ModelError> {
        LinearRegression::new(self.n_x)
    }
}

impl LinearRegressionBuilder {
    pub fn new() -> Self {
        Self { n_x: 0 }
    }

    pub fn n_input_features(&mut self, n_x: usize) -> &mut Self {
        self.n_x = n_x;
        self
    }
}
