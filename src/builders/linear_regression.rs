use crate::builders::builder::Builder;
use crate::core::error::ModelError;
use crate::core::types::{DefInput, DefOutput};
use crate::model::linear_regression::LinearRegression;
use crate::model::ml_model::OptimizableModel;

pub struct LinearRegressionBuilder {
    n_x: usize,
}

impl Builder<DefInput, DefOutput> for LinearRegressionBuilder {
    fn build(&self) -> Result<Box<dyn OptimizableModel<DefInput, DefOutput>>, ModelError> {
        let model = LinearRegression::new(self.n_x)?;
        Ok(Box::new(model))
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
