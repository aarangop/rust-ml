use crate::builders::builder::Builder;
use crate::core::activations::activation_functions::ActivationFn;
use crate::core::error::ModelError;
use crate::core::types::{Matrix, Vector};
use crate::model::logistic_regression::LogisticRegression;

pub struct LogisticRegressionBuilder {
    n_features: usize,
    activation_fn: ActivationFn,
}

impl LogisticRegressionBuilder {
    pub fn new() -> Self {
        Self {
            n_features: 1,
            activation_fn: ActivationFn::Sigmoid,
        }
    }

    pub fn n_features(mut self, n_features: usize) -> Self {
        self.n_features = n_features;
        self
    }

    pub fn activation_function(mut self, activation_function: ActivationFn) -> Self {
        self.activation_fn = activation_function;
        self
    }
}

impl Builder<LogisticRegression, Matrix, Vector> for LogisticRegressionBuilder {
    fn build(&self) -> Result<LogisticRegression, ModelError> {
        Ok(LogisticRegression::new(self.n_features, self.activation_fn))
    }
}
