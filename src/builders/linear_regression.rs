/// Builder implementation for LinearRegression models.
///
/// This module provides a builder pattern implementation for creating LinearRegression
/// models with customizable configurations. The builder allows for fluent API-style
/// configuration of model parameters before construction.
use crate::builders::builder::Builder;
use crate::core::error::ModelError;
use crate::core::types::{Matrix, Vector};
use crate::model::linear_regression::LinearRegression;

/// Builder for creating LinearRegression models with customizable configurations.
///
/// The LinearRegressionBuilder provides methods to configure the properties of a
/// LinearRegression model before it is instantiated, following the Builder design pattern.
///
/// # Fields
///
/// * `n_x` - The number of input features for the linear regression model
///
/// # Examples
///
pub struct LinearRegressionBuilder {
    /// Number of input features (independent variables) for the model
    n_x: usize,
}

impl Builder<LinearRegression, Matrix, Vector> for LinearRegressionBuilder {
    /// Builds and returns a new LinearRegression model with the configured parameters.
    ///
    /// # Returns
    ///
    /// * `Result<LinearRegression, ModelError>` - A new LinearRegression instance with the
    ///   specified number of input features, or an error if construction fails
    fn build(&self) -> Result<LinearRegression, ModelError> {
        LinearRegression::new(self.n_x)
    }
}

impl LinearRegressionBuilder {
    /// Creates a new LinearRegressionBuilder with default parameter values.
    ///
    /// The default number of input features is set to 0 and must be configured
    /// before building the model.
    ///
    /// # Returns
    ///
    /// * `Self` - A new LinearRegressionBuilder instance with default settings
    pub fn new() -> Self {
        Self { n_x: 0 }
    }

    /// Sets the number of input features for the linear regression model.
    ///
    /// # Arguments
    ///
    /// * `n_x` - The number of independent variables (features) in the input data
    ///
    /// # Returns
    ///
    /// * `&mut Self` - Reference to self for method chaining
    pub fn n_input_features(&mut self, n_x: usize) -> &mut Self {
        self.n_x = n_x;
        self
    }
}

impl Default for LinearRegressionBuilder {
    /// Creates a new LinearRegressionBuilder with default parameter values.
    ///
    /// The default number of input features is set to 0 and must be configured
    /// before building the model.
    ///
    /// # Returns
    ///
    /// * `Self` - A new LinearRegressionBuilder instance with default settings
    fn default() -> Self {
        Self::new()
    }
}
