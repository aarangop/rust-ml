/// Builder implementation for LogisticRegression models.
///
/// This module provides a builder pattern implementation for creating LogisticRegression
/// models with customizable configurations, such as feature count and activation function.
/// The builder allows for fluent API-style configuration of model parameters before construction.
use crate::builders::builder::Builder;
use crate::core::activations::activation_functions::ActivationFn;
use crate::core::error::ModelError;
use crate::core::types::{Matrix, Vector};
use crate::model::logistic_regression::LogisticRegression;

// use crate::model::logistic_regression::LogisticRegression;

/// Builder for creating LogisticRegression models with customizable configurations.
///
/// The LogisticRegressionBuilder provides methods to configure the properties of a
/// LogisticRegression model before it is instantiated, following the Builder design pattern.
///
/// # Fields
///
/// * `n_features` - The number of input features for the logistic regression model
/// * `activation_fn` - The activation function to use (default: Sigmoid)
///
/// # Examples
///
/// ```
/// use rust_ml::model::logistic_regression::LogisticRegression;
/// use rust_ml::core::activations::activation_functions::ActivationFn;
/// use rust_ml::builders::builder::Builder;
///
/// // Create a logistic regression model with 4 features and sigmoid activation
/// let model = LogisticRegression::builder()
///     .n_features(4)
///     .activation_function(ActivationFn::Sigmoid)
///     .build()
///     .unwrap();
/// ```
pub struct LogisticRegressionBuilder {
    /// Number of input features for the model
    n_features: usize,
    /// Activation function to be used in the model
    activation_fn: ActivationFn,
    /// Classification threshold
    threshold: f64,
}

impl LogisticRegressionBuilder {
    /// Creates a new LogisticRegressionBuilder with default parameter values.
    ///
    /// The default configuration uses 1 feature and the Sigmoid activation function.
    ///
    /// # Returns
    ///
    /// * `Self` - A new LogisticRegressionBuilder instance with default settings
    pub fn new() -> Self {
        Self {
            n_features: 1,
            activation_fn: ActivationFn::Sigmoid,
            threshold: 0.5,
        }
    }

    /// Sets the number of input features for the logistic regression model.
    ///
    /// # Arguments
    ///
    /// * `n_features` - The number of independent variables (features) in the input data
    ///
    /// # Returns
    ///
    /// * `Self` - Builder instance with updated feature count for method chaining
    pub fn n_features(mut self, n_features: usize) -> Self {
        self.n_features = n_features;
        self
    }

    /// Sets the activation function to use in the logistic regression model.
    ///
    /// While sigmoid is the traditional activation function for logistic regression,
    /// other functions like ReLU or Tanh could be used for specific use cases.
    ///
    /// # Arguments
    ///
    /// * `activation_function` - The activation function to use
    ///
    /// # Returns
    ///
    /// * `Self` - Builder instance with updated activation function for method chaining
    pub fn activation_function(mut self, activation_function: ActivationFn) -> Self {
        self.activation_fn = activation_function;
        self
    }
    /// Sets the classification threshold for the logistic regression model.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold value for classifying predictions (between 0 and 1)
    ///
    /// # Returns
    ///
    /// * `Self` - Builder instance with updated threshold for method chaining
    pub fn threshold(mut self, threshold: f64) -> Self {
        if threshold < 0.0 || threshold > 1.0 {
            panic!("Threshold must be between 0 and 1");
        }
        self.threshold = threshold;
        self
    }
}

impl Builder<LogisticRegression, Matrix, Vector> for LogisticRegressionBuilder {
    /// Builds and returns a new LogisticRegression model with the configured parameters.
    ///
    /// # Returns
    ///
    /// * `Result<LogisticRegression, ModelError>` - A new LogisticRegression instance with the
    ///   specified configuration, or an error if construction fails
    fn build(&self) -> Result<LogisticRegression, ModelError> {
        Ok(LogisticRegression::new(
            self.n_features,
            self.activation_fn,
            self.threshold,
        ))
    }
}
