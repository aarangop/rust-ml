use crate::bench::regression_metrics::RegressionMetrics;
use crate::builders::linear_regression::LinearRegressionBuilder;
use crate::core::error::ModelError;
use crate::core::param_manager::ParamManager;
use crate::core::types::{Matrix, ModelParams, Vector};
use crate::model::core::base::{BackwardPropagation, BaseModel, ForwardPropagation};
use crate::model::core::optimizable_model::OptimizableModel;
use crate::model::core::regression_model::RegressionModel;
use ndarray::{Array1, Array2, ArrayView, Ix1, IxDyn};

/// A Linear Regression model that fits the equation y = Wx + b
///
/// This model predicts a continuous value output based on input features
/// using a linear relationship where W is the weight vector and b is the bias.
/// Linear regression is one of the most fundamental machine learning algorithms
/// used for predicting numerical values by establishing a linear relationship
/// between the independent variables (features) and the dependent variable (target).
///
/// The model minimizes the Mean Squared Error (MSE) between predictions and actual values.
#[derive(Debug)]
pub struct LinearRegression {
    /// The weights vector (W) for the linear model representing the coefficients
    /// for each feature in the input data
    weights: Vector,
    /// The bias term (b) for the linear model representing the y-intercept
    /// of the linear equation
    bias: Vector,
}

impl LinearRegression {
    /// Creates a new LinearRegression model with zero-initialized weights and bias
    ///
    /// Initializes a linear regression model with all weights set to zero and bias
    /// set to zero. This creates an untrained model that can be later optimized
    /// using various training algorithms.
    ///
    /// # Arguments
    /// * `n_x` - Number of input features in the dataset
    ///
    /// # Returns
    /// * `Result<Self, ModelError>` - A new LinearRegression instance or an error
    /// if the initialization fails
    ///
    /// # Example
    /// ```
    /// use rust_ml::model::linear_regression::LinearRegression;
    ///
    /// // Create a linear regression model with 3 input features
    /// let model = LinearRegression::new(3).unwrap();
    /// ```
    pub fn new(n_x: usize) -> Result<Self, ModelError> {
        let weights = Array1::<f64>::zeros(n_x);
        let bias = Array1::<f64>::from_elem(1, 0.0);
        Ok(Self { weights, bias })
    }
    /// Returns a builder for creating a LinearRegression with custom configuration
    ///
    /// The builder pattern allows for more flexible initialization of the model
    /// with various optional parameters and configurations.
    ///
    /// # Returns
    /// * `LinearRegressionBuilder` - A builder for LinearRegression with fluent API
    ///
    /// # Example
    /// ```
    /// use rust_ml::model::linear_regression::LinearRegression;
    ///
    /// // Create a model using the builder
    /// let model = LinearRegression::builder()
    ///     .with_feature_count(4)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder() -> LinearRegressionBuilder {
        LinearRegressionBuilder::new()
    }
}

impl ParamManager for LinearRegression {
    /// Gets all model parameters
    ///
    /// Retrieves all trainable parameters (weights and bias) from the model
    /// as a key-value store where keys are "W" for weights and "b" for bias.
    ///
    /// # Returns
    /// * `ModelParams` - HashMap containing the model parameters
    fn get_params(&self) -> ModelParams {
        let mut params = ModelParams::new();
        params.insert("W".to_string(), self.weights.clone().into_dyn());
        params.insert("b".to_string(), self.bias.clone().into_dyn());
        params
    }

    /// Updates the model parameters
    ///
    /// Sets the model parameters (weights and bias) based on the provided values.
    /// This is typically used during model optimization/training to update the
    /// parameters based on calculated gradients.
    ///
    /// # Arguments
    /// * `params` - HashMap containing the parameters to update with keys "W" and "b"
    fn update_params(&mut self, params: ModelParams) {
        if params.contains_key("W") {
            let weights = params
                .get("W")
                .unwrap()
                .clone()
                .into_dimensionality::<Ix1>()
                .unwrap();
            self.weights = weights;
        }
        if params.contains_key("b") {
            self.bias = params
                .get("b")
                .unwrap()
                .clone()
                .into_dimensionality::<Ix1>()
                .unwrap();
        }
    }

    /// Gets a specific model parameter by key
    ///
    /// Retrieves a specific parameter from the model based on its key identifier.
    ///
    /// # Arguments
    /// * `key` - The key of the parameter to retrieve ("W" for weights, "b" for bias)
    ///
    /// # Returns
    /// * `Result<ArrayView<f64, IxDyn>, ModelError>` - The parameter value or an error
    /// if the key is invalid
    fn get_param(&self, key: &str) -> Result<ArrayView<f64, IxDyn>, ModelError> {
        match key {
            "W" => Ok(self.weights.view().into_dyn()),
            "b" => Ok(self.bias.view().into_dyn()),
            _ => Err(ModelError::InvalidParameter(key.to_string())),
        }
    }
}

impl OptimizableModel<Matrix, Vector> for LinearRegression {}

impl ForwardPropagation<Matrix, Vector> for LinearRegression {
    /// Computes the forward pass of linear regression: y = Wx + b
    ///
    /// Performs the forward computation step for linear regression by calculating
    /// the linear combination of inputs and weights, then adding the bias term.
    /// The formula is: y_hat = W^T * x + b
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m) where n_x is the number of features
    ///         and m is the number of examples/batch size
    ///
    /// # Returns
    /// * `Result<(Vector, Option<ModelParams>), ModelError>` - Tuple containing:
    ///   - The predicted values (y_hat)
    ///   - Option<ModelParams>: None for LinearRegression (no cache needed)
    ///
    /// # Errors
    /// Returns a ModelError if the dimensions of the input are incompatible with
    /// the model's weights.
    fn compute_forward_propagation(
        &self,
        x: &Matrix,
    ) -> Result<(Vector, Option<ModelParams>), ModelError> {
        // Check that dimensions are compatible.
        if x.shape()[0] != self.weights.shape()[0] {
            return Err(ModelError::Dimensions {
                dim1: x.shape().to_vec(),
                dim2: self.weights.shape().to_vec(),
            });
        }

        let w = self
            .get_param("W")?
            .clone()
            .into_dimensionality::<Ix1>()
            .unwrap();
        let b = self.get_param("b")?.clone()[0];
        // Weights param has shape (n_x, 1), while x is (n_x, m).
        // In order to get the linear combination we need to compute w.T.dot(x) or w.T @ X
        let linear_term = w.t().dot(x);
        Ok((linear_term + b, None))
    }
}

impl BackwardPropagation<Array2<f64>, Array1<f64>> for LinearRegression {
    /// Computes the backward pass to get gradients for linear regression
    ///
    /// Calculates the gradients of the cost function with respect to the weights and bias.
    /// For linear regression with MSE loss, the gradients are:
    /// - dW = (1/m) * X * (y_hat - y)^T
    /// - db = (1/m) * sum(y_hat - y)
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m,)
    /// * `output_gradient` - Gradient from the output layer (typically y_hat - y)
    /// * `cache` - Optional cached values from forward pass (not used in LinearRegression)
    ///
    /// # Returns
    /// * `Result<ModelParams, ModelError>` - HashMap containing gradients "dW" and "db"
    fn compute_backward_propagation(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        output_gradient: &Vector,
        cache: Option<ModelParams>,
    ) -> Result<ModelParams, ModelError> {
        // Get size of training set
        let m = x.shape()[1] as f64;
        let _ = cache; // Unused in this implementation

        // Compute gradients
        let dz = output_gradient.clone();
        let dw = x.dot(&dz.t()) / m;
        let db = dz.sum() / m;

        // Create gradients hashmap
        let mut gradients = ModelParams::new();
        gradients.insert("dW".to_string(), dw.into_dyn());
        gradients.insert("db".to_string(), Array1::<f64>::from_elem(1, db).into_dyn());
        Ok(gradients)
    }
}

impl BaseModel<Matrix, Vector> for LinearRegression {
    /// Predicts output values for given input features
    ///
    /// Applies the linear regression model to make predictions on new data.
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    ///
    /// # Returns
    /// * `Result<Vector, ModelError>` - Predicted values of shape (m,)
    fn predict(&self, x: &Matrix) -> Result<Vector, ModelError> {
        let (y_hat, _) = self.compute_forward_propagation(x)?;
        Ok(y_hat)
    }

    /// Computes the Mean Squared Error cost between predictions and target values
    ///
    /// Calculates the cost using the formula: J = (1/2m) * Σ(y_hat - y)²
    /// where m is the number of examples
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m,)
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The computed cost value
    fn compute_cost(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let (y_hat, _) = self.compute_forward_propagation(x)?;
        let m = x.len() as f64;
        Ok((&y_hat - y).mapv(|v| v.powi(2)).sum() / (2.0 * m))
    }

    /// Computes the gradient of the cost with respect to the output predictions
    ///
    /// For Mean Squared Error, the gradient dJ/dy_hat is: (1/m) * (y_hat - y)
    /// where m is the number of examples
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m,)
    ///
    /// # Returns
    /// * `Result<Vector, ModelError>` - The gradient of the cost function
    fn compute_gradient(&self, x: &Matrix, y: &Vector) -> Result<Vector, ModelError> {
        let (y_hat, _) = self.compute_forward_propagation(x)?;
        let m = x.len() as f64;
        let dy = (y_hat - y).sum() / m;
        // The compute gradient returns an array, so we'll return a 'broadcast' array with the value
        // of `dy`.
        Ok(Array1::from_elem(y.len(), dy))
    }
}

impl RegressionModel<Matrix, Vector> for LinearRegression {
    /// Calculates the Mean Squared Error between predictions and target values
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m,)
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The MSE value
    fn mse(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_hat = self.predict(x)?;
        let m = x.len() as f64;
        Ok((&y_hat - y).mapv(|v| v.powi(2)).sum() / m)
    }

    /// Calculates the Root Mean Squared Error between predictions and target values
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m,)
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The RMSE value (square root of MSE)
    fn rmse(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_hat = self.predict(x)?;
        let m = x.len() as f64;
        let rmse = ((&y_hat - y).mapv(|v| v.powi(2)).sum() / m).sqrt();
        Ok(rmse)
    }

    /// Calculates the R-squared (coefficient of determination) for the model
    ///
    /// R² represents the proportion of variance in the dependent variable
    /// that is predictable from the independent variables.
    /// R² = 1 - (MSE / Variance of y)
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m,)
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The R² value between 0 and 1
    fn r2(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_hat = self.predict(x)?;
        let numerator = self.mse(x, y)?;
        let denominator = (y_hat - y.mean().unwrap()).powi(2).sum();
        Ok(1.0 - numerator / denominator)
    }

    /// Computes a complete set of regression metrics for model evaluation
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m,)
    ///
    /// # Returns
    /// * `Result<RegressionMetrics, ModelError>` - Struct containing MSE, RMSE, and R²
    fn compute_metrics(&self, x: &Matrix, y: &Vector) -> Result<RegressionMetrics, ModelError> {
        let mse = self.mse(x, y)?;
        let rmse = self.rmse(x, y)?;
        let r2 = self.r2(x, y)?;
        Ok(RegressionMetrics { mse, rmse, r2 })
    }
}

#[cfg(test)]
mod forward_propagation_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_compute_forward_propagation_single_feature() {
        let mut lr = LinearRegression::new(1).unwrap();

        // Set weights and bias explicitly
        let mut params = ModelParams::new();
        params.insert("W".to_string(), array![2.0].into_dyn());
        params.insert("b".to_string(), array![1.0].into_dyn());
        lr.update_params(params);

        let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let (output, cache) = lr.compute_forward_propagation(&x).unwrap();

        let expected = Array1::from_vec(vec![3.0, 5.0, 7.0]); // 1*2 + 1, 2*2 + 1, 3*2 + 1
        assert_abs_diff_eq!(
            output.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            epsilon = 1e-10
        );
        assert!(cache.is_none());
    }

    #[test]
    fn test_compute_forward_propagation_multiple_features() {
        let mut lr = LinearRegression::new(3).unwrap();

        // Set weights and bias explicitly
        let mut params = ModelParams::new();
        params.insert("W".to_string(), array![1.0, 2.0, 3.0].into_dyn());
        params.insert("b".to_string(), array![0.5].into_dyn());
        lr.update_params(params);

        let x = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let (output, cache) = lr.compute_forward_propagation(&x).unwrap();

        // (1.0 * 0.1 + 2.0 * 0.3 + 3.0 * 0.5) + 0.5 = 2.7
        // (1.0 * 0.2 + 2.0 * 0.4 + 3.0 * 0.6) + 0.5 = 3.3
        let expected = Array1::from_vec(vec![2.7, 3.3]);
        assert_abs_diff_eq!(
            output.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            epsilon = 1e-10
        );
        assert!(cache.is_none());
    }

    #[test]
    fn test_compute_forward_propagation_zeros() {
        let lr = LinearRegression::new(2).unwrap();

        // With default zero weights and bias
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (output, cache) = lr.compute_forward_propagation(&x).unwrap();

        let expected = Array1::from_vec(vec![0.0, 0.0]); // All zeros with zero weights and bias
        assert_abs_diff_eq!(
            output.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            epsilon = 1e-10
        );
        assert!(cache.is_none());
    }
}

#[cfg(test)]
mod backward_propagation_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_compute_backward_propagation_single_feature() {
        let lr = LinearRegression::new(1).unwrap();

        // Input data
        let x = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0]);

        // Output gradient: this is y_hat - y, assume it's given for the test
        let output_gradient = Array1::from_vec(vec![0.1, -0.2, 0.3]);

        // Perform backward propagation
        let gradients = lr
            .compute_backward_propagation(&x, &y, &output_gradient, None)
            .unwrap();

        // Check if dW gradient is computed correctly
        let dw = gradients.get("dW").unwrap();
        let expected_dw = array![0.1, -0.2, 0.3].into_dyn();
        assert_abs_diff_eq!(
            dw.as_slice().unwrap(),
            expected_dw.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_compute_backward_propagation_multiple_features() {
        let lr = LinearRegression::new(3).unwrap();

        // Input data
        let x = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0]);

        // Output gradient: this is y_hat - y, assume it's given for the test
        let output_gradient = Array1::from_vec(vec![0.1, -0.2]);

        // Perform backward propagation
        let gradients = lr
            .compute_backward_propagation(&x, &y, &output_gradient, None)
            .unwrap();

        // Check if dW gradient is computed correctly
        let dw = gradients.get("dW").unwrap();
        let expected_dw = Array1::from_vec(vec![0.1, -0.2]).into_dyn();
        assert_abs_diff_eq!(
            dw.as_slice().unwrap(),
            expected_dw.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }
}

#[cfg(test)]
mod linear_regression_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_predict() {
        let mut lr = LinearRegression::new(2).unwrap();

        // Setting explicit weights and bias
        let mut params = ModelParams::new();
        params.insert("W".to_string(), array![2.0, 3.0].into_dyn());
        params.insert("b".to_string(), array![0.5].into_dyn());
        lr.update_params(params);

        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y_hat = lr.predict(&x).unwrap();

        let expected = Array1::from_vec(vec![8.5, 14.5]); // (1*2 + 2*3 + 0.5), (3*2 + 4*3 + 0.5)
        assert_abs_diff_eq!(
            y_hat.as_slice().unwrap(),
            expected.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_compute_cost() {
        let mut lr = LinearRegression::new(2).unwrap();

        // Setting weights and bias explicitly
        let mut params = ModelParams::new();
        params.insert("W".to_string(), array![1.0, 1.0].into_dyn());
        params.insert("b".to_string(), array![0.0].into_dyn());
        lr.update_params(params);

        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 7.0]);
        let cost = lr.compute_cost(&x, &y).unwrap();

        let expected_cost = 0.0; // Perfect prediction
        assert_abs_diff_eq!(cost, expected_cost, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_gradient() {
        let mut lr = LinearRegression::new(2).unwrap();

        // Setting weights and bias explicitly
        let mut params = ModelParams::new();
        params.insert("W".to_string(), array![1.0, 1.0].into_dyn());
        params.insert("b".to_string(), array![0.0].into_dyn());
        lr.update_params(params);

        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 7.0]);
        let gradient = lr.compute_gradient(&x, &y).unwrap();

        // Gradient should be 0 (perfect prediction)
        let expected_gradient = Array1::zeros(2);
        assert_abs_diff_eq!(
            gradient.as_slice().unwrap(),
            expected_gradient.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_invalid_input_dimensions() {
        let mut lr = LinearRegression::new(3).unwrap();

        // Setting weights and bias explicitly
        let mut params = ModelParams::new();
        params.insert("W".to_string(), array![1.0, 1.0, 1.0].into_dyn());
        params.insert("b".to_string(), array![0.0].into_dyn());
        lr.update_params(params);

        // Input dimensions don't match weights
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 7.0]);
        let result = lr.compute_cost(&x, &y);

        assert!(result.is_err());
    }
}
