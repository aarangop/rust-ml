use crate::bench::regression_metrics::RegressionMetrics;
use crate::builders::linear_regression::LinearRegressionBuilder;
use crate::core::error::ModelError;
use crate::core::types::{Matrix, Scalar, Vector};
use crate::model::core::base::{BaseModel, DLModel};
use crate::model::core::param_collection::{GradientCollection, ParamCollection};
use crate::model::core::regression_model::RegressionModel;
use ndarray::{Array0, Array1, ArrayView, ArrayView0, ArrayView1};

/// A Linear Regression model that fits the equation y = W.T @ x + b
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
    pub w: Vector,
    /// The bias term (b) for the linear model representing the y-intercept
    /// of the linear equation
    pub b: Scalar,
    /// Gradient of the cost function J with respect to the weights
    dw: Vector,
    /// Gradient of the cost function J with respect to the bias
    db: Scalar,
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
    pub fn new(n_x: usize) -> Result<Self, ModelError> {
        let weights = Array1::<f64>::zeros(n_x);
        let bias = Array0::<f64>::from_elem((), 0.0);
        Ok(Self {
            w: weights,
            b: bias,
            dw: Array1::<f64>::zeros(n_x),
            db: Array0::<f64>::from_elem((), 0.0),
        })
    }
    /// Returns a builder for creating a LinearRegression with custom configuration
    ///
    /// The builder pattern allows for more flexible initialization of the model
    /// with various optional parameters and configurations.
    ///
    /// # Returns
    /// * `LinearRegressionBuilder` - A builder for LinearRegression with fluent API
    ///
    pub fn builder() -> LinearRegressionBuilder {
        LinearRegressionBuilder::new()
    }
}

impl ParamCollection for LinearRegression {
    fn get<D: ndarray::Dimension>(
        &self,
        key: &str,
    ) -> Result<ndarray::ArrayView<f64, D>, ModelError> {
        match key {
            "weights" => Ok(self.w.view().into_dimensionality::<D>().unwrap()),
            "bias" => Ok(self.b.view().into_dimensionality::<D>().unwrap()),
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }

    fn set<D: ndarray::Dimension>(
        &mut self,
        key: &str,
        value: ndarray::ArrayView<f64, D>,
    ) -> Result<(), ModelError> {
        match key {
            "weights" => {
                self.w
                    .assign(&value.into_dimensionality::<ndarray::Ix1>().unwrap());
                Ok(())
            }
            "bias" => {
                self.b
                    .assign(&value.into_dimensionality::<ndarray::Ix0>().unwrap());
                Ok(())
            }
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }

    fn param_iter(&self) -> Vec<(&str, ndarray::ArrayView<f64, ndarray::IxDyn>)> {
        vec![
            ("weights", self.w.view().into_dyn()),
            ("bias", self.b.view().into_dyn()),
        ]
    }

    fn get_mut<D: ndarray::Dimension>(
        &mut self,
        key: &str,
    ) -> Result<ndarray::ArrayViewMut<f64, D>, ModelError> {
        match key {
            "weights" => Ok(self.w.view_mut().into_dimensionality::<D>().unwrap()),
            "bias" => Ok(self.b.view_mut().into_dimensionality::<D>().unwrap()),
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }
}

impl GradientCollection for LinearRegression {
    fn get_gradient<D: ndarray::Dimension>(
        &self,
        key: &str,
    ) -> Result<ArrayView<f64, D>, ModelError> {
        match key {
            "weights" => Ok(self.dw.view().into_dimensionality::<D>().unwrap()),
            "bias" => Ok(self.db.view().into_dimensionality::<D>().unwrap()),
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }

    fn set_gradient<D: ndarray::Dimension>(
        &mut self,
        key: &str,
        value: ndarray::ArrayView<f64, D>,
    ) -> Result<(), ModelError> {
        match key {
            "weights" => {
                self.dw
                    .assign(&value.into_dimensionality::<ndarray::Ix1>().unwrap());
                Ok(())
            }
            "bias" => {
                self.db
                    .assign(&value.into_dimensionality::<ndarray::Ix0>().unwrap());
                Ok(())
            }
            _ => Err(ModelError::KeyError(key.to_string())),
        }
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
        let w: ArrayView1<f64> = self.get("weights")?;
        let b: ArrayView0<f64> = self.get("bias")?;
        // For matrix-vector multiplication, transpose weights if needed or use a more specific method
        let y_hat = w.t().dot(x) + b;
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
        let y_hat = self.predict(x)?;
        let m = x.len() as f64;
        Ok((&y_hat - y).mapv(|v| v.powi(2)).sum() / (2.0 * m))
    }
}

impl DLModel<Matrix, Vector> for LinearRegression {
    fn forward(&self, input: &Matrix) -> Result<Vector, ModelError> {
        let weights: ArrayView1<f64> = self.get("weights")?;
        let bias: ArrayView0<f64> = self.get("bias")?;
        let y_hat = input.dot(&weights) + bias;
        Ok(y_hat)
    }

    fn backward(&mut self, input: &Matrix, output_grad: &Vector) -> Result<(), ModelError> {
        let batch_size = input.shape()[1] as f64;

        // Compute weights grads dw
        let dw: Vector = input.dot(&output_grad.t()) / batch_size;
        let dw: ArrayView1<f64> = dw.view();

        // Compute bias grad db
        let db = output_grad.sum() / batch_size;
        let binding = Scalar::from_elem((), db);
        let db: ArrayView0<f64> = binding.view();

        // Update gradients
        self.set_gradient("weights", dw)?;
        self.set_gradient("bias", db)?;

        Ok(())
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
    fn compute_output_gradient(&self, x: &Matrix, y: &Vector) -> Result<Vector, ModelError> {
        let y_hat = self.forward(x)?;
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
