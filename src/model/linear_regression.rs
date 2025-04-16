use crate::bench::regression_metrics::RegressionMetrics;
use crate::builders::linear_regression::LinearRegressionBuilder;
use crate::core::error::ModelError;
use crate::core::types::{Matrix, Scalar, Vector};
use crate::model::core::base::{BaseModel, OptimizableModel};
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
    fn get<D: ndarray::Dimension>(&self, key: &str) -> Result<ArrayView<f64, D>, ModelError> {
        match key {
            "weights" => Ok(self.w.view().into_dimensionality::<D>().unwrap()),
            "bias" => Ok(self.b.view().into_dimensionality::<D>().unwrap()),
            _ => Err(ModelError::KeyError(key.to_string())),
        }
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

    fn set<D: ndarray::Dimension>(
        &mut self,
        key: &str,
        value: ArrayView<f64, D>,
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

    fn param_iter(&self) -> Vec<(&str, ArrayView<f64, ndarray::IxDyn>)> {
        vec![
            ("weights", self.w.view().into_dyn()),
            ("bias", self.b.view().into_dyn()),
        ]
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
        value: ArrayView<f64, D>,
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
    /// * `Result<Vector, ModelError>` - Predicted values of shape (m, )
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
    /// * `y` - Target values of shape (m, )
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The computed cost value
    fn compute_cost(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_hat = self.predict(x)?;
        let m = x.len() as f64;
        let cost = (&y_hat - y).powi(2).sum() / (2.0 * m);
        Ok(cost)
    }
}

#[cfg(test)]
mod lr_base_model_tests {
    use crate::model::core::base::BaseModel;
    use crate::model::linear_regression::LinearRegression;
    use ndarray::{arr0, arr1, arr2};

    #[test]
    fn test_predict() {
        let mut lr = LinearRegression::new(2).unwrap();
        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let weights = arr1(&[1.0, 2.0]);
        let bias = arr0(0.0);
        // Set weights and bias explicitly.
        lr.w = weights;
        lr.b = bias;
        // y = [1.0 * 1.0 + 2.0*4.0, 1.0 * 2.0 + 2.0 * 5.0, 1.0 * 3.0 + 2.0 * 6.0]
        // y = [1 + 8, 2 + 10, 3 + 12]
        // y = [9, 12, 15]
        let y = arr1(&[9.0, 12.0, 15.0]);

        let y_hat = lr.predict(&x).unwrap();

        assert_eq!(&y, &y_hat);
    }
}

impl OptimizableModel<Matrix, Vector> for LinearRegression {
    fn forward(&self, input: &Matrix) -> Result<Vector, ModelError> {
        let y_hat = &self.w.t().dot(input) + &self.b;
        Ok(y_hat)
    }

    fn backward(&mut self, input: &Matrix, output_grad: &Vector) -> Result<(), ModelError> {
        let m = input.shape()[1] as f64;

        // Compute weights grads dw
        let dw: Vector = input.dot(&output_grad.t()) / m;
        let dw: ArrayView1<f64> = dw.view();

        // Compute bias grad db.
        // Mind that in linear regression db = dy = output_grad, and output_grad is a scalar
        // wrapped in a vector of size 1.
        let db = output_grad.sum() / m;
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
    /// where m is the number of examples.
    ///
    /// In linear regression the output gradient is a scalar. However, for compatibility
    /// with the OptimizableModel trait, we return a vector of size 1.
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m, )
    ///
    /// # Returns
    /// * `Result<Vector, ModelError>` - The gradient of the cost function
    fn compute_output_gradient(&self, x: &Matrix, y: &Vector) -> Result<Vector, ModelError> {
        let y_hat = self.forward(x)?;
        let dy = &y_hat - y;
        // The compute gradient returns an array, so we'll return a 'broadcast' array with the value
        // of `dy`.
        Ok(dy)
    }
}

#[cfg(test)]
mod lr_optimizable_model_tests {
    use ndarray::{arr0, arr1, arr2, ArrayView0, ArrayView1};

    use crate::{
        builders::builder::Builder,
        model::core::{base::OptimizableModel, param_collection::GradientCollection},
    };

    use super::LinearRegression;

    #[test]
    /// Test the forward propagation of the LinearRegression model
    fn test_forward_propagation() {
        let n_features = 2;
        let mut model = LinearRegression::builder()
            .n_input_features(n_features)
            .build()
            .unwrap();

        // Initialize weights and bias
        let weights = arr1(&[1.0, 2.0]);
        model.w.assign(&weights);
        let bias = arr0(0.0);
        model.b.assign(&bias);

        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        // y = [1.0 * 1.0 + 2.0*4.0, 1.0 * 2.0 + 2.0 * 5.0, 1.0 * 3.0 + 2.0 * 6.0]
        // y = [1 + 8, 2 + 10, 3 + 12]
        let y = arr1(&[9.0, 12.0, 15.0]);
        // Perform forward propagation
        let y_hat = model.forward(&x).unwrap();
        // Check if the predicted values match the expected values
        assert_eq!(y_hat, y);
        // Check if the weights and bias are unchanged
        assert_eq!(model.w, weights);
        assert_eq!(model.b, bias);
    }

    #[test]
    fn test_compute_output_grad() {
        let n_features = 2;
        let mut model = LinearRegression::builder()
            .n_input_features(n_features)
            .build()
            .unwrap();

        // Initialize weights and bias
        let weights = arr1(&[1.0, 2.0]);
        model.w.assign(&weights);
        let bias = arr0(0.0);
        model.b.assign(&bias);

        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let y = arr1(&[8.0, 11.0, 14.0]);
        // y_hat - y = [1.0, 1.0, 1.0]
        // dy = (y_hat - y).sum() / m
        // dy = 3.0 / 3.0 = 1
        let expected_grad = 1.0;
        // Compute the output gradient
        let output_grad = model.compute_output_gradient(&x, &y).unwrap();
        // Check if the output gradient is computed correctly

        assert_eq!(output_grad[0], expected_grad);
    }

    #[test]
    fn test_backward_propagation() {
        let n_features = 2;
        let mut model = LinearRegression::builder()
            .n_input_features(n_features)
            .build()
            .unwrap();

        // Initialize weights and bias
        let weights = arr1(&[1.0, 2.0]);
        model.w.assign(&weights);
        let bias = arr0(0.0);
        model.b.assign(&bias);

        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let y = arr1(&[8.0, 11.0, 14.0]);
        // Compute the output gradient
        let output_grad = model.compute_output_gradient(&x, &y).unwrap();
        // Perform backward propagation
        model.backward(&x, &output_grad).unwrap();
        // Compute expected gradients
        // dy = (y_hat - y) = [1.0, 1.0, 1.0]
        // expected_dw = (1/m) * (x @ dy.T)
        // expected_dw = (1/3) * ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] @ [1.0, 1.0, 1.0].T)
        // expected_dw = (1/3) * [[1.0*1.0 + 2.0*1.0 + 3.0*1.0], [4.0*1.0 + 5.0*1.0 + 6.0*1.0]]
        // expected_dw = (1/3) * [[6.0], [15.0]]
        let expected_dw = arr1(&[2.0, 5.0]);
        let expected_db = arr0(1.0);

        // Check if the gradients are computed correctly
        let dw: ArrayView1<f64> = model.get_gradient("weights").unwrap();
        let db: ArrayView0<f64> = model.get_gradient("bias").unwrap();
        assert_eq!(dw, expected_dw);
        assert_eq!(db, expected_db);
    }
}

impl RegressionModel<Matrix, Vector> for LinearRegression {
    /// Calculates the Mean Squared Error between predictions and target values
    ///
    /// # Arguments
    /// * `x` - Input features of shape (n_x, m)
    /// * `y` - Target values of shape (m, )
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
    /// * `y` - Target values of shape (m, )
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
    /// * `y` - Target values of shape (m, )
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
