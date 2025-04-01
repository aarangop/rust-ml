//! # Linear Regression Model
//! 
//! This module provides a linear regression implementation that conforms to the MLModel trait.
//! 
//! ## Overview
//! 
//! Linear regression is a simple approach for predicting a continuous value based on
//! one or more input features. The model uses the formula y = wx + b, where:
//! - w: weight vector
//! - x: input features
//! - b: bias term
//! 
//! ## Usage
//! 
//! ```
//! use crate::linear_regression::model::LinearRegression;
//! use crate::model::base::MLModel;
//! use ndarray::{arr1, arr2};
//! 
//! // Initialize a linear regression model with 2 features
//! let mut model = LinearRegression::new(2, 0.01, 100);
//! 
//! // Prepare training data
//! let x = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
//! let y = arr1(&[3.0, 7.0, 11.0]);
//! 
//! // Train the model
//! // model.fit(&x, &y, 0.01, 1000);
//! 
//! // Make predictions
//! let predictions = model.predict(&x).unwrap();
//! ```
//! 
//! ## Implementation Details
//! 
//! The model is initialized with small random weights and uses gradient descent
//! for optimization during training.

use ndarray::{Array1, Array2};
use rand::{rng, Rng};

use crate::model::{base::MLModel, error::{DimensionsError, ModelError}};
use ndarray::{arr1, arr2};

/// A simple linear regression model.
///
/// This structure represents a linear regression model with learnable parameters
/// (weights and bias). It can be trained using gradient descent.
///
/// # Fields
///
/// * `weights` - A 1D array of coefficients for each feature in the input data.
/// * `bias` - The intercept term in the linear model.
/// * `learning_rate` - Step size for gradient descent optimization.
/// * `epochs` - Number of training iterations over the entire dataset.
///
/// # Example
///
/// ```
/// let model = LinearRegression::new(features_count, 0.01, 1000);
/// model.fit(&x_train, &y_train);
/// let predictions = model.predict(&x_test);
/// ```
pub struct LinearRegression {
  weights: Array1<f64>,
  bias: f64,
}

impl LinearRegression {
  /// Creates a new `LinearRegression` model.
  ///
  /// # Arguments
  ///
  /// * `n_x` - The number of features in the input data.
  /// * `learning_rate` - The step size at each iteration while moving toward a minimum of the loss function.
  /// * `epochs` - The number of complete passes through the training dataset.
  ///
  /// # Returns
  ///
  /// A new instance of `LinearRegression` with weights initialized randomly between 0 and 0.01,
  /// bias initialized to 0, and the specified learning rate and number of epochs.
  pub fn new (n_x: usize) -> LinearRegression {
    let mut rng = rng();
    let weights = Array1::<f64>::from_iter((0..n_x)
      .map(|_| rng.random::<f64>())) * 0.01;

    LinearRegression {
      weights,
      bias: 0_f64,
    }
  }

  pub fn weights(&self) -> &Array1<f64> {
    &self.weights
  }

  pub fn bias(&self) -> f64 {
    self.bias
  }
}

#[cfg(test)]
mod linear_regression_tests {
  use super::*;
  
  #[test]
  fn test_initialization() {
    let n_x = 5;
    
    let model = LinearRegression::new(n_x);
    
    assert_eq!(model.weights.len(), n_x);
    assert_eq!(model.bias, 0.0);
    
    // Check that weights are initialized to small values
    for weight in model.weights.iter() {
      assert!(weight.abs() < 0.01);
    }
  }
}

#[cfg(test)]
mod ml_model_implementation_tests {

  use super::*;

  #[test]
  fn test_compute_cost() {
    // Initialize model
    let mut model = LinearRegression::new(2);
    // Set weights and bias 
    model.weights = arr1(&[2.0, 3.0]);
    model.bias = 1.0;
    // Set input and labeled data
    let x = arr2(&[[5.0, 1.0], [2.0, 7.0]]);
    let y = arr1(&[15.0, 20.0]);
    // The expected cost is calculated as follows:
    // J = 1/(2*m) * sum((y_hat - y)^2)
    // where y_hat = w.T * x + b, so
    // y_hat = [2.0, 3.0].T * [[5.0, 1.0], [2.0, 7.0]] + 1.0 
    // y_hat = [2.0*5.0 + 3.0*2.0 + 1.0, 2.0*1.0 + 3.0*7.0 + 1.0]
    // y_hat = [17.0, 24.0]
    // The cost is then:
    // J = 1/(2*2) * ((17.0 - 15.0)^2 + (24.0 - 20.0)^2)
    // J = 1/4 * (2.0^2 + 4.0^2) = 1/4 * (4.0 + 16.0) = 5.0
    let expected_cost = 5.0;
  
    let cost = model.compute_cost(&x, &y).unwrap();

    assert_eq!(expected_cost, cost);
  }

  #[test]
  fn test_fit() {

    // Initialize model
    let mut model = LinearRegression::new(2);
    // Setup dummy training data and labeled data.
    let x = arr2(&[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]);
    let y = arr1(&[3.0, 7.0, 11.0]);

    let expected_weights = arr1(&[0.83922521, 2.38188848]);
    let expected_bias = 9.05674000725192;

    // Train the model
    model.fit(&x, &y, 0.01, 1000).unwrap();

    // Check that weights and bias are updated
    assert!(model.weights.iter().all(|&w| w != 0.0));
    assert!(model.bias != 0.0);
    
    // Check that weights and bias are close to expected values
    for (w, ew) in model.weights.iter().zip(expected_weights.iter()) {
      assert!((w - ew).abs() < 0.01);
    }
    assert!((model.bias - expected_bias).abs() < 0.01);
  }

  #[test]
  fn test_prediction_dimensions() {
    let model = LinearRegression::new(2);
    let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let predictions = model.predict(&x).unwrap();
    assert_eq!(predictions.len(), 2);
  }

  #[test]
  fn test_prediction_values() {
    // Create a LinearRegression with specific weights and bias for testing
    let mut model = LinearRegression::new(2);
    // Manually set the weights and bias to known values for testing
    model.weights = arr1(&[1.0, 2.0]);
    model.bias = 1.0;

    let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    // Calculate expected predictions: y = w^T * x + b
    // for x = [[1.0, 2.0], [3.0, 4.0]], w = [1.0, 2.0], b = 1.0
    // y[0] = 1 * 1 + 2 * 3 + 1 = 8
    // y[1] = 1 * 2 + 2 * 4 + 1 = 11
    let expected_predictions = arr1(&[8.0, 11.0]);
    let predictions = model.predict(&x).unwrap();

    assert_eq!(predictions, expected_predictions);
  }
}

impl MLModel for LinearRegression {
  /// Compute the least squares cost function for the current model state and input `x`.
  fn compute_cost(&self, x: &Array2<f64>, y: &ndarray::Array1<f64>) -> Result<f64, ModelError> {
    // Get number of examples 
    let m = x.shape()[1];

    // Make prediction
    let y_hat = self.predict(x)?;

    // Compute cost using least squared error (y_hat - y) ^2 as loss function.
    let j = 1.0/(2.0 * m as f64) * (y_hat - y).pow2().sum();
    Ok(j) 
  }

  /// Perform gradient descent and update the weights interatively.
  fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, learning_rate: f64, epochs: usize) -> Result<(), ModelError> {
    // Check that dimensions of x and y are compatible.
    if x.shape()[0] != self.weights.len() {
      return Err(ModelError::Dimensions(DimensionsError::new(
        x.shape().to_vec(),
        self.weights.shape().to_vec(),
      )));
    }

    if y.shape()[0] != x.shape()[1] {
      return Err(ModelError::Dimensions(DimensionsError::new(
        y.shape().to_vec(),
        x.shape().to_vec(),
      )));
    }

    let m = x.shape()[1] as f64;

    for i in 0..epochs {
      // 'Forward propagation'
      let y_hat = self.predict(x)?;
      
      // Compute 'dz/
      let dz = y - &y_hat;

      // Compute gradients
      let dw = -x.dot(&dz) / m;
      let db = -y_hat.sum() / m;

      // Update parameters
      self.weights = &self.weights - learning_rate * &dw;
      self.bias -= learning_rate * db;

      // Compute cost
      let cost = self.compute_cost(x, y)?;

      if i % 500 == 0 {
        println!("{}.\tEpoch - cost:\t{:.4}", i, cost);
      }
    }

    Ok(())
  }

  /// Compute a batch prediction `y_hat` from input `x`. The input vector `x` must be of compatible dimensions with the weights vector in the model. If the model has `n_x` features, and `x` has `m` training examples, then `x` must be of shape [n_x, m], while the weights will be of shape [n_x, 1]
  fn predict(&self, x: &Array2<f64>) -> Result<ndarray::Array1<f64>, ModelError> {
    // Check that dimensions of x and weights are compatible.
    if x.shape()[0] != self.weights.shape()[0] {
      return Err(ModelError::Dimensions(DimensionsError::new(
        x.shape().to_vec(),
        self.weights.shape().to_vec(),
      )));
    } 
    // Compute the prediction
    let z = self.weights.t().dot(x) + self.bias;
    Ok(z)
  }
}