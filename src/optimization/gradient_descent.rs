/// Gradient Descent optimizer implementation for machine learning models.
///
/// This module provides an implementation of the gradient descent optimization algorithm
/// for training machine learning models. Gradient descent works by iteratively adjusting
/// model parameters in the direction that minimizes the cost function.
use crate::core::error::ModelError;
use crate::core::types::{Matrix, ModelParams, Vector};
use crate::model::core::optimizable_model::OptimizableModel;
use crate::optimization::core::optimizer::Optimizer;

/// A standard gradient descent optimizer.
///
/// Gradient descent is an optimization algorithm that iteratively adjusts parameters
/// to minimize a cost function by moving in the direction of the steepest decrease
/// in the cost function.
///
/// # Fields
/// * `learning_rate` - The step size for each iteration of gradient descent
/// * `epochs` - The number of complete passes through the training dataset
/// * `cost_history` - Records the cost value after each parameter update
pub struct GradientDescent {
    learning_rate: f64,
    epochs: usize,
    pub cost_history: Vec<f64>,
}

impl GradientDescent {
    /// Creates a new GradientDescent optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - The step size for each parameter update
    /// * `epochs` - The number of complete passes through the training dataset
    ///
    /// # Returns
    /// A new GradientDescent instance
    pub fn new(learning_rate: f64, epochs: usize) -> Self {
        Self {
            learning_rate,
            epochs,
            cost_history: Vec::new(),
        }
    }
}

impl Optimizer<Matrix, Vector> for GradientDescent {
    /// Fits the model to the training data using gradient descent algorithm.
    ///
    /// This method updates the model parameters by computing gradients
    /// and adjusting the parameters in the direction that minimizes the cost function.
    ///
    /// # Arguments
    /// * `model` - The machine learning model to optimize
    /// * `x` - The input training data
    /// * `y` - The expected output values
    ///
    /// # Returns
    /// * `Ok(())` if optimization completes successfully
    /// * `Err(ModelError)` if an error occurs during optimization
    fn fit(
        &mut self,
        model: &mut dyn OptimizableModel<Matrix, Vector>,
        x: &Matrix,
        y: &Vector,
    ) -> Result<(), ModelError> {
        for _ in 0..self.epochs {
            // Get the gradient of the cost function in relation to the model predictions.
            let output_grad = model.compute_gradient(x, y)?;
            let (_, cache) = model.compute_forward_propagation(x)?;
            // Obtain the gradients from backward propagation.
            let gradients = model.compute_backward_propagation(x, y, &output_grad, cache)?;

            // Compute the new parameter values.
            let params = model.get_params();
            let mut new_params = ModelParams::new();
            for (key, value) in params.iter() {
                // Get respective gradient for the current parameter
                let gradient_key = format!("d{}", key);
                let gradient = gradients
                    .get(&gradient_key)
                    .ok_or(ModelError::GradientNotFound(gradient_key))?;
                let new_value = value - self.learning_rate * gradient;
                new_params.insert(key.to_string(), new_value);

                let cost = model.compute_cost(x, y)?;
                self.cost_history.push(cost);
            }

            // Update the model parameters
            model.update_params(new_params);
        }
        Ok(())
    }
}
