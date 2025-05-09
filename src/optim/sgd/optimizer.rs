/// Gradient Descent optimizer implementation for machine learning models.
///
/// This module provides an implementation of the gradient descent optim algorithm
/// for training machine learning models. Gradient descent works by iteratively adjusting
/// model parameters in the direction that minimizes the cost function.
use crate::core::error::ModelError;
use crate::core::types::{Matrix, Vector};
use crate::model::core::base::OptimizableModel;
use crate::optim::core::optimizer::Optimizer;
use crate::optim::core::state::OptimizerState;
use crate::optim::sgd::state::GradientDescentState;

/// A standard gradient descent optimizer.
///
/// Gradient descent is an optim algorithm that iteratively adjusts parameters
/// to minimize a cost function by moving in the direction of the steepest decrease
/// in the cost function.
///
/// # Fields
/// * `learning_rate` - The step size for each iteration of gradient descent
/// * `epochs` - The number of complete passes through the training dataset
/// * `cost_history` - Records the cost value after each parameter update
pub struct GradientDescent<Input, Output, M: OptimizableModel<Input, Output>> {
    epochs: usize,
    pub cost_history: Vec<f64>,
    state: GradientDescentState<Input, Output, M>,
}

impl<Input, Output, M: OptimizableModel<Input, Output>> GradientDescent<Input, Output, M> {
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
            epochs,
            cost_history: Vec::new(),
            state: GradientDescentState::new(learning_rate),
        }
    }
}

impl<M: OptimizableModel<Matrix, Vector>> Optimizer<Matrix, Vector, M>
    for GradientDescent<Matrix, Vector, M>
{
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
    /// * `Ok(())` if optim completes successfully
    /// * `Err(ModelError)` if an error occurs during optim
    fn fit(&mut self, model: &mut M, x: &Matrix, y: &Vector) -> Result<(), ModelError> {
        for _ in 0..self.epochs {
            // Compute cost
            let cost = model.compute_cost(x, y)?;

            self.cost_history.push(cost);

            // Compute output gradient (includes forward prop)
            let output_gradient = model.compute_output_gradient(x, y)?;

            // Compute gradients using backward propagation
            model.backward(x, &output_gradient)?;

            // Update model parameters using optimizer state
            self.state.update_weights(model)?;
        }
        Ok(())
    }
}
