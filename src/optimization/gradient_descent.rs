use crate::core::error::ModelError;
use crate::core::types::{Matrix, ModelParams, Vector};
use crate::model::ml_model::OptimizableModel;
use crate::optimization::optimizer::Optimizer;

pub struct GradientDescent {
    learning_rate: f64,
    epochs: usize,
    pub cost_history: Vec<f64>,
}

impl GradientDescent {
    pub fn new(learning_rate: f64, epochs: usize) -> Self {
        Self {
            learning_rate,
            epochs,
            cost_history: Vec::new(),
        }
    }
}

impl Optimizer<Matrix, Vector> for GradientDescent {
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
