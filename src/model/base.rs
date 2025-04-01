use ndarray::{Array1, Array2, ArrayBase};

use super::error::{DimensionsError, ModelError};


/// Model trait to provide common functionality across different types of models.
pub trait MLModel {
    /// Train the model using input `x` and output `y`.
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>, learning_rate: f64, epochs: usize) -> Result<(), ModelError>;
    /// Compute the cost function for the current model state and input `x`.
    fn compute_cost(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64, ModelError>;
    /// Compute a prediction from input `x`, returning an array of outputs `y_hat`.
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, ModelError>;
}
