/// Model trait to provide common functionality across different types of models.
pub trait MLModel {
    /// Train the model using input `x` and output `y`.
    fn train(&mut self, x: Array2<f64>, y: Array1<f64>, learning_rate: f64, epochs: usize);
    /// Compute the cost function for the current model state and input `x`.
    fn compute_cost(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64;
}