use crate::core::error::ModelError;
use crate::core::types::ModelParams;

pub trait BaseModel<Input, Output> {
    /// Predicts an output value based on the input data.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data to make a prediction on
    ///
    /// # Returns
    ///
    /// The predicted output value
    fn predict(&self, x: &Input) -> Result<Output, ModelError>;

    /// Computes the cost (or loss) between the predicted output and the actual output.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `y` - The expected output/target value
    ///
    /// # Returns
    ///
    /// The computed cost as a floating point value
    fn compute_cost(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;

    /// Compute gradient returns the gradient of the cost function in relation to the model's prediction.
    ///
    /// # Arguments
    ///
    /// * `x` - The input data
    /// * `y` - The expected output/target value
    ///
    /// # Returns
    ///
    /// The gradient of the cost function with respect to the model's predictions,
    /// which has the same shape as the output.
    fn compute_gradient(&self, x: &Input, y: &Output) -> Result<Output, ModelError>;
}

pub trait ForwardPropagation<Input, Output> {
    /// Computes forward propagation with the current parameter weights and the provided input `x`.
    /// Returns the prediction and a hashmap containing intermediate activations.
    fn compute_forward_propagation(
        &self,
        x: &Input,
    ) -> Result<(Output, Option<ModelParams>), ModelError>;
}

pub trait BackwardPropagation<Input, Output> {
    /// Perform backward propagation and returns the gradients for the model parameters.
    fn compute_backward_propagation(
        &self,
        x: &Input,
        y: &Output,
        output_gradients: &Output,
        cache: Option<ModelParams>,
    ) -> Result<ModelParams, ModelError>;
}
