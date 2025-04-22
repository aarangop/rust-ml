use crate::core::error::ModelError;

use super::param_collection::{GradientCollection, ParamCollection};

pub trait BaseModel<Input, Output> {
    /// Predicts an output value based on the input data.
    ///
    /// # Returns
    ///
    /// The predicted output value
    fn predict(&mut self, x: &Input) -> Result<Output, ModelError>;

    /// Computes the cost (or loss) between the predicted output and the actual output.
    ///
    /// # Returns
    ///
    /// The computed cost as a floating point value
    fn compute_cost(&mut self, x: &Input, y: &Output) -> Result<f64, ModelError>;

    // Returns whether the model is initialized or not.
    fn model_is_initialized(&self) -> bool;

    /// Performs any initialization logic necessary for the model.
    /// This is typically called before training or inference, and
    /// may include input and output data for models to generate internal data structures.
    fn initialize_model(
        &mut self,
        _: Option<&Input>,
        _: Option<&Output>,
    ) -> Result<(), ModelError> {
        Ok(())
    }
}

pub trait OptimizableModel<Input, Output>:
    BaseModel<Input, Output> + ParamCollection + GradientCollection
{
    /// Forward pass through the model.
    fn forward(&mut self, input: &Input) -> Result<Output, ModelError>;

    /// Backward pass to compute gradients.
    fn backward(&mut self, input: &Input, output_grad: &Output) -> Result<(), ModelError>;

    /// Computes the gradient of the cost with respect to the output predictions
    fn compute_output_gradient(&self, x: &Input, y: &Output) -> Result<Output, ModelError>;
}
