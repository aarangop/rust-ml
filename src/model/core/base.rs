use crate::core::error::ModelError;

use super::param_collection::{GradientCollection, ParamCollection};

pub trait BaseModel<Input, Output> {
    /// Predicts an output value based on the input data.
    ///
    /// # Returns
    ///
    /// The predicted output value
    fn predict(&self, x: &Input) -> Result<Output, ModelError>;

    /// Computes the cost (or loss) between the predicted output and the actual output.
    ///
    /// # Returns
    ///
    /// The computed cost as a floating point value
    fn compute_cost(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
}

pub trait OptimizableModel<Input, Output>:
    BaseModel<Input, Output> + ParamCollection + GradientCollection
{
    /// Forward pass through the model.
    fn forward(&self, input: &Input) -> Result<Output, ModelError>;

    /// Backward pass to compute gradients.
    fn backward(&mut self, input: &Input, output_grad: &Output) -> Result<(), ModelError>;

    /// Computes the gradient of the cost with respect to the output predictions
    fn compute_output_gradient(&self, x: &Input, y: &Output) -> Result<Output, ModelError>;
}
