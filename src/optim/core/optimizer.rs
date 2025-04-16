/// Module containing optim traits and implementations for machine learning models.
use crate::core::error::ModelError;
use crate::model::core::base::OptimizableModel;


/// The `Optimizer` trait defines the interface for optim algorithms.
///
/// Implementations of this trait can be used to train machine learning models
/// that conform to the `OptimizableModel` trait. Different optim strategies
/// can be implemented to find optimal parameters for a given model.
///
/// # Type Parameters
///
/// * `Input` - The type of input data used to train the model
/// * `Output` - The type of output data that the model produces
pub trait Optimizer<Input, Output, M: OptimizableModel<Input, Output>> {
    /// Fits the provided model to the training data.
    ///
    /// # Arguments
    ///
    /// * `model` - A mutable reference to a model that implements the `DLModel` trait
    /// * `x` - A reference to the input data
    /// * `y` - A reference to the expected output data
    ///
    /// # Returns
    ///
    /// * `Result<(), ModelError>` - Ok(()) if fitting was successful, or an error if it failed
    fn fit(&mut self, model: &mut M, x: &Input, y: &Output) -> Result<(), ModelError>;
}
