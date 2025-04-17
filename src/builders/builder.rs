/// Builder pattern trait for machine learning models.
///
/// This module implements the Builder design pattern for creating machine learning models
/// with customizable configurations. The Builder pattern separates the construction of complex
/// objects from their representation, allowing the same construction process to create
/// different representations.
///
/// # Type Parameters
///
/// * `M` - The machine learning model type that implements BaseModel
/// * `Input` - The input data type for the model (typically Matrix)
/// * `Output` - The output data type for the model (typically Vector)
use crate::core::error::ModelError;
use crate::model::core::base::BaseModel;

/// A trait for implementing the Builder pattern for machine learning models.
///
/// This trait ensures that any builder can construct a specific model type
/// through a consistent interface. Builders allow for flexible configuration
/// of model parameters before instantiation.
pub trait Builder<M, Input, Output>
where
    M: BaseModel<Input, Output>,
{
    /// Builds and returns a new model instance with the configured parameters.
    ///
    /// # Returns
    ///
    /// * `Result<M, ModelError>` - The constructed model if successful, or an error if the
    ///   construction fails (e.g., due to invalid configuration parameters)
    fn build(&self) -> Result<M, ModelError>;
}
