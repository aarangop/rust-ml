use ndarray::ShapeError;
use std::convert::From;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Incompatible dimensions:  {dim1:?}, and {dim2:?}")]
    Dimensions { dim1: Vec<usize>, dim2: Vec<usize> },

    #[error("Key error: {0}")]
    KeyError(String),

    #[error("Invalid configuration: {0}")]
    Configuration(String),

    #[error("Invalid parameter: Requested parameter {0} not found.")]
    InvalidParameter(String),

    #[error("Gradient not found: {0} not found in gradients hash map.")]
    GradientNotFound(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Convergence failed: {0}")]
    Convergence(String),

    #[error("Dimensionality error: {0}")]
    DimensionalityError(String),

    #[error("Shape error: {0}")]
    ShapeError(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Uninitialized model: {0}")]
    UninitializedModel(String),
}

// Implement conversion from ShapeError to ModelError
impl From<ShapeError> for ModelError {
    fn from(err: ShapeError) -> Self {
        ModelError::ShapeError(format!("{}", err))
    }
}
