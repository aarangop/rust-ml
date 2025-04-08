use crate::core::param_manager::ParamManager;
use crate::model::core::base::{BackwardPropagation, BaseModel, ForwardPropagation};

/// A trait for models that can be optimized through a training process.
///
/// This trait combines several fundamental capabilities required for machine learning models
/// that can be trained using gradient-based optimization techniques:
///
/// * `BaseModel<Input, Output>`: Provides the basic model interface
/// * `ForwardPropagation<Input, Output>`: Enables forward pass computation to make predictions
/// * `BackwardPropagation<Input, Output>`: Supports gradient computation through backward propagation
/// * `ParamManager`: Handles parameter management (access, update, iteration, etc.)
///
/// Types implementing this trait can be passed to optimizers for training.
///
/// # Type Parameters
/// * `Input`: The type of input the model accepts
/// * `Output`: The type of output the model produces
pub trait OptimizableModel<Input, Output>:
    BaseModel<Input, Output>
    + ForwardPropagation<Input, Output>
    + BackwardPropagation<Input, Output>
    + ParamManager
{
}
