use crate::{core::error::ModelError, model::core::{base::DLModel, param_collection::{GradientCollection, ParamCollection}}};

/// OptimizerState trait defines the interface for managing the state of an optimizer.
/// It provides methods to initialize the optimizer state and update the model weights.
/// This trait is useful to implement custom parameter update rules, as used in different
/// optimization algorithms, like RMSProp, Adam, etc.
pub trait OptimizerState<Input, Output, M: DLModel<Input, Output>> {
    fn initialize(&self) {}
    // fn update_weights(&self, params: &mut P, grads: &G);
    fn update_weights(&mut self, model: &mut M) -> Result<(), ModelError>;
}