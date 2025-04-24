use crate::core::error::ModelError;
use ndarray::{Array, ArrayView, ArrayViewMut, Dimension, IxDyn};
use std::fmt::Debug;

/// Provides access to parameters.
pub trait ParamCollection: Debug + Send + Sync {
    /// Get a reference to a specific parameter with strong typing.
    fn get<D: Dimension>(&self, key: &str) -> Result<ArrayView<f64, D>, ModelError>;

    fn get_mut<D: Dimension>(&mut self, key: &str) -> Result<ArrayViewMut<f64, D>, ModelError>;

    /// Set the value of a parameter.
    fn set<D: Dimension>(&mut self, key: &str, value: ArrayView<f64, D>) -> Result<(), ModelError>;

    /// Iterate over all parameters.
    fn param_iter(&self) -> Vec<(&str, ArrayView<f64, IxDyn>)>;
}

pub trait GradientCollection {
    /// Get a reference to a specific gradient with strong typing.
    fn get_gradient<D: Dimension>(&self, key: &str) -> Result<ArrayView<f64, D>, ModelError>;

    /// Set the value of a gradient.
    fn set_gradient<D: Dimension>(
        &mut self,
        key: &str,
        value: Array<f64, D>,
    ) -> Result<(), ModelError>;
}
