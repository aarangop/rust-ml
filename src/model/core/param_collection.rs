use crate::core::error::ModelError;
use ndarray::{Array, ArrayView, ArrayViewMut, Dimension, IxDyn};
use std::fmt::Debug;
use std::marker::PhantomData;

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
        value: ArrayView<f64, D>,
    ) -> Result<(), ModelError>;
}

/// Helper trait with utility methods for handling parameters of different dimensions

pub struct ParamUpdateHelper<P: ParamCollection, G: GradientCollection> {
    _phantom: PhantomData<P>,
    _phantom2: PhantomData<G>,
}

impl<P: ParamCollection, G: GradientCollection> ParamUpdateHelper<P, G> {
    /// Apply a function to parameters and their corresponding gradients
    pub fn new() -> ParamUpdateHelper<P, G> {
        ParamUpdateHelper {
            _phantom2: PhantomData,
            _phantom: PhantomData,
        }
    }
    pub fn apply_to_param_grad<F, R>(
        &self,
        params: &mut P,
        grads: &G,
        key: &str,
        f: F,
    ) -> Result<R, ModelError>
    where
        F: Fn(ArrayView<f64, IxDyn>, ArrayView<f64, IxDyn>) -> Result<R, ModelError>,
    {
        // Get parameter and gradient views in their dynamic representation
        let param_view = params.get::<IxDyn>(key)?;
        let grad_view = grads.get_gradient::<IxDyn>(key)?;

        // Apply the function
        f(param_view, grad_view)
    }

    /// Update parameters with a function that transforms parameter and gradient views
    pub fn update_param<F>(
        &self,
        params: &mut P,
        grads: &G,
        key: &str,
        f: F,
    ) -> Result<(), ModelError>
    where
        F: Fn(
            ArrayView<f64, IxDyn>,
            ArrayView<f64, IxDyn>,
        ) -> Result<Array<f64, IxDyn>, ModelError>,
    {
        let param_view = params.get::<IxDyn>(key)?;
        let grad_view = grads.get_gradient::<IxDyn>(key)?;

        // Apply the transformation function
        let new_param = f(param_view, grad_view)?;

        // Update the parameter with the new value
        params.set(key, new_param.view())
    }
}
