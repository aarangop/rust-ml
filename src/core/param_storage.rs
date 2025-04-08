use crate::core::error::ModelError;
use ndarray::{Array, ArrayView, IxDyn};
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::{Debug, Display};

/// Trait for type-safe parameter storage and access in ML models
///
/// This trait provides a more type-safe alternative to string-based parameter management,
/// allowing each model to define its own parameter key type.
pub trait ParameterStorage {
    /// The parameter key type that identifies model parameters
    /// Must implement basic traits for use in HashMaps and debugging
    type ParamKey: Clone + Eq + Hash + Debug;
    
    /// Gets a view of a parameter by its key
    ///
    /// # Arguments
    /// * `key` - The parameter key to access
    ///
    /// # Returns
    /// * `Result<ArrayView<f64, IxDyn>, ModelError>` - The parameter view or an error
    fn get_param(&self, key: &Self::ParamKey) -> Result<ArrayView<f64, IxDyn>, ModelError>;
    
    /// Gets a mutable reference to a parameter by its key
    ///
    /// # Arguments
    /// * `key` - The parameter key to access
    ///
    /// # Returns
    /// * `Result<&mut Array<f64, IxDyn>, ModelError>` - Mutable reference to the parameter or an error
    fn get_param_mut(&mut self, key: &Self::ParamKey) -> Result<&mut Array<f64, IxDyn>, ModelError>;
    
    /// Updates a parameter with a new value
    ///
    /// # Arguments
    /// * `key` - The parameter key to update
    /// * `value` - The new parameter value
    ///
    /// # Returns
    /// * `Result<(), ModelError>` - Success or an error
    fn update_param(&mut self, key: &Self::ParamKey, value: Array<f64, IxDyn>) -> Result<(), ModelError>;
    
    /// Gets all parameters as a HashMap of keys to views
    ///
    /// # Returns
    /// * `HashMap<Self::ParamKey, ArrayView<f64, IxDyn>>` - All parameters
    fn get_all_params(&self) -> HashMap<Self::ParamKey, ArrayView<f64, IxDyn>>;
}
