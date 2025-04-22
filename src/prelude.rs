/// The prelude module provides a convenient way to import common components of the library.
///
/// # Example
/// ```rust
/// use rust_ml::prelude::*;
///
/// // Now you can use imported types without full path qualification
/// // let model = LinearRegression::new(2);
/// // let optimizer = GradientDescent::new(0.01, 1000);
/// ```
// Re-export common types, traits, and functions for convenient imports
// Users should be able to import everything they need with just `use rust_ml::prelude::*;`
// Re-export core types
pub use crate::core::types::{Matrix, Vector};

// Re-export activation functions
pub use crate::core::activations::activation_functions::ActivationFn;

// Re-export error types
pub use crate::core::error::ModelError;

// Re-export models
pub use crate::model::single_layer_classifier::SingleLayerClassifier;

// Re-export builders
pub mod single_layer_classifier {
    pub use crate::builders::single_layer_classifier::SingleLayerClassifierBuilder;
}
