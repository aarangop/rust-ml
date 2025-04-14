use ndarray::{Array, Array0, Array1, Array2, ArrayView0, ArrayView1, ArrayView2, IxDyn};
use std::collections::HashMap;

/// ModelParams is an alias for a hashmap that can be used to store model parameters,
/// or other values, like gradients.
///
/// # Naming Convention
///
/// By convention, we follow these naming patterns:
/// - Uppercase letters for matrices: `W`, `W1`, `A2`, etc.
/// - Lowercase letters for vectors: `b`, `z2`, `w`, etc.
/// - Gradients use the same convention with a leading 'd' to indicate derivatives:
///   - `dW2` is the derivative of the cost function with respect to `W2`
///   - `db1` is the derivative of the cost function with respect to `b1`
///
/// # Examples
///
/// ```rust
/// let mut params = ModelParams::new();
///
/// // Storing weight matrices for a neural network with 2 layers
/// params.insert("W1".to_string(), weight_matrix_layer1); // First layer weights
/// params.insert("W2".to_string(), weight_matrix_layer2); // Second layer weights
///
/// // Storing bias vectors
/// params.insert("b1".to_string(), bias_vector_layer1);
/// params.insert("b2".to_string(), bias_vector_layer2);
///
/// // During backpropagation, storing gradients
/// params.insert("dW1".to_string(), gradient_of_weights_layer1);
/// params.insert("dW2".to_string(), gradient_of_weights_layer2);
/// params.insert("db1".to_string(), gradient_of_bias_layer1);
/// params.insert("db2".to_string(), gradient_of_bias_layer2);
///
/// // Activations may also be stored during forward propagation
/// params.insert("A1".to_string(), activation_layer1);
/// ```
///
/// This convention makes it easy to match parameters with their corresponding
/// gradients during optim steps (e.g., `W1` pairs with `dW1`).
pub type ModelParams = HashMap<String, Array<f64, IxDyn>>;
pub type Matrix = Array2<f64>;
pub type Vector = Array1<f64>;
pub type Scalar = Array0<f64>;

