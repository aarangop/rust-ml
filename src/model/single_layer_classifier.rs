use crate::core::activations::activation::Activation;
use crate::core::activations::leaky_relu::LeakyReLU;
use crate::core::activations::relu::ReLU;
use crate::core::activations::sigmoid::Sigmoid;
use crate::core::activations::tanh::Tanh;
use crate::model::core::base::{BaseModel, OptimizableModel};
use crate::model::core::param_collection::{GradientCollection, ParamCollection};
use crate::prelude::single_layer_classifier::SingleLayerClassifierBuilder;
use crate::prelude::*;
use ndarray::{Array1, ArrayView, Dimension, arr1, arr2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use uuid::Uuid;

/// Cache for storing intermediate values during forward pass.
#[derive(Debug, Clone)]
pub struct SingleLayerClassifierCache {
    pub a1: Option<Matrix>,
    pub z1: Option<Matrix>,
    pub a2: Option<Matrix>,
    pub z2: Option<Matrix>,
    pub cache_id: Option<Uuid>,
}

/// Single hidden layer classifier model.
/// This model is a simple feedforward neural network with one hidden layer.
///
/// # Structure
/// This neural network consists of:
/// * An input layer with n_features nodes
/// * A single hidden layer with n_hidden_nodes nodes using a configurable activation function
/// * An output layer with a single node using a configurable activation function
///
/// # Parameters
/// * `w1` - Weight matrix between input and hidden layer (size: n_hidden_nodes × n_features)
/// * `b1` - Bias vector for hidden layer (size: n_hidden_nodes)
/// * `w2` - Weight matrix between hidden and output layer (size: 1 × n_hidden_nodes)
/// * `b2` - Bias vector for output layer (size: n_hidden_nodes)
/// * `output_layer_activation_fn` - Activation function applied to the output layer
/// * `hidden_layer_activation_fn` - Activation function applied to the hidden layer
/// * `threshold` - Classification threshold for binary classification
#[derive(Debug, Clone)]
pub struct SingleLayerClassifier {
    pub w1: Matrix,
    pub b1: Vector,
    pub w2: Matrix,
    pub b2: Vector,
    output_layer_activation_fn: ActivationFn,
    hidden_layer_activation_fn: ActivationFn,
    threshold: f64,
    cache: Option<SingleLayerClassifierCache>,
    current_cache_id: Option<Uuid>,
}

impl SingleLayerClassifier {
    /// Creates a new single layer neural network classifier.
    ///
    /// # Parameters
    /// * `n_features` - The number of input features.
    /// * `n_hidden_nodes` - The number of nodes in the hidden layer.
    /// * `threshold` - Classification threshold between 0.0 and 1.0.
    ///                Outputs above this value are classified as positive.
    /// * `output_layer_activation_fn` - Activation function for the output layer.
    /// * `hidden_layer_activation_fn` - Activation function for the hidden layer.
    ///
    /// # Returns
    /// * `Result<SingleLayerClassifier, ModelError>` - A new classifier instance or an error.
    ///
    /// # Errors
    /// Returns `ModelError::InvalidParameter` if the threshold is not between 0.0 and 1.0.
    ///
    /// # Details
    /// Initializes the weights using a standard normal distribution and biases with zeros.
    ///
    /// # Example
    /// ```
    /// use rust_ml::prelude::*;
    /// let classifier = SingleLayerClassifier::new(
    ///     4, // features
    ///     10, // hidden nodes
    ///     0.5, // threshold
    ///     ActivationFn::Sigmoid,
    ///     ActivationFn::ReLU
    /// )?;
    /// ```
    ///
    /// Alternatively, use the builder pattern:
    /// ```
    /// use rust_ml::builders::builder::Builder;
    /// use rust_ml::prelude::SingleLayerClassifier;
    /// let classifier = SingleLayerClassifier::builder()
    ///     // configure with builder methods
    ///     .build()?;
    /// ```
    pub fn new(
        n_features: usize,
        n_hidden_nodes: usize,
        threshold: f64,
        output_layer_activation_fn: ActivationFn,
        hidden_layer_activation_fn: ActivationFn,
    ) -> Result<SingleLayerClassifier, ModelError> {
        // Check that the threshold is between 0.0 and 1.0
        if !(0.0..=1.0).contains(&threshold) {
            return Err(ModelError::InvalidParameter(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Initialize weights and biases using a normal distribution for weights, and zeros for biases
        let distribution = Normal::new(0.0, 1.0).unwrap();
        let w1 = Matrix::random((n_hidden_nodes, n_features), distribution);
        let b1 = Vector::zeros(n_hidden_nodes);
        let w2 = Matrix::random((1, n_hidden_nodes), distribution);
        let b2 = Vector::zeros(1);

        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            output_layer_activation_fn,
            hidden_layer_activation_fn,
            threshold,
            cache: None,
            current_cache_id: None,
        })
    }

    /// Creates a builder for configuring a new SingleLayerClassifier.
    ///
    /// This method returns a builder that allows for a fluent interface
    /// to configure and construct a new classifier instance.
    ///
    /// # Returns
    /// A new SingleLayerClassifierBuilder instance with default settings.
    ///
    /// # Example
    /// ```
    /// use rust_ml::builders::builder::Builder;
    /// use rust_ml::prelude::SingleLayerClassifier;
    ///
    /// let classifier = SingleLayerClassifier::builder()
    ///     .n_features(4)
    ///     .n_hidden_nodes(10)
    ///     .threshold(0.5)
    ///     .build()?;
    /// ```
    pub fn builder() -> SingleLayerClassifierBuilder {
        SingleLayerClassifierBuilder::default()
    }

    /// Returns the activation for a given input z.
    pub fn compute_activation(&self, z: &Matrix, activation_fn: ActivationFn) -> Matrix {
        match activation_fn {
            ActivationFn::Sigmoid => Sigmoid::activate(z),
            ActivationFn::ReLU => ReLU::activate(z),
            ActivationFn::Tanh => Tanh::activate(z),
            ActivationFn::LeakyReLU => LeakyReLU::activate(z),
        }
    }

    /// Computes the linear activation for a given input x, weights w, and bias b.
    /// Given the following dimensions:
    /// * x: (n_features, m)
    /// * w: (n_nodes, n_features)
    /// * b: (n_nodes)
    ///
    /// The dimensions of the output will be:
    ///
    /// * (n_nodes, m)
    pub fn compute_linear_activation(
        &self,
        x: &Matrix,
        w: &Matrix,
        b: &Vector,
    ) -> Result<Matrix, ModelError> {
        let m = x.shape()[1];
        let b = b.to_shape((b.len(), 1))?;
        let n_nodes = w.shape()[0];
        let b = b
            .broadcast((n_nodes, m))
            .ok_or(ModelError::ShapeError("Broadcasting failed".to_string()))?;
        let z = w.dot(x) + b;
        Ok(z)
    }

    pub fn cache(&self) -> Option<&SingleLayerClassifierCache> {
        self.cache.as_ref()
    }

    pub fn set_cache(&mut self, cache: SingleLayerClassifierCache) {
        self.cache = Some(cache);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::builder::Builder;

    #[test]
    fn test_new_valid_parameters() {
        let result =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU);

        assert!(result.is_ok());
        let classifier = result.unwrap();

        assert_eq!(classifier.w1.shape(), &[10, 4]);
        assert_eq!(classifier.b1.len(), 10);
        assert_eq!(classifier.w2.shape(), &[1, 10]);
        assert_eq!(classifier.b2.len(), 1);
        assert_eq!(classifier.threshold, 0.5);
        assert!(classifier.cache().is_none());
    }

    #[test]
    fn test_new_invalid_threshold() {
        let result =
            SingleLayerClassifier::new(4, 10, 1.5, ActivationFn::Sigmoid, ActivationFn::ReLU);

        assert!(result.is_err());
        match result {
            Err(ModelError::InvalidParameter(msg)) => {
                assert!(msg.contains("Threshold must be between 0.0 and 1.0"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_builder_pattern() {
        let result = SingleLayerClassifier::builder()
            .n_features(4)
            .n_hidden_nodes(10)
            .threshold(0.5)
            .output_layer_activation_fn(ActivationFn::Sigmoid)
            .output_layer_activation_fn(ActivationFn::ReLU)
            .build();

        assert!(result.is_ok());

        let classifier = result.unwrap();

        assert_eq!(classifier.w1.shape(), &[10, 4]);
        assert_eq!(classifier.b1.len(), 10);
        assert_eq!(classifier.w2.shape(), &[1, 10]);
        assert_eq!(classifier.b2.len(), 1);
    }

    /// Test the compute_activation method
    #[test]
    fn test_compute_activation() {
        let classifier =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        let input = Matrix::from_shape_vec((10, 1), vec![0.1; 10]).unwrap();
        let result = classifier.compute_activation(&input, ActivationFn::Sigmoid);
        assert_eq!(result.shape(), &[10, 1]);
    }

    /// Test the compute_linear_activation method
    #[test]
    fn test_compute_linear_activation() {
        let classifier =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        let x = Matrix::from_shape_vec((4, 2), vec![0.1; 8]).unwrap();
        let result = classifier.compute_linear_activation(&x, &classifier.w1, &classifier.b1);
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.shape(), &[10, 2]);
    }
}

impl BaseModel<Matrix, Matrix> for SingleLayerClassifier {
    /// Predicts output values for given input features.
    ///
    /// This method performs a forward pass through the neural network to generate predictions.
    ///
    /// # Parameters
    /// * `x` - Input feature matrix where each columns represents a sample
    ///
    /// # Returns
    /// * `Result<Vector, ModelError>` - Vector of prediction values (before thresholding) or an error
    ///
    /// # Errors
    /// Returns `ModelError` if matrix dimensions are incompatible
    fn predict(&mut self, x: &Matrix) -> Result<Matrix, ModelError> {
        let a2 = self.forward(x)?;
        let y_hat = a2.map(|v| if v > &self.threshold { 1.0 } else { 0.0 });
        Ok(y_hat)
    }

    /// Computes the loss/cost for predictions against actual values.
    ///
    /// This method calculates the binary cross-entropy loss for binary classification,
    /// which measures how well the model's predictions match the true labels.
    ///
    /// # Parameters
    /// * `x` - Input feature matrix where each column represents a sample
    /// * `y` - Target vector with ground truth labels (0 or 1)
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The average loss across all samples or an error
    ///
    /// # Errors
    /// Returns `ModelError` if dimensions are incompatible or if the forward pass fails
    fn compute_cost(&mut self, x: &Matrix, y: &Matrix) -> Result<f64, ModelError> {
        // Perform forward pass to get predictions
        let predictions = self.forward(x)?;

        // Ensure y and predictions have compatible dimensions
        if y.shape() != predictions.shape() {
            return Err(ModelError::ShapeError(format!(
                "Target shape {:?} does not match predictions shape {:?}",
                y.shape(),
                predictions.shape()
            )));
        }

        // Compute binary cross-entropy loss
        // L = -1/m * sum(y * log(a) + (1-y) * log(1-a))
        let m = y.len() as f64;
        let epsilon = 1e-15; // Small constant to avoid log(0)

        // Clip predictions to avoid numerical instability
        let predictions = predictions.mapv(|a| a.max(epsilon).min(1.0 - epsilon));

        // Calculate element-wise terms of the loss function
        let term1 = y * &predictions.mapv(|a| a.ln());
        let term2 = (1.0 - y) * &predictions.mapv(|a| (1.0 - a).ln());

        // Calculate total loss
        let total_loss = -1.0 * (term1 + term2).sum() / m;

        Ok(total_loss)
    }

    fn model_is_initialized(&self) -> bool {
        if self.cache.is_none() {
            return false;
        }
        true
    }

    /// Initializes the model with input and output data by creating the necessary cache.
    fn initialize_model(
        &mut self,
        x: Option<&Matrix>,
        y: Option<&Matrix>,
    ) -> Result<(), ModelError> {
        let x = x.ok_or(ModelError::InvalidParameter(
            "Input data is required for initialization".to_string(),
        ))?;
        let y = y.ok_or(ModelError::InvalidParameter(
            "Output data is required for initialization".to_string(),
        ))?;
        let m = x.shape()[1];
        let z1 = Matrix::zeros((self.w1.shape()[0], m));
        let a1 = Matrix::zeros((self.w1.shape()[0], m));
        let z2 = Matrix::zeros((self.w2.shape()[0], m));
        let a2 = Matrix::zeros((self.w2.shape()[0], m));
        let cache = SingleLayerClassifierCache {
            a1: Some(a1),
            z1: Some(z1),
            a2: Some(a2),
            z2: Some(z2),
            cache_id: None,
        };
        self.set_cache(cache);
        Ok(())
    }
}

#[cfg(test)]
mod base_model_tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn test_predict() {
        // Create a classifier with known parameters
        let mut classifier = SingleLayerClassifier::new(
            2,   // 2 features
            3,   // 3 hidden nodes
            0.5, // threshold
            ActivationFn::Sigmoid,
            ActivationFn::ReLU,
        )
        .unwrap();

        // Set predetermined weights to get deterministic outputs
        classifier.w1 = Matrix::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 5.0, 6.0]).unwrap();
        classifier.b1 = Vector::from_vec(vec![0.1, 0.2, 0.3]);
        classifier.w2 = Matrix::from_shape_vec((1, 3), vec![0.1, 0.2, 0.3]).unwrap();
        classifier.b2 = Vector::from_vec(vec![0.1]);

        // Input data with 2 samples
        let input = Matrix::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Expected probability output: [0.65432483, 0.65889896]
        let expected_predictions = arr2(&[[1.0, 1.0]]);

        let result = classifier.predict(&input);
        assert!(result.is_ok());

        // Get the predictions
        let predictions = result.unwrap();
        assert_eq!(predictions, expected_predictions);

        // Check the shape of the predictions
        assert_eq!(predictions.shape(), &[1, 2]);
        // All predictions should be either 0.0 or 1.0 (binary classification)
        for val in predictions.iter() {
            assert!(*val == 0.0 || *val == 1.0);
        }
    }

    #[test]
    fn test_predict_with_threshold() {
        // Create two classifiers with different thresholds
        let mut classifier_low_threshold =
            SingleLayerClassifier::new(2, 3, 0.3, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        let mut classifier_high_threshold =
            SingleLayerClassifier::new(2, 3, 0.7, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        // Set identical weights for both classifiers
        let w1 = Matrix::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        let b1 = Vector::from_vec(vec![0.1, 0.2, 0.3]);
        let w2 = Matrix::from_shape_vec((1, 3), vec![0.7, 0.8, 0.9]).unwrap();
        let b2 = Vector::from_vec(vec![0.1]);

        classifier_low_threshold.w1 = w1.clone();
        classifier_low_threshold.b1 = b1.clone();
        classifier_low_threshold.w2 = w2.clone();
        classifier_low_threshold.b2 = b2.clone();

        classifier_high_threshold.w1 = w1;
        classifier_high_threshold.b1 = b1;
        classifier_high_threshold.w2 = w2;
        classifier_high_threshold.b2 = b2;

        // Input data
        let input = Matrix::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        // The low threshold classifier should be more likely to predict 1s
        let pred_low = classifier_low_threshold.predict(&input).unwrap();
        let pred_high = classifier_high_threshold.predict(&input).unwrap();

        // For this particular input and weights, we expect different predictions
        // based on the threshold (this is based on the sigmoid output being between 0.3 and 0.7)
        // This is less brittle than checking for specific values
        assert!(pred_low.sum() >= pred_high.sum());
    }

    #[test]
    fn test_compute_cost() {
        // Create a classifier
        let mut classifier =
            SingleLayerClassifier::new(2, 3, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        // Create simple test data
        let x = Matrix::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

        // For perfect predictions, cost should be close to 0
        let _ = Matrix::from_shape_vec((1, 3), vec![1.0, 0.0, 1.0]).unwrap();
        let perfect_y = Matrix::from_shape_vec((1, 3), vec![1.0, 0.0, 1.0]).unwrap();

        let cost_perfect = classifier.compute_cost(&x, &perfect_y).unwrap();
        assert_relative_eq!(cost_perfect, 0.0, epsilon = 1e-10);

        // For completely wrong predictions, cost should be high
        let _ = Matrix::from_shape_vec((1, 3), vec![0.01, 0.99, 0.01]).unwrap();
        let wrong_y = Matrix::from_shape_vec((1, 3), vec![0.99, 0.01, 0.99]).unwrap();

        let cost_wrong = classifier.compute_cost(&x, &wrong_y).unwrap();
        assert!(cost_wrong > 1.0); // Binary cross-entropy should be high for wrong predictions
    }

    #[test]
    fn test_compute_cost_validation() {
        let mut classifier =
            SingleLayerClassifier::new(2, 3, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        // Input data
        let x = Matrix::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        // Target with incorrect shape
        let y_wrong_shape = Matrix::from_shape_vec((2, 2), vec![1.0, 0.0, 1.0, 0.0]).unwrap();

        // The compute_cost function should return an error due to shape mismatch
        let result = classifier.compute_cost(&x, &y_wrong_shape);
        assert!(result.is_err());

        if let Err(ModelError::ShapeError(_)) = result {
            // Good, we got the expected error type
        } else {
            panic!("Expected ShapeError, got {:?}", result);
        }
    }
}

impl ParamCollection for SingleLayerClassifier {
    /// Retrieves a read-only view of the specified parameter.
    ///
    /// This method provides access to the model's parameters by name.
    ///
    /// # Parameters
    /// * `key` - The name of the parameter to retrieve: "w1", "b1", "w2", or "b2"
    ///
    /// # Returns
    /// * `Result<ndarray::ArrayView<f64, D>, ModelError>` - A view of the parameter or an error
    ///
    /// # Errors
    /// Returns `ModelError::InvalidParameter` if the key doesn't match any parameter,
    /// or `ShapeError` if the dimension conversion fails.
    fn get<D: ndarray::Dimension>(
        &self,
        key: &str,
    ) -> Result<ndarray::ArrayView<f64, D>, ModelError> {
        match key {
            "w1" => Ok(self.w1.view().into_dimensionality::<D>()?),
            "b1" => Ok(self.b1.view().into_dimensionality::<D>()?),
            "w2" => Ok(self.w2.view().into_dimensionality::<D>()?),
            "b2" => Ok(self.b2.view().into_dimensionality::<D>()?),
            _ => Err(ModelError::InvalidParameter(format!(
                "Invalid parameter key: {}",
                key
            ))),
        }
    }

    /// Retrieves a mutable view of the specified parameter.
    ///
    /// This method provides mutable access to the model's parameters by name,
    /// allowing for parameter updates during training.
    ///
    /// # Parameters
    /// * `key` - The name of the parameter to retrieve: "w1", "b1", "w2", or "b2"
    ///
    /// # Returns
    /// * `Result<ndarray::ArrayViewMut<f64, D>, ModelError>` - A mutable view of the parameter or an error
    ///
    /// # Errors
    /// Returns `ModelError::InvalidParameter` if the key doesn't match any parameter,
    /// or `ShapeError` if the dimension conversion fails.
    fn get_mut<D: ndarray::Dimension>(
        &mut self,
        key: &str,
    ) -> Result<ndarray::ArrayViewMut<f64, D>, ModelError> {
        match key {
            "w1" => Ok(self.w1.view_mut().into_dimensionality::<D>()?),
            "b1" => Ok(self.b1.view_mut().into_dimensionality::<D>()?),
            "w2" => Ok(self.w2.view_mut().into_dimensionality::<D>()?),
            "b2" => Ok(self.b2.view_mut().into_dimensionality::<D>()?),
            _ => Err(ModelError::InvalidParameter(format!(
                "Invalid parameter key: {}",
                key
            ))),
        }
    }

    /// Sets a parameter to the provided value.
    ///
    /// This method updates a parameter with new values, performing shape validation.
    ///
    /// # Parameters
    /// * `key` - The name of the parameter to update: "w1", "b1", "w2", or "b2"
    /// * `value` - The new values to assign to the parameter
    ///
    /// # Returns
    /// * `Result<(), ModelError>` - Success or an error
    ///
    /// # Errors
    /// Returns `ModelError::InvalidParameter` if the key doesn't match any parameter,
    /// or `ShapeError` if the shapes are incompatible.
    fn set<D: ndarray::Dimension>(
        &mut self,
        key: &str,
        value: ndarray::ArrayView<f64, D>,
    ) -> Result<(), ModelError> {
        match key {
            "w1" => {
                self.w1.assign(&value.to_shape(self.w1.shape())?);
                Ok(())
            }
            "b1" => {
                self.b1.assign(&value.to_shape(self.b1.shape())?);
                Ok(())
            }
            "w2" => {
                self.w2.assign(&value.to_shape(self.w2.shape())?);
                Ok(())
            }
            "b2" => {
                self.b2.assign(&value.to_shape(self.b2.shape())?);
                Ok(())
            }
            _ => Err(ModelError::InvalidParameter(format!(
                "Invalid parameter key: {}",
                key
            ))),
        }
    }

    /// Returns an iterator over all parameters.
    ///
    /// # Returns
    /// A vector of tuples containing parameter names and their corresponding array views.
    fn param_iter(&self) -> Vec<(&str, ndarray::ArrayView<f64, ndarray::IxDyn>)> {
        vec![
            ("w1", self.w1.view().into_dyn()),
            ("b1", self.b1.view().into_dyn()),
            ("w2", self.w2.view().into_dyn()),
            ("b2", self.b2.view().into_dyn()),
        ]
    }
}

#[cfg(test)]
mod param_collection_tests {
    use crate::{
        model::core::param_collection::ParamCollection,
        prelude::{ActivationFn, Matrix, SingleLayerClassifier},
    };

    #[test]
    fn test_param_collection_get() {
        let classifier =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        let w1 = classifier.get::<ndarray::Ix2>("w1");
        assert!(w1.is_ok());
        assert_eq!(w1.unwrap().shape(), &[10, 4]);

        let invalid = classifier.get::<ndarray::Ix2>("invalid");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_param_collection_set() {
        let mut classifier =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        let new_w1 = Matrix::zeros((10, 4));
        let result = classifier.set("w1", new_w1.view());
        assert!(result.is_ok());

        let wrong_shape = Matrix::zeros((5, 5));
        let result = classifier.set("w1", wrong_shape.view());
        assert!(result.is_err());

        let invalid = classifier.set("invalid", new_w1.view());
        assert!(invalid.is_err());
    }

    #[test]
    fn test_param_iter() {
        let classifier =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        let params = classifier.param_iter();
        assert_eq!(params.len(), 4);
        assert_eq!(params[0].0, "w1");
        assert_eq!(params[1].0, "b1");
        assert_eq!(params[2].0, "w2");
        assert_eq!(params[3].0, "b2");
    }
}

impl GradientCollection for SingleLayerClassifier {
    fn get_gradient<D: Dimension>(&self, key: &str) -> Result<ArrayView<f64, D>, ModelError> {
        match key {
            "w1" => Ok(self.w1.view().into_dimensionality::<D>()?),
            "b1" => Ok(self.b1.view().into_dimensionality::<D>()?),
            "w2" => Ok(self.w2.view().into_dimensionality::<D>()?),
            "b2" => Ok(self.b2.view().into_dimensionality::<D>()?),
            _ => Err(ModelError::InvalidParameter(format!(
                "Invalid parameter key: {}",
                key
            ))),
        }
    }

    fn set_gradient<D: Dimension>(
        &mut self,
        key: &str,
        value: ArrayView<f64, D>,
    ) -> Result<(), ModelError> {
        match key {
            "w1" => {
                self.w1.assign(&value.to_shape(self.w1.shape())?);
                Ok(())
            }
            "b1" => {
                self.b1.assign(&value.to_shape(self.b1.shape())?);
                Ok(())
            }
            "w2" => {
                self.w2.assign(&value.to_shape(self.w2.shape())?);
                Ok(())
            }
            "b2" => {
                self.b2.assign(&value.to_shape(self.b2.shape())?);
                Ok(())
            }
            _ => Err(ModelError::InvalidParameter(format!(
                "Invalid parameter key: {}",
                key
            ))),
        }
    }
}
#[cfg(test)]
mod gradient_collection_tests {
    use ndarray::Array;

    use crate::{
        model::core::param_collection::GradientCollection,
        prelude::{ActivationFn, Matrix, SingleLayerClassifier, Vector},
    };

    #[test]
    fn test_gradient_collection_get() {
        let classifier =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        let w1_grad = classifier.get_gradient::<ndarray::Ix2>("w1");
        assert!(w1_grad.is_ok());
        assert_eq!(w1_grad.unwrap().shape(), &[10, 4]);

        let b1_grad = classifier.get_gradient::<ndarray::Ix1>("b1");
        assert!(b1_grad.is_ok());
        assert_eq!(b1_grad.unwrap().shape(), &[10]);

        let invalid = classifier.get_gradient::<ndarray::Ix2>("invalid");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_gradient_collection_set() {
        let mut classifier =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        let new_w1_grad = Matrix::zeros((10, 4));
        let result = classifier.set_gradient("w1", new_w1_grad.view());
        assert!(result.is_ok());

        let new_b1_grad = Vector::ones(10);
        let result = classifier.set_gradient("b1", new_b1_grad.view());
        assert!(result.is_ok());

        // Test with wrong shape
        let wrong_shape = Matrix::zeros((5, 5));
        let result = classifier.set_gradient("w1", wrong_shape.view());
        assert!(result.is_err());

        // Test with invalid key
        let invalid = classifier.set_gradient("invalid", new_w1_grad.view());
        assert!(invalid.is_err());
    }

    #[test]
    fn test_gradient_dimension_conversion() {
        let mut classifier =
            SingleLayerClassifier::new(4, 10, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        // Test getting with dynamic dimensions
        let w2_grad = classifier.get_gradient::<ndarray::IxDyn>("w2");
        assert!(w2_grad.is_ok());
        assert_eq!(w2_grad.unwrap().shape(), &[1, 10]);

        // Test setting with dynamic dimensions
        let new_b2_grad = Array::ones(ndarray::IxDyn(&[10]));
        let result = classifier.set_gradient("b2", new_b2_grad.view());
        assert!(result.is_ok());
    }
}

impl OptimizableModel<Matrix, Matrix> for SingleLayerClassifier {
    /// Performs a forward pass through the model.
    fn forward(&mut self, input: &Matrix) -> Result<Matrix, ModelError> {
        // Forward pass through the network
        // Shape of input: (n_features, n_samples)
        // Shape of w1: (n_hidden_nodes, n_features)
        // Shape of b1: (n_hidden_nodes)
        // Shape of z1: (n_hidden_nodes, n_samples)
        // z1 = w1 @ input + b1
        // (n_hidden_nodes, n_samples) = (n_hidden_nodes, n_features) @ (n_features, n_samples) + (n_hidden_nodes)
        // Extract the number of samples, and the number of input features
        let z1 = self.compute_linear_activation(input, &self.w1, &self.b1)?;
        let a1 = self.compute_activation(&z1, self.hidden_layer_activation_fn);
        let z2 = self.compute_linear_activation(&a1, &self.w2, &self.b2)?;
        let a2 = self.compute_activation(&z2, self.output_layer_activation_fn);
        // Create new cache
        let cache = SingleLayerClassifierCache {
            a1: Some(a1),
            z1: Some(z1),
            a2: Some(a2.clone()), // Changed from Some(&a2) to avoid lifetime issues
            z2: Some(z2),
            cache_id: Some(Uuid::new_v4()),
        };
        self.set_cache(cache);
        // a2 is the output of the model
        Ok(a2)
    }

    fn backward(&mut self, _: &Matrix, _: &Matrix) -> Result<(), ModelError> {
        Ok(())
    }

    fn compute_output_gradient(&mut self, _: &Matrix, _: &Matrix) -> Result<Matrix, ModelError> {
        Ok(arr2(&[[0.0]]))
    }
}

#[cfg(test)]
mod optimizable_model_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_forward_dimensions() {
        let mut classifier = SingleLayerClassifier::new(
            3, // 3 features
            5, // 5 hidden nodes
            0.5,
            ActivationFn::Sigmoid,
            ActivationFn::ReLU,
        )
        .unwrap();

        // Input with 3 features and 2 samples
        let input = Matrix::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

        let result = classifier.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        // Should return one value per sample
        assert_eq!(output.len(), 2);
        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_forward_with_relu_sigmoid() {
        // Create a model with controlled weights for deterministic testing
        let mut classifier = SingleLayerClassifier::new(
            2, // 2 features
            3, // 3 hidden nodes
            0.5,
            ActivationFn::Sigmoid,
            ActivationFn::ReLU,
        )
        .unwrap();

        // Set predetermined weights and biases
        classifier.w1 = Matrix::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        classifier.b1 = Vector::from_vec(vec![0.1, 0.2, 0.3]);
        classifier.w2 = Matrix::from_shape_vec((1, 3), vec![0.7, 0.8, 0.9]).unwrap();
        classifier.b2 = Vector::from_vec(vec![0.1, 0.2, 0.3]); // Note: only first value used

        // Single sample input
        let input = Matrix::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        let result = classifier.forward(&input);
        assert!(result.is_ok());

        // Manual calculation:
        // z1 = w1 * x + b1 = [0.1*1.0 + 0.2*2.0 + 0.1, 0.3*1.0 + 0.4*2.0 + 0.2, 0.5*1.0 + 0.6*2.0 + 0.3] = [0.6, 1.3, 2.0]
        // a1 = ReLU(z1) = [0.6, 1.3, 2.0] (all positive)
        // z2 = w2 * a1 + b2[0] = 0.7*0.6 + 0.8*1.3 + 0.9*2.0 + 0.1 = 0.42 + 1.04 + 1.8 + 0.1 = 3.36
        // a2 = sigmoid(z2) = 1/(1+exp(-3.36)) ≈ 0.966

        let expected = Vector::from_vec(vec![0.966]);
        let output = result.unwrap();

        assert_relative_eq!(output[[0, 0]], expected[0], epsilon = 1e-3);
    }

    #[test]
    fn test_forward_with_different_activations() {
        // Test with tanh in hidden layer and linear in output layer
        let mut classifier =
            SingleLayerClassifier::new(2, 3, 0.5, ActivationFn::ReLU, ActivationFn::Sigmoid)
                .unwrap();

        // Set predetermined weights and biases
        classifier.w1 = Matrix::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        classifier.b1 = Vector::from_vec(vec![0.1, 0.2, 0.3]);
        classifier.w2 = Matrix::from_shape_vec((1, 3), vec![0.7, 0.8, 0.9]).unwrap();
        classifier.b2 = Vector::from_vec(vec![0.1, 0.2, 0.3]);

        let input = Matrix::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();

        let result = classifier.forward(&input);
        assert!(result.is_ok());

        // With different activation functions, the result will differ
        // This test checks that the forward function runs successfully with various activation functions
    }

    #[test]
    fn test_forward_batch_processing() {
        let mut classifier =
            SingleLayerClassifier::new(2, 4, 0.5, ActivationFn::Sigmoid, ActivationFn::ReLU)
                .unwrap();

        // Batch of 3 samples with 2 features each
        let input = Matrix::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

        let result = classifier.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        // Should have one prediction per sample
        assert_eq!(output.len(), 3);

        // All outputs should be between 0 and 1 (sigmoid activation)
        for val in output.iter() {
            assert!(0.0 <= *val && *val <= 1.0);
        }
    }
}
