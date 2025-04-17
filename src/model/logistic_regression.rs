use crate::bench::classification_metrics::ClassificationMetrics;
use crate::builders::logistic_regression::LogisticRegressionBuilder;
use crate::core::activations::activation::Activation;
use crate::core::activations::activation_functions::ActivationFn;
use crate::core::activations::leaky_relu::LeakyReLU;
use crate::core::activations::relu::ReLU;
use crate::core::activations::sigmoid::Sigmoid;
use crate::core::activations::tanh::Tanh;
use crate::core::error::ModelError;
use crate::core::types::{Matrix, Vector};
use crate::model::core::base::{BaseModel, OptimizableModel};
use crate::model::core::classification_model::ClassificationModel;
use crate::model::core::param_collection::{GradientCollection, ParamCollection};
use ndarray::{ArrayView, ArrayViewMut, Dimension, IxDyn};

/// A logistic regression model for binary classification.
///
/// This model uses a linear combination of features and weights, followed by an activation function,
/// to predict the probability of an input belonging to a positive class.
#[derive(Debug)]
pub struct LogisticRegression {
    /// Model weights for each feature
    pub weights: Vector,
    /// Bias term (intercept)
    pub bias: Vector,
    /// Activation function used for prediction
    pub activation_fn: ActivationFn,
}

impl LogisticRegression {
    /// Creates a new LogisticRegression model with the specified number of features and activation function.
    ///
    /// # Arguments
    ///
    /// * `n_features` - The number of input features
    /// * `activation_fn` - The activation function to use
    ///
    /// # Returns
    ///
    /// A new LogisticRegression instance with weights initialized to zeros
    pub fn new(n_features: usize, activation_fn: ActivationFn) -> Self {
        let weights = Vector::zeros(n_features);
        let bias = Vector::from_elem(1, 0.0);
        Self {
            weights,
            bias,
            activation_fn,
        }
    }

    /// Returns a builder for creating LogisticRegression models with custom configurations.
    ///
    /// # Returns
    ///
    /// A new LogisticRegressionBuilder instance
    pub fn builder() -> LogisticRegressionBuilder {
        LogisticRegressionBuilder::new()
    }

    /// Computes the activation for the given input vector.
    ///
    /// # Arguments
    ///
    /// * `z` - Input vector
    ///
    /// # Returns
    ///
    /// The result of applying the activation function to the input vector
    fn compute_activation(&self, z: &Vector) -> Result<Vector, ModelError> {
        match self.activation_fn {
            ActivationFn::Sigmoid => Ok(Sigmoid::activate(z)),
            ActivationFn::ReLU => Ok(ReLU::activate(z)),
            ActivationFn::Tanh => Ok(Tanh::activate(z)),
            ActivationFn::LeakyReLU => Ok(LeakyReLU::activate(z)),
        }
    }

    fn compute_z(&self, x: &Matrix) -> Result<Vector, ModelError> {
        let z = self.weights.t().dot(x) + &self.bias;
        Ok(z)
    }

    /// Computes the derivative of the activation function for the given input vector.
    ///
    /// # Arguments
    ///
    /// * `z` - Input vector
    ///
    /// # Returns
    ///
    /// The derivative of the activation function applied to the input
    fn compute_derivative(&self, z: &Vector) -> Result<Vector, ModelError> {
        match self.activation_fn {
            ActivationFn::Sigmoid => Ok(Sigmoid::derivative(z)),
            ActivationFn::ReLU => Ok(ReLU::derivative(z)),
            ActivationFn::Tanh => Ok(Tanh::derivative(z)),
            ActivationFn::LeakyReLU => Ok(LeakyReLU::derivative(z)),
        }
    }
}

impl ParamCollection for LogisticRegression {
    fn get<D: Dimension>(&self, key: &str) -> Result<ArrayView<f64, D>, ModelError> {
        match key {
            "weights" => Ok(self.weights.view().into_dimensionality::<D>()?),
            "bias" => Ok(self.bias.view().into_dimensionality::<D>()?),
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }

    fn get_mut<D: Dimension>(&mut self, key: &str) -> Result<ArrayViewMut<f64, D>, ModelError> {
        match key {
            "weights" => Ok(self.weights.view_mut().into_dimensionality::<D>()?),
            "bias" => Ok(self.bias.view_mut().into_dimensionality::<D>()?),
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }

    fn set<D: Dimension>(&mut self, key: &str, value: ArrayView<f64, D>) -> Result<(), ModelError> {
        match key {
            "weights" => {
                self.weights.assign(&value.to_shape(self.weights.shape())?);
                Ok(())
            }
            "bias" => {
                self.bias.assign(&value.to_shape(self.bias.shape())?);
                Ok(())
            }
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }

    fn param_iter(&self) -> Vec<(&str, ArrayView<f64, IxDyn>)> {
        vec![
            ("weights", self.weights.view().into_dyn()),
            ("bias", self.bias.view().into_dyn()),
        ]
    }
}

impl GradientCollection for LogisticRegression {
    fn get_gradient<D: Dimension>(&self, key: &str) -> Result<ArrayView<f64, D>, ModelError> {
        match key {
            "weights" => Ok(self.weights.view().into_dimensionality::<D>()?),
            "bias" => Ok(self.bias.view().into_dimensionality::<D>()?),
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }

    fn set_gradient<D: Dimension>(
        &mut self,
        key: &str,
        value: ArrayView<f64, D>,
    ) -> Result<(), ModelError> {
        match key {
            "weights" => {
                self.weights.assign(&value.to_shape(self.weights.shape())?);
                Ok(())
            }
            "bias" => {
                self.bias.assign(&value.to_shape(self.bias.shape())?);
                Ok(())
            }
            _ => Err(ModelError::KeyError(key.to_string())),
        }
    }
}

impl OptimizableModel<Matrix, Vector> for LogisticRegression {
    fn forward(&self, input: &Matrix) -> Result<Vector, ModelError> {
        let z = self.compute_z(input)?;
        let a = self.compute_activation(&z)?;
        Ok(a)
    }

    fn backward(&mut self, input: &Matrix, dz: &Vector) -> Result<(), ModelError> {
        let m = input.shape()[1] as f64;

        // Calculate gradients
        // For matrix dimensions to work correctly:
        // - input has shape (n_features, n_samples)
        // - dz has shape (n_samples,)
        // - to get dw with shape (n_features,), we need to do input.dot(dz)
        let dw = input.dot(dz) / m;
        let db = dz.sum() / m;

        // Set the gradients
        self.set_gradient("weights", dw.view())?;
        self.set_gradient("bias", ArrayView::from(&[db]))?;

        Ok(())
    }

    /// Computes the gradient of the loss function with regards to the prediction (dJ/dy)
    ///
    /// # Arguments
    ///
    /// * `x` - Input feature matrix
    /// * `y` - Expected output vector
    ///
    /// # Returns
    ///
    /// The gradient vector
    fn compute_output_gradient(&self, x: &Matrix, y: &Vector) -> Result<Vector, ModelError> {
        // Forward pass to get predictions
        let z = self.compute_z(x)?;
        let y_hat = self.compute_activation(&z)?;

        // Compute the derivative of the activation function
        let g_prime_of_z = self.compute_derivative(&z)?;

        // Compute the gradient of the loss function with respect to the predictions
        let dy = (1.0 - y) / (1.0 - &y_hat) - y / &y_hat;

        // Apply the chain rule to get the gradient of the loss function with respect to z
        // dz = dy * g'(z)
        let dz = dy * g_prime_of_z;
        Ok(dz)
    }
}

#[cfg(test)]
mod optimizable_model_tests {
    use ndarray::{arr1, arr2, ArrayView1};

    use crate::core::activations::activation::Activation;
    use crate::core::activations::activation_functions::ActivationFn;
    use crate::core::activations::sigmoid::Sigmoid;
    use crate::core::types::{Matrix, Scalar};
    use crate::model::core::base::OptimizableModel;
    use crate::model::core::param_collection::GradientCollection;
    use crate::model::logistic_regression::LogisticRegression;

    #[test]
    fn test_logistic_regression_forward_sigmoid() {
        // Create model
        let mut model = LogisticRegression::new(3, ActivationFn::Sigmoid);
        // Assign weights and bias explicitly.
        let weights = arr1(&[0.5, -0.2, 0.1]);
        let bias = Scalar::from_elem((), 0.2);
        model.weights.assign(&weights);
        model.bias.assign(&bias);
        // Create input data
        let input = Matrix::zeros((3, 3));
        // Compute expected output.
        let z = model.weights.t().dot(&input) + bias;
        let a = Sigmoid::activate(&z);
        let expected_output = a;

        // Forward pass
        let output = model.forward(&input).unwrap();

        assert_eq!(output.shape(), [3]);
        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_compute_output_gradient() {
        // Create model with sigmoid activation
        let mut model = LogisticRegression::new(2, ActivationFn::Sigmoid);
        let weights = arr1(&[0.5, -0.3]);
        let bias = Scalar::from_elem((), 0.1);
        model.weights.assign(&weights);
        model.bias.assign(&bias);

        // Create test data
        let x = arr2(&[[0.2, 0.7], [0.3, 0.5]]);
        let y = arr1(&[0.5, 1.0]); // Expected outputs

        // Compute forward pass to get predictions
        let y_hat = model.forward(&x).unwrap();
        // Manually compute the gradient using the formula for sigmoid
        // dz = y_hat - y 
        let expected_dz = &y_hat - &y;
        // Get the computed gradient
        let dz = model.compute_output_gradient(&x, &y).unwrap();

        // Check dimensions and values (with a small epsilon for floating point comparison)
        assert_eq!(dz.shape(), expected_dz.shape());

        for (a, b) in dz.iter().zip(expected_dz.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_backward() {
        // Create model with sigmoid activation
        let mut model = LogisticRegression::new(2, ActivationFn::Sigmoid);
        let weights = arr1(&[0.5, -0.3]);
        let bias = Scalar::from_elem((), 0.1);
        model.weights.assign(&weights);
        model.bias.assign(&bias);

        // Create test data
        let x = arr2(&[[0.2, 0.7], [0.3, 0.5]]); // 2x2 matrix with 2 features and 2 samples

        // Create a test gradient vector
        let dz = arr1(&[0.1, -0.2]);

        println!("x.shape(): {:?}", x.shape());
        println!("dz.shape(): {:?}", dz.shape());

        // Call backward
        model.backward(&x, &dz).unwrap();

        // Manually compute the expected gradients
        // For weights: dw = (x.dot(dz)) / m where m is number of samples
        let m = x.shape()[1] as f64;
        let expected_dw = x.dot(&dz) / m;

        // For bias: db = sum(dz) / m
        let expected_db = dz.sum() / m;

        // Get the gradients from the model
        let actual_dw: ArrayView1<f64> = model.get_gradient("weights").unwrap();
        let actual_db: ArrayView1<f64> = model.get_gradient("bias").unwrap();
        let actual_db_value = actual_db[0];

        // Check dimensions and values (with a small epsilon for floating point comparison)
        assert!(
            (actual_db_value - expected_db).abs() < 1e-5,
            "Expected {}, got {}",
            expected_db,
            actual_db_value
        );

        for (a, b) in actual_dw.iter().zip(expected_dw.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {}, got {}", b, a);
        }
    }
}
/// Implementation of BaseModel trait for LogisticRegression
impl BaseModel<Matrix, Vector> for LogisticRegression {
    /// Makes binary predictions for the given input data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input feature matrix
    ///
    /// # Returns
    ///
    /// A vector of predictions (0.0 or 1.0)
    fn predict(&self, x: &Matrix) -> Result<Vector, ModelError> {
        let bias = self.bias[0];
        let z = self.weights.dot(x) + bias;
        let a = self.compute_activation(&z)?;
        let y_hat = a.mapv(|x| if x >= 0.5 { 1.0 } else { 0.0 });
        Ok(y_hat)
    }

    /// Computes the cost/loss for the given input and expected output.
    ///
    /// # Arguments
    ///
    /// * `x` - Input feature matrix
    /// * `y` - Expected output vector
    ///
    /// # Returns
    ///
    /// The computed loss value
    fn compute_cost(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_hat = self.predict(x)?;
        let cost = y * y_hat.ln() + (1.0 - y) * (1.0 - y_hat).ln();
        Ok(cost.sum())
    }
}

/// Implementation of ClassificationModel trait for LogisticRegression
impl ClassificationModel<Matrix, Vector> for LogisticRegression {
    /// Calculates the accuracy of the model on the given data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input feature matrix
    /// * `y` - Expected output vector
    ///
    /// # Returns
    ///
    /// The accuracy as a value between 0.0 and 1.0
    fn accuracy(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_pred = self.predict(x)?;
        let y_pred_binary = y_pred.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 });
        let correct = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| (pred - actual).abs() < f64::EPSILON)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    /// Calculates the loss (binary cross-entropy) of the model on the given data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input feature matrix
    /// * `y` - Expected output vector
    ///
    /// # Returns
    ///
    /// The computed loss value
    fn loss(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        // Binary cross-entropy loss
        let y_pred = self.predict(x)?;
        let epsilon = 1e-15; // prevent log(0)
        let y_pred = y_pred.mapv(|val| val.max(epsilon).min(1.0 - epsilon));
        let loss = y
            .iter()
            .zip(y_pred.iter())
            .map(|(y_i, y_pred_i)| -y_i * y_pred_i.ln() - (1.0 - y_i) * (1.0 - y_pred_i).ln())
            .sum::<f64>()
            / y.len() as f64;
        Ok(loss)
    }

    /// Calculates the recall (sensitivity) of the model on the given data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input feature matrix
    /// * `y` - Expected output vector
    ///
    /// # Returns
    ///
    /// The recall as a value between 0.0 and 1.0
    fn recall(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_pred = self.predict(x)?;
        let y_pred_binary = y_pred.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 });

        let true_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual > 0.5)
            .count();

        let actual_positives = y.iter().filter(|&&actual| actual > 0.5).count();

        if actual_positives == 0 {
            return Ok(0.0);
        }

        Ok(true_positives as f64 / actual_positives as f64)
    }

    /// Calculates the F1 score of the model on the given data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input feature matrix
    /// * `y` - Expected output vector
    ///
    /// # Returns
    ///
    /// The F1 score as a value between 0.0 and 1.0
    fn f1_score(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_pred = self.predict(x)?;
        let y_pred_binary = y_pred.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 });

        let true_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual > 0.5)
            .count() as f64;

        let false_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual <= 0.5)
            .count() as f64;

        let false_negatives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred <= 0.5 && *actual > 0.5)
            .count() as f64;

        let precision = if true_positives + false_positives == 0.0 {
            0.0
        } else {
            true_positives / (true_positives + false_positives)
        };

        let recall = if true_positives + false_negatives == 0.0 {
            0.0
        } else {
            true_positives / (true_positives + false_negatives)
        };

        if precision + recall == 0.0 {
            return Ok(0.0);
        }

        Ok(2.0 * precision * recall / (precision + recall))
    }

    /// Computes all classification metrics for the model on the given data.
    ///
    /// # Arguments
    ///
    /// * `x` - Input feature matrix
    /// * `y` - Expected output vector
    ///
    /// # Returns
    ///
    /// A ClassificationMetrics struct containing accuracy, loss, precision, recall and F1 score
    fn compute_metrics(&self, x: &Matrix, y: &Vector) -> Result<ClassificationMetrics, ModelError> {
        let accuracy = self.accuracy(x, y)?;
        let loss = self.loss(x, y)?;
        let recall = self.recall(x, y)?;
        let f1 = self.f1_score(x, y)?;

        // Calculate precision
        let y_pred = self.predict(x)?;
        let y_pred_binary = y_pred.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 });

        let true_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual > 0.5)
            .count() as f64;

        let false_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual <= 0.5)
            .count() as f64;

        let precision = if true_positives + false_positives == 0.0 {
            0.0
        } else {
            true_positives / (true_positives + false_positives)
        };

        Ok(ClassificationMetrics {
            accuracy,
            loss,
            precision,
            recall,
            f1_score: f1,
        })
    }
}
