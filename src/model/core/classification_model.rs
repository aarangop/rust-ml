use crate::bench::classification_metrics::ClassificationMetrics;
use crate::core::error::ModelError;

/// A trait for models that perform classification tasks.
///
/// This trait defines common evaluation metrics and functionality for classification models,
/// allowing for standardized performance assessment across different implementations.
///
/// # Type Parameters
/// * `Input`: The type of input data the model accepts
/// * `Output`: The type of output data against which predictions are compared
pub trait ClassificationModel<Input, Output> {
    /// Calculates the accuracy of the model on the given data.
    ///
    /// Accuracy is defined as the proportion of correct predictions among the total number of predictions.
    ///
    /// # Arguments
    /// * `x` - The input data
    /// * `y` - The expected output/ground truth
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The calculated accuracy score or an error
    fn accuracy(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;

    /// Calculates the loss of the model on the given data.
    ///
    /// The specific loss function depends on the model implementation.
    ///
    /// # Arguments
    /// * `x` - The input data
    /// * `y` - The expected output/ground truth
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The calculated loss value or an error
    fn loss(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;

    /// Calculates the recall score of the model on the given data.
    ///
    /// Recall (also known as sensitivity) measures the proportion of actual positives
    /// that were correctly identified by the model.
    ///
    /// # Arguments
    /// * `x` - The input data
    /// * `y` - The expected output/ground truth
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The calculated recall score or an error
    fn recall(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;

    /// Calculates the F1 score of the model on the given data.
    ///
    /// F1 score is the harmonic mean of precision and recall, providing a balance
    /// between the two metrics.
    ///
    /// # Arguments
    /// * `x` - The input data
    /// * `y` - The expected output/ground truth
    ///
    /// # Returns
    /// * `Result<f64, ModelError>` - The calculated F1 score or an error
    fn f1_score(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;

    /// Computes multiple evaluation metrics for the model on the given data.
    ///
    /// This method allows for efficient calculation of multiple metrics in a single pass
    /// through the data, potentially optimizing performance when multiple metrics are needed.
    ///
    /// # Arguments
    /// * `x` - The input data
    /// * `y` - The expected output/ground truth
    ///
    /// # Returns
    /// * `Result<ModelParams, ModelError>` - A collection of calculated metrics or an error
    fn compute_metrics(&self, x: &Input, y: &Output) -> Result<ClassificationMetrics, ModelError>;
}
