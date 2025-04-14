use crate::bench::core::error::ProfilerError;
use crate::bench::core::train_metrics::TrainMetrics;

/// Trait for profiling and benchmarking machine learning models during training and evaluation.
///
/// This trait defines methods for collecting performance metrics during model training and evaluation.
/// It allows for consistent measurement of training time and model performance metrics across
/// different model types and optim strategies.
///
/// # Type Parameters
///
/// * `Model` - The machine learning model type being profiled
/// * `Opt` - The optimizer type used for training
/// * `Input` - The input data type (features)
/// * `Output` - The output data type (targets/labels)
pub trait Profiler<Model, Opt, Input, Output> {
    /// The type of evaluation metrics returned by the profiler
    type EvalMetrics;

    /// Profiles the training process of a model, collecting training time and evaluation metrics.
    ///
    /// This method measures the time taken for training while also computing performance metrics
    /// on the provided data.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the model being trained
    /// * `optimizer` - Mutable reference to the optimizer used for training
    /// * `x` - Reference to input features
    /// * `y` - Reference to output targets
    ///
    /// # Returns
    ///
    /// A tuple containing training metrics (including training time) and evaluation metrics
    /// specific to the model type, or a ProfilerError if an error occurs during profiling.
    fn profile_training(
        &self,
        model: &mut Model,
        optimizer: &mut Opt,
        x: &Input,
        y: &Output,
    ) -> Result<(TrainMetrics, Self::EvalMetrics), ProfilerError>;

    /// Profiles the evaluation process of a model, computing performance metrics.
    ///
    /// This method evaluates the model on the provided data and returns metrics
    /// specific to the model type.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the model being evaluated
    /// * `x` - Reference to input features
    /// * `y` - Reference to output targets
    ///
    /// # Returns
    ///
    /// Evaluation metrics specific to the model type, or a ProfilerError if an error occurs
    /// during evaluation.
    fn profile_evaluation(
        &self,
        model: &mut Model,
        x: &Input,
        y: &Output,
    ) -> Result<Self::EvalMetrics, ProfilerError>;
}
