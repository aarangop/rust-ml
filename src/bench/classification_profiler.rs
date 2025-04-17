use crate::bench::classification_metrics::ClassificationMetrics;
use crate::bench::core::error::ProfilerError;
use crate::bench::core::profiler::Profiler;
use crate::bench::core::train_metrics::TrainMetrics;
use crate::model::core::base::BaseModel;
use crate::model::core::classification_model::ClassificationModel;
use crate::optim::core::optimizer::Optimizer;
use std::marker::PhantomData;
use std::time::Instant;

/// A profiler for classification models that measures training time and computes classification metrics.
///
/// This struct implements the `Profiler` trait specifically for classification models,
/// providing performance assessment through metrics such as accuracy, precision, recall,
/// and F1 score.
///
/// # Type Parameters
///
/// * `Model` - The classification model type being profiled
/// * `Opt` - The optimizer type used for training
/// * `Input` - The input data type (features)
/// * `Output` - The output data type (labels/classes)
pub struct ClassificationProfiler<Model, Opt, Input, Output> {
    _phantom: std::marker::PhantomData<(Model, Opt, Input, Output)>,
}

impl<Model, Opt, Input, Output> ClassificationProfiler<Model, Opt, Input, Output> {
    /// Creates a new ClassificationProfiler instance.
    ///
    /// # Returns
    ///
    /// A new ClassificationProfiler instance.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<Model, Opt, Input, Output> Profiler<Model, Opt, Input, Output>
    for ClassificationProfiler<Model, Opt, Input, Output>
where
    Model: BaseModel<Input, Output> + ClassificationModel<Input, Output>,
    Opt: Optimizer<Input, Output, Model>,
{
    type EvalMetrics = ClassificationMetrics;

    /// Profiles the training process of a classification model.
    ///
    /// Measures the time taken to train the model and computes classification metrics
    /// on the provided data.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the classification model being trained
    /// * `optimizer` - Mutable reference to the optimizer used for training
    /// * `x` - Reference to input features
    /// * `y` - Reference to target labels
    ///
    /// # Returns
    ///
    /// A tuple containing training metrics (including training time) and classification metrics
    /// (accuracy, precision, recall, F1 score), or a ProfilerError if an error occurs.
    fn train(
        &self,
        model: &mut Model,
        optimizer: &mut Opt,
        x: &Input,
        y: &Output,
    ) -> Result<(TrainMetrics, Self::EvalMetrics), ProfilerError> {
        let tick = Instant::now();
        optimizer.fit(model, x, y)?;
        let tock = Instant::now();

        // Store elapsed time and create struct with training metrics.
        let elapsed = tock.duration_since(tick).as_secs_f64();
        let train_metrics = TrainMetrics::new(elapsed);

        // Compute model evaluation metrics.
        let eval_metrics = model.compute_metrics(x, y)?;

        Ok((train_metrics, eval_metrics))
    }

    /// Profiles the evaluation process of a classification model.
    ///
    /// Computes classification metrics for the model on the provided data.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the classification model being evaluated
    /// * `x` - Reference to input features
    /// * `y` - Reference to target labels
    ///
    /// # Returns
    ///
    /// Classification metrics (accuracy, precision, recall, F1 score), or a ProfilerError if an error occurs.
    fn profile_evaluation(
        &self,
        model: &mut Model,
        x: &Input,
        y: &Output,
    ) -> Result<Self::EvalMetrics, ProfilerError> {
        todo!()
    }
}
