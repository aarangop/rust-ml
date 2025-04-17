use crate::bench::core::error::ProfilerError;
use crate::bench::core::profiler::Profiler;
use crate::bench::core::train_metrics::TrainMetrics;
use crate::bench::regression_metrics::RegressionMetrics;
use crate::model::core::base::BaseModel;
use crate::model::core::regression_model::RegressionModel;
use crate::optim::core::optimizer::Optimizer;
use std::marker::PhantomData;
use std::time::Instant;

/// A profiler for regression models that measures training time and computes regression metrics.
///
/// This struct implements the `Profiler` trait specifically for regression models,
/// providing performance assessment through metrics such as Mean Squared Error (MSE),
/// Root Mean Squared Error (RMSE), and coefficient of determination (R²).
///
/// # Type Parameters
///
/// * `Model` - The regression model type being profiled
/// * `Opt` - The optimizer type used for training
/// * `Input` - The input data type (features)
/// * `Output` - The output data type (target values)
pub struct RegressionProfiler<Model, Opt, Input, Output> {
    _phantom: PhantomData<(Model, Opt, Input, Output)>,
}

impl<Model, Opt, Input, Output> RegressionProfiler<Model, Opt, Input, Output> {
    /// Creates a new RegressionProfiler instance.
    ///
    /// # Returns
    ///
    /// A new RegressionProfiler instance.
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<Model, Opt, Input, Output> Default for RegressionProfiler<Model, Opt, Input, Output> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Model, Opt, Input, Output> Profiler<Model, Opt, Input, Output>
    for RegressionProfiler<Model, Opt, Input, Output>
where
    Model: BaseModel<Input, Output> + RegressionModel<Input, Output>,
    Opt: Optimizer<Input, Output, Model>,
{
    type EvalMetrics = RegressionMetrics;

    /// Profiles the training process of a regression model.
    ///
    /// Measures the time taken to train the model and computes regression metrics
    /// on the provided data.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the regression model being trained
    /// * `optimizer` - Mutable reference to the optimizer used for training
    /// * `x` - Reference to input features
    /// * `y` - Reference to target values
    ///
    /// # Returns
    ///
    /// A tuple containing training metrics (including training time) and regression metrics
    /// (MSE, RMSE, R²), or a ProfilerError if an error occurs.
    fn train(
        &self,
        model: &mut Model,
        optimizer: &mut Opt,
        x: &Input,
        y: &Output,
    ) -> Result<(TrainMetrics, Self::EvalMetrics), ProfilerError> {
        // Train model and measure training time.
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

    /// Profiles the evaluation process of a regression model.
    ///
    /// Computes regression metrics for the model on the provided data.
    ///
    /// # Arguments
    ///
    /// * `model` - Mutable reference to the regression model being evaluated
    /// * `x` - Reference to input features
    /// * `y` - Reference to target values
    ///
    /// # Returns
    ///
    /// Regression metrics (MSE, RMSE, R²), or a ProfilerError if an error occurs.
    fn profile_evaluation(
        &self,
        model: &mut Model,
        x: &Input,
        y: &Output,
    ) -> Result<Self::EvalMetrics, ProfilerError> {
        Ok(model.compute_metrics(x, y)?)
    }
}
