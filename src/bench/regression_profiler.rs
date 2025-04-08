use crate::bench::core::error::ProfilerError;
use crate::bench::core::profiler::Profiler;
use crate::bench::core::train_metrics::TrainMetrics;
use crate::bench::regression_metrics::RegressionMetrics;
use crate::model::core::optimizable_model::OptimizableModel;
use crate::model::core::regression_model::RegressionModel;
use crate::optimization::core::optimizer::Optimizer;
use std::marker::PhantomData;
use std::time::Instant;

pub struct RegressionProfiler<Model, Opt, Input, Output> {
    _phantom: PhantomData<(Model, Opt, Input, Output)>,
}

impl<Model, Opt, Input, Output> RegressionProfiler<Model, Opt, Input, Output> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<Model, Opt, Input, Output> Profiler<Model, Opt, Input, Output>
    for RegressionProfiler<Model, Opt, Input, Output>
where
    Model: OptimizableModel<Input, Output> + RegressionModel<Input, Output>,
    Opt: Optimizer<Input, Output>,
{
    type TrainMetrics = TrainMetrics;
    type EvalMetrics = RegressionMetrics;

    fn profile_training(
        &self,
        model: &mut Model,
        optimizer: &mut Opt,
        x: &Input,
        y: &Output,
    ) -> Result<(Self::TrainMetrics, Self::EvalMetrics), ProfilerError> {
        // Train model and measure training time.
        let tick = Instant::now();
        optimizer.fit(model, x, y)?;
        let tock = Instant::now();

        // Store elapsed time and create struct with training metrics.
        let elapsed = tock.duration_since(tick).as_secs_f64();
        let train_metrics = Self::TrainMetrics::new(elapsed);

        // Compute model evaluation metrics.
        let eval_metrics = model.compute_metrics(x, y)?;

        Ok((train_metrics, eval_metrics))
    }

    fn profile_evaluation(
        &self,
        model: &mut Model,
        x: &Input,
        y: &Output,
    ) -> Result<Self::EvalMetrics, ProfilerError> {
        Ok(model.compute_metrics(x, y)?)
    }
}
