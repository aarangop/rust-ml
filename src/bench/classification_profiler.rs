use crate::bench::classification_metrics::ClassificationMetrics;
use crate::bench::core::error::ProfilerError;
use crate::bench::core::profiler::Profiler;
use crate::bench::core::train_metrics::TrainMetrics;
use crate::model::core::classification_model::ClassificationModel;
use crate::model::core::optimizable_model::OptimizableModel;
use crate::optimization::core::optimizer::Optimizer;
use std::marker::PhantomData;
use std::time::Instant;

pub struct ClassificationProfiler<Model, Opt, Input, Output> {
    _phantom: std::marker::PhantomData<(Model, Opt, Input, Output)>,
}

impl<Model, Opt, Input, Output> ClassificationProfiler<Model, Opt, Input, Output> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<Model, Opt, Input, Output> Profiler<Model, Opt, Input, Output>
    for ClassificationProfiler<Model, Opt, Input, Output>
where
    Model: OptimizableModel<Input, Output> + ClassificationModel<Input, Output>,
    Opt: Optimizer<Input, Output>,
{
    type EvalMetrics = ClassificationMetrics;

    fn profile_training(
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

    fn profile_evaluation(
        &self,
        model: &mut Model,
        x: &Input,
        y: &Output,
    ) -> Result<Self::EvalMetrics, ProfilerError> {
        todo!()
    }
}
