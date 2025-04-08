use crate::bench::core::error::ProfilerError;
use crate::bench::core::train_metrics::TrainMetrics;

pub trait Profiler<Model, Opt, Input, Output> {
    type EvalMetrics;

    fn profile_training(
        &self,
        model: &mut Model,
        optimizer: &mut Opt,
        x: &Input,
        y: &Output,
    ) -> Result<(TrainMetrics, Self::EvalMetrics), ProfilerError>;

    fn profile_evaluation(
        &self,
        model: &mut Model,
        x: &Input,
        y: &Output,
    ) -> Result<Self::EvalMetrics, ProfilerError>;
}
