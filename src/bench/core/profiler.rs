use crate::bench::core::error::ProfilerError;

pub trait Profiler<Model, Opt, Input, Output> {
    type TrainMetrics;
    type EvalMetrics;

    fn profile_training(
        &self,
        model: &mut Model,
        optimizer: &mut Opt,
        x: &Input,
        y: &Output,
    ) -> Result<(Self::TrainMetrics, Self::EvalMetrics), ProfilerError>;

    fn profile_evaluation(
        &self,
        model: &mut Model,
        x: &Input,
        y: &Output,
    ) -> Result<Self::EvalMetrics, ProfilerError>;
}
