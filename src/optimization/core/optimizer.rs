use crate::core::error::ModelError;
use crate::model::core::optimizable_model::OptimizableModel;

pub trait Optimizer<Input, Output> {
    fn fit(
        &mut self,
        model: &mut dyn OptimizableModel<Input, Output>,
        x: &Input,
        y: &Output,
    ) -> Result<(), ModelError>;
}
