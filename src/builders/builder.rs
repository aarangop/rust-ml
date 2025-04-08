use crate::core::error::ModelError;
use crate::model::core::optimizable_model::OptimizableModel;

pub trait Builder<M, Input, Output>
where
    M: OptimizableModel<Input, Output>,
{
    fn build(&self) -> Result<M, ModelError>;
}
