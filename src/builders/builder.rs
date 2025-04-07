use crate::core::error::ModelError;
use crate::model::ml_model::OptimizableModel;

pub trait Builder<Input, Output> {
    fn build(&self) -> Result<Box<dyn OptimizableModel<Input, Output>>, ModelError>;
}
