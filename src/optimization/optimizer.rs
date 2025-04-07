use crate::core::error::ModelError;
use crate::model::ml_model::OptimizableModel;

pub trait Optimizer<Input, Output, Model: ?Sized> {
    fn fit(&mut self, model: &mut Model, x: &Input, y: &Output) -> Result<(), ModelError>
    where
        Model: OptimizableModel<Input, Output>;
}
