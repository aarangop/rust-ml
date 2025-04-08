use crate::core::error::ModelError;
use crate::model::core::base::BaseModel;

pub trait Builder<M, Input, Output>
where
    M: BaseModel<Input, Output>,
{
    fn build(&self) -> Result<M, ModelError>;
}
