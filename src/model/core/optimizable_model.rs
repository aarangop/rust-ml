use crate::core::param_manager::ParamManager;
use crate::model::core::base::{BackwardPropagation, BaseModel, ForwardPropagation};

pub trait OptimizableModel<Input, Output>:
    BaseModel<Input, Output>
    + ForwardPropagation<Input, Output>
    + BackwardPropagation<Input, Output>
    + ParamManager
{
}
