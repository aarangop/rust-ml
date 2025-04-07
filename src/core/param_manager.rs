use crate::core::error::ModelError;
use crate::core::types::ModelParams;
use ndarray::{ArrayView, IxDyn};

pub trait ParamManager {
    fn get_params(&self) -> ModelParams;
    fn update_params(&mut self, params: ModelParams);
    fn get_param(&self, key: &str) -> Result<ArrayView<f64, IxDyn>, ModelError>; // Use array view because the returned parameters should be read-only.
}
