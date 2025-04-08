use crate::core::error::ModelError;
use crate::core::types::ModelParams;

pub trait ClassificationModel<Input, Output> {
    fn accuracy(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn loss(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn recall(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn f1_score(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn compute_metrics(&self, x: &Input, y: &Output) -> Result<ModelParams, ModelError>;
}
