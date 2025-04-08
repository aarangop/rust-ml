use crate::core::error::ModelError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProfilerError {
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

impl From<ModelError> for ProfilerError {
    fn from(err: ModelError) -> Self {
        ProfilerError::OptimizationError(err.to_string())
    }
}
