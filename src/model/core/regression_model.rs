use crate::bench::regression_metrics::RegressionMetrics;
use crate::core::error::ModelError;

pub trait RegressionModel<Input, Output> {
    fn mse(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn rmse(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn r2(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn compute_metrics(&self, x: &Input, y: &Output) -> Result<RegressionMetrics, ModelError>;
}
