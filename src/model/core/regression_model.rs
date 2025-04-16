use crate::bench::regression_metrics::RegressionMetrics;
use crate::core::error::ModelError;
use crate::model::core::base::OptimizableModel;

/// A trait defining common metrics and evaluation methods for regression models.
///
/// This trait should be implemented by any model that performs regression tasks,
/// allowing standardized evaluation through common regression metrics.
///
/// # Type Parameters
///
/// * `Input` - The type of the input data (features).
/// * `Output` - The type of the output data (target values).
///
/// # Methods
///
/// * `mse` - Calculates the Mean Squared Error between predictions and actual values.
/// * `rmse` - Calculates the Root Mean Squared Error between predictions and actual values.
/// * `r2` - Calculates the coefficient of determination (R²) score.
/// * `compute_metrics` - Calculates a comprehensive set of regression metrics.
///
/// # Errors
///
/// Methods return `Result<_, ModelError>` to handle cases where metric calculation
/// might fail, such as with empty inputs or numerical issues.
pub trait RegressionModel<Input, Output>: OptimizableModel<Input, Output> {
    fn mse(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn rmse(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn r2(&self, x: &Input, y: &Output) -> Result<f64, ModelError>;
    fn compute_metrics(&self, x: &Input, y: &Output) -> Result<RegressionMetrics, ModelError>;
}
