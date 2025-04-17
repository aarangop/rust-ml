/// Performance metrics for regression models.
///
/// This struct contains common evaluation metrics used to assess the performance
/// of regression models, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
/// and coefficient of determination (R²).
#[derive(Debug)]
pub struct RegressionMetrics {
    /// Mean Squared Error - the average of the squared differences between predicted and actual values.
    /// MSE = (1/n) * Σ(y_pred - y_true)².
    /// Lower values indicate better fit, with 0 being a perfect fit.
    pub mse: f64,

    /// Root Mean Squared Error - the square root of MSE, which provides an error measure
    /// in the same units as the target variable.
    /// RMSE = √MSE.
    /// Lower values indicate better fit, with 0 being a perfect fit.
    pub rmse: f64,

    /// Coefficient of determination (R²) - measures the proportion of variance in the
    /// dependent variable that is predictable from the independent variable(s).
    /// Range: (-∞, 1.0], where:
    /// - 1.0 indicates a perfect fit (all variance is explained by the model)
    /// - 0.0 indicates the model performs no better than a horizontal line (mean of y)
    /// - Negative values indicate the model performs worse than a horizontal line
    pub r2: f64,
}
