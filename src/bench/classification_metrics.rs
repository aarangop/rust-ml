#[derive(Debug)]
pub struct ClassificationMetrics {
    pub loss: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}
