/// Performance metrics for classification models.
///
/// This struct contains common evaluation metrics used to assess the performance
/// of classification models, including accuracy, precision, recall, and F1 score.
#[derive(Debug)]
pub struct ClassificationMetrics {
    /// The loss value (e.g., cross-entropy loss) from the model's predictions.
    pub loss: f64,

    /// The accuracy of the model, measured as the proportion of correctly
    /// classified instances out of the total instances.
    /// Range: [0.0, 1.0], where 1.0 means perfect classification.
    pub accuracy: f64,

    /// The precision of the model, measured as the ratio of true positives to
    /// the sum of true positives and false positives.
    /// Range: [0.0, 1.0], where 1.0 means no false positives.
    pub precision: f64,

    /// The recall (sensitivity) of the model, measured as the ratio of true positives
    /// to the sum of true positives and false negatives.
    /// Range: [0.0, 1.0], where 1.0 means no false negatives.
    pub recall: f64,

    /// The F1 score, which is the harmonic mean of precision and recall.
    /// Range: [0.0, 1.0], where 1.0 means perfect precision and recall.
    /// F1 = 2 * (precision * recall) / (precision + recall)
    pub f1_score: f64,
}
