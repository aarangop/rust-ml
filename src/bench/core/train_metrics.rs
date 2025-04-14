/// Metrics related to the model training process.
///
/// This struct tracks performance metrics related to the training process itself,
/// such as training time, rather than the model's predictive performance.
#[derive(Debug)]
pub struct TrainMetrics {
    /// Time taken to train the model, measured in seconds.
    training_time: f64,
}

impl TrainMetrics {
    /// Creates a new TrainMetrics instance.
    ///
    /// # Arguments
    ///
    /// * `training_time` - The time taken to train the model, in seconds.
    ///
    /// # Returns
    ///
    /// A new TrainMetrics instance with the provided training time.
    pub fn new(training_time: f64) -> Self {
        Self { training_time }
    }

    /// Gets the training time in seconds.
    ///
    /// # Returns
    ///
    /// The time taken to train the model, in seconds.
    pub fn training_time(&self) -> f64 {
        self.training_time
    }
}
