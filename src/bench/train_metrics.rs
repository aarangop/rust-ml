#[derive(Debug)]
pub struct TrainMetrics {
    training_time: f64,
}

impl TrainMetrics {
    pub fn new(training_time: f64) -> Self {
        Self { training_time }
    }

    pub fn training_time(&self) -> f64 {
        self.training_time
    }
}
