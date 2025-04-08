use crate::bench::classification_metrics::ClassificationMetrics;
use crate::builders::logistic_regression::LogisticRegressionBuilder;
use crate::core::activations::activation::Activation;
use crate::core::activations::activation_functions::ActivationFn;
use crate::core::activations::leaky_relu::LeakyReLU;
use crate::core::activations::relu::ReLU;
use crate::core::activations::sigmoid::Sigmoid;
use crate::core::activations::tanh::Tanh;
use crate::core::error::ModelError;
use crate::core::param_manager::ParamManager;
use crate::core::types::{Matrix, ModelParams, Vector};
use crate::model::core::base::{BackwardPropagation, BaseModel, ForwardPropagation};
use crate::model::core::classification_model::ClassificationModel;
use crate::model::core::optimizable_model::OptimizableModel;
use ndarray::{ArrayView, IxDyn};

pub struct LogisticRegression {
    weights: Vector,
    bias: Vector,
    activation_fn: ActivationFn,
}

impl LogisticRegression {
    pub fn new(n_features: usize, activation_fn: ActivationFn) -> Self {
        let weights = Vector::zeros(n_features);
        let bias = Vector::from_elem(1, 0.0);
        Self {
            weights,
            bias,
            activation_fn,
        }
    }

    pub fn builder() -> LogisticRegressionBuilder {
        LogisticRegressionBuilder::new()
    }

    fn compute_activation(&self, z: &Vector) -> Result<Vector, ModelError> {
        match self.activation_fn {
            ActivationFn::Sigmoid => Ok(Sigmoid::activate(z)),
            ActivationFn::ReLU => Ok(ReLU::activate(z)),
            ActivationFn::Tanh => Ok(Tanh::activate(z)),
            ActivationFn::LeakyReLU => Ok(LeakyReLU::activate(z)),
        }
    }

    fn compute_derivative(&self, z: &Vector) -> Result<Vector, ModelError> {
        match self.activation_fn {
            ActivationFn::Sigmoid => Ok(Sigmoid::derivative(z)),
            ActivationFn::ReLU => Ok(ReLU::derivative(z)),
            ActivationFn::Tanh => Ok(Tanh::derivative(z)),
            ActivationFn::LeakyReLU => Ok(LeakyReLU::derivative(z)),
        }
    }
}

impl BaseModel<Matrix, Vector> for LogisticRegression {
    fn predict(&self, x: &Matrix) -> Result<Vector, ModelError> {
        let bias = self.bias[0];
        let z = self.weights.dot(x) + bias;
        let a = self.compute_activation(&z)?;
        let y_hat = a.mapv(|x| if x >= 0.5 { 1.0 } else { 0.0 });
        Ok(y_hat)
    }

    fn compute_cost(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_hat = self.predict(x)?;
        let cost = y * y_hat.ln() + (1.0 - y) * (1.0 - y_hat).ln();
        Ok(cost.sum())
    }

    fn compute_gradient(&self, x: &Matrix, y: &Vector) -> Result<Vector, ModelError> {
        let y_hat = self.predict(x)?;
        Ok(y_hat - y)
    }
}

impl ForwardPropagation<Matrix, Vector> for LogisticRegression {
    fn compute_forward_propagation(
        &self,
        x: &Matrix,
    ) -> Result<(Vector, Option<ModelParams>), ModelError> {
        let bias = self.bias[0];
        let z = self.weights.dot(x) + bias;
        let a = self.compute_activation(&z)?;
        let mut cache = ModelParams::new();
        cache.insert("z".to_string(), z.into_dyn());
        Ok((a, Some(cache)))
    }
}

impl BackwardPropagation<Matrix, Vector> for LogisticRegression {
    fn compute_backward_propagation(
        &self,
        x: &Matrix,
        y: &Vector,
        output_gradients: &Vector,
        cache: Option<ModelParams>,
    ) -> Result<ModelParams, ModelError> {
        let dz = output_gradients;
        let dw = x.dot(dz);
        let db = dz.sum();
        let mut grads = ModelParams::new();
        grads.insert("dW".to_string(), dw.into_dyn());
        grads.insert("db".to_string(), Vector::from_elem(1, db).into_dyn());
        Ok(grads)
    }
}

impl ParamManager for LogisticRegression {
    fn get_params(&self) -> ModelParams {
        todo!()
    }

    fn update_params(&mut self, params: ModelParams) {
        todo!()
    }

    fn get_param(&self, key: &str) -> Result<ArrayView<f64, IxDyn>, ModelError> {
        todo!()
    }
}

impl OptimizableModel<Matrix, Vector> for LogisticRegression {}

impl ClassificationModel<Matrix, Vector> for LogisticRegression {
    fn accuracy(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_pred = self.predict(x)?;
        let y_pred_binary = y_pred.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 });
        let correct = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| (pred - actual).abs() < f64::EPSILON)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    fn loss(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        // Binary cross-entropy loss
        let y_pred = self.predict(x)?;
        let epsilon = 1e-15; // prevent log(0)
        let y_pred = y_pred.mapv(|val| val.max(epsilon).min(1.0 - epsilon));
        let loss = y
            .iter()
            .zip(y_pred.iter())
            .map(|(y_i, y_pred_i)| -y_i * y_pred_i.ln() - (1.0 - y_i) * (1.0 - y_pred_i).ln())
            .sum::<f64>()
            / y.len() as f64;
        Ok(loss)
    }

    fn recall(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_pred = self.predict(x)?;
        let y_pred_binary = y_pred.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 });

        let true_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual > 0.5)
            .count();

        let actual_positives = y.iter().filter(|&&actual| actual > 0.5).count();

        if actual_positives == 0 {
            return Ok(0.0);
        }

        Ok(true_positives as f64 / actual_positives as f64)
    }

    fn f1_score(&self, x: &Matrix, y: &Vector) -> Result<f64, ModelError> {
        let y_pred = self.predict(x)?;
        let y_pred_binary = y_pred.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 });

        let true_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual > 0.5)
            .count() as f64;

        let false_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual <= 0.5)
            .count() as f64;

        let false_negatives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred <= 0.5 && *actual > 0.5)
            .count() as f64;

        let precision = if true_positives + false_positives == 0.0 {
            0.0
        } else {
            true_positives / (true_positives + false_positives)
        };

        let recall = if true_positives + false_negatives == 0.0 {
            0.0
        } else {
            true_positives / (true_positives + false_negatives)
        };

        if precision + recall == 0.0 {
            return Ok(0.0);
        }

        Ok(2.0 * precision * recall / (precision + recall))
    }

    fn compute_metrics(&self, x: &Matrix, y: &Vector) -> Result<ClassificationMetrics, ModelError> {
        let accuracy = self.accuracy(x, y)?;
        let loss = self.loss(x, y)?;
        let recall = self.recall(x, y)?;
        let f1 = self.f1_score(x, y)?;

        // Calculate precision
        let y_pred = self.predict(x)?;
        let y_pred_binary = y_pred.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 });

        let true_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual > 0.5)
            .count() as f64;

        let false_positives = y_pred_binary
            .iter()
            .zip(y.iter())
            .filter(|&(pred, actual)| *pred > 0.5 && *actual <= 0.5)
            .count() as f64;

        let precision = if true_positives + false_positives == 0.0 {
            0.0
        } else {
            true_positives / (true_positives + false_positives)
        };

        Ok(ClassificationMetrics {
            accuracy,
            loss,
            precision,
            recall,
            f1_score: f1,
        })
    }
}
