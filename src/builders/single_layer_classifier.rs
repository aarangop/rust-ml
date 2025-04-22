use crate::core::activations::activation_functions::ActivationFn;
use crate::core::types::Matrix;
use crate::prelude::{ModelError, SingleLayerClassifier};

use crate::builders::builder::Builder;

pub struct SingleLayerClassifierBuilder {
    n_features: usize,
    n_hidden_nodes: usize,
    threshold: f64,
    output_layer_activation: ActivationFn,
    hidden_layer_activation: ActivationFn,
}

impl SingleLayerClassifierBuilder {
    pub fn n_features(mut self, n_features: usize) -> Self {
        self.n_features = n_features;
        self
    }

    pub fn output_layer_activation_fn(mut self, activation: ActivationFn) -> Self {
        self.output_layer_activation = activation;
        self
    }

    pub fn hidden_layer_activation_fn(mut self, activation: ActivationFn) -> Self {
        self.hidden_layer_activation = activation;
        self
    }

    pub fn n_hidden_nodes(mut self, n_hidden_nodes: usize) -> Self {
        self.n_hidden_nodes = n_hidden_nodes;
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for SingleLayerClassifierBuilder {
    fn default() -> Self {
        Self {
            n_features: 2,
            n_hidden_nodes: 2,
            threshold: 0.5,
            output_layer_activation: ActivationFn::Sigmoid,
            hidden_layer_activation: ActivationFn::ReLU,
        }
    }
}

impl Builder<SingleLayerClassifier, Matrix, Matrix> for SingleLayerClassifierBuilder {
    fn build(&self) -> Result<SingleLayerClassifier, ModelError> {
        Ok(SingleLayerClassifier::new(
            self.n_features,
            self.n_hidden_nodes,
            self.threshold,
            self.output_layer_activation,
            self.hidden_layer_activation,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_builder() {
        let builder = SingleLayerClassifierBuilder::default();
        assert_eq!(builder.n_features, 2);
        assert_eq!(builder.n_hidden_nodes, 2);
        assert_eq!(builder.threshold, 0.5);
        match builder.output_layer_activation {
            ActivationFn::Sigmoid => {}
            _ => panic!("Default output activation should be Sigmoid"),
        }
        match builder.hidden_layer_activation {
            ActivationFn::ReLU => {}
            _ => panic!("Default hidden activation should be ReLU"),
        }
    }

    #[test]
    fn test_builder_methods() {
        let builder = SingleLayerClassifierBuilder::default()
            .n_features(5)
            .n_hidden_nodes(10)
            .output_layer_activation_fn(ActivationFn::Tanh)
            .hidden_layer_activation_fn(ActivationFn::Sigmoid);

        assert_eq!(builder.n_features, 5);
        assert_eq!(builder.n_hidden_nodes, 10);
        match builder.output_layer_activation {
            ActivationFn::Tanh => {}
            _ => panic!("Output activation should be Tanh"),
        }
        match builder.hidden_layer_activation {
            ActivationFn::Sigmoid => {}
            _ => panic!("Hidden activation should be Sigmoid"),
        }
    }

    #[test]
    /// Test the build method of the SingleLayerClassifierBuilder
    fn test_builder_build() {
        let builder = SingleLayerClassifierBuilder::default()
            .n_features(3)
            .n_hidden_nodes(5)
            .output_layer_activation_fn(ActivationFn::Sigmoid)
            .hidden_layer_activation_fn(ActivationFn::LeakyReLU);

        assert!(builder.build().is_ok());
    }
}
