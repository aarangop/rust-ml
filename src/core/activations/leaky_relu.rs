use crate::core::activations::activation::Activation;
use ndarray::{Array, Ix1};

pub struct LeakyReLU;

impl Activation<Ix1> for LeakyReLU {
    fn activate(z: &Array<f64, Ix1>) -> Array<f64, Ix1> {
        let alpha = 0.01;
        z.mapv(|x| if x > 0.0 { x } else { alpha * x })
    }

    fn derivative(z: &Array<f64, Ix1>) -> Array<f64, Ix1> {
        let alpha = 0.01;
        z.mapv(|x| if x > 0.0 { 1.0 } else { alpha })
    }
}
