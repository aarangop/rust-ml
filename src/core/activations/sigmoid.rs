use crate::core::activations::activation::Activation;
use ndarray::{Array, Array1, Array2, Ix1, Ix2};

pub struct Sigmoid;

impl Activation<Ix1> for Sigmoid {
    fn activate(z: &Array1<f64>) -> Array1<f64> {
        1.0 / (1.0 + (-z).exp())
    }

    fn derivative(z: &Array1<f64>) -> Array1<f64> {
        let y_hat = Self::activate(z);
        &y_hat * (1.0 - &y_hat)
    }
}

impl Activation<Ix2> for Sigmoid {
    fn activate(z: &Array2<f64>) -> Array2<f64> {
        z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn derivative(z: &Array<f64, Ix2>) -> Array<f64, Ix2> {
        let y_hat = Self::activate(z);
        &y_hat * (1.0 - &y_hat)
    }
}
