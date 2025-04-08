use crate::core::activations::activation::Activation;
use crate::core::types::Vector;
use ndarray::Ix1;

pub struct Tanh;

impl Activation<Ix1> for Tanh {
    fn activate(z: &Vector) -> Vector {
        z.mapv(|x| x.tanh())
    }

    fn derivative(z: &Vector) -> Vector {
        z.mapv(|x| 1.0 - x.tanh().powi(2))
    }
}
