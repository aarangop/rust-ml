use crate::core::activations::activation::Activation;
use crate::core::types::Vector;
use ndarray::Ix1;

pub struct ReLU;

impl Activation<Ix1> for ReLU {
    fn activate(z: &Vector) -> Vector {
        z.mapv(|x| x.max(0.0))
    }

    fn derivative(z: &Vector) -> Vector {
        z.map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}
