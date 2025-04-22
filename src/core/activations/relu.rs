use crate::core::types::Vector;
use crate::{core::activations::activation::Activation, prelude::Matrix};
use ndarray::{Ix1, Ix2};

pub struct ReLU;

impl Activation<Ix1> for ReLU {
    fn activate(z: &Vector) -> Vector {
        z.mapv(|x| x.max(0.0))
    }

    fn derivative(z: &Vector) -> Vector {
        z.map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests_relu_vector {
    use ndarray::{Array1, array};

    use super::*;

    #[test]
    fn test_relu_activate_vector() {
        let input = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = array![0.0, 0.0, 0.0, 1.0, 2.0];
        let output = ReLU::activate(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_derivative_vector() {
        let input = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = array![0.0, 0.0, 0.0, 1.0, 1.0];
        let output = ReLU::derivative(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_activate_empty() {
        let input: Array1<f64> = array![];
        let expected: Array1<f64> = array![];
        let output = ReLU::activate(&input);
        assert_eq!(output, expected);
    }
}

impl Activation<Ix2> for ReLU {
    fn activate(z: &Matrix) -> Matrix {
        z.mapv(|x| x.max(0.0))
    }

    fn derivative(z: &Matrix) -> Matrix {
        z.map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests_relu_matrix {
    use ndarray::{Array2, array};

    use crate::core::activations::{activation::Activation, relu::ReLU};

    #[test]
    fn test_relu_activate_matrix() {
        let input = array![[-2.0, -1.0], [0.0, 1.0], [2.0, -3.0]];
        let expected = array![[0.0, 0.0], [0.0, 1.0], [2.0, 0.0]];
        let output = ReLU::activate(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_derivative_matrix() {
        let input = array![[-2.0, -1.0], [0.0, 1.0], [2.0, -3.0]];
        let expected = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let output = ReLU::derivative(&input);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_activate_zeros() {
        let input = Array2::zeros((3, 3));
        let expected = Array2::zeros((3, 3));
        let output = ReLU::activate(&input);
        assert_eq!(output, expected);
    }
}
