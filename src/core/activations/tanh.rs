use crate::core::types::Vector;
use crate::{core::activations::activation::Activation, prelude::Matrix};
use ndarray::{Ix1, Ix2};

pub struct Tanh;

impl Activation<Ix1> for Tanh {
    fn activate(z: &Vector) -> Vector {
        (z.exp() - (-z).exp()) / (z.exp() + (-z).exp())
    }

    fn derivative(z: &Vector) -> Vector {
        let a = (z.exp() - (-z).exp()) / (z.exp() + (-z).exp());
        1.0 - a.powi(2)
    }
}

#[cfg(test)]
mod tests_tanh_vector {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_tanh_activate() {
        let input = array![0.0, 1.0, -1.0, 2.0, -2.0];
        let expected = array![
            0.0,
            0.7615941559557649,
            -0.7615941559557649,
            0.9640275800758169,
            -0.9640275800758169
        ];

        let output = Tanh::activate(&input);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tanh_derivative() {
        let input = array![0.0, 1.0, -1.0, 2.0, -2.0];
        let expected = array![
            1.0,
            0.41997434161402614,
            0.41997434161402614,
            0.07065082485316443,
            0.07065082485316443
        ];

        let output = Tanh::derivative(&input);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tanh_activate_large_values() {
        let input = array![10.0, -10.0, 100.0, -100.0];
        let expected = array![0.9999999958776927, -0.9999999958776927, 1.0, -1.0];

        let output = Tanh::activate(&input);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_tanh_derivative_large_values() {
        let input = array![10.0, -10.0, 100.0, -100.0];
        let expected = array![8.24461455076283e-9, 8.24461455076283e-9, 0.0, 0.0];

        let output = Tanh::derivative(&input);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_tanh_activate_empty() {
        let input: Vector = array![];
        let output = Tanh::activate(&input);
        assert_eq!(output.len(), 0);
    }
}

impl Activation<Ix2> for Tanh {
    fn activate(z: &Matrix) -> Matrix {
        (z.exp() - (-z).exp()) / (z.exp() + (-z).exp())
    }

    fn derivative(z: &Matrix) -> Matrix {
        let a = (z.exp() - (-z).exp()) / (z.exp() + (-z).exp());
        1.0 - a.powi(2)
    }
}
#[cfg(test)]
mod tests_tanh_matrix {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    use super::*;

    #[test]
    fn test_tanh_activate_matrix() {
        let input = array![[0.0, 1.0], [-1.0, 2.0]];
        let expected = array![
            [0.0, 0.7615941559557649],
            [-0.7615941559557649, 0.9640275800758169]
        ];

        let output = Tanh::activate(&input);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tanh_derivative_matrix() {
        let input = array![[0.0, 1.0], [-1.0, 2.0]];
        let expected = array![
            [1.0, 0.41997434161402614],
            [0.41997434161402614, 0.07065082485316443]
        ];

        let output = Tanh::derivative(&input);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tanh_activate_large_values_matrix() {
        let input = array![[10.0, -10.0], [100.0, -100.0]];
        let expected = array![[0.9999999958776927, -0.9999999958776927], [1.0, -1.0]];

        let output = Tanh::activate(&input);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_tanh_derivative_large_values_matrix() {
        let input = array![[10.0, -10.0], [100.0, -100.0]];
        let expected = array![[8.24461455076283e-9, 8.24461455076283e-9], [0.0, 0.0]];

        let output = Tanh::derivative(&input);

        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_tanh_activate_empty_matrix() {
        let input: Array2<f64> = Array2::zeros((0, 0));
        let output = Tanh::activate(&input);
        assert_eq!(output.shape(), &[0, 0]);
    }
}
