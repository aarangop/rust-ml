use crate::core::activations::activation::Activation;
use ndarray::{Array, Ix1, Ix2};

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

impl Activation<Ix2> for LeakyReLU {
    fn activate(z: &Array<f64, Ix2>) -> Array<f64, Ix2> {
        let alpha = 0.01;
        z.mapv(|x| if x > 0.0 { x } else { alpha * x })
    }

    fn derivative(z: &Array<f64, Ix2>) -> Array<f64, Ix2> {
        let alpha = 0.01;
        z.mapv(|x| if x > 0.0 { 1.0 } else { alpha })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_leaky_relu_activate() {
        let input = array![2.0, -3.0, 0.0, 5.0, -1.0];
        let expected = array![2.0, -0.03, 0.0, 5.0, -0.01];

        let output = LeakyReLU::activate(&input);

        assert_eq!(output.len(), expected.len());
        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let input = array![2.0, -3.0, 0.0, 5.0, -1.0];
        let expected = array![1.0, 0.01, 1.0, 1.0, 0.01];

        let output = LeakyReLU::derivative(&input);

        assert_eq!(output.len(), expected.len());
        for (a, b) in output.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_leaky_relu_zero() {
        let input = array![0.0];

        // For x = 0, LeakyReLU should return 0
        let activate_output = LeakyReLU::activate(&input);
        assert_abs_diff_eq!(activate_output[0], 0.0, epsilon = 1e-10);

        // For x = 0, derivative should be 1.0 (treating 0 as positive)
        let derivative_output = LeakyReLU::derivative(&input);
        assert_abs_diff_eq!(derivative_output[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_leaky_relu_matrix_activate() {
        let input = array![[2.0, -3.0], [0.0, -1.0]];
        let expected = array![[2.0, -0.03], [0.0, -0.01]];

        let output = LeakyReLU::activate(&input);

        assert_eq!(output.shape(), expected.shape());
        for ((i, j), val) in output.indexed_iter() {
            assert_abs_diff_eq!(val, &expected[[i, j]], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_leaky_relu_matrix_derivative() {
        let input = array![[2.0, -3.0], [0.0, 5.0]];
        let expected = array![[1.0, 0.01], [1.0, 1.0]];

        let output = LeakyReLU::derivative(&input);

        assert_eq!(output.shape(), expected.shape());
        for ((i, j), val) in output.indexed_iter() {
            assert_abs_diff_eq!(val, &expected[[i, j]], epsilon = 1e-10);
        }
    }
}
