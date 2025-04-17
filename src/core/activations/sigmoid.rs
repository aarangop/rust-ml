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
        1.0 / (1.0 + (-z).exp())
    }

    fn derivative(z: &Array<f64, Ix2>) -> Array<f64, Ix2> {
        let y_hat = Self::activate(z);
        &y_hat * (1.0 - &y_hat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2};
    use ndarray_rand::rand_distr::num_traits::Float;

    #[test]
    fn test_sigmoid_activate_vector() {
        // Test with a vector input
        let input = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = Sigmoid::activate(&input);

        // Expected values calculated using sigmoid formula: 1 / (1 + exp(-x))
        let expected = arr1(&[
            0.11920292, // sigmoid(-2.0)
            0.26894142, // sigmoid(-1.0)
            0.5,        // sigmoid(0.0)
            0.73105858, // sigmoid(1.0)
            0.88079708, // sigmoid(2.0)
        ]);

        assert_eq!(result.len(), expected.len());

        // Compare with a small epsilon for floating point precision
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_sigmoid_derivative_vector() {
        // Test with a vector input
        let input = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = Sigmoid::derivative(&input);

        // Expected values calculated using sigmoid derivative formula: sigmoid(x) * (1 - sigmoid(x))
        let sigmoid_values = arr1(&[
            0.11920292, // sigmoid(-2.0)
            0.26894142, // sigmoid(-1.0)
            0.5,        // sigmoid(0.0)
            0.73105858, // sigmoid(1.0)
            0.88079708, // sigmoid(2.0)
        ]);

        let expected = sigmoid_values.mapv(|v| v * (1.0 - v));

        assert_eq!(result.len(), expected.len());

        // Compare with a small epsilon for floating point precision
        for (a, b) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_sigmoid_activate_matrix() {
        // Test with a matrix input
        let input = arr2(&[[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);

        let result = Sigmoid::activate(&input);

        // Expected values calculated using sigmoid formula: 1 / (1 + exp(-x))
        let expected = arr2(&[
            [0.11920292, 0.26894142, 0.5],
            [0.73105858, 0.88079708, 0.95257413],
        ]);

        assert_eq!(result.shape(), expected.shape());

        // Compare with a small epsilon for floating point precision
        for ((i, j), value) in result.indexed_iter() {
            assert_abs_diff_eq!(value, &expected[[i, j]], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_sigmoid_derivative_matrix() {
        // Test with a matrix input
        let input = arr2(&[[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);

        let result = Sigmoid::derivative(&input);

        // Expected values calculated using sigmoid derivative formula: sigmoid(x) * (1 - sigmoid(x))
        let sigmoid_values = arr2(&[
            [0.11920292, 0.26894142, 0.5],
            [0.73105858, 0.88079708, 0.95257413],
        ]);

        let expected = sigmoid_values.mapv(|v| v * (1.0 - v));

        assert_eq!(result.shape(), expected.shape());

        // Compare with a small epsilon for floating point precision
        for ((i, j), value) in result.indexed_iter() {
            assert_abs_diff_eq!(value, &expected[[i, j]], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_sigmoid_properties() {
        // Test that sigmoid(0) = 0.5
        let zero = arr1(&[0.0]);
        assert_abs_diff_eq!(Sigmoid::activate(&zero)[0], 0.5, epsilon = 1e-8);

        // Test symmetry property: sigmoid(-x) = 1 - sigmoid(x)
        let x = arr1(&[1.0, 2.0, 3.0]);
        let neg_x = arr1(&[-1.0, -2.0, -3.0]);

        let sigmoid_x = Sigmoid::activate(&x);
        let sigmoid_neg_x = Sigmoid::activate(&neg_x);

        for (i, value) in sigmoid_neg_x.iter().enumerate() {
            assert_abs_diff_eq!(value, &(1.0 - sigmoid_x[i]), epsilon = 1e-8);
        }

        // Test that derivative is maximum at x = 0
        let points = arr1(&[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]);
        let derivatives = Sigmoid::derivative(&points);

        // Maximum value should be at x = 0 (which is 0.25)
        let max_derivative = derivatives.iter().fold(0.0, |max, &val| max.max(val));
        assert_abs_diff_eq!(max_derivative, 0.25, epsilon = 1e-8);
        assert_abs_diff_eq!(derivatives[3], 0.25, epsilon = 1e-8); // index 3 is x = 0
    }
}
