use ndarray::{Array1, Array2};
use rand::{Rng, thread_rng};


pub struct Model {
    weights: Array1<f64>,
    bias: f64,
}

impl Model {
  pub fn new(n_x: usize) -> Model {
    println!("Creating new logistic regression model with {} features.", n_x);
    // Initialize weights as random values between 0 and 0.1
    let mut rng = rand::rng();
    let weights = Array1::<f64>::from_iter((0..n_x)
      .map(|_| rng.random::<f64>() * 0.1));
    Model {
      weights,
      bias: 0.0,
    }
  }

  /// Train the model using gradient descent, given input `x` (n_x, m), output `y` (m, 1), a learning rate and a number of epochs (or iterations).
  pub fn train(&mut self, x: Array2<f64>, y: Array1<f64>, learning_rate: f64, epochs: usize) {
    // Ensure the dimensions of the provided input and target match the model's weights.
    assert_eq!(x.shape()[0], self.weights.len(), "Input features do not match model weights. Expected {} features, got {}", self.weights.shape()[0], x.shape()[0]);
    assert_eq!(y.shape()[0], x.shape()[1], "Target shape doesn't match input shape, expected {} got {}", x.shape()[1], y.shape()[0]);

    println!("Starting logistic regression model training.");

    let m = x.shape()[1] as f64;
    for i in 0..epochs {

      // We need to compute the gradients dw, db, and dz
      // For that we first need the activation values.
      let a = self.forward_propagation(&x);
      let dz = a - &y;
      let dw = x.dot(&dz.t())/(m as f64)/m;
      let db = dz.sum()/&(m as f64)/m;

      // Update parameters
      let diff_w = learning_rate * dw;
      self.weights = &self.weights - diff_w;
      self.bias = self.bias - learning_rate * db;
      let cost = self.compute_cost(&x, &y);
      if i % 100 == 0 {
        println!("{}. Epoch - cost: {:.4}", i, cost);
      }
    } 
  }

  /// Compute the cost function for the current model state and input `x`.
  pub fn compute_cost(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
    // Extract the number of examples to normalize the summation later.
    let m = x.shape()[1];
    // Compute loss for each example
    let y_hat = self.forward_propagation(&x); 
    let l = -y * y_hat.ln() + (1_f64 - y) * (1_f64 - y_hat).ln();
    let j = l.sum()/m as f64;
    j
  }

  /// Compute forward propagation for training examples x.
  fn forward_propagation(&self, x: &Array2<f64>) -> Array1<f64> {
    let z = self.weights.t().dot(x) + self.bias;
    Model::sigmoid(z)  
  }

  /// Compute the sigmoid function for the argument `z`, obtained as `w.T * X + b`.
  fn sigmoid(z: Array1<f64>) -> Array1<f64> {
    z.map(|&x| {
      if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
      } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
      }
    })
  }

  ///
  pub fn accuracy(&self, x_test: Array2<f64>, y_test: Array1<f64>) -> f64 {
    assert_eq!(x_test.shape()[0], self.weights.len(), "Input features do not match model weights. Expected {} features, got {}", self.weights.shape()[0], x_test.shape()[0]);
    assert_eq!(y_test.shape()[0], x_test.shape()[1], "Target shape doesn't match input shape, expected {} got {}", x_test.shape()[1], y_test.shape()[0]);

    println!("Computing accuracy for {} features and {} samples.", x_test.shape()[0], x_test.shape()[1]);

    let m = x_test.shape()[1] as f64;
    let y_hat = self.forward_propagation(&x_test);
    let y_hat = y_hat.map(|&x| if x > 0.5 {1.0} else {0.0});
    let correct = y_hat
      .iter()
      .zip(y_test.iter()).filter(|(a, b)| a == b)
      .count() as f64;
    correct/m
  }
}