use ndarray::{arr1, Array1, Array2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use rust_ml::bench::core::profiler::Profiler;
use rust_ml::bench::regression_profiler::RegressionProfiler;
use rust_ml::builders::builder::Builder;
use rust_ml::core::types::{Matrix, Vector};
use rust_ml::model::linear_regression::LinearRegression;
use rust_ml::optim::sgd::optimizer::GradientDescent;

type GD = GradientDescent<Matrix, Vector, LinearRegression>;

fn main() {
    let mut model = LinearRegression::builder(/* &LinearRegression */)
        .n_input_features(3)
        .build()
        .unwrap();

    let mut gd: GD = GradientDescent::new(0.01, 10000);

    // Create a sample data set.
    // Create some weights and bias
    let weights = arr1(&[1.5, 2.0, 0.8]);
    let bias = 3.7;

    // Create a random input matrix with the number of features as rows, and the number of samples as columns.
    let inputs = Array2::random((3, 20), Normal::new(0., 1.).unwrap());
    // Calculate the target from the linear combination of the weights and biases
    let target = weights.dot(&inputs) + bias;

    let profiler: RegressionProfiler<LinearRegression, GD, Array2<f64>, Array1<f64>> =
        RegressionProfiler::new();

    let (train_metrics, eval_metrics) = profiler
        .profile_training(&mut model, &mut gd, &inputs, &target)
        .unwrap();

    let weights_delta = (weights - &model.w).powi(2).sqrt();
    let bias_delta = (bias - &model.b).powi(2).sqrt();

    println!("train metrics: \n{:?}", train_metrics);
    println!("model metrics: \n{:?}", eval_metrics);
    println!("model weights: \n{:?}", model.w);
    println!("model bias: \n{:?}", model.b);
    println!("weights delta: \n{:?}", weights_delta);
    println!("bias delta: \n{:?}", bias_delta);
}
