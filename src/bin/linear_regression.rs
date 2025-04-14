use ndarray::{arr1, arr2, Array1, Array2};
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

    let x = arr2(&[[1., 2., 3.], [1., 5., 9.], [1., 8., 7.]]);
    let y = arr1(&[6., 75., 57.]);

    let profiler: RegressionProfiler<LinearRegression, GD, Array2<f64>, Array1<f64>> =
        RegressionProfiler::new();

    let (train_metrics, eval_metrics) = profiler
        .profile_training(&mut model, &mut gd, &x, &y)
        .unwrap();

    println!("train metrics: \n{:?}", train_metrics);
    println!("model metrics: \n{:?}", eval_metrics);
}
