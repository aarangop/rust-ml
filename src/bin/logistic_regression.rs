use ndarray::{arr1, arr2};
use rust_ml::bench::classification_profiler::ClassificationProfiler;
use rust_ml::bench::core::profiler::Profiler;
use rust_ml::builders::builder::Builder;
use rust_ml::core::activations::activation_functions::ActivationFn;
use rust_ml::core::types::{Matrix, Vector};
use rust_ml::model::logistic_regression::LogisticRegression;
use rust_ml::optimization::gradient_descent::GradientDescent;

fn main() {
    let mut model = LogisticRegression::builder()
        .n_features(3)
        .activation_function(ActivationFn::Sigmoid)
        .build()
        .unwrap();

    let mut gd = GradientDescent::new(0.01, 10000);

    let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let y = arr1(&[0.0, 1.0]);

    let profiler: ClassificationProfiler<LogisticRegression, GradientDescent, Matrix, Vector> =
        ClassificationProfiler::new();

    let (train_metrics, eval_metrics) = profiler
        .profile_training(&mut model, &mut gd, &x, &y)
        .unwrap();

    println!("{:?}", train_metrics);
    println!("{:?}", eval_metrics);
}
