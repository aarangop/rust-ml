use ndarray::{Axis, Ix1};
use polars::prelude::{Float64Type, IndexOrder};
use rust_ml::bench::core::profiler::Profiler;
use rust_ml::bench::regression_profiler::RegressionProfiler;
use rust_ml::builders::builder::Builder;
use rust_ml::core::types::{Matrix, Vector};
use rust_ml::model::linear_regression::LinearRegression;
use rust_ml::optim::sgd::optimizer::GradientDescent;
use rust_ml::utils::data::{load_dataset, shuffle_split};
use std::path::PathBuf;

type GD = GradientDescent<Matrix, Vector, LinearRegression>;

/// This executable showcases the usage of the linear regression model, using the advertising
/// dataset.
///
/// The advertising dataset is composed of three features: TV, radio and print advertising budgets,
/// and the sales. The goal is to predict the sales based on the advertising budgets.
fn main() {
    println!("Linear Regression Example");
    println!("=========================\n");
    println!(
        "This example demonstrates the usage of the linear regression model, using the advertising dataset."
    );
    println!(
        "The advertising dataset is composed of three features: TV, radio and print advertising budgets, and the sales."
    );
    println!("The goal is to predict the sales based on the advertising budgets.");

    // Load the advertising dataset
    let dataset_path = PathBuf::from("./datasets/advertising.csv");
    let df = load_dataset(dataset_path).unwrap();
    // Print the first 5 rows of the dataset
    println!("First 5 rows of the dataset: \n{:?}", df.head(Some(5)));

    // Select features and target
    let x = df
        .select(["TV", "Radio", "Newspaper"])
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap();
    let y = df
        .select(["Sales"])
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap()
        .column(0)
        .to_owned()
        .into_dimensionality::<Ix1>()
        .unwrap();

    let (x_train, y_train, x_test, y_test) = shuffle_split(&x, &y, 0.8, 42);

    println!("\nTraining set size: {:?}", x_train.nrows());
    println!("Testing set size: {:?}", x_test.nrows());
    println!("Shape of target samples: {:?}", y_train.shape());

    // Create the model
    let n_features = x_train.shape()[1];
    let mut model = LinearRegression::builder()
        .n_input_features(n_features)
        .build()
        .unwrap();

    let mut gd: GD = GradientDescent::new(0.01, 1000);

    // Instantiate a profiler for linear regression and gradient descent.
    let profiler: RegressionProfiler<LinearRegression, GD, Matrix, Vector> =
        RegressionProfiler::new();

    let (train_metrics, eval_metrics) = profiler
        .profile_training(&mut model, &mut gd, &x_train.t().to_owned(), &y_train)
        .unwrap();

    println!("train metrics: \n{:?}", train_metrics);
    println!("model metrics: \n{:?}", eval_metrics);
    println!("model weights: \n{:?}", model.w);
    println!("model bias: \n{:?}", model.b);

    // It turns out that with this particular dataset, the cost does not decrease.
    // If we look at the data the features are on different scales. What if we normalized it?
    println!("Normalize the dataset, as the model doesn't learn without normalization.");
    
    let x = (&x - x.mean_axis(Axis(0)).unwrap()) / x.std_axis(Axis(0), 1.0);
    let y = (&y - y.mean().unwrap()) / y.std(1.0);
    let (x_train, y_train, x_test, y_test) = shuffle_split(&x, &y, 0.8, 42);

    let (train_metrics, eval_metrics) = profiler
        .profile_training(&mut model, &mut gd, &x_train.t().to_owned(), &y_train)
        .unwrap();

    println!("train metrics: \n{:?}", train_metrics);
    println!("model metrics: \n{:?}", eval_metrics);
    println!("model weights: \n{:?}", model.w);
    println!("model bias: \n{:?}", model.b);
}
