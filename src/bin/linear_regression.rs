use ndarray::Axis;
use rand::distr::{Distribution, Uniform};
use rust_ml::bench::core::profiler::Profiler;
use rust_ml::bench::regression_profiler::RegressionProfiler;
use rust_ml::builders::builder::Builder;
use rust_ml::core::types::{Matrix, Vector};
use rust_ml::model::core::base::BaseModel;
use rust_ml::model::core::regression_model::RegressionModel;
use rust_ml::model::linear_regression::LinearRegression;
use rust_ml::optim::sgd::optimizer::GradientDescent;
use rust_ml::utils::data::{get_features_and_target, load_dataset, shuffle_split};
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
    let features = vec!["TV", "Radio", "Newspaper"];
    let target = "Sales";
    let (x, y) = get_features_and_target(&df, features, target).unwrap();

    // It turns out that with this particular dataset, the cost does not decrease.
    // If we look at the data the features are on different scales. What if we normalized it?
    println!(
        "With this dataset the model has trouble learning without normalization, so we'll normalize it."
    );
    // Compute mean and standard deviation for the whole dataset.
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let x_std = x.std_axis(Axis(0), 1.0);
    let y_mean = y.mean_axis(Axis(0)).unwrap();
    let y_std = y.std_axis(Axis(0), 1.0);

    println!("\nMean of x: {:?}", &x_mean);
    println!("Std of x: {:?}", &x_std);
    println!("\nMean of y: {:?}", &y_mean);
    println!("Std of y: {:?}", &y_std);
    println!();

    // Normalize dataset
    let x = (&x - &x_mean) / &x_std;
    let y = (&y - &y_mean) / &y_std;
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

    println!("The dataset has been normalized, try training again./n");

    // Train the model again.
    let (training_metrics, training_performance) = profiler
        .profile_training(&mut model, &mut gd, &x_train.t().to_owned(), &y_train)
        .unwrap();
    let test_performance = model
        .compute_metrics(&x_test.t().to_owned(), &y_test)
        .unwrap();

    println!("train metrics: \n{:?}", training_metrics);
    println!("model weights: \n{:?}", model.w);
    println!("model bias: \n{:?}", model.b);
    println!("model train metrics: \n{:?}", training_performance);
    println!("model test metrics: \n{:?}\n", test_performance);

    // Make some predictions
    println!("Make some predictions with the test set and compare the output.");

    // Get a random sample from the test set
    let step = Uniform::new(0, x_test.shape()[0] - 1).unwrap();
    let mut rng = rand::rng();
    let sample_idx: usize = step.sample(&mut rng);

    // Take the sample and convert it to the expected shape for prediction (n_features, 1)
    let pred_x_norm = x_test
        .row(sample_idx)
        .to_shape((n_features, 1))
        .unwrap()
        .to_owned();
    let pred_y_norm = model.predict(&pred_x_norm).unwrap();

    let pred_x = pred_x_norm * &x_std + &x_mean;
    let pred_y = pred_y_norm * &y_std + &y_mean;

    println!("Sample prediction:");
    println!("TV budget: {:?}", &pred_x[[0, 0]]);
    println!("Radio budget: {:?}", &pred_x[[1, 0]]);
    println!("Newspaper budget: {:?}", &pred_x[[2, 0]]);
    println!("Predicted sales: {:?}", &pred_y[0]);
}
