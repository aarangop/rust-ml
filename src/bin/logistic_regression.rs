use ndarray::Axis;
use rust_ml::{
    bench::{classification_profiler::ClassificationProfiler, core::profiler::Profiler},
    builders::builder::Builder,
    core::{
        activations::activation_functions::ActivationFn,
        types::{Matrix, Vector},
    },
    model::{
        core::classification_model::ClassificationModel, logistic_regression::LogisticRegression,
    },
    optim::sgd::optimizer::GradientDescent,
    utils::data::{get_features_and_target, load_dataset, shuffle_split},
};
use std::path::PathBuf;

type GD = GradientDescent<Matrix, Vector, LogisticRegression>;

fn main() {
    println!("Logistic Regression Example");
    println!("=========================\n");
    println!();

    println!(
        "This example demonstrates the usage of the logistic regression classifier, using the diabetes dataset."
    );
    println!("The diabetes dataset is composed of 8 features and a target.");
    println!("The goal is to predict the presence of diabetes based on the features.");
    println!("The dataset is composed of 768 samples.");

    // Load the diabetes dataset
    let dataset_path = PathBuf::from("datasets/diabetes-dataset.csv");
    let df = load_dataset(dataset_path).unwrap();
    // Print the first 5 rows of the dataset
    println!("First 5 rows of the dataset:\n{:?}", df.head(Some(5)));

    // The target is the Outcome column, the rest are features.
    let target = "Outcome";
    let features: Vec<&str> = df
        .get_columns()
        .iter()
        .filter(|&col| col.name() != target)
        .map(|col| col.name().as_str())
        .collect();
    println!("Features: {:?}", features);
    println!("Target: {:?}", target);
    let (x, y) = get_features_and_target(&df, features, target).unwrap();

    // Normalize the features
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let x_std = x.std_axis(Axis(0), 1.0);

    let x = (&x - &x_mean) / &x_std;
    let (x_train, y_train, x_test, y_test) = shuffle_split(&x, &y, 0.8, 42);

    println!("\nTraining set size: {:?}", x_train.nrows());
    println!("Testing set size: {:?}", x_test.nrows());

    // Create the model
    let n_features = x_train.shape()[1];
    let mut model = LogisticRegression::builder()
        .n_features(n_features)
        .activation_function(ActivationFn::Sigmoid)
        .build()
        .unwrap();

    // Create the optimizer
    let mut gd: GD = GradientDescent::new(0.01, 2000, None, None);

    // Create the profiler
    let profiler: ClassificationProfiler<LogisticRegression, GD, Matrix, Vector> =
        ClassificationProfiler::new();

    // Train the model using the profiler to extract the metrics.
    let (training_metrics, training_performance) = profiler
        .train(&mut model, &mut gd, &x_train.t().to_owned(), &y_train)
        .unwrap();

    let test_performance = model
        .compute_metrics(&x_test.t().to_owned(), &y_test)
        .unwrap();

    println!("\nTraining set performance");
    println!("=========================");
    println!("Training time: {:?}", training_metrics.training_time());
    println!("Training accuracy: {:?}", training_performance.accuracy);
    println!("Training loss: {:?}", training_performance.loss);
    println!("Training precision: {:?}", training_performance.precision);
    println!("Training recall: {:?}", training_performance.recall);
    println!("Training F1 score: {:?}", training_performance.f1_score);
    println!();
    println!("Test set perfomance");
    println!("=========================");
    println!("Test accuracy: {:?}", test_performance.accuracy);
    println!("Test precision: {:?}", test_performance.precision);
    println!("Test recall: {:?}", test_performance.recall);
    println!("Test F1 score: {:?}", test_performance.f1_score);

    println!();
}
