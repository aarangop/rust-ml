use std::path::PathBuf;

use ndarray::Axis;
use polars::{
    error::PolarsError,
    prelude::{Float64Type, IndexOrder},
};
use rust_ml::optim::sgd::optimizer::GradientDescent;
use rust_ml::prelude::{Matrix, SingleLayerClassifier};
use rust_ml::utils::data::{load_dataset, shuffle_split};

type GD = GradientDescent<Matrix, Matrix, SingleLayerClassifier>;
/// In this example, we will try using the SingleLayerClassifier
/// from the rust_ml library to classify a simple dataset.
/// To compare the performance of the model against simple logistic regression,
/// we will also use the diabetes dataset.
fn main() -> Result<(), PolarsError> {
    println!("Single Layer Classifier Example\n");
    println!("=============================\n");

    // Load the dataset
    let dataset_path = PathBuf::from("datasets/diabetes-dataset.csv");
    let df = load_dataset(dataset_path).unwrap();

    // Print the first few rows of the dataset
    println!("First five rows of the dataset:");
    println!("{:?}", df.head(Some(5)));

    // Extract features and target
    let target = df.column("Outcome")?;
    // Convert the target column to a Vec<f64> first, then to Array1
    let target_vec: Vec<f64> = target
        .i64()?
        .iter()
        .map(|opt_val| opt_val.map(|val| val as f64).unwrap_or(0.0))
        .collect();
    // Create an Array1 from the Vec
    let target = ndarray::Array1::from(target_vec);
    let features = df.drop("Outcome")?;

    // Normalize the features
    let features = features.to_ndarray::<Float64Type>(IndexOrder::C).unwrap();
    let mean = features.mean_axis(Axis(0)).unwrap();
    let std = features.std_axis(Axis(0), 1.0);
    let features = (&features - &mean) / &std;

    // Create train and test sets.
    // let (x_train, y_train, x_test, y_test) = shuffle_split(&features, &target, 0.8, 42);

    // Initialize the optimizer
    let gd: GD = GradientDescent::new(0.01, 1000);

    Ok(())
}
