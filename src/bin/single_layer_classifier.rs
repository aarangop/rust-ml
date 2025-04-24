use std::path::PathBuf;

use ndarray::{Array2, Axis};
use polars::{
    error::PolarsError,
    prelude::{Float64Type, IndexOrder},
};
use rust_ml::{
    bench::classification_profiler::ClassificationProfiler, bench::core::profiler::Profiler,
    builders::builder::Builder, optim::sgd::optimizer::GradientDescent, prelude::ActivationFn,
};
use rust_ml::{
    prelude::Matrix,
    utils::data::{load_dataset, shuffle_split},
};
use rust_ml::{
    prelude::SingleLayerClassifier,
    vis::progress_bar::{init_progress_bar, progress_bar_callback},
};

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
    let (x_train, y_train, x_test, y_test) = shuffle_split(&features, &target, 0.8, 42);

    // Reshape the data to match what the model expects
    let x_train = x_train.t().to_owned();
    let x_test = x_test.t().to_owned();
    let y_train = y_train.to_shape((1, y_train.len())).unwrap().to_owned();
    let _y_test = y_test.to_shape((1, y_test.len())).unwrap().to_owned();

    // Print the shapes of the training and testing sets
    println!("Training set shape: {:?}", x_train.shape());
    println!("Testing set shape: {:?}", x_test.shape());
    println!("Training target shape: {:?}", y_train.shape());
    println!("Testing target shape: {:?}", _y_test.shape());

    // Initialize the optimizer
    let epochs = 1000;
    let mut gd: GD = GradientDescent::new(0.01, epochs, None, Some(progress_bar_callback));

    // Initialize the progress bar
    init_progress_bar(epochs);

    // Initialize the model
    let mut model = SingleLayerClassifier::builder()
        .n_features(x_train.shape()[0])
        .n_hidden_nodes(10)
        .hidden_layer_activation_fn(ActivationFn::ReLU)
        .output_layer_activation_fn(ActivationFn::Sigmoid)
        .build()
        .unwrap();

    // Initialize a profiler
    let profiler: ClassificationProfiler<SingleLayerClassifier, GD, Array2<f64>, Array2<f64>> =
        ClassificationProfiler::new();

    let (_training_metrics, _eval_metrics) = profiler
        .train(&mut model, &mut gd, &x_train, &y_train)
        .unwrap();

    Ok(())
}
