use rust_ml::linear_regression::model::LinearRegression;
use rust_ml::model::base::MLModel;
use ndarray::{Array1, Array2, Ix2};
use polars::prelude::*;
use std::path::Path;
use rand::prelude::*;

fn read_dataset_from_csv(path: &Path) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))?
        .finish()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the advertising dataset
    let path = Path::new("./datasets/advertising.csv");
    let df = read_dataset_from_csv(path)?;
    println!("Dataset loaded with shape: {:?}", df.shape());

    // Shuffle the dataset
    let m = df.height() as u32;
    let mut indices: Vec<u32> = (0..m).collect();
    let mut rng = rand::rng();
    indices.shuffle(&mut rng);

    // Create series from indices
    let idx = Series::new("idx".into(), indices);

    // Take rows in shuffled order
    let shuffled_df = df.take(idx.u32()?)?;

    // Split dataset into training and test sets (80/20 split)
    let split_point = (shuffled_df.height() as f64 * 0.8) as usize;

    let train_df = shuffled_df.slice(0, split_point);
    let test_df = shuffled_df
        .slice(split_point as i64, shuffled_df.height() - split_point);

    println!("Data split into {} training and {} test samples", 
             train_df.height(), test_df.height());

    // Separate features and target column
    let target_column = "Sales";

    let train_features_df = train_df.drop(target_column)?;
    let train_target_df = train_df.column(target_column)?;
    let test_features_df = test_df.drop(target_column)?;
    let test_target_df = test_df.column(target_column)?;

    // Convert features into 2D array (features x samples)
    let train_features_array: Array2<f64> = train_features_df
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)?
        .into_dimensionality::<Ix2>()?
        .t()
        .to_owned();
    let test_features_array: Array2<f64> = test_features_df
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)?
        .into_dimensionality::<Ix2>()?
        .t()
        .to_owned();

    // Convert targets into 1D array
    let train_target_array: Array1<f64> = train_target_df
        .cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .filter_map(|x| x)
        .collect::<Array1<f64>>();
    let test_target_array: Array1<f64> = test_target_df
        .cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .filter_map(|x| x)
        .collect::<Array1<f64>>();

    // Print dimensions for debugging
    println!("Train features shape: {:?}", train_features_array.shape());
    println!("Train target shape: {:?}", train_target_array.shape());

    // Initialize the model
    let n_features = train_features_array.shape()[0];
    println!("Initializing linear regression model with {} features", n_features);
    let mut model = LinearRegression::new(n_features);

    // Train the model
    println!("Training the model...");
    let learning_rate = 0.00001;
    let epochs = 10000;
    model.fit(&train_features_array, &train_target_array, learning_rate, epochs)?;

    // Evaluate on test set
    let test_predictions = model.predict(&test_features_array)?;
    
    // Calculate MSE (Mean Squared Error)
    let mse = (&test_predictions - &test_target_array).mapv(|x| x.powi(2)).mean().unwrap();
    println!("Mean Squared Error on test set: {:.4}", mse);
    let test_set_cost = model.compute_cost(
      &test_features_array, &test_target_array)
      .unwrap();
    println!("Total cost on test set: {:.4}", test_set_cost);
    
    // Print model coefficients
    println!("\nModel Parameters:");
    println!("Weights: {:?}", model.weights());
    println!("Bias: {:.4}", model.bias());

    Ok(())
}