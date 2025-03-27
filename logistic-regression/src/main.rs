use logreg::logistic_regression::Model;
use ndarray::{Array1, Array2, Ix1, Ix2};
use polars::prelude::*;
use std::path::Path;
use rand::{rng, seq::SliceRandom};
pub mod logreg;

fn read_dataset_from_csv(path: &Path) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))?
        .finish()
}

fn main() -> Result<(), PolarsError> {
    let path = Path::new("diabetes-dataset.csv");
    let df = read_dataset_from_csv(path).unwrap();
    println!("Read with shape: {:?}", df.shape());

    // Shuffle the dataset
    // First create shuffled indices
    let m = df.height() as u32;
    let mut indices: Vec<u32> = (0..m).collect();
    let mut rng = rng();
    indices.shuffle(&mut rng);

    // Create series from indices
    let idx = Series::new("idx".into(), indices);

    // Take rows in shuffled order
    let shuffled_df = df.take(idx.u32().unwrap()).unwrap();

    let split_point = (shuffled_df.height() as f64 * 0.8) as usize;
    let train_df = shuffled_df.slice(0, split_point);
    let test_df = shuffled_df
        .slice(split_point as i64, shuffled_df.height());

    // Separate features and target column
    let target_column = "Outcome";

    let train_features_df = train_df.drop(target_column)?;
    let train_target_df = train_df.column(target_column)?;
    let test_features_df = test_df.drop(target_column)?;
    let test_target_df = test_df.column(target_column)?;

    // Convert features into 2D array (features x samples)
    let train_features_array: Array2<f64> = train_features_df
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)?
        .into_dimensionality::<Ix2>().unwrap()
        .t()
        .to_owned();

    let train_target_array: Array1<f64> = train_target_df
        .cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .filter_map(|x| x)
        .collect::<Array1<f64>>();

    let test_features_array: Array2<f64> = test_features_df
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)?
        .into_dimensionality::<Ix2>().unwrap()
        .t()
        .to_owned();
    let test_target_array: Array1<f64> = test_target_df
        .cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .filter_map(|x| x)
        .collect::<Array1<f64>>();

    // Print dimensions for debugging.
    println!("Train features shape: {:?}", train_features_array.shape());
    println!("Train target shape: {:?}", train_target_array.shape());

    // Initialize the model
    println!("Initializing model");
    let n_features = train_features_array.shape()[0];
    let mut model = Model::new(n_features);

    // Train the model
    model.train(train_features_array, train_target_array, 0.05, 10000);

    // Test the model
    let accuracy = model.accuracy(test_features_array, test_target_array);
    println!("Accuracy: {:.2}", accuracy); 

    Ok(())
}
