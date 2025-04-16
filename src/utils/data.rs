use std::path::PathBuf;

use ndarray::{Axis, Ix1};
use ndarray_rand::rand::{seq::SliceRandom, SeedableRng};
use polars::prelude::*;

use crate::core::types::{Matrix, Vector};

pub fn load_dataset(path: PathBuf) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(path))?
        .finish()
}

#[cfg(test)]
mod load_dataset_tests {
    use std::path::PathBuf;

    use crate::utils::data::load_dataset;

    #[test]
    fn test_load_dataset_existing_path() {
        let path = PathBuf::from("./datasets/advertising.csv");
        println!("path exists: {:?}", path.exists());
        let df = load_dataset(path);
        assert!(df.is_ok());
    }

    #[test]
    fn test_load_dataset_with_non_existing_path() {
        let path = PathBuf::from("./data/non_existing.csv");
        let df = load_dataset(path);
        assert!(df.is_err());
    }
}

pub fn shuffle_split(
    x: &Matrix,
    y: &Vector,
    train_perc: f64,
    seed: i32,
) -> (Matrix, Vector, Matrix, Vector) {
    // Create a seedable range and use the provided seed
    let mut rng = ndarray_rand::rand::rngs::StdRng::seed_from_u64(seed as u64);

    // Shuffle the indices of the dataset
    let n_samples = x.nrows();
    let indices: Vec<usize> = (0..n_samples).collect();
    let shuffled_indices: Vec<usize> = indices
        .choose_multiple(&mut rng, n_samples)
        .cloned()
        .collect();

    // Calculate the split index
    let split_index = (n_samples as f64 * train_perc).round() as usize;

    // Split the dataset into training and testing sets
    let x_train = x.select(Axis(0), &shuffled_indices[..split_index]);
    let y_train = y.select(Axis(0), &shuffled_indices[..split_index]);
    let x_test = x.select(Axis(0), &shuffled_indices[split_index..]);
    let y_test = y.select(Axis(0), &shuffled_indices[split_index..]);

    (x_train, y_train, x_test, y_test)
}

#[cfg(test)]
mod shuffle_split_tests {
    use crate::utils::data::shuffle_split;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_shuffle_split_train_test_ratio() {
        // x is a (10, 2) matrix
        let x = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
        ]);
        // y is a (10, ) vector
        let y = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        // A shuffle split with 75% training data should return 7 values for the train sets and 3 for the test sets.
        let (x_train, y_train, x_test, y_test) = shuffle_split(&x, &y, 0.7, 42);

        assert_eq!(x_train.nrows(), 7);
        assert_eq!(y_train.len(), 7);
        assert_eq!(x_test.nrows(), 3);
        assert_eq!(y_test.len(), 3);
    }

    #[test]
    fn test_shuffle_split_returns_sets_in_random_order() {
        // Create a sample dataset
        let x = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
        ]);
        let y = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        // Split the dataset using two different seeds
        let (x_train_1, y_train_1, _, _) = shuffle_split(&x, &y, 0.7, 42);
        let (x_train_2, y_train_2, _, _) = shuffle_split(&x, &y, 0.7, 100);

        // Check that the training sets are different, which indicates shuffling occurred
        let mut sets_are_different = false;

        // Compare each row in the training sets to see if they're different
        for i in 0..x_train_1.nrows() {
            if x_train_1.row(i) != x_train_2.row(i) {
                sets_are_different = true;
                break;
            }
        }

        // Same size training sets should have been created with different content
        assert_eq!(x_train_1.nrows(), 7);
        assert_eq!(x_train_2.nrows(), 7);
        assert_eq!(y_train_1.len(), 7);
        assert_eq!(y_train_2.len(), 7);
        assert!(
            sets_are_different,
            "Training sets should be different when using different seeds"
        );
    }
}

pub fn get_features_and_target(df:&DataFrame, features: Vec<&str>, target: &str) -> PolarsResult<(Matrix, Vector)> {
    let x = df
        .select(features.into_iter())
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap();
    let y = df
        .select([target])
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::Fortran)
        .unwrap()
        .column(0)
        .to_owned()
        .into_dimensionality::<Ix1>()
        .unwrap();

    Ok((x, y))
}

#[cfg(test)]
mod get_features_and_target_tests {
    use std::path::PathBuf;
    use crate::utils::data::{load_dataset, get_features_and_target};

    #[test]
    fn test_get_features_and_target() {
        let path = PathBuf::from("./datasets/advertising.csv");
        let df = load_dataset(path).unwrap();
        let features = vec!["TV", "Radio", "Newspaper"];
        let target = "Sales";

        let (x, y) = get_features_and_target(&df, features, target).unwrap();

        assert_eq!(x.nrows(), 200);
        assert_eq!(x.ncols(), 3);
        assert_eq!(y.len(), 200);
    }
}