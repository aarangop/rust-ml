## Overview

Rust-ML is a practice project that aims to combine Rust programming skills with machine learning concepts. The project
leverages Rust's performance characteristics, memory safety guarantees, and expressive type system to build reliable
machine learning tools while providing an opportunity to deepen understanding of both Rust and machine learning
algorithms.

### Objectives

- Implement common machine learning algorithms in pure Rust
- Provide high-performance implementations suitable for production use
- Ensure memory safety and thread safety through Rust's ownership model
- Create an ergonomic API that is easy to use for both ML beginners and experts
- Support integration with existing data processing pipelines

## Project Structure

The project is organized into several key modules:

### Core (`src/core/`)

Contains fundamental data structures and utilities:

- `types.rs`: Defines common types used throughout the library like `ModelParams`
- `param_storage.rs`: Provides storage mechanisms for model parameters
- `param_manager.rs`: Manages parameter operations and transformations
- `error.rs`: Defines error types and handling for the core module

### Model (`src/model/`)

Houses machine learning model implementations:

- `core/`: Contains base traits and interfaces for models:
    - `base.rs`: Defines base model traits
    - `optimizable_model.rs`: Trait for models that can be optimized
    - `regression_model.rs`: Trait for regression models
    - `classification_model.rs`: Trait for classification models

- `linear_regression.rs`: Implementation of linear regression model
- Other model implementations

### Optimization (`src/optimization/`)

Provides optimization algorithms for training models:

- `core/`:
    - `optimizer.rs`: Defines the Optimizer trait and related interfaces

- `gradient_descent.rs`: Implementation of gradient descent optimization

### Benchmarking (`src/bench/`)

Tools for profiling and evaluating model performance:

- `core/`:
    - `profiler.rs`: Base trait for profiling model training and evaluation
    - `error.rs`: Error handling for profiling operations
    - `train_metrics.rs`: Defines metrics for training evaluation

- `regression_profiler.rs`: Profiler for regression models
- `regression_metrics.rs`: Metrics specific to regression models

### Builders (`src/builders/`)

Factory patterns for creating and configuring models:

- `builder.rs`: Defines builder interfaces
- `linear_regression.rs`: Builder for linear regression models

### Examples (`src/bin/`)

Executable examples demonstrating library usage:

- `linear_regression.rs`: Example usage of linear regression
- `sl_classifier.rs`: Example of supervised learning classifier

## Error Handling

The library employs a robust error handling strategy using Rust's `Result` type and the `thiserror` crate:

- Domain-specific error types are defined for each module
- Error conversion is facilitated with the `From` trait
- The `?` operator is used for ergonomic error propagation

This approach ensures clear error messages and type-safe error handling throughout the codebase.

## Getting Started

### Prerequisites

- Rust (stable version 1.86.0 or higher recommended)
- Cargo (comes with Rust installation)

### Installation

Add Rust-ML to your project by including it in your `Cargo.toml`:

``` toml
[dependencies]
rust-ml = "0.1.0"
```

Or clone the repository to contribute or run examples:

``` bash
git clone https://github.com/username/rust-ml.git
cd rust-ml
cargo build
```

### Running Tests

``` bash
cargo test
```

### Running Benchmarks

``` bash
cargo bench
```

## Example Usage

### Linear Regression

``` rust
use rust_ml::model::linear_regression::LinearRegression;
use rust_ml::model::ml_model::RegressionModel;
use rust_ml::optimization::gradient_descent::GradientDescent;
use rust_ml::optimization::optimizer::Optimizer;
use ndarray::{Array1, Array2};

fn main() {
    // Create sample data
    let x_train = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]).unwrap();
    let y_train = Array1::from_vec(vec![6.0, 9.0, 12.0]);

    // Initialize model and optimizer
    let mut model = LinearRegression::new(2); // 2 features
    let mut optimizer = GradientDescent::new(0.01, 1000); // learning rate and iterations
    
    // Train model
    optimizer.fit(&mut model, &x_train, &y_train).expect("Training failed");

    // Make predictions
    let x_test = Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap();
    let predictions = model.predict(&x_test);

    println!("Prediction: {:?}", predictions);
}
```

## Profiling Models

The library provides profiling capabilities to measure model performance:

``` rust
use rust_ml::bench::regression_profiler::RegressionProfiler;
use rust_ml::bench::profiler::Profiler;
use rust_ml::model::linear_regression::LinearRegression;
use rust_ml::optimization::gradient_descent::GradientDescent;

fn main() {
    // Initialize data, model and optimizer
    // [...]
    
    // Create profiler
    let profiler = RegressionProfiler::<LinearRegression, GradientDescent, _, _>::new();
    
    // Profile training
    let (train_metrics, eval_metrics) = profiler
        .profile_training(&mut model, &mut optimizer, &x_train, &y_train)
        .expect("Profiling failed");
        
    println!("Training metrics: {:?}", train_metrics);
    println!("Evaluation metrics: {:?}", eval_metrics);
}
```

## Dependencies

The project relies on several high-quality Rust crates:

- `thiserror`: For ergonomic error handling
- `ndarray`: For efficient numerical computations
- `ndarray-rand`: For random matrix generation
- `polars`: For data manipulation
- `rand`: For random number generation
- `approx`: For approximate floating-point comparisons

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request