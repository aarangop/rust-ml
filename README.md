# Rust-ML

A machine learning library implemented in Rust, focusing on performance, safety,
and ergonomic API design.

## Overview

Rust-ML is a practice project that aims to combine Rust programming skills with
machine learning concepts. The project leverages Rust's performance
characteristics, memory safety guarantees, and expressive type system to build
reliable machine learning tools while providing an opportunity to deepen
understanding of both Rust and machine learning algorithms.

### Objectives

- Implement common machine learning algorithms in pure Rust
- Provide high-performance implementations suitable for production use
- Ensure memory safety and thread safety through Rust's ownership model
- Create an ergonomic API that is easy to use for both ML beginners and experts
- Support integration with existing data processing pipelines

## Project Structure

The crate is structured as a library that exposes different machine learning
models. Currently, only logistic regression and linear regression models are
implemented. The project also provides different entry points as examples to
demonstrate how to use these models in practice.

Additionally, there's a notebooks folder where data analysis is performed using
Python, allowing for more familiar exploratory data analysis and providing a way
to benchmark performance between the Python and Rust implementations.

## Getting Started

### Prerequisites

- Rust (stable version 1.56 or higher recommended)
- Cargo (comes with Rust installation)

### Installation

Add Rust-ML to your project by including it in your `Cargo.toml`:

```toml
[dependencies]
rust-ml = "0.1.0"
```

Or clone the repository to contribute or run examples:

```bash
git clone https://github.com/username/rust-ml.git
cd rust-ml
cargo build
```

### Running Tests

```bash
cargo test
```

### Running Benchmarks

```bash
cargo bench
```

## Example Usage

### Linear Regression

```rust
use rust_ml::algorithms::linear::LinearRegression;
use rust_ml::preprocessing::StandardScaler;

fn main() {
    // Create sample data
    let x_train = vec![
        vec![1.0, 2.0],
        vec![2.0, 3.0],
        vec![3.0, 4.0],
    ];
    let y_train = vec![6.0, 9.0, 12.0];

    // Standardize features
    let mut scaler = StandardScaler::new();
    let x_scaled = scaler.fit_transform(&x_train);

    // Train model
    let mut model = LinearRegression::new();
    model.fit(&x_scaled, &y_train);

    // Make predictions
    let x_test = vec![vec![4.0, 5.0]];
    let x_test_scaled = scaler.transform(&x_test);
    let predictions = model.predict(&x_test_scaled);

    println!("Prediction: {:?}", predictions);
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for
details.
