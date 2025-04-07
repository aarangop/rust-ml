use ndarray::{arr1, arr2};
use rust_ml::builders::builder::Builder;
use rust_ml::model::linear_regression::LinearRegression;
use rust_ml::optimization::gradient_descent::GradientDescent;
use rust_ml::optimization::optimizer::Optimizer;

fn main() {
    let mut model = LinearRegression::builder()
        .n_input_features(3)
        .build()
        .unwrap();

    let mut gd = GradientDescent::new(0.01, 1000);

    let x = arr2(&[[1., 2., 3.], [1., 5., 9.], [1., 8., 7.]]);
    let y = arr1(&[5., 6., 7.]);

    println!("x.shape: {:?}", x.shape());
    println!("y.shape: {:?}", y.shape());
    println!("x: \n{:?}", x);
    println!("y: \n{:?}", y);

    let cost = model.compute_cost(&x, &y).unwrap();

    println!("untrained cost: \n{:?}", cost);

    // Train the model
    gd.fit(&mut *model, &x, &y).unwrap();

    let cost = model.compute_cost(&x, &y).unwrap();
    println!("trained cost: \n{:?}", cost);

    let y_hat = model.predict(&x).unwrap();
    println!("y_hat: \n{:?}", y_hat);
    println!("y: \n{:?}", y);
}
