use ndarray::{Array, Dimension};

pub trait Activation<D>
where
    D: Dimension,
{
    fn activate(z: &Array<f64, D>) -> Array<f64, D>;
    fn derivative(z: &Array<f64, D>) -> Array<f64, D>;
}
