use ndarray::{Array, Array0, Array1, Array2, ArrayView2, IxDyn};
use std::collections::HashMap;

pub type ParamsHashmap = HashMap<String, Array<f64, IxDyn>>;
pub type Matrix = Array2<f64>;
pub type MatrixView<'a> = ArrayView2<'a, f64>;
pub type Vector = Array1<f64>;
pub type VectorView = ArrayView2<'static, f64>;
pub type Scalar = Array0<f64>;
pub type ScalarView = ArrayView2<'static, f64>;
