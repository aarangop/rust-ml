use crate::core::error::ModelError;
use crate::core::types::{Matrix, Vector};
use crate::model::core::base::DLModel;
use crate::optim::core::state::OptimizerState;
use ndarray::{Array, ArrayView, ArrayViewMut, IxDyn};

pub struct GradientDescentState<Input, Output, M: DLModel<Input, Output>> {
    learning_rate: f64,
    _phantom_model: std::marker::PhantomData<M>,
    _phantom_input: std::marker::PhantomData<Input>,
    _phantom_output: std::marker::PhantomData<Output>,
}

impl<Input, Output, M: DLModel<Input, Output>> GradientDescentState<Input, Output, M> {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            _phantom_model: std::marker::PhantomData,
            _phantom_input: std::marker::PhantomData,
            _phantom_output: std::marker::PhantomData,
        }
    }

    // Helper function to update a parameter with gradient descent
    fn gradient_descent_update(
        &self,
        param: ArrayView<f64, ndarray::IxDyn>,
        grad: ArrayView<f64, ndarray::IxDyn>,
    ) -> Result<Array<f64, ndarray::IxDyn>, ModelError> {
        // Create a new owned array for the updated parameter
        let mut new_param = param.to_owned();

        // Apply gradient descent update: param = param - learning_rate * gradient
        new_param -= &(&grad * self.learning_rate);

        Ok(new_param)
    }
}

impl<M: DLModel<Matrix, Vector>> OptimizerState<Matrix, Vector, M>
    for GradientDescentState<Matrix, Vector, M> {
        fn update_weights(&mut self, model: &mut M) -> Result<(), ModelError> {
            // Collect all parameter updates first
            let mut updates = Vec::new();
            
            // Iterate over all parameters and calculate updates
            for (key, param_view) in model.param_iter() {
                // Get the corresponding gradient view
                let grad_view: ArrayView<f64, IxDyn> = model.get_gradient(key)?;
                
                // Check the dimensions of the parameter
                if param_view.ndim() != grad_view.ndim() {
                    return Err(ModelError::DimensionalityError("Parameter and gradient dimensions do not match".to_string()));
                }
                // Try to convert both parameter and gradient to dynamic dimensions
                let param_view = param_view
                    .into_dimensionality::<ndarray::IxDyn>()
                    .map_err(|_| ModelError::DimensionalityError("Failed to convert parameter to dynamic dimensions".to_string()))?;
                let grad_view = grad_view
                    .into_dimensionality::<ndarray::IxDyn>()
                    .map_err(|_| ModelError::DimensionalityError("Failed to convert gradient to dynamic dimensions".to_string()))?;
    
                // Calculate the updated parameter using the gradient descent update rule
                let updated_param = &param_view - (self.learning_rate * &grad_view);
                updates.push((key.to_string(), updated_param.to_owned()));
            }
            
            // Apply all updates after the loop
            for (key, updated_param) in updates {
                let mut current_param: ArrayViewMut<f64, IxDyn> = model.get_mut(&key)?;
                current_param.assign(&updated_param);
            }
    
            Ok(())
    }
}
