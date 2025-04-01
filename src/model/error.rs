
#[derive(Debug)]
pub struct DimensionsError {
  pub shape1: Vec<usize>,
  pub shape2: Vec<usize>,
}

#[derive(Debug)]
pub enum ModelError {
  Dimensions(DimensionsError),
  // Add more error variants as needed
}

impl std::fmt::Display for DimensionsError {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "Incompatible dimensions {:?}, and {:?}", self.shape1, self.shape2)
  }
}

impl std::fmt::Display for ModelError {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    match self {
      ModelError::Dimensions(err) => write!(f, "{}", err),
      // Handle other variants
    }
  }
}

impl std::error::Error for ModelError {}

impl DimensionsError {
  pub fn new(shape1: Vec<usize>, shape2: Vec<usize>) -> Self {
    Self {
      shape1,
      shape2
    }
  }
}

impl From<DimensionsError> for ModelError {
  fn from(error: DimensionsError) -> Self {
    ModelError::Dimensions(error)
  }
}
#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_dimensions_error_creation() {
    let shape1 = vec![2, 3];
    let shape2 = vec![4, 5];
    let error = DimensionsError::new(shape1.clone(), shape2.clone());
    
    assert_eq!(error.shape1, shape1);
    assert_eq!(error.shape2, shape2);
  }

  #[test]
  fn test_dimensions_error_display() {
    let error = DimensionsError::new(vec![2, 3], vec![4, 5]);
    let message = format!("{}", error);

    assert!(message.contains("Incompatible dimensions"));
    assert!(message.contains("[2, 3]"));
    assert!(message.contains("[4, 5]"));
  }

  #[test]
  fn test_model_error_from_dimensions_error() {
    let dim_error = DimensionsError::new(vec![1, 2], vec![3, 4]);
    let model_error = ModelError::from(dim_error);
    
    match model_error {
      ModelError::Dimensions(err) => {
        assert_eq!(err.shape1, vec![1, 2]);
        assert_eq!(err.shape2, vec![3, 4]);
      }
      _ => panic!("Expected ModelError::Dimensions variant"),
    }
  }

  #[test]
  fn test_model_error_display() {
    let dim_error = DimensionsError::new(vec![10, 5], vec![5, 2]);
    let model_error = ModelError::from(dim_error);
    let message = format!("{}", model_error);
    
    assert!(message.contains("Incompatible dimensions"));
    assert!(message.contains("[10, 5]"));
    assert!(message.contains("[5, 2]"));
  }
}
