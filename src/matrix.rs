use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "matrisedata har {} verdier, forventet {rows} rader * {cols} kolonner = {}",
            data.len(),
            rows * cols
        );
        Self { data, rows, cols }
    }
}

impl Deref for Matrix {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.data
    }
}

impl DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_creates_a_matrix_of_the_requested_shape() {
        let matrix = Matrix::zeros(2, 3);

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.data, vec![0.0; 6]);
    }

    #[test]
    fn from_vec_keeps_the_provided_data_and_shape() {
        let matrix = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix[2], 3.0);
    }

    #[test]
    #[should_panic]
    fn from_vec_panics_when_data_does_not_match_shape() {
        Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn derefs_to_a_slice_for_indexing_and_iteration() {
        let mut matrix = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        matrix[0] = 9.0;

        assert_eq!(matrix.iter().sum::<f32>(), 9.0 + 2.0 + 3.0 + 4.0);
    }
}
