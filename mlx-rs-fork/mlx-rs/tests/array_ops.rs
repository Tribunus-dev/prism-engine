use mlx_rs::{Array, ArrayOps};

#[test]
fn test_array_ops_reshape() {
    let a = Array::zeros::<f32>(&[2, 3]).unwrap();
    let reshaped = ArrayOps::reshape(&a, &[3, 2]);
    assert_eq!(reshaped.shape(), &[3, 2]);
}

#[test]
fn test_array_ops_transpose() {
    let a = Array::zeros::<f32>(&[2, 3, 4]).unwrap();
    let transposed = ArrayOps::transpose(&a, &[2, 0, 1]);
    assert_eq!(transposed.shape(), &[4, 2, 3]);
}

#[test]
fn test_array_ops_slice() {
    let a = Array::zeros::<f32>(&[10, 10]).unwrap();
    let sliced = ArrayOps::slice(&a, &[2, 2], &[8, 8], &[2, 2]);
    assert_eq!(sliced.shape(), &[3, 3]);
}
